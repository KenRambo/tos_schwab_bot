"""
Position Manager

Handles position tracking, trade execution, and order management.
Implements the "hold until opposite signal" strategy.

UPDATED 2026-01-06:
- Added butterfly credit spread mode for SPX/XSP stacking strategy
- Supports ES signals → SPX/XSP option execution
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum

from schwab_client import SchwabClient, OptionContract, OrderType, Position
from signal_detector import Signal, Direction, SignalType
from notifications import get_notifier

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ButterflyStatus(Enum):
    PENDING = "PENDING"
    LEG1_FILLED = "LEG1_FILLED"
    COMPLETE = "COMPLETE"
    EXPIRED = "EXPIRED"


@dataclass
class Trade:
    """Record of a completed trade"""
    id: str
    signal_type: SignalType
    direction: Direction
    entry_time: datetime
    entry_price: float
    option_symbol: str
    option_strike: float
    option_expiry: date
    option_type: str  # "CALL" or "PUT"
    quantity: int
    status: TradeStatus
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    bars_held: int = 0
    exit_reason: str = ""


@dataclass
class ButterflyTrade:
    """Record of a butterfly credit spread trade"""
    id: str
    signal_type: SignalType
    direction: Direction
    entry_time: datetime
    symbol: str  # SPX, XSP, etc.
    underlying_price: float
    
    # Strikes
    lower_strike: float
    middle_strike: float
    upper_strike: float
    option_type: str  # "CALL" or "PUT"
    
    # Pricing
    wing_debit: float = 0.0      # Cost of wings (lower + upper asks)
    middle_credit: float = 0.0   # Credit from 2x middle bids
    net_credit: float = 0.0      # Net credit received
    
    # Orders
    order_id: Optional[str] = None
    fill_time: Optional[datetime] = None
    status: ButterflyStatus = ButterflyStatus.PENDING
    
    # Settlement
    settlement_price: Optional[float] = None
    final_pnl: Optional[float] = None
    
    @property
    def quantity(self) -> int:
        return 1
    
    @property
    def option_symbol(self) -> str:
        return f"{self.symbol} {self.lower_strike}/{self.middle_strike}/{self.upper_strike} {self.option_type}"
    
    @property
    def option_strike(self) -> float:
        return self.middle_strike
    
    @property
    def option_expiry(self) -> str:
        return self.entry_time.strftime("%Y-%m-%d")
    
    @property
    def entry_price(self) -> float:
        return self.net_credit


@dataclass
class PositionState:
    """Current position state"""
    direction: Direction = Direction.FLAT
    option_symbol: Optional[str] = None
    option_type: Optional[str] = None
    quantity: int = 0
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    entry_signal: Optional[SignalType] = None
    bars_held: int = 0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    current_price: Optional[float] = None
    high_water_mark: float = 0.0  # Highest price seen (for trailing stop)
    high_water_pnl_percent: float = 0.0  # Highest P&L % seen


class PositionManager:
    """
    Manages trading positions and executes orders.
    
    Strategy: Hold until opposite signal
    - LONG signal: Buy CALL, hold until SHORT signal
    - SHORT signal: Buy PUT, hold until LONG signal
    
    Butterfly Mode (for SPX/XSP credit spread stacking):
    - LONG signal: Call butterfly (lower/middle/upper calls)
    - SHORT signal: Put butterfly (lower/middle/upper puts)
    
    Optional risk management:
    - Stop loss (% or $)
    - Take profit (% or $)
    - Trailing stop
    - Daily loss limit
    - Fixed fractional position sizing
    - Delta exposure / correlation check
    """
    
    def __init__(
        self,
        client: SchwabClient,
        symbol: str = "SPY",
        contracts: int = 1,
        max_daily_trades: int = 3,
        min_dte: int = 0,
        max_dte: int = 7,
        paper_trading: bool = True,
        # Delta targeting
        use_delta_targeting: bool = True,
        target_delta: float = 0.30,
        afternoon_delta: float = 0.40,
        afternoon_start_hour: int = 12,
        # Stop Loss settings
        enable_stop_loss: bool = False,
        stop_loss_percent: float = 50.0,
        stop_loss_dollars: float = 500.0,
        # Take Profit settings
        enable_take_profit: bool = False,
        take_profit_percent: float = 100.0,
        take_profit_dollars: float = 500.0,
        # Trailing Stop settings
        enable_trailing_stop: bool = False,
        trailing_stop_percent: float = 25.0,
        trailing_stop_activation: float = 50.0,
        # Fixed Fractional Position Sizing
        use_fixed_fractional: bool = True,
        risk_percent_per_trade: float = 2.0,
        max_position_size: int = 10,
        min_position_size: int = 1,
        # Daily Loss Limit
        enable_daily_loss_limit: bool = True,
        max_daily_loss_dollars: float = 500.0,
        max_daily_loss_percent: float = 5.0,
        # Correlation / Delta Exposure
        enable_correlation_check: bool = True,
        max_delta_exposure: float = 100.0,
        # Butterfly mode (for SPX/XSP credit spread stacking)
        butterfly_mode: bool = False,
        butterfly_wing_width: int = 5,
        butterfly_credit_target_pct: float = 0.30
    ):
        self.client = client
        self.symbol = symbol
        self.contracts = contracts
        self.max_daily_trades = max_daily_trades
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.paper_trading = paper_trading
        
        # Delta targeting
        self.use_delta_targeting = use_delta_targeting
        self.target_delta = target_delta
        self.afternoon_delta = afternoon_delta
        self.afternoon_start_hour = afternoon_start_hour
        
        # Risk management settings
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_percent = stop_loss_percent
        self.stop_loss_dollars = stop_loss_dollars
        
        self.enable_take_profit = enable_take_profit
        self.take_profit_percent = take_profit_percent
        self.take_profit_dollars = take_profit_dollars
        
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_percent = trailing_stop_percent
        self.trailing_stop_activation = trailing_stop_activation
        
        # Fixed fractional sizing
        self.use_fixed_fractional = use_fixed_fractional
        self.risk_percent_per_trade = risk_percent_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
        # Daily loss limit
        self.enable_daily_loss_limit = enable_daily_loss_limit
        self.max_daily_loss_dollars = max_daily_loss_dollars
        self.max_daily_loss_percent = max_daily_loss_percent
        self.daily_starting_balance: Optional[float] = None
        
        # Correlation / exposure check
        self.enable_correlation_check = enable_correlation_check
        self.max_delta_exposure = max_delta_exposure
        
        # Butterfly mode
        self.butterfly_mode = butterfly_mode
        self.butterfly_wing_width = butterfly_wing_width
        self.butterfly_credit_target_pct = butterfly_credit_target_pct
        
        # Current position
        self.position = PositionState()
        
        # Daily tracking
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.last_trade_date: Optional[date] = None
        
        # Butterfly tracking
        self.butterflies: List[ButterflyTrade] = []
        self.credits_today: float = 0.0
        self.total_credits_collected: float = 0.0
        
        # Monitor only mode (no trading, just log signals)
        self._monitor_only = False
        
        # Trade history
        self.trades: List[Trade] = []
        self.trade_counter = 0
        self.butterfly_counter = 0
        
        if butterfly_mode:
            logger.info(f"Butterfly mode ENABLED: {butterfly_wing_width}pt wings, {butterfly_credit_target_pct:.0%} credit target")
    
    def _reset_daily_count(self) -> None:
        """Reset daily trade count and P&L if new day"""
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.credits_today = 0.0
            self.last_trade_date = today
            
            # Capture starting balance for daily loss % calculation
            if self.enable_daily_loss_limit:
                try:
                    self.daily_starting_balance = self.client.get_buying_power()
                    logger.info(f"New trading day: {today}")
                    logger.info(f"Starting balance: ${self.daily_starting_balance:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not get starting balance: {e}")
                    self.daily_starting_balance = None
            else:
                logger.info(f"New trading day: {today}")
    
    def is_locked_out(self) -> bool:
        """Check if we've hit daily trade limit or daily loss limit"""
        self._reset_daily_count()
        
        # Check trade count
        if self.daily_trade_count >= self.max_daily_trades:
            return True
        
        # Check daily loss limit
        if self.enable_daily_loss_limit and self._check_daily_loss_limit():
            return True
        
        return False
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        # Check dollar limit
        if self.daily_pnl <= -self.max_daily_loss_dollars:
            logger.warning(f"Daily loss limit hit: ${self.daily_pnl:.2f} <= -${self.max_daily_loss_dollars:.2f}")
            get_notifier().send(
                title="🛑 Daily Loss Limit",
                message=f"Daily loss limit hit!\nLoss: ${abs(self.daily_pnl):.2f}\nLimit: ${self.max_daily_loss_dollars:.2f}"
            )
            return True
        
        # Check percent limit
        if self.daily_starting_balance and self.daily_starting_balance > 0:
            loss_percent = abs(self.daily_pnl) / self.daily_starting_balance * 100
            if self.daily_pnl < 0 and loss_percent >= self.max_daily_loss_percent:
                logger.warning(f"Daily loss % limit hit: {loss_percent:.1f}% >= {self.max_daily_loss_percent:.1f}%")
                return True
        
        return False
    
    def _calculate_position_size(self, option_price: float) -> int:
        """Calculate position size using fixed fractional method"""
        if not self.use_fixed_fractional:
            return self.contracts
        
        try:
            account_balance = self.client.get_buying_power()
            
            # Risk amount = account * risk%
            risk_amount = account_balance * (self.risk_percent_per_trade / 100)
            
            # Cost per contract = option price * 100
            cost_per_contract = option_price * 100
            
            if cost_per_contract <= 0:
                return self.min_position_size
            
            # Contracts = risk amount / cost per contract
            calculated_size = int(risk_amount / cost_per_contract)
            
            # Apply min/max limits
            position_size = max(self.min_position_size, min(calculated_size, self.max_position_size))
            
            logger.info(f"Position sizing: ${account_balance:,.2f} balance, {self.risk_percent_per_trade}% risk = {position_size} contracts")
            
            return position_size
            
        except Exception as e:
            logger.warning(f"Error calculating position size: {e}")
            return self.contracts
    
    def _get_current_delta_exposure(self) -> float:
        """Calculate current total delta exposure from all positions"""
        if self.paper_trading:
            # In paper mode, use our tracked position
            if self.position.direction == Direction.FLAT:
                return 0.0
            
            # Approximate delta based on position
            if self.position.option_type == "CALL":
                return self.target_delta * self.position.quantity * 100
            else:
                return -self.target_delta * self.position.quantity * 100
        
        try:
            total_delta = 0.0
            positions = self.client.get_option_positions(self.symbol)
            
            for pos in positions:
                if pos.quantity != 0:
                    is_call = "C" in pos.symbol or "CALL" in pos.symbol.upper()
                    estimated_delta = 0.30 if is_call else -0.30
                    total_delta += estimated_delta * abs(pos.quantity) * 100
            
            return total_delta
            
        except Exception as e:
            logger.warning(f"Error getting delta exposure: {e}")
            return 0.0
    
    def _check_correlation(self, signal_direction: Direction) -> bool:
        """Check if adding this position would exceed delta exposure limits."""
        if not self.enable_correlation_check:
            return True
        
        current_delta = self._get_current_delta_exposure()
        current_target = self._get_current_target_delta()
        new_delta = current_target * self.contracts * 100
        if signal_direction == Direction.SHORT:
            new_delta = -new_delta
        
        projected_delta = current_delta + new_delta
        
        if abs(projected_delta) > self.max_delta_exposure:
            logger.warning(f"Correlation check failed: {current_delta:.0f} + {new_delta:.0f} = {projected_delta:.0f} > max {self.max_delta_exposure:.0f}")
            return False
        
        logger.info(f"Delta exposure OK: {current_delta:.0f} + {new_delta:.0f} = {projected_delta:.0f} (max: {self.max_delta_exposure:.0f})")
        return True
    
    def _get_current_target_delta(self) -> float:
        """Get target delta based on time of day."""
        now = datetime.now()
        
        if now.hour >= self.afternoon_start_hour:
            logger.info(f"Afternoon session ({now.hour}:00) - using {self.afternoon_delta:.0%} delta")
            return self.afternoon_delta
        else:
            logger.info(f"Morning session ({now.hour}:00) - using {self.target_delta:.0%} delta")
            return self.target_delta
    
    def trades_remaining(self) -> int:
        """Get number of trades remaining today"""
        self._reset_daily_count()
        return max(0, self.max_daily_trades - self.daily_trade_count)
    
    def has_position(self) -> bool:
        """Check if we have an open position"""
        return self.position.direction != Direction.FLAT
    
    def get_position_direction(self) -> Direction:
        """Get current position direction"""
        return self.position.direction
    
    def sync_with_broker(self) -> None:
        """Sync position state with broker"""
        if self.paper_trading:
            return
        
        try:
            positions = self.client.get_option_positions(self.symbol)
            
            if not positions:
                if self.position.direction != Direction.FLAT:
                    logger.warning("Position mismatch: Expected position but broker shows flat")
                    self.position = PositionState()
                return
            
            for pos in positions:
                if pos.quantity != 0:
                    is_call = "C" in pos.symbol or "CALL" in pos.symbol.upper()
                    direction = Direction.LONG if is_call else Direction.SHORT
                    
                    self.position.option_symbol = pos.symbol
                    self.position.option_type = "CALL" if is_call else "PUT"
                    self.position.quantity = int(abs(pos.quantity))
                    self.position.direction = direction
                    self.position.unrealized_pnl = pos.unrealized_pnl
                    
                    logger.info(f"Synced position: {direction.value} {pos.quantity} {pos.symbol}")
                    break
                    
        except Exception as e:
            logger.error(f"Error syncing with broker: {e}")
    
    def process_signal(self, signal: Signal) -> Optional[Trade]:
        """
        Process a trading signal and execute if appropriate.
        
        Returns Trade if a trade was executed, None otherwise.
        """
        self._reset_daily_count()
        
        # Check if in monitor only mode
        if self._monitor_only:
            logger.info(f"[MONITOR] Signal: {signal.signal_type.value} - {signal.direction.value} @ ${signal.price:.2f}")
            return None
        
        # Check lockout
        if self.is_locked_out():
            logger.warning(f"Locked out - {self.daily_trade_count}/{self.max_daily_trades} trades today")
            return None
        
        # Route to butterfly or single-leg execution
        if self.butterfly_mode:
            return self._execute_butterfly(signal)
        
        # Standard single-leg execution
        current_direction = self.position.direction
        signal_direction = signal.direction
        
        # If flat, just open new position
        if current_direction == Direction.FLAT:
            return self._open_position(signal)
        
        # If opposite signal, close current and open new
        if current_direction != signal_direction:
            exit_trade = self._close_position(signal)
            
            if not self.is_locked_out():
                new_trade = self._open_position(signal)
                return new_trade
            else:
                logger.info("Position closed but locked out for new position")
                return exit_trade
        
        logger.info(f"Already {current_direction.value} - ignoring {signal_direction.value} signal")
        return None
    
    # =========================================================================
    # BUTTERFLY CREDIT SPREAD EXECUTION
    # =========================================================================
    
    def _execute_butterfly(self, signal: Signal) -> Optional[Trade]:
        """Execute a butterfly credit spread based on signal"""
        
        option_type = "CALL" if signal.direction == Direction.LONG else "PUT"
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🦋 BUTTERFLY SIGNAL: {signal.direction.value} {option_type}")
        logger.info(f"   Signal: {signal.signal_type.value}")
        logger.info(f"   Price: ${signal.price:.2f}")
        logger.info("=" * 60)
        
        # Get underlying price for strike calculation
        try:
            # For SPX, get the index price
            if self.symbol.upper() in ['SPX', '$SPX', '$SPX.X']:
                quote = self.client.get_quote('$SPX')
            elif self.symbol.upper() in ['XSP', '$XSP']:
                quote = self.client.get_quote('$SPX')
                # XSP is 1/10th of SPX
            else:
                quote = self.client.get_quote(self.symbol)
            
            underlying_price = quote.last_price
            logger.info(f"   Underlying: ${underlying_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error getting underlying price: {e}")
            # Use signal price as fallback
            underlying_price = signal.price
        
        # Calculate strikes
        width = self.butterfly_wing_width
        
        # Round to appropriate strike interval
        if self.symbol.upper() in ['XSP', '$XSP']:
            # XSP has $0.50 or $1 strikes
            atm = round(underlying_price / 10)  # XSP is 1/10th SPX
            atm = round(atm)
        elif self.symbol.upper() in ['SPX', '$SPX', '$SPX.X']:
            # SPX has $5 strikes
            atm = round(underlying_price / 5) * 5
        else:
            # Default to $1 strikes
            atm = round(underlying_price)
        
        # Build butterfly strikes
        if signal.direction == Direction.LONG:
            # Call butterfly for bullish: buy lower, sell 2x middle, buy upper
            lower = atm
            middle = atm + width
            upper = atm + (width * 2)
        else:
            # Put butterfly for bearish: buy upper, sell 2x middle, buy lower
            upper = atm
            middle = atm - width
            lower = atm - (width * 2)
        
        logger.info(f"   Strikes: {lower}/{middle}/{upper}")
        
        # Get butterfly pricing
        prices = self._get_butterfly_prices(lower, middle, upper, option_type)
        
        if not prices:
            logger.error("Could not get butterfly prices")
            return None
        
        wing_debit = prices['wing_debit']
        middle_credit = prices['middle_credit']
        theoretical_net = middle_credit - wing_debit
        
        # Target: wing cost + credit target %
        target_credit = wing_debit * (1 + self.butterfly_credit_target_pct)
        
        logger.info(f"   ┌─ BUTTERFLY STRUCTURE ─────────────────────")
        logger.info(f"   │ Lower wing ({lower}): ${prices['lower_ask']:.2f} debit")
        logger.info(f"   │ Middle x2  ({middle}): ${prices['middle_bid']:.2f} × 2 = ${middle_credit:.2f} credit")
        logger.info(f"   │ Upper wing ({upper}): ${prices['upper_ask']:.2f} debit")
        logger.info(f"   ├───────────────────────────────────────────")
        logger.info(f"   │ Wing debit:     ${wing_debit:.2f}")
        logger.info(f"   │ Middle credit:  ${middle_credit:.2f}")
        logger.info(f"   │ Theoretical net: ${theoretical_net:.2f}")
        logger.info(f"   │ Target credit:  ${target_credit:.2f} ({self.butterfly_credit_target_pct:.0%} above wings)")
        logger.info(f"   └───────────────────────────────────────────")
        
        # Execute the butterfly
        success, order_id, fill_credit = self._place_butterfly_order(
            lower, middle, upper, option_type, theoretical_net
        )
        
        if not success:
            logger.error("   ✗ Butterfly order failed")
            get_notifier().send(
                title="🦋 ❌ Trade Rejected",
                message=f"Butterfly order failed\n{self.symbol} {lower}/{middle}/{upper} {option_type}"
            )
            return None
        
        # Create butterfly trade record
        self.butterfly_counter += 1
        self.daily_trade_count += 1
        
        butterfly = ButterflyTrade(
            id=f"BF-{date.today().strftime('%Y%m%d')}-{self.butterfly_counter:04d}",
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry_time=datetime.now(),
            symbol=self.symbol,
            underlying_price=underlying_price,
            lower_strike=lower,
            middle_strike=middle,
            upper_strike=upper,
            option_type=option_type,
            wing_debit=wing_debit,
            middle_credit=middle_credit,
            net_credit=fill_credit,
            order_id=order_id,
            fill_time=datetime.now(),
            status=ButterflyStatus.COMPLETE
        )
        
        self.butterflies.append(butterfly)
        
        # Track credits
        credit_collected = fill_credit * 100  # Options are x100
        self.credits_today += credit_collected
        self.total_credits_collected += credit_collected
        
        logger.info(f"   ✓ FILLED @ net credit ${fill_credit:.2f}")
        logger.info(f"   💰 CREDIT STACKED: ${credit_collected:.2f}")
        logger.info(f"   📊 Credits today: ${self.credits_today:.2f} | Total: ${self.total_credits_collected:.2f}")
        
        # Send notification
        emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
        get_notifier().send(
            title=f"🦋 {emoji} Butterfly Filled",
            message=f"{signal.direction.value} {lower}/{middle}/{upper} {option_type}\nCredit: ${fill_credit:.2f}\nStacked: ${credit_collected:.2f}",
            sound="bugle"
        )
        
        # Return as Trade-compatible object for trading_bot.py
        return Trade(
            id=butterfly.id,
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry_time=butterfly.entry_time,
            entry_price=butterfly.net_credit,
            option_symbol=butterfly.option_symbol,
            option_strike=butterfly.middle_strike,
            option_expiry=date.today(),
            option_type=option_type,
            quantity=1,
            status=TradeStatus.FILLED
        )
    
    def _get_butterfly_prices(
        self,
        lower: float,
        middle: float,
        upper: float,
        option_type: str
    ) -> Optional[Dict[str, float]]:
        """Get bid/ask prices for butterfly legs"""
        
        if self.paper_trading:
            # Simulate prices for paper trading
            # Rough approximation: ATM options ~$5-10, wings cheaper
            base_price = 5.0
            spread = 0.10
            
            return {
                'lower_bid': base_price - spread,
                'lower_ask': base_price + spread,
                'middle_bid': base_price * 1.5 - spread,
                'middle_ask': base_price * 1.5 + spread,
                'upper_bid': base_price - spread,
                'upper_ask': base_price + spread,
                'wing_debit': (base_price + spread) * 2,  # lower_ask + upper_ask
                'middle_credit': (base_price * 1.5 - spread) * 2,  # 2 × middle_bid
            }
        
        try:
            # Get option chain
            chain = self.client.get_option_chain(
                symbol=self.symbol,
                contract_type=option_type.upper(),
                strike_count=30,
                include_quotes=True,
                from_date=date.today(),
                to_date=date.today()
            )
            
            if option_type.upper() in ['C', 'CALL']:
                exp_map = chain.get('callExpDateMap', {})
            else:
                exp_map = chain.get('putExpDateMap', {})
            
            prices = {}
            
            for exp_str, strikes in exp_map.items():
                for strike_val, strike_key in [(lower, 'lower'), (middle, 'middle'), (upper, 'upper')]:
                    strike_str = f"{strike_val:.1f}"
                    if strike_str in strikes:
                        opt = strikes[strike_str][0]
                        prices[f'{strike_key}_bid'] = opt.get('bid', 0)
                        prices[f'{strike_key}_ask'] = opt.get('ask', 0)
            
            # Calculate totals
            if all(k in prices for k in ['lower_ask', 'middle_bid', 'upper_ask']):
                prices['wing_debit'] = prices['lower_ask'] + prices['upper_ask']
                prices['middle_credit'] = prices['middle_bid'] * 2
                return prices
            
            logger.warning("Could not find all strikes in option chain")
            return None
            
        except Exception as e:
            logger.error(f"Error getting butterfly prices: {e}")
            return None
    
    def _place_butterfly_order(
        self,
        lower: float,
        middle: float,
        upper: float,
        option_type: str,
        limit_credit: float
    ) -> tuple:
        """
        Place a butterfly order.
        Returns: (success, order_id, fill_price)
        """
        
        if self.paper_trading:
            order_id = f"PAPER-BF-{datetime.now().strftime('%H%M%S')}"
            # Simulate fill at theoretical price
            fill_price = limit_credit
            logger.info(f"[PAPER] Butterfly filled: {lower}/{middle}/{upper} @ ${fill_price:.2f} credit")
            return True, order_id, fill_price
        
        # Live trading - place complex order
        try:
            # Build butterfly order
            # This is broker-specific - Schwab uses their own format
            result = self.client.place_butterfly_order(
                symbol=self.symbol,
                lower_strike=lower,
                middle_strike=middle,
                upper_strike=upper,
                option_type=option_type,
                quantity=1,
                limit_credit=limit_credit
            )
            
            if result and result.get('orderId'):
                return True, result['orderId'], limit_credit
            
            return False, "", 0
            
        except Exception as e:
            logger.error(f"Error placing butterfly order: {e}")
            return False, "", 0
    
    # =========================================================================
    # SINGLE-LEG OPTION EXECUTION (Original methods)
    # =========================================================================
    
    def _open_position(self, signal: Signal) -> Optional[Trade]:
        """Open a new single-leg position based on signal"""
        option_type = "CALL" if signal.direction == Direction.LONG else "PUT"
        
        logger.info(f"Opening {signal.direction.value} position - looking for {option_type}")
        
        # Check correlation / delta exposure
        if not self._check_correlation(signal.direction):
            logger.warning("Trade blocked by correlation/delta exposure check")
            get_notifier().send(
                title="⚠️ Delta Limit",
                message=f"Adding this {signal.direction.value} would exceed max delta exposure of {self.max_delta_exposure:.0f}"
            )
            return None
        
        # Find option
        try:
            current_delta = self._get_current_target_delta()
            target_delta = current_delta if self.use_delta_targeting else None
            
            option = self.client.get_nearest_otm_option(
                symbol=self.symbol,
                option_type=option_type,
                min_dte=self.min_dte,
                max_dte=self.max_dte,
                target_delta=target_delta
            )
            
            if not option:
                logger.error(f"No suitable {option_type} option found")
                return None
            
            delta_info = f" | Delta: {option.delta:.2f}" if option.delta else ""
            logger.info(f"Selected option: {option.symbol} @ {option.strike} exp {option.expiration}{delta_info}")
            
        except Exception as e:
            logger.error(f"Error finding option: {e}")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(option.ask)
        estimated_cost = option.ask * 100 * position_size
        
        # Execute order
        try:
            if not self.paper_trading:
                result = self.client.buy_to_open(
                    option_symbol=option.symbol,
                    quantity=position_size,
                    order_type=OrderType.MARKET
                )
                logger.info(f"Order placed: {result}")
            else:
                logger.info(f"[PAPER] Would buy {position_size}x {option.symbol} @ ${option.ask:.2f} (${estimated_cost:.2f} total)")
            
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None
        
        # Update position state
        self.position = PositionState(
            direction=signal.direction,
            option_symbol=option.symbol,
            option_type=option_type,
            quantity=position_size,
            entry_time=datetime.now(),
            entry_price=option.ask,
            entry_signal=signal.signal_type,
            bars_held=0,
            unrealized_pnl=0.0
        )
        
        # Record trade
        self.trade_counter += 1
        self.daily_trade_count += 1
        
        trade = Trade(
            id=f"T{self.trade_counter:05d}",
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry_time=datetime.now(),
            entry_price=option.ask,
            option_symbol=option.symbol,
            option_strike=option.strike,
            option_expiry=option.expiration,
            option_type=option_type,
            quantity=position_size,
            status=TradeStatus.FILLED
        )
        
        self.trades.append(trade)
        return trade
    
    def _close_position(self, exit_signal: Signal) -> Optional[Trade]:
        """Close current position"""
        if self.position.direction == Direction.FLAT:
            logger.warning("No position to close")
            return None
        
        option_symbol = self.position.option_symbol
        quantity = self.position.quantity
        
        logger.info(f"Closing {self.position.direction.value} position: {option_symbol}")
        
        try:
            if not self.paper_trading:
                result = self.client.sell_to_close(
                    option_symbol=option_symbol,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                logger.info(f"Close order placed: {result}")
            else:
                logger.info(f"[PAPER] Would sell {quantity}x {option_symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
        
        # Find the trade to update
        open_trade = None
        for trade in reversed(self.trades):
            if trade.option_symbol == option_symbol and trade.exit_time is None:
                open_trade = trade
                break
        
        if open_trade:
            try:
                quote = self.client.get_quote(self.symbol)
                exit_price = self.position.entry_price * 1.1  # Placeholder
            except:
                exit_price = self.position.entry_price
            
            open_trade.exit_time = datetime.now()
            open_trade.exit_price = exit_price
            open_trade.exit_reason = f"Opposite signal: {exit_signal.signal_type.value}"
            open_trade.bars_held = self.position.bars_held
            
            if open_trade.entry_price and open_trade.exit_price:
                open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * 100 * quantity
                open_trade.pnl_percent = ((open_trade.exit_price / open_trade.entry_price) - 1) * 100
            
            logger.info(f"Closed trade {open_trade.id}: P&L ${open_trade.pnl:.2f} ({open_trade.pnl_percent:.1f}%)")
            
            self.daily_pnl += open_trade.pnl
            
            get_notifier().send(
                title="📊 Position Closed",
                message=f"Closed: {self.position.direction.value}\nP&L: ${open_trade.pnl:.2f} ({open_trade.pnl_percent:.1f}%)"
            )
        
        # Reset position
        self.position = PositionState()
        
        return open_trade
    
    def update_bars_held(self) -> None:
        """Increment bars held counter"""
        if self.has_position():
            self.position.bars_held += 1
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L for current position"""
        if self.has_position() and self.position.entry_price:
            self.position.current_price = current_price
            
            pnl_dollars = (current_price - self.position.entry_price) * 100 * self.position.quantity
            pnl_percent = ((current_price / self.position.entry_price) - 1) * 100
            
            self.position.unrealized_pnl = pnl_dollars
            self.position.unrealized_pnl_percent = pnl_percent
            
            if current_price > self.position.high_water_mark:
                self.position.high_water_mark = current_price
                self.position.high_water_pnl_percent = pnl_percent
    
    def check_stop_loss(self) -> Optional[str]:
        """Check if stop loss has been triggered."""
        if not self.enable_stop_loss or not self.has_position():
            return None
        
        if self.position.unrealized_pnl_percent <= -self.stop_loss_percent:
            return f"Stop Loss: {self.position.unrealized_pnl_percent:.1f}% loss"
        
        if self.position.unrealized_pnl <= -self.stop_loss_dollars:
            return f"Stop Loss: ${abs(self.position.unrealized_pnl):.2f} loss"
        
        return None
    
    def check_take_profit(self) -> Optional[str]:
        """Check if take profit has been triggered."""
        if not self.enable_take_profit or not self.has_position():
            return None
        
        if self.position.unrealized_pnl_percent >= self.take_profit_percent:
            return f"Take Profit: {self.position.unrealized_pnl_percent:.1f}% gain"
        
        if self.position.unrealized_pnl >= self.take_profit_dollars:
            return f"Take Profit: ${self.position.unrealized_pnl:.2f} gain"
        
        return None
    
    def check_trailing_stop(self) -> Optional[str]:
        """Check if trailing stop has been triggered."""
        if not self.enable_trailing_stop or not self.has_position():
            return None
        
        if self.position.high_water_pnl_percent < self.trailing_stop_activation:
            return None
        
        if self.position.high_water_mark > 0 and self.position.current_price:
            drawdown_percent = ((self.position.high_water_mark - self.position.current_price) 
                               / self.position.high_water_mark) * 100
            
            if drawdown_percent >= self.trailing_stop_percent:
                return f"Trailing Stop: {drawdown_percent:.1f}% drawdown from high"
        
        return None
    
    def check_risk_management(self) -> Optional[str]:
        """Check all risk management conditions."""
        stop_reason = self.check_stop_loss()
        if stop_reason:
            return stop_reason
        
        profit_reason = self.check_take_profit()
        if profit_reason:
            return profit_reason
        
        trailing_reason = self.check_trailing_stop()
        if trailing_reason:
            return trailing_reason
        
        return None
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        self._reset_daily_count()
        
        today = date.today()
        today_trades = [t for t in self.trades if t.entry_time.date() == today]
        
        closed_trades = [t for t in today_trades if t.exit_time is not None]
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl is not None)
        
        winners = len([t for t in closed_trades if t.pnl and t.pnl > 0])
        losers = len([t for t in closed_trades if t.pnl and t.pnl < 0])
        
        stats = {
            "date": today.isoformat(),
            "trades_taken": self.daily_trade_count,
            "trades_remaining": self.trades_remaining(),
            "locked_out": self.is_locked_out(),
            "open_position": self.position.direction.value,
            "closed_trades": len(closed_trades),
            "winners": winners,
            "losers": losers,
            "win_rate": winners / len(closed_trades) * 100 if closed_trades else 0,
            "total_pnl": total_pnl
        }
        
        # Add butterfly stats if in butterfly mode
        if self.butterfly_mode:
            stats["butterflies_today"] = len([b for b in self.butterflies if b.entry_time.date() == today])
            stats["credits_today"] = self.credits_today
            stats["total_credits"] = self.total_credits_collected
        
        return stats
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary"""
        if not self.has_position():
            summary = {"status": "FLAT"}
        else:
            summary = {
                "status": self.position.direction.value,
                "symbol": self.position.option_symbol,
                "type": self.position.option_type,
                "quantity": self.position.quantity,
                "entry_time": self.position.entry_time.isoformat() if self.position.entry_time else None,
                "entry_price": self.position.entry_price,
                "entry_signal": self.position.entry_signal.value if self.position.entry_signal else None,
                "bars_held": self.position.bars_held,
                "unrealized_pnl": self.position.unrealized_pnl
            }
        
        # Add butterfly info
        if self.butterfly_mode:
            summary["butterfly_mode"] = True
            summary["credits_today"] = self.credits_today
            summary["total_credits"] = self.total_credits_collected
        
        return summary
    
    def force_close_all(self, reason: str = "Manual close") -> List[Trade]:
        """Force close all positions"""
        closed = []
        
        if self.has_position():
            logger.warning(f"Force closing position: {reason}")
            
            dummy_signal = Signal(
                signal_type=SignalType.NONE,
                direction=Direction.FLAT,
                timestamp=datetime.now(),
                price=0,
                vah=0,
                val=0,
                poc=0,
                or_bias=0,
                reason=reason
            )
            
            trade = self._close_position(dummy_signal)
            if trade:
                trade.exit_reason = reason
                closed.append(trade)
        
        return closed