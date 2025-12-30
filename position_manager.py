"""
Position Manager

Handles position tracking, trade execution, and order management.
Implements the "hold until opposite signal" strategy.
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
        max_delta_exposure: float = 100.0
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
        
        # Current position
        self.position = PositionState()
        
        # Daily tracking
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.last_trade_date: Optional[date] = None
        
        # Monitor only mode (no trading, just log signals)
        self._monitor_only = False
        
        # Trade history
        self.trades: List[Trade] = []
        self.trade_counter = 0
    
    def _reset_daily_count(self) -> None:
        """Reset daily trade count and P&L if new day"""
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
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
                f"Daily loss limit hit!\nLoss: ${abs(self.daily_pnl):.2f}\nLimit: ${self.max_daily_loss_dollars:.2f}",
                title="🛑 Daily Loss Limit",
                priority=get_notifier().send.__self__.__class__.__bases__[0]  # NORMAL
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
            # Calls have positive delta, puts have negative
            if self.position.option_type == "CALL":
                return self.target_delta * self.position.quantity * 100  # Delta per contract * 100 shares
            else:
                return -self.target_delta * self.position.quantity * 100
        
        try:
            total_delta = 0.0
            positions = self.client.get_option_positions(self.symbol)
            
            for pos in positions:
                if pos.quantity != 0:
                    # Get option details for delta
                    # Note: This is an approximation - real delta would come from option chain
                    is_call = "C" in pos.symbol or "CALL" in pos.symbol.upper()
                    estimated_delta = 0.30 if is_call else -0.30  # Use target delta as estimate
                    total_delta += estimated_delta * abs(pos.quantity) * 100
            
            return total_delta
            
        except Exception as e:
            logger.warning(f"Error getting delta exposure: {e}")
            return 0.0
    
    def _check_correlation(self, signal_direction: Direction) -> bool:
        """
        Check if adding this position would exceed delta exposure limits.
        Returns True if OK to trade, False if would exceed limits.
        """
        if not self.enable_correlation_check:
            return True
        
        current_delta = self._get_current_delta_exposure()
        
        # Estimate delta of new position using current target delta
        current_target = self._get_current_target_delta()
        new_delta = current_target * self.contracts * 100
        if signal_direction == Direction.SHORT:
            new_delta = -new_delta
        
        projected_delta = current_delta + new_delta
        
        if abs(projected_delta) > self.max_delta_exposure:
            logger.warning(f"Correlation check failed: Current delta {current_delta:.0f}, new would add {new_delta:.0f}, total {projected_delta:.0f} > max {self.max_delta_exposure:.0f}")
            return False
        
        logger.info(f"Delta exposure OK: {current_delta:.0f} + {new_delta:.0f} = {projected_delta:.0f} (max: {self.max_delta_exposure:.0f})")
        return True
    
    def _get_current_target_delta(self) -> float:
        """
        Get target delta based on time of day.
        
        Morning (before afternoon_start_hour): target_delta (default 30Δ)
        Afternoon (after afternoon_start_hour): afternoon_delta (default 40Δ)
        """
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
                # No positions - we're flat
                if self.position.direction != Direction.FLAT:
                    logger.warning("Position mismatch: Expected position but broker shows flat")
                    self.position = PositionState()
                return
            
            # Find our option position
            for pos in positions:
                if pos.quantity != 0:
                    # Determine direction from option type
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
            logger.info(f"[MONITOR] Would trade {signal.direction.value} but in monitor-only mode")
            return None
        
        # Check lockout
        if self.is_locked_out():
            logger.warning(f"Locked out - {self.daily_trade_count}/{self.max_daily_trades} trades today")
            return None
        
        # Determine what we need to do
        current_direction = self.position.direction
        signal_direction = signal.direction
        
        # If flat, just open new position
        if current_direction == Direction.FLAT:
            return self._open_position(signal)
        
        # If opposite signal, close current and open new
        if current_direction != signal_direction:
            # Close existing position
            exit_trade = self._close_position(signal)
            
            # Check if we can open new position (might be at limit now)
            if not self.is_locked_out():
                new_trade = self._open_position(signal)
                return new_trade
            else:
                logger.info("Position closed but locked out for new position")
                return exit_trade
        
        # Same direction signal - already positioned correctly
        logger.info(f"Already {current_direction.value} - ignoring {signal_direction.value} signal")
        return None
    
    def _open_position(self, signal: Signal) -> Optional[Trade]:
        """Open a new position based on signal"""
        option_type = "CALL" if signal.direction == Direction.LONG else "PUT"
        
        logger.info(f"Opening {signal.direction.value} position - looking for {option_type}")
        
        # Check correlation / delta exposure before proceeding
        if not self._check_correlation(signal.direction):
            logger.warning("Trade blocked by correlation/delta exposure check")
            get_notifier().trade_rejected(
                "Delta exposure limit",
                f"Adding this {signal.direction.value} would exceed max delta exposure of {self.max_delta_exposure:.0f}"
            )
            return None
        
        # Find option (by delta or nearest OTM)
        try:
            # Determine delta based on time of day
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
        
        # Calculate position size using fixed fractional method
        position_size = self._calculate_position_size(option.ask)
        
        # Check buying power before placing order (live trading only)
        estimated_cost = option.ask * 100 * position_size
        if not self.paper_trading:
            try:
                buying_power = self.client.get_buying_power()
                logger.info(f"Buying power: ${buying_power:,.2f} | Estimated cost: ${estimated_cost:,.2f} ({position_size} contracts)")
                
                if buying_power < estimated_cost:
                    logger.error("")
                    logger.error("╔════════════════════════════════════════════════════════════╗")
                    logger.error("║              ❌ INSUFFICIENT BUYING POWER                  ║")
                    logger.error("╠════════════════════════════════════════════════════════════╣")
                    logger.error(f"║  Available:  ${buying_power:>12,.2f}                            ║")
                    logger.error(f"║  Required:   ${estimated_cost:>12,.2f}                            ║")
                    logger.error(f"║  Shortfall:  ${estimated_cost - buying_power:>12,.2f}                            ║")
                    logger.error("╚════════════════════════════════════════════════════════════╝")
                    logger.error("")
                    
                    # Send push notification
                    get_notifier().buying_power_warning(buying_power, estimated_cost)
                    
                    # Prompt user for action
                    action = self._prompt_insufficient_funds_action()
                    
                    if action == 'paper':
                        logger.info("Switching to PAPER TRADING mode")
                        self.paper_trading = True
                        # Continue to paper trade this signal
                        logger.info(f"[PAPER] Would buy {position_size}x {option.symbol} @ ${option.ask:.2f} (${estimated_cost:.2f} total)")
                    elif action == 'monitor':
                        logger.info("Continuing in MONITOR ONLY mode - signals will be logged but not traded")
                        self._monitor_only = True
                        return None
                    elif action == 'skip':
                        logger.info("Skipping this trade - will attempt next signal")
                        return None
                    else:  # stop
                        logger.info("Stopping bot as requested")
                        raise SystemExit("User requested stop due to insufficient funds")
                    
            except SystemExit:
                raise
            except Exception as e:
                logger.warning(f"Could not check buying power: {e}")
        
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
            error_msg = str(e)
            
            # Parse common rejection reasons
            if "Symbol is expired" in error_msg:
                logger.error("❌ ORDER REJECTED: Option has expired")
                logger.error("   The selected option expiration has passed.")
                logger.error("   This can happen after hours - will retry with next expiration.")
            elif "buying power" in error_msg.lower() or "insufficient" in error_msg.lower():
                logger.error("❌ ORDER REJECTED: Insufficient buying power")
                logger.error("   Your account doesn't have enough funds for this trade.")
                logger.error(f"   Estimated cost: ${estimated_cost:.2f}")
            elif "market closed" in error_msg.lower() or "not open" in error_msg.lower():
                logger.error("❌ ORDER REJECTED: Market is closed")
                logger.error("   Options can only be traded during market hours (9:30 AM - 4:00 PM ET)")
            elif "invalid" in error_msg.lower():
                logger.error(f"❌ ORDER REJECTED: Invalid order - {error_msg}")
            else:
                logger.error(f"❌ ORDER REJECTED: {error_msg}")
            
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
    
    def _prompt_insufficient_funds_action(self) -> str:
        """Prompt user for action when insufficient buying power"""
        print("")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                    WHAT WOULD YOU LIKE TO DO?              ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print("║  [1] Switch to PAPER TRADING mode                          ║")
        print("║      - Continue running, simulate trades without real $    ║")
        print("║                                                            ║")
        print("║  [2] Continue in MONITOR ONLY mode                         ║")
        print("║      - Log signals but don't trade (recommended)           ║")
        print("║                                                            ║")
        print("║  [3] SKIP this trade                                       ║")
        print("║      - Skip this signal, try again on next signal          ║")
        print("║                                                            ║")
        print("║  [4] STOP the bot                                          ║")
        print("║      - Shut down completely                                ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print("")
        
        while True:
            try:
                choice = input("Enter choice [1-4] (default: 2): ").strip()
                
                if choice == '' or choice == '2':
                    return 'monitor'
                elif choice == '1':
                    return 'paper'
                elif choice == '3':
                    return 'skip'
                elif choice == '4':
                    return 'stop'
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except (EOFError, KeyboardInterrupt):
                print("\nDefaulting to monitor mode...")
                return 'monitor'
    
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
            # Get current price for P&L calc
            try:
                quote = self.client.get_quote(self.symbol)
                exit_price = self.position.entry_price * 1.1  # Placeholder
            except:
                exit_price = self.position.entry_price
            
            open_trade.exit_time = datetime.now()
            open_trade.exit_price = exit_price
            open_trade.exit_reason = f"Opposite signal: {exit_signal.signal_type.value}"
            open_trade.bars_held = self.position.bars_held
            
            # Calculate P&L
            if open_trade.entry_price and open_trade.exit_price:
                open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * 100 * quantity
                open_trade.pnl_percent = ((open_trade.exit_price / open_trade.entry_price) - 1) * 100
            
            logger.info(f"Closed trade {open_trade.id}: P&L ${open_trade.pnl:.2f} ({open_trade.pnl_percent:.1f}%)")
            
            # Update daily P&L tracking
            self.daily_pnl += open_trade.pnl
            
            # Send close notification
            get_notifier().trade_closed(
                direction=self.position.direction.value,
                pnl=open_trade.pnl,
                pnl_percent=open_trade.pnl_percent,
                reason=open_trade.exit_reason
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
            
            # Calculate P&L
            pnl_dollars = (current_price - self.position.entry_price) * 100 * self.position.quantity
            pnl_percent = ((current_price / self.position.entry_price) - 1) * 100
            
            self.position.unrealized_pnl = pnl_dollars
            self.position.unrealized_pnl_percent = pnl_percent
            
            # Update high water mark for trailing stop
            if current_price > self.position.high_water_mark:
                self.position.high_water_mark = current_price
                self.position.high_water_pnl_percent = pnl_percent
    
    def check_stop_loss(self) -> Optional[str]:
        """
        Check if stop loss has been triggered.
        Returns exit reason if triggered, None otherwise.
        """
        if not self.enable_stop_loss or not self.has_position():
            return None
        
        # Check percent stop
        if self.position.unrealized_pnl_percent <= -self.stop_loss_percent:
            return f"Stop Loss: {self.position.unrealized_pnl_percent:.1f}% loss (limit: -{self.stop_loss_percent}%)"
        
        # Check dollar stop
        if self.position.unrealized_pnl <= -self.stop_loss_dollars:
            return f"Stop Loss: ${abs(self.position.unrealized_pnl):.2f} loss (limit: ${self.stop_loss_dollars})"
        
        return None
    
    def check_take_profit(self) -> Optional[str]:
        """
        Check if take profit has been triggered.
        Returns exit reason if triggered, None otherwise.
        """
        if not self.enable_take_profit or not self.has_position():
            return None
        
        # Check percent target
        if self.position.unrealized_pnl_percent >= self.take_profit_percent:
            return f"Take Profit: {self.position.unrealized_pnl_percent:.1f}% gain (target: {self.take_profit_percent}%)"
        
        # Check dollar target
        if self.position.unrealized_pnl >= self.take_profit_dollars:
            return f"Take Profit: ${self.position.unrealized_pnl:.2f} gain (target: ${self.take_profit_dollars})"
        
        return None
    
    def check_trailing_stop(self) -> Optional[str]:
        """
        Check if trailing stop has been triggered.
        Returns exit reason if triggered, None otherwise.
        """
        if not self.enable_trailing_stop or not self.has_position():
            return None
        
        # Only activate trailing stop after minimum gain
        if self.position.high_water_pnl_percent < self.trailing_stop_activation:
            return None
        
        # Calculate drawdown from high water mark
        if self.position.high_water_mark > 0 and self.position.current_price:
            drawdown_percent = ((self.position.high_water_mark - self.position.current_price) 
                               / self.position.high_water_mark) * 100
            
            if drawdown_percent >= self.trailing_stop_percent:
                return (f"Trailing Stop: {drawdown_percent:.1f}% drawdown from high "
                       f"(activated at {self.position.high_water_pnl_percent:.1f}% gain)")
        
        return None
    
    def check_risk_management(self) -> Optional[str]:
        """
        Check all risk management conditions.
        Returns exit reason if any triggered, None otherwise.
        """
        # Check in order of priority
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
    
    def update_position_price(self) -> Optional[str]:
        """
        Update position with current option price and check risk management.
        Returns exit reason if risk management triggered, None otherwise.
        """
        if not self.has_position() or not self.position.option_symbol:
            return None
        
        try:
            # Get current option price
            # Note: This is simplified - ideally we'd get the option quote directly
            # For now, we estimate based on underlying movement
            quote = self.client.get_quote(self.symbol)
            
            # Very rough estimate: option price moves ~50% of underlying for ATM
            # This is a placeholder - real implementation should get actual option quote
            if self.position.entry_price:
                underlying_move = quote.last_price - self.position.entry_price
                estimated_option_price = self.position.entry_price + (underlying_move * 0.5)
                estimated_option_price = max(0.01, estimated_option_price)  # Floor at $0.01
                
                self.update_unrealized_pnl(estimated_option_price)
                
                # Check risk management
                return self.check_risk_management()
                
        except Exception as e:
            logger.error(f"Error updating position price: {e}")
        
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
        
        return {
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
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary"""
        if not self.has_position():
            return {"status": "FLAT"}
        
        return {
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
    
    def force_close_all(self, reason: str = "Manual close") -> List[Trade]:
        """Force close all positions"""
        closed = []
        
        if self.has_position():
            logger.warning(f"Force closing position: {reason}")
            
            # Create dummy signal for closing
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