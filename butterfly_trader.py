"""
SPX/XSP Butterfly Credit Stacker - Production Trading Bot

Features:
1. Auto-selects SPX or XSP based on account equity
2. Improved realistic option pricing model
3. Paper trading mode for testing
4. Integrated signal detection from ES futures
5. Real-time position monitoring
6. Complete butterfly even on abort (no day trades)

Usage:
    python butterfly_trader.py                    # Live trading
    python butterfly_trader.py --paper            # Paper trading
    python butterfly_trader.py --backtest --days 60  # Backtest
"""

import os
import sys
import time
import logging
import argparse
import math
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('butterfly_trader.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ButterflyConfig:
    """Configuration for butterfly trading"""
    
    # Instrument Selection
    spx_min_equity: float = 25000.0       # Min equity to trade SPX ($25k)
    xsp_min_equity: float = 2500.0        # Min equity to trade XSP ($2.5k)
    
    # Butterfly Structure
    wing_width: int = 5                    # Points between strikes
    credit_target_pct: float = 0.30        # 30% credit target
    abort_loss_pct: float = 0.30           # Complete if wings down 30%
    
    # Position Sizing
    max_risk_per_trade_pct: float = 0.02   # Risk 2% of equity per trade
    max_concurrent_butterflies: int = 10   # Max open at once
    max_daily_butterflies: int = 20        # Max per day
    
    # Timing
    eod_cutoff_minutes: int = 15           # Complete all 15 min before close
    market_open: dt_time = field(default_factory=lambda: dt_time(9, 30))
    market_close: dt_time = field(default_factory=lambda: dt_time(16, 0))
    
    # Signal Settings
    signal_cooldown_bars: int = 17
    use_or_bias: bool = True
    
    # Paper Trading
    paper_trading: bool = False
    paper_starting_equity: float = 10000.0
    
    # Monitoring
    poll_interval_seconds: int = 2


# =============================================================================
# IMPROVED OPTION PRICER
# =============================================================================

class RealisticOptionPricer:
    """
    Improved option pricing model for 0DTE SPX/XSP options.
    
    Key improvements over simplified model:
    1. Uses actual Black-Scholes approximation
    2. Accounts for 0DTE rapid time decay
    3. Realistic bid-ask spreads based on moneyness
    4. IV smile/skew approximation
    """
    
    def __init__(self, base_iv: float = 0.12, risk_free_rate: float = 0.05):
        self.base_iv = base_iv
        self.rf = risk_free_rate
    
    def _norm_cdf(self, x: float) -> float:
        """Cumulative normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _norm_pdf(self, x: float) -> float:
        """Normal probability density"""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    
    def _get_iv_for_strike(self, underlying: float, strike: float, base_iv: float) -> float:
        """
        Get implied volatility with smile/skew adjustment.
        OTM puts have higher IV (skew), ATM lowest, OTM calls slightly higher.
        """
        moneyness = strike / underlying
        
        # IV smile: higher for OTM options
        if moneyness < 0.97:  # OTM puts
            skew_adj = 0.03 * (0.97 - moneyness) * 10  # Up to +3% IV for deep OTM puts
        elif moneyness > 1.03:  # OTM calls
            skew_adj = 0.01 * (moneyness - 1.03) * 10  # Smaller adj for OTM calls
        else:  # Near ATM
            skew_adj = 0
        
        return base_iv + skew_adj
    
    def black_scholes(
        self,
        underlying: float,
        strike: float,
        tte: float,  # Time to expiry in years
        iv: float,
        option_type: str  # 'C' or 'P'
    ) -> float:
        """Calculate Black-Scholes option price"""
        
        if tte <= 0:
            # At expiration - intrinsic only
            if option_type == 'C':
                return max(0, underlying - strike)
            else:
                return max(0, strike - underlying)
        
        # Black-Scholes
        d1 = (math.log(underlying / strike) + (self.rf + 0.5 * iv**2) * tte) / (iv * math.sqrt(tte))
        d2 = d1 - iv * math.sqrt(tte)
        
        if option_type == 'C':
            price = underlying * self._norm_cdf(d1) - strike * math.exp(-self.rf * tte) * self._norm_cdf(d2)
        else:
            price = strike * math.exp(-self.rf * tte) * self._norm_cdf(-d2) - underlying * self._norm_cdf(-d1)
        
        return max(0.01, price)
    
    def get_delta(
        self,
        underlying: float,
        strike: float,
        tte: float,
        iv: float,
        option_type: str
    ) -> float:
        """Calculate option delta"""
        if tte <= 0:
            if option_type == 'C':
                return 1.0 if underlying > strike else 0.0
            else:
                return -1.0 if underlying < strike else 0.0
        
        d1 = (math.log(underlying / strike) + (self.rf + 0.5 * iv**2) * tte) / (iv * math.sqrt(tte))
        
        if option_type == 'C':
            return self._norm_cdf(d1)
        else:
            return self._norm_cdf(d1) - 1
    
    def get_option_price(
        self,
        underlying: float,
        strike: float,
        option_type: str,
        minutes_to_expiry: int,
        iv_override: float = None
    ) -> Tuple[float, float, float]:
        """
        Get bid, ask, and mid price for an option.
        
        Returns: (bid, ask, mid)
        """
        # Time to expiry in years (390 min/day, 252 days/year)
        tte = max(minutes_to_expiry / (252 * 390), 0.00001)
        
        # Get IV with smile adjustment
        iv = iv_override or self._get_iv_for_strike(underlying, strike, self.base_iv)
        
        # Calculate theoretical mid price
        mid = self.black_scholes(underlying, strike, tte, iv, option_type)
        
        # Calculate delta for spread sizing
        delta = abs(self.get_delta(underlying, strike, tte, iv, option_type))
        
        # Bid-ask spread based on delta and price
        # ATM (high delta) = tighter spread, OTM (low delta) = wider
        # Low priced options = wider spread percentage
        
        if mid < 0.50:
            spread_pct = 0.20  # 20% spread for cheap options
        elif mid < 2.00:
            spread_pct = 0.10  # 10% spread
        elif mid < 5.00:
            spread_pct = 0.05  # 5% spread
        else:
            spread_pct = 0.03  # 3% spread for expensive options
        
        # Widen spread for low delta (OTM) options
        if delta < 0.20:
            spread_pct *= 1.5
        
        spread = max(0.05, mid * spread_pct)
        
        bid = max(0.01, mid - spread / 2)
        ask = mid + spread / 2
        
        return bid, ask, mid
    
    def get_butterfly_prices(
        self,
        underlying: float,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        option_type: str,
        minutes_to_expiry: int
    ) -> Dict[str, float]:
        """
        Get complete pricing for a butterfly spread.
        
        Returns dict with all relevant prices.
        """
        lower_bid, lower_ask, lower_mid = self.get_option_price(
            underlying, lower_strike, option_type, minutes_to_expiry
        )
        middle_bid, middle_ask, middle_mid = self.get_option_price(
            underlying, middle_strike, option_type, minutes_to_expiry
        )
        upper_bid, upper_ask, upper_mid = self.get_option_price(
            underlying, upper_strike, option_type, minutes_to_expiry
        )
        
        # Wing debit: buy lower and upper at ask
        wing_debit = lower_ask + upper_ask
        
        # Potential credit: sell 2x middle at bid
        middle_credit = middle_bid * 2
        
        # Net at current prices
        net = middle_credit - wing_debit
        
        # Theoretical net (mid prices)
        theo_net = (middle_mid * 2) - (lower_mid + upper_mid)
        
        return {
            'lower_bid': lower_bid,
            'lower_ask': lower_ask,
            'lower_mid': lower_mid,
            'middle_bid': middle_bid,
            'middle_ask': middle_ask,
            'middle_mid': middle_mid,
            'upper_bid': upper_bid,
            'upper_ask': upper_ask,
            'upper_mid': upper_mid,
            'wing_debit': wing_debit,
            'middle_credit': middle_credit,
            'net': net,
            'theo_net': theo_net
        }


# =============================================================================
# INSTRUMENT SELECTOR
# =============================================================================

class InstrumentSelector:
    """
    Selects between SPX and XSP based on account equity.
    
    SPX: $100 multiplier, requires more capital
    XSP: $100 multiplier but 1/10th the index value (SPX/10)
    """
    
    SPX_MULTIPLIER = 100
    XSP_MULTIPLIER = 100  # Same multiplier, but index is 1/10th
    
    def __init__(self, config: ButterflyConfig):
        self.config = config
    
    def select_instrument(self, equity: float, spx_price: float) -> Tuple[str, float, int]:
        """
        Select instrument based on equity.
        
        Args:
            equity: Account equity/buying power
            spx_price: Current SPX cash index price
        
        Returns: (symbol, underlying_price_for_strikes, multiplier)
        """
        xsp_price = spx_price / 10
        
        # Calculate approximate max risk per butterfly
        # Max loss = wing_width (if net debit)
        spx_max_risk = self.config.wing_width * self.SPX_MULTIPLIER  # $500 for 5-pt
        
        # Check if we can afford SPX
        max_risk_allowed = equity * self.config.max_risk_per_trade_pct
        
        if equity >= self.config.spx_min_equity and max_risk_allowed >= spx_max_risk:
            return '$SPX', spx_price, self.SPX_MULTIPLIER
        elif equity >= self.config.xsp_min_equity:
            return '$XSP', xsp_price, self.XSP_MULTIPLIER
        else:
            return None, 0, 0  # Can't afford either


# =============================================================================
# BUTTERFLY POSITION
# =============================================================================

class ButterflyStatus(Enum):
    PENDING_WINGS = "PENDING_WINGS"
    MONITORING = "MONITORING"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"


@dataclass
class ButterflyPosition:
    """A butterfly position"""
    id: str
    symbol: str  # $SPX or $XSP
    direction: Direction
    signal_type: SignalType
    entry_time: datetime
    underlying_price: float
    
    # Strikes
    lower_strike: float
    middle_strike: float
    upper_strike: float
    
    # Prices (per point, multiply by 100 for contract value)
    wing_debit: float = 0.0
    middle_credit: float = 0.0
    net_credit: float = 0.0
    
    # Status
    status: ButterflyStatus = ButterflyStatus.PENDING_WINGS
    fill_time: Optional[datetime] = None
    was_early_exit: bool = False
    early_exit_reason: str = ""
    
    # Order IDs (for live trading)
    wing_order_id: Optional[str] = None
    middle_order_id: Optional[str] = None
    
    # Settlement
    settlement_price: Optional[float] = None
    settlement_value: Optional[float] = None
    final_pnl: Optional[float] = None
    
    @property
    def option_type(self) -> str:
        return 'C' if self.direction == Direction.LONG else 'P'
    
    @property
    def wing_width(self) -> float:
        return self.middle_strike - self.lower_strike
    
    @property
    def multiplier(self) -> int:
        return 100
    
    @property
    def max_loss_dollars(self) -> float:
        """Max loss if net debit"""
        if self.net_credit >= 0:
            return 0
        return abs(self.net_credit) * self.multiplier
    
    @property
    def max_profit_dollars(self) -> float:
        """Max profit if settles at middle strike"""
        return (self.wing_width + self.net_credit) * self.multiplier
    
    def calculate_settlement(self, settlement_price: float) -> float:
        """Calculate P&L at settlement"""
        self.settlement_price = settlement_price
        
        # Butterfly payoff
        if self.direction == Direction.LONG:
            # Call butterfly
            if settlement_price <= self.lower_strike:
                intrinsic = 0
            elif settlement_price >= self.upper_strike:
                intrinsic = 0
            elif settlement_price <= self.middle_strike:
                intrinsic = settlement_price - self.lower_strike
            else:
                intrinsic = self.upper_strike - settlement_price
        else:
            # Put butterfly
            if settlement_price >= self.upper_strike:
                intrinsic = 0
            elif settlement_price <= self.lower_strike:
                intrinsic = 0
            elif settlement_price >= self.middle_strike:
                intrinsic = self.upper_strike - settlement_price
            else:
                intrinsic = settlement_price - self.lower_strike
        
        self.settlement_value = intrinsic
        self.final_pnl = (self.net_credit + intrinsic) * self.multiplier
        self.status = ButterflyStatus.EXPIRED
        
        return self.final_pnl
    
    def __str__(self):
        opt = "CALL" if self.direction == Direction.LONG else "PUT"
        return f"{self.symbol} {opt} Fly {self.lower_strike}/{self.middle_strike}/{self.upper_strike}"


# =============================================================================
# PAPER TRADING SIMULATOR
# =============================================================================

class PaperTradingSimulator:
    """
    Simulates order execution for paper trading.
    """
    
    def __init__(self, starting_equity: float, pricer: RealisticOptionPricer):
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.pricer = pricer
        self.positions: List[ButterflyPosition] = []
        self.order_counter = 0
        self.trade_log: List[Dict] = []
    
    def get_equity(self) -> float:
        return self.equity
    
    def place_wing_order(
        self,
        position: ButterflyPosition,
        underlying_price: float,
        minutes_to_expiry: int
    ) -> Tuple[bool, str, float]:
        """
        Simulate placing wing order.
        Returns: (success, order_id, fill_price)
        """
        prices = self.pricer.get_butterfly_prices(
            underlying_price,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            position.option_type,
            minutes_to_expiry
        )
        
        self.order_counter += 1
        order_id = f"PAPER-{self.order_counter:06d}"
        
        # Simulate fill at ask (we're buying)
        fill_price = prices['wing_debit']
        
        # Deduct from equity (margin/buying power)
        cost = fill_price * position.multiplier
        if cost > self.equity:
            return False, "", 0
        
        self.equity -= cost
        
        self.trade_log.append({
            'time': datetime.now().isoformat(),
            'action': 'BUY_WINGS',
            'position_id': position.id,
            'order_id': order_id,
            'fill_price': fill_price,
            'cost': cost
        })
        
        return True, order_id, fill_price
    
    def place_middle_order(
        self,
        position: ButterflyPosition,
        underlying_price: float,
        minutes_to_expiry: int,
        limit_price: float = None
    ) -> Tuple[bool, str, float]:
        """
        Simulate placing middle strike order (sell 2x).
        Returns: (success, order_id, fill_price)
        """
        prices = self.pricer.get_butterfly_prices(
            underlying_price,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            position.option_type,
            minutes_to_expiry
        )
        
        middle_bid = prices['middle_bid']
        
        # Check if limit would fill
        if limit_price and middle_bid < limit_price:
            return False, "", 0
        
        self.order_counter += 1
        order_id = f"PAPER-{self.order_counter:06d}"
        
        # Fill at bid
        fill_price = middle_bid * 2  # 2x contracts
        
        # Add to equity (we're selling)
        credit = fill_price * position.multiplier
        self.equity += credit
        
        # Also return the original wing cost since butterfly is now complete
        self.equity += position.wing_debit * position.multiplier
        
        self.trade_log.append({
            'time': datetime.now().isoformat(),
            'action': 'SELL_MIDDLE',
            'position_id': position.id,
            'order_id': order_id,
            'fill_price': fill_price,
            'credit': credit
        })
        
        return True, order_id, fill_price
    
    def settle_position(self, position: ButterflyPosition, settlement_price: float) -> float:
        """Settle a position at expiration"""
        pnl = position.calculate_settlement(settlement_price)
        self.equity += pnl
        
        self.trade_log.append({
            'time': datetime.now().isoformat(),
            'action': 'SETTLEMENT',
            'position_id': position.id,
            'settlement_price': settlement_price,
            'pnl': pnl
        })
        
        return pnl


# =============================================================================
# BUTTERFLY TRADER BOT
# =============================================================================

class ButterflyTrader:
    """
    Main trading bot for butterfly credit stacking.
    """
    
    def __init__(
        self,
        config: ButterflyConfig,
        schwab_client=None,
        paper_mode: bool = False
    ):
        self.config = config
        self.client = schwab_client
        self.paper_mode = paper_mode or config.paper_trading
        
        # Pricing
        self.pricer = RealisticOptionPricer()
        
        # Instrument selection
        self.instrument_selector = InstrumentSelector(config)
        
        # Paper trading simulator
        if self.paper_mode:
            self.simulator = PaperTradingSimulator(config.paper_starting_equity, self.pricer)
            logger.info(f"Paper trading mode: Starting equity ${config.paper_starting_equity:,.2f}")
        else:
            self.simulator = None
        
        # Signal detector
        self.detector = SignalDetector(
            signal_cooldown_bars=config.signal_cooldown_bars,
            use_or_bias_filter=config.use_or_bias,
            rth_only=True
        )
        
        # State
        self.positions: List[ButterflyPosition] = []
        self.pending_positions: List[ButterflyPosition] = []
        self.daily_butterflies = 0
        self.daily_pnl = 0.0
        self._position_counter = 0
        self.running = False
        
        # Current instrument
        self.current_symbol = None
        # Current prices
        self.current_es_price = 0
        self.current_spx_price = 0
    
    def get_es_price(self) -> float:
        """Get current ES futures price (used for signal detection)"""
        if self.paper_mode:
            return self.current_es_price or 6010.0
        else:
            quote = self.client.get_quote('/ES')
            return quote.last_price
    
    def get_spx_price(self) -> float:
        """Get current SPX cash index price (used for strike selection)"""
        if self.paper_mode:
            return self.current_spx_price or 6000.0
        else:
            quote = self.client.get_quote('$SPX')
            return quote.last_price
    
    def select_instrument(self) -> Tuple[str, float]:
        """Select SPX or XSP based on equity, fetch actual SPX price for strikes"""
        equity = self.get_equity()
        spx_price = self.get_spx_price()  # Get real SPX price, not converted from ES
        
        symbol, price, multiplier = self.instrument_selector.select_instrument(equity, spx_price)
        
        if symbol is None:
            logger.warning(f"Insufficient equity (${equity:,.2f}) to trade SPX or XSP")
            return None, 0
        
        self.current_symbol = symbol
        self.current_spx_price = spx_price
        
        logger.info(f"Selected {symbol} at ${price:.2f} (SPX: ${spx_price:.2f}, Equity: ${equity:,.2f})")
        return symbol, price
    
    def process_signal(self, signal: Signal) -> Optional[ButterflyPosition]:
        """Process a trading signal"""
        
        # Check daily limit
        if self.daily_butterflies >= self.config.max_daily_butterflies:
            logger.warning("Daily butterfly limit reached")
            return None
        
        # Check concurrent limit
        active = [p for p in self.positions if p.status in [ButterflyStatus.MONITORING, ButterflyStatus.FILLED]]
        if len(active) >= self.config.max_concurrent_butterflies:
            logger.warning("Max concurrent butterflies reached")
            return None
        
        # Select instrument
        symbol, underlying_price = self.select_instrument()
        if symbol is None:
            return None
        
        # Calculate minutes to close
        now = datetime.now()
        close_time = datetime.combine(now.date(), self.config.market_close)
        minutes_to_close = max(0, (close_time - now).total_seconds() / 60)
        
        if minutes_to_close < self.config.eod_cutoff_minutes:
            logger.warning("Too close to market close")
            return None
        
        # Calculate strikes
        strike_rounding = 5 if '$SPX' in symbol else 0.5  # XSP uses smaller increments
        atm = round(underlying_price / strike_rounding) * strike_rounding
        width = self.config.wing_width
        
        if signal.direction == Direction.LONG:
            lower = atm
            middle = atm + width
            upper = atm + (width * 2)
        else:
            lower = atm - (width * 2)
            middle = atm - width
            upper = atm
        
        # Create position
        self._position_counter += 1
        position = ButterflyPosition(
            id=f"BF-{now.strftime('%Y%m%d')}-{self._position_counter:04d}",
            symbol=symbol,
            direction=signal.direction,
            signal_type=signal.signal_type,
            entry_time=now,
            underlying_price=underlying_price,
            lower_strike=lower,
            middle_strike=middle,
            upper_strike=upper
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🦋 NEW BUTTERFLY: {position}")
        logger.info(f"   Signal: {signal.signal_type.value}")
        logger.info(f"   Underlying: ${underlying_price:.2f}")
        logger.info("=" * 60)
        
        # Place wing order
        success = self._place_wing_order(position, underlying_price, int(minutes_to_close))
        
        if success:
            self.pending_positions.append(position)
            return position
        else:
            logger.error(f"Failed to place wing order")
            return None
    
    def _place_wing_order(
        self,
        position: ButterflyPosition,
        underlying_price: float,
        minutes_to_expiry: int
    ) -> bool:
        """Place order to buy wings"""
        
        if self.paper_mode:
            success, order_id, fill_price = self.simulator.place_wing_order(
                position, underlying_price, minutes_to_expiry
            )
        else:
            # Real order - TODO: implement
            success, order_id, fill_price = False, "", 0
            logger.warning("Live trading not yet implemented")
        
        if success:
            position.wing_order_id = order_id
            position.wing_debit = fill_price
            position.status = ButterflyStatus.MONITORING
            
            target_credit = fill_price * (1 + self.config.credit_target_pct)
            target_middle = target_credit / 2
            
            logger.info(f"   ✓ Wings filled: ${fill_price:.2f}")
            logger.info(f"   Target credit: ${target_credit:.2f} (middle bid ≥ ${target_middle:.2f})")
            
            return True
        else:
            logger.error(f"   ✗ Wing order failed")
            return False
    
    def monitor_pending(self, underlying_price: float) -> None:
        """Monitor pending positions for credit target"""
        
        now = datetime.now()
        close_time = datetime.combine(now.date(), self.config.market_close)
        minutes_to_close = max(0, (close_time - now).total_seconds() / 60)
        
        still_pending = []
        
        for position in self.pending_positions:
            filled = self._check_and_fill(position, underlying_price, int(minutes_to_close))
            
            if filled:
                self.positions.append(position)
                self.daily_butterflies += 1
            else:
                still_pending.append(position)
        
        self.pending_positions = still_pending
    
    def _check_and_fill(
        self,
        position: ButterflyPosition,
        underlying_price: float,
        minutes_to_expiry: int
    ) -> bool:
        """Check if position hits target or should early exit"""
        
        prices = self.pricer.get_butterfly_prices(
            underlying_price,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            position.option_type,
            minutes_to_expiry
        )
        
        middle_bid = prices['middle_bid']
        potential_credit = middle_bid * 2
        target_credit = position.wing_debit * (1 + self.config.credit_target_pct)
        
        # Check if target hit
        if potential_credit >= target_credit:
            logger.info(f"   🎯 TARGET HIT! Middle bid: ${middle_bid:.2f}")
            return self._complete_butterfly(position, underlying_price, minutes_to_expiry, middle_bid)
        
        # Check abort conditions
        
        # 1. Wings lost too much
        current_wing_value = prices['lower_bid'] + prices['upper_bid']
        if current_wing_value < position.wing_debit * (1 - self.config.abort_loss_pct):
            logger.warning(f"   ⚠️ WING STOP: Completing at current price")
            return self._complete_butterfly(position, underlying_price, minutes_to_expiry, middle_bid, "WING_STOP")
        
        # 2. EOD approaching
        if minutes_to_expiry < self.config.eod_cutoff_minutes:
            logger.warning(f"   ⚠️ EOD: Completing at current price")
            return self._complete_butterfly(position, underlying_price, minutes_to_expiry, middle_bid, "EOD")
        
        return False
    
    def _complete_butterfly(
        self,
        position: ButterflyPosition,
        underlying_price: float,
        minutes_to_expiry: int,
        middle_bid: float,
        early_exit_reason: str = None
    ) -> bool:
        """Complete butterfly by selling 2x middle"""
        
        if self.paper_mode:
            success, order_id, fill_price = self.simulator.place_middle_order(
                position, underlying_price, minutes_to_expiry
            )
        else:
            # Real order - TODO
            success, order_id, fill_price = False, "", 0
        
        if success:
            position.middle_order_id = order_id
            position.middle_credit = fill_price
            position.net_credit = fill_price - position.wing_debit
            position.status = ButterflyStatus.FILLED
            position.fill_time = datetime.now()
            
            if early_exit_reason:
                position.was_early_exit = True
                position.early_exit_reason = early_exit_reason
            
            if position.net_credit >= 0:
                logger.info(f"   ✓ BUTTERFLY COMPLETE!")
                logger.info(f"   Net Credit: ${position.net_credit:.2f} ({position.net_credit/position.wing_debit*100:.1f}%)")
            else:
                logger.warning(f"   ✓ BUTTERFLY COMPLETE (net debit)")
                logger.warning(f"   Net Debit: ${abs(position.net_credit):.2f}")
            
            logger.info(f"   Status: Riding to expiration 🎢")
            return True
        else:
            logger.error(f"   ✗ Failed to complete butterfly")
            return False
    
    def settle_day(self, settlement_price: float) -> float:
        """Settle all positions at end of day"""
        total_pnl = 0.0
        
        for position in self.positions:
            if position.status == ButterflyStatus.FILLED:
                if self.paper_mode:
                    pnl = self.simulator.settle_position(position, settlement_price)
                else:
                    pnl = position.calculate_settlement(settlement_price)
                
                total_pnl += pnl
                
                emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                logger.info(f"   {emoji} {position}: ${pnl:,.2f}")
        
        self.daily_pnl = total_pnl
        return total_pnl
    
    def log_summary(self) -> None:
        """Log daily summary"""
        filled = [p for p in self.positions if p.status in [ButterflyStatus.FILLED, ButterflyStatus.EXPIRED]]
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 DAILY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   Butterflies: {len(filled)}")
        logger.info(f"   Successful (30% target): {sum(1 for p in filled if not p.was_early_exit)}")
        logger.info(f"   Early Exits: {sum(1 for p in filled if p.was_early_exit)}")
        
        total_credits = sum(p.net_credit * p.multiplier for p in filled)
        total_pnl = sum(p.final_pnl or 0 for p in filled)
        
        logger.info(f"   Total Credits: ${total_credits:,.2f}")
        logger.info(f"   Total P&L: ${total_pnl:,.2f}")
        
        if self.paper_mode:
            logger.info(f"   Current Equity: ${self.simulator.get_equity():,.2f}")
            logger.info(f"   Session P&L: ${self.simulator.get_equity() - self.simulator.starting_equity:,.2f}")
        
        logger.info("=" * 60)


# =============================================================================
# BACKTESTER
# =============================================================================

def run_backtest(config: ButterflyConfig, bars: List[Bar]) -> Dict:
    """Run backtest with realistic option pricing"""
    
    logger.info(f"Starting backtest with {len(bars)} bars")
    
    pricer = RealisticOptionPricer()
    detector = SignalDetector(
        signal_cooldown_bars=config.signal_cooldown_bars,
        use_or_bias_filter=config.use_or_bias,
        rth_only=True
    )
    
    # Group bars by date
    bars_by_date = defaultdict(list)
    for bar in bars:
        bars_by_date[bar.timestamp.date()].append(bar)
    
    dates = sorted(bars_by_date.keys())
    
    # Results tracking
    all_butterflies = []
    daily_results = {}
    
    for day in dates:
        day_bars = sorted(bars_by_date[day], key=lambda b: b.timestamp)
        
        # In live trading, we'd fetch $SPX price directly
        # For backtest with ES data, approximate SPX = ES - 10
        # (In production, fetch actual SPX bars or use real-time SPX quote)
        es_settlement = day_bars[-1].close if day_bars else 0
        spx_settlement = es_settlement - 10  # Approximation for backtest
        
        detector.reset_session()
        
        pending = []
        filled = []
        
        market_close = datetime.combine(day, config.market_close)
        
        for bar in day_bars:
            bar_time = bar.timestamp
            if bar_time.tzinfo:
                bar_time = bar_time.replace(tzinfo=None)
            
            minutes_to_close = max(0, (market_close - bar_time).total_seconds() / 60)
            
            # ES bar for signal detection
            es_price = bar.close
            # SPX price for option strikes (in live, fetch $SPX; here approximate)
            spx_price = es_price - 10
            
            # Process signal using ES data
            signal = detector.add_bar(bar)
            
            # Check pending butterflies using SPX price for options
            still_pending = []
            for bf in pending:
                prices = pricer.get_butterfly_prices(
                    spx_price, bf.lower_strike, bf.middle_strike, bf.upper_strike,
                    bf.option_type, int(minutes_to_close)
                )
                
                middle_bid = prices['middle_bid']
                target = bf.wing_debit * (1 + config.credit_target_pct)
                
                if middle_bid * 2 >= target:
                    # Target hit
                    bf.middle_credit = middle_bid * 2
                    bf.net_credit = bf.middle_credit - bf.wing_debit
                    bf.status = ButterflyStatus.FILLED
                    filled.append(bf)
                elif minutes_to_close < config.eod_cutoff_minutes:
                    # EOD - complete anyway
                    bf.middle_credit = middle_bid * 2
                    bf.net_credit = bf.middle_credit - bf.wing_debit
                    bf.status = ButterflyStatus.FILLED
                    bf.was_early_exit = True
                    filled.append(bf)
                elif prices['lower_bid'] + prices['upper_bid'] < bf.wing_debit * (1 - config.abort_loss_pct):
                    # Wing stop
                    bf.middle_credit = middle_bid * 2
                    bf.net_credit = bf.middle_credit - bf.wing_debit
                    bf.status = ButterflyStatus.FILLED
                    bf.was_early_exit = True
                    filled.append(bf)
                else:
                    still_pending.append(bf)
            
            pending = still_pending
            
            # New signal
            if signal and signal.direction in [Direction.LONG, Direction.SHORT]:
                if minutes_to_close >= config.eod_cutoff_minutes:
                    # ES price from bar, convert to SPX for butterfly strikes
                    es_price = bar.close
                    spx_price = es_price - 10  # ES trades ~10 pts above SPX
                    
                    # Create butterfly on SPX strikes
                    atm = round(spx_price / 5) * 5
                    width = config.wing_width
                    
                    if signal.direction == Direction.LONG:
                        lower, middle, upper = atm, atm + width, atm + (width * 2)
                    else:
                        lower, middle, upper = atm - (width * 2), atm - width, atm
                    
                    prices = pricer.get_butterfly_prices(
                        spx_price, lower, middle, upper,
                        'C' if signal.direction == Direction.LONG else 'P',
                        int(minutes_to_close)
                    )
                    
                    bf = ButterflyPosition(
                        id=f"BT-{day}-{len(all_butterflies)+1}",
                        symbol='$SPX',
                        direction=signal.direction,
                        signal_type=signal.signal_type,
                        entry_time=bar.timestamp,
                        underlying_price=spx_price,
                        lower_strike=lower,
                        middle_strike=middle,
                        upper_strike=upper,
                        wing_debit=prices['wing_debit'],
                        status=ButterflyStatus.MONITORING
                    )
                    pending.append(bf)
        
        # EOD: complete remaining pending
        for bf in pending:
            prices = pricer.get_butterfly_prices(
                spx_settlement, bf.lower_strike, bf.middle_strike, bf.upper_strike,
                bf.option_type, 1
            )
            bf.middle_credit = prices['middle_bid'] * 2
            bf.net_credit = bf.middle_credit - bf.wing_debit
            bf.status = ButterflyStatus.FILLED
            bf.was_early_exit = True
            filled.append(bf)
        
        # Settle at SPX price
        day_pnl = 0
        for bf in filled:
            bf.calculate_settlement(spx_settlement)
            day_pnl += bf.final_pnl or 0
        
        all_butterflies.extend(filled)
        daily_results[day] = {
            'butterflies': len(filled),
            'pnl': day_pnl
        }
        
        if filled:
            logger.info(f"{day}: {len(filled)} butterflies, P&L: ${day_pnl:,.2f}")
    
    # Calculate summary
    total_pnl = sum(bf.final_pnl or 0 for bf in all_butterflies)
    wins = sum(1 for bf in all_butterflies if (bf.final_pnl or 0) > 0)
    
    return {
        'start_date': dates[0],
        'end_date': dates[-1],
        'trading_days': len(dates),
        'total_butterflies': len(all_butterflies),
        'successful_fills': sum(1 for bf in all_butterflies if not bf.was_early_exit),
        'early_exits': sum(1 for bf in all_butterflies if bf.was_early_exit),
        'total_pnl': total_pnl,
        'win_rate': wins / len(all_butterflies) if all_butterflies else 0,
        'avg_pnl': total_pnl / len(all_butterflies) if all_butterflies else 0,
        'best_day': max(r['pnl'] for r in daily_results.values()) if daily_results else 0,
        'worst_day': min(r['pnl'] for r in daily_results.values()) if daily_results else 0,
        'daily_results': daily_results,
        'butterflies': all_butterflies
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Butterfly Credit Stacker')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--days', type=int, default=30, help='Days for backtest')
    parser.add_argument('--equity', type=float, default=10000, help='Starting equity')
    parser.add_argument('--demo', action='store_true', help='Demo with synthetic data')
    
    args = parser.parse_args()
    
    config = ButterflyConfig(
        paper_trading=args.paper,
        paper_starting_equity=args.equity
    )
    
    print("\n" + "=" * 60)
    print("🦋 SPX/XSP BUTTERFLY CREDIT STACKER")
    print("=" * 60)
    
    if args.backtest or args.demo:
        # Generate or fetch data
        if args.demo:
            print("Running backtest with synthetic data...")
            import random
            random.seed(42)
            
            bars = []
            price = 600.0
            
            for day_offset in range(args.days):
                day = date.today() - timedelta(days=args.days - day_offset)
                
                for bar_idx in range(78):
                    hour = 9 + (bar_idx * 5 + 30) // 60
                    minute = (bar_idx * 5 + 30) % 60
                    
                    ts = datetime.combine(day, dt_time(hour, minute))
                    change = random.gauss(0, 0.5)
                    price += change
                    
                    bars.append(Bar(
                        timestamp=ts,
                        open=price,
                        high=price + abs(random.gauss(0, 0.3)),
                        low=price - abs(random.gauss(0, 0.3)),
                        close=price + random.gauss(0, 0.1),
                        volume=random.randint(50000, 200000)
                    ))
        else:
            print(f"Fetching {args.days} days of data...")
            from backtest import fetch_historical_data
            raw_bars = fetch_historical_data('/ES', args.days)
            bars = [Bar(
                timestamp=b['datetime'],
                open=b['open'],
                high=b['high'],
                low=b['low'],
                close=b['close'],
                volume=b['volume']
            ) for b in raw_bars]
        
        results = run_backtest(config, bars)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Trading Days: {results['trading_days']}")
        print(f"\nButterflies: {results['total_butterflies']}")
        print(f"  Hit 30% Target: {results['successful_fills']}")
        print(f"  Early Exits: {results['early_exits']}")
        print(f"\nTotal P&L: ${results['total_pnl']:,.2f}")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Avg P&L: ${results['avg_pnl']:,.2f}")
        print(f"Best Day: ${results['best_day']:,.2f}")
        print(f"Worst Day: ${results['worst_day']:,.2f}")
        
    else:
        # Live or paper trading
        if args.paper:
            print(f"Paper trading mode - Starting equity: ${args.equity:,.2f}")
        else:
            print("Live trading mode")
        
        trader = ButterflyTrader(config, paper_mode=args.paper)
        
        # Select instrument
        symbol, price = trader.select_instrument()
        if symbol:
            print(f"Selected: {symbol} @ ${price:.2f}")
        else:
            print("Insufficient equity to trade")


if __name__ == "__main__":
    main()