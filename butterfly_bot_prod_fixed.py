#!/usr/bin/env python3
"""
SPX/XSP Butterfly Credit Stacker - Production Bot (FIXED)

Features:
- Auto-selects SPX or XSP based on equity
- Small account mode (<$200): Legs into butterflies with debit spread first
- Pushover notifications for all trades
- Runs as background daemon
- Real option pricing from Schwab API
- FIXED: Properly polls ES bars and feeds to signal detector

Usage:
    python butterfly_bot_prod_fixed.py                    # Normal mode
    python butterfly_bot_prod_fixed.py --paper            # Paper trading
    ./run_butterfly_bot.sh start                          # Background daemon
    ./run_butterfly_bot.sh stop                           # Stop daemon
    ./run_butterfly_bot.sh status                         # Check status
"""

# Load environment variables first (for Pushover keys, etc.)
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import signal
import logging
import argparse
import random
import atexit
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType
from notifications import get_notifier

# Setup logging
LOG_FILE = os.path.join(os.path.dirname(__file__), 'butterfly_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ButterflyConfig:
    """Configuration for butterfly trading"""
    
    # Account Thresholds
    spx_min_equity: float = 25000.0       # Trade SPX if equity >= $25k
    xsp_min_equity: float = 2500.0        # Trade XSP if equity >= $2.5k
    leg_in_threshold: float = 200.0       # Use legging if equity < $200
    
    # Butterfly Structure
    wing_width: int = 5                    # Points between strikes
    
    # Credit Targets
    normal_credit_target_pct: float = 0.30    # 30% for normal mode
    leg_in_credit_target_pct: float = 0.10    # 10% for small account legging
    
    # Position Limits
    max_concurrent_positions: int = 10
    max_daily_trades: int = 20
    
    # Timing
    market_open: dt_time = field(default_factory=lambda: dt_time(9, 30))
    market_close: dt_time = field(default_factory=lambda: dt_time(16, 0))
    eod_cutoff_minutes: int = 15
    
    # Signal Settings  
    signal_cooldown_bars: int = 17
    use_or_bias: bool = True
    
    # Polling
    bar_interval_seconds: int = 300  # 5 min bars
    poll_interval_seconds: int = 5


# =============================================================================
# POSITION TYPES
# =============================================================================

class LegStatus(Enum):
    PENDING = "PENDING"
    LEG1_FILLED = "LEG1_FILLED"      # Debit spread filled
    COMPLETE = "COMPLETE"            # Full butterfly
    EXPIRED = "EXPIRED"


@dataclass 
class ButterflyPosition:
    """Tracks a butterfly position (full or legged)"""
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
    
    # Mode
    is_legged: bool = False  # True if using small account legging
    
    # Leg 1: Debit Spread (buy lower, sell middle) or full wings
    leg1_debit: float = 0.0
    leg1_order_id: Optional[str] = None
    leg1_fill_time: Optional[datetime] = None
    
    # Leg 2: Credit Spread (sell middle, buy upper) - only for legged mode
    leg2_credit: float = 0.0
    leg2_order_id: Optional[str] = None
    leg2_fill_time: Optional[datetime] = None
    
    # Full butterfly (non-legged mode)
    wing_debit: float = 0.0
    middle_credit: float = 0.0
    net_credit: float = 0.0
    
    # Status
    status: LegStatus = LegStatus.PENDING
    
    # Settlement
    settlement_price: Optional[float] = None
    final_pnl: Optional[float] = None
    
    @property
    def option_type(self) -> str:
        return 'C' if self.direction == Direction.LONG else 'P'
    
    @property
    def total_cost(self) -> float:
        """Total cost basis"""
        if self.is_legged:
            return self.leg1_debit - self.leg2_credit
        return self.wing_debit - self.middle_credit
    
    def __str__(self):
        opt = "CALL" if self.direction == Direction.LONG else "PUT"
        mode = "LEGGED" if self.is_legged else "FULL"
        return f"{self.symbol} {opt} {self.lower_strike}/{self.middle_strike}/{self.upper_strike} ({mode})"


# =============================================================================
# SCHWAB OPTION INTERFACE
# =============================================================================

class SchwabOptionTrader:
    """Interface for trading SPX/XSP options via Schwab"""
    
    def __init__(self, client, paper_mode: bool = False):
        self.client = client
        self.paper_mode = paper_mode
        self._chain_cache = {}
        self._cache_time = None
        self._paper_equity = 10000.0
        self._paper_es_price = 6050.0
    
    def get_equity(self) -> float:
        """Get account equity/buying power"""
        if self.paper_mode:
            return self._paper_equity
        return self.client.get_buying_power()
    
    def get_spx_price(self) -> float:
        """Get current SPX price"""
        if self.paper_mode:
            return self._paper_es_price
        quote = self.client.get_quote('$SPX')
        return quote.last_price
    
    def get_es_price(self) -> float:
        """Get current ES futures price"""
        if self.paper_mode:
            return self._paper_es_price
        quote = self.client.get_quote('/ES')
        return quote.last_price
    
    def get_price_history(self, symbol: str, **kwargs) -> dict:
        """Get price history (candles) for a symbol"""
        if self.paper_mode:
            return self._get_price_history_paper(symbol, **kwargs)
        return self.client.get_price_history(symbol, **kwargs)
    
    def _get_price_history_paper(self, symbol: str, **kwargs) -> dict:
        """Simulate price history for paper trading"""
        now = datetime.now()
        base_price = self._paper_es_price
        
        candles = []
        temp_price = base_price - 25  # Start lower and walk up
        
        for i in range(60):
            bar_time = now - timedelta(minutes=5 * (60 - i))
            
            # Skip non-RTH hours
            if bar_time.time() < dt_time(9, 30) or bar_time.time() >= dt_time(16, 0):
                continue
            
            # Random walk with slight upward bias
            change = random.uniform(-3, 3.5)
            temp_price += change
            
            o = temp_price
            h = o + random.uniform(0.5, 4)
            l = o - random.uniform(0.5, 4)
            c = random.uniform(l, h)
            v = random.randint(8000, 25000)
            
            candles.append({
                'datetime': int(bar_time.timestamp() * 1000),
                'open': round(o, 2),
                'high': round(h, 2),
                'low': round(l, 2),
                'close': round(c, 2),
                'volume': v
            })
        
        # Update paper ES price to last close
        if candles:
            self._paper_es_price = candles[-1]['close']
        
        return {'candles': candles}
    
    def get_option_chain(self, symbol: str = '$SPX') -> Dict:
        """Get option chain with caching"""
        now = datetime.now()
        
        if (self._cache_time and 
            (now - self._cache_time).seconds < 30):
            return self._chain_cache.get(symbol, {})
        
        if self.paper_mode:
            # Return empty chain for paper mode
            return {'callExpDateMap': {}, 'putExpDateMap': {}}
        
        chain = self.client.get_option_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=20,
            include_quotes=True,
            from_date=date.today(),
            to_date=date.today()
        )
        
        self._chain_cache[symbol] = chain
        self._cache_time = now
        return chain
    
    def get_option_price(
        self,
        symbol: str,
        strike: float,
        option_type: str
    ) -> Tuple[float, float]:
        """Get bid/ask for an option. Returns (bid, ask)"""
        
        if self.paper_mode:
            # Simulate option prices based on distance from ATM
            atm = self._paper_es_price
            distance = abs(strike - atm)
            base_price = max(0.10, 5.0 - (distance * 0.1))
            spread = base_price * 0.05
            return round(base_price - spread, 2), round(base_price + spread, 2)
        
        chain = self.get_option_chain(symbol)
        
        if option_type.upper() in ['C', 'CALL']:
            exp_map = chain.get('callExpDateMap', {})
        else:
            exp_map = chain.get('putExpDateMap', {})
        
        for exp_str, strikes in exp_map.items():
            strike_str = f"{strike:.1f}"
            if strike_str in strikes:
                opt = strikes[strike_str][0]
                return opt.get('bid', 0), opt.get('ask', 0)
        
        return 0, 0
    
    def get_butterfly_prices(
        self,
        symbol: str,
        lower: float,
        middle: float,
        upper: float,
        option_type: str
    ) -> Dict[str, float]:
        """Get all prices for a butterfly"""
        
        l_bid, l_ask = self.get_option_price(symbol, lower, option_type)
        m_bid, m_ask = self.get_option_price(symbol, middle, option_type)
        u_bid, u_ask = self.get_option_price(symbol, upper, option_type)
        
        return {
            'lower_bid': l_bid, 'lower_ask': l_ask,
            'middle_bid': m_bid, 'middle_ask': m_ask,
            'upper_bid': u_bid, 'upper_ask': u_ask,
            'wing_debit': l_ask + u_ask,
            'middle_credit': m_bid * 2,
            'debit_spread_cost': l_ask - m_bid,  # Buy lower, sell middle
            'credit_spread_credit': m_bid - u_ask,  # Sell middle, buy upper
        }
    
    def place_debit_spread(
        self,
        symbol: str,
        buy_strike: float,
        sell_strike: float,
        option_type: str,
        limit_price: float
    ) -> Tuple[bool, str, float]:
        """
        Place a debit spread order.
        Returns: (success, order_id, fill_price)
        """
        if self.paper_mode:
            order_id = f"PAPER-DS-{datetime.now().strftime('%H%M%S')}"
            # Simulate fill at limit
            self._paper_equity -= (limit_price * 100)
            logger.info(f"[PAPER] Debit spread filled: {buy_strike}/{sell_strike} @ ${limit_price:.2f}")
            return True, order_id, limit_price
        
        # Real order - TODO: implement Schwab order API
        logger.warning("Live trading not yet implemented")
        return False, "", 0
    
    def place_credit_spread(
        self,
        symbol: str,
        sell_strike: float,
        buy_strike: float,
        option_type: str,
        limit_price: float
    ) -> Tuple[bool, str, float]:
        """
        Place a credit spread order.
        Returns: (success, order_id, fill_price)
        """
        if self.paper_mode:
            order_id = f"PAPER-CS-{datetime.now().strftime('%H%M%S')}"
            # Simulate fill at limit
            self._paper_equity += (limit_price * 100)
            logger.info(f"[PAPER] Credit spread filled: {sell_strike}/{buy_strike} @ ${limit_price:.2f}")
            return True, order_id, limit_price
        
        # Real order - TODO: implement
        logger.warning("Live trading not yet implemented")
        return False, "", 0
    
    def place_butterfly(
        self,
        symbol: str,
        lower: float,
        middle: float,
        upper: float,
        option_type: str,
        limit_credit: float
    ) -> Tuple[bool, str, float]:
        """
        Place a full butterfly order.
        Returns: (success, order_id, fill_price)
        """
        if self.paper_mode:
            order_id = f"PAPER-BF-{datetime.now().strftime('%H%M%S')}"
            prices = self.get_butterfly_prices(symbol, lower, middle, upper, option_type)
            fill = prices['middle_credit'] - prices['wing_debit']
            logger.info(f"[PAPER] Butterfly filled: {lower}/{middle}/{upper} @ ${fill:.2f}")
            return True, order_id, fill
        
        # Real order - TODO
        return False, "", 0


# =============================================================================
# MAIN BOT
# =============================================================================

class ButterflyBot:
    """Main butterfly trading bot"""
    
    def __init__(
        self,
        config: ButterflyConfig,
        trader: SchwabOptionTrader,
        paper_mode: bool = False
    ):
        self.config = config
        self.trader = trader
        self.paper_mode = paper_mode
        self.notifier = get_notifier()
        
        # Signal detector
        self.detector = SignalDetector(
            signal_cooldown_bars=config.signal_cooldown_bars,
            use_or_bias_filter=config.use_or_bias,
            rth_only=True
        )
        
        # State
        self.positions: List[ButterflyPosition] = []
        self.pending_leg2: List[ButterflyPosition] = []  # Waiting for leg 2
        self.daily_trades = 0
        self._position_counter = 0
        self.running = False
        
        # Stats
        self.signals_today = 0
        self.trades_today = 0
        self.pnl_today = 0.0
        
        # Credit tracking - the key metric for butterfly stacking
        self.total_credits_collected = 0.0  # Cumulative credits from all completed butterflies
        self.credits_today = 0.0            # Credits collected today
        self.completed_butterflies = 0       # Count of fully completed butterflies
        
        # Bar tracking
        self.last_processed_bar_time: Optional[datetime] = None
        
        # Track instrument for notifications
        self.current_symbol = '$XSP'  # Will be updated on first trade
        
        # Shutdown tracking - prevent duplicate notifications
        self._shutdown_complete = False
    
    def select_instrument(self) -> Tuple[str, float]:
        """Select SPX or XSP based on equity"""
        equity = self.trader.get_equity()
        spx_price = self.trader.get_spx_price()
        
        if equity >= self.config.spx_min_equity:
            return '$SPX', spx_price
        elif equity >= self.config.xsp_min_equity:
            return '$XSP', spx_price / 10
        else:
            # Can still trade with legging
            return '$XSP', spx_price / 10
    
    def should_leg_in(self) -> bool:
        """Check if we should use legging mode"""
        equity = self.trader.get_equity()
        return equity < self.config.leg_in_threshold
    
    def fetch_es_bars(self, count: int = 60) -> List[Bar]:
        """Fetch recent ES futures bars from Schwab"""
        try:
            candles = self.trader.get_price_history(
                symbol='/ES',
                period_type='day',
                period=1,
                frequency_type='minute',
                frequency=5,
                need_extended_hours=False
            )
            
            bars = []
            for candle in candles.get('candles', [])[-count:]:
                # Schwab returns timestamp in milliseconds
                ts = datetime.fromtimestamp(candle['datetime'] / 1000)
                bars.append(Bar(
                    timestamp=ts,
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume']
                ))
            
            return bars
        except Exception as e:
            logger.error(f"Error fetching ES bars: {e}")
            return []
    
    def load_historical_bars(self) -> None:
        """Load historical bars to initialize detector state"""
        logger.info("Loading historical bars to initialize detector...")
        
        bars = self.fetch_es_bars(count=60)
        
        if not bars:
            logger.warning("No historical bars available")
            return
        
        # Feed all but the last bar with signals suppressed
        for bar in bars[:-1]:
            self.detector.add_bar(bar, suppress_signals=True)
        
        logger.info(f"Loaded {len(bars)-1} historical bars")
        
        # Set last processed bar time
        if bars:
            self.last_processed_bar_time = bars[-2].timestamp if len(bars) > 1 else None
        
        # Log current state
        state = self.detector.get_state_summary()
        logger.info(f"Detector initialized:")
        logger.info(f"   OR Bias: {state['or_bias']}")
        logger.info(f"   OR High: {state['or_high']:.2f}, OR Low: {state['or_low']:.2f}")
        logger.info(f"   VAH: {state['vah']:.2f}, VAL: {state['val']:.2f}, POC: {state['poc']:.2f}")
        logger.info(f"   Cooldown clear: {state['cooldown_clear']}")
    
    def get_current_bar_boundary(self) -> datetime:
        """Get the start time of the current 5-min bar"""
        now = datetime.now()
        # Round down to nearest 5 minutes
        minutes = (now.minute // 5) * 5
        return now.replace(minute=minutes, second=0, microsecond=0)
    
    def process_signal(self, signal: Signal, es_price: float) -> Optional[ButterflyPosition]:
        """Process a trading signal"""
        
        # Check limits
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return None
        
        active = [p for p in self.positions if p.status != LegStatus.EXPIRED]
        if len(active) >= self.config.max_concurrent_positions:
            logger.warning("Max concurrent positions reached")
            return None
        
        # Check time
        now = datetime.now()
        close_time = datetime.combine(now.date(), self.config.market_close)
        minutes_to_close = (close_time - now).total_seconds() / 60
        
        if minutes_to_close < self.config.eod_cutoff_minutes:
            logger.warning("Too close to market close")
            return None
        
        # Select instrument
        symbol, underlying_price = self.select_instrument()
        use_legging = self.should_leg_in()
        
        # Calculate strikes
        width = self.config.wing_width
        if symbol == '$XSP':
            atm = round(underlying_price / 0.5) * 0.5  # XSP rounds to 0.5
        else:
            atm = round(underlying_price / 5) * 5  # SPX rounds to 5
        
        if signal.direction == Direction.LONG:
            lower, middle, upper = atm, atm + width, atm + (width * 2)
        else:
            lower, middle, upper = atm - (width * 2), atm - width, atm
        
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
            upper_strike=upper,
            is_legged=use_legging
        )
        
        # Log and notify
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🦋 NEW SIGNAL: {position}")
        logger.info(f"   Signal: {signal.signal_type.value}")
        logger.info(f"   Direction: {signal.direction.value}")
        logger.info(f"   Reason: {signal.reason}")
        logger.info(f"   Mode: {'LEGGING' if use_legging else 'FULL'}")
        logger.info("=" * 60)
        
        # Update current symbol for notifications
        self.current_symbol = symbol
        
        # Send butterfly signal notification
        emoji = "📈" if signal.direction == Direction.LONG else "📉"
        self.notifier.send(
            message=f"{signal.direction.value} @ ${signal.price:.2f}\n{signal.signal_type.value}",
            title=f"🦋 {emoji} {symbol} Signal",
            sound="cashregister"
        )
        
        # Execute
        if use_legging:
            success = self._execute_leg1(position)
        else:
            success = self._execute_full_butterfly(position)
        
        if success:
            self.signals_today += 1
            self.trades_today += 1
            return position
        
        return None
    
    def _execute_leg1(self, position: ButterflyPosition) -> bool:
        """Execute leg 1: debit spread (buy lower, sell middle)"""
        
        opt_type = position.option_type
        prices = self.trader.get_butterfly_prices(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            opt_type
        )
        
        # Debit spread: buy lower at ask, sell 1 middle at bid
        leg1_debit = prices['debit_spread_cost']
        
        # Show how the full butterfly will be built
        logger.info(f"   ┌─ LEGGING INTO BUTTERFLY ───────────────────")
        logger.info(f"   │ LEG 1 (Debit Spread):")
        logger.info(f"   │   Buy  {position.lower_strike}: ${prices['lower_ask']:.2f}")
        logger.info(f"   │   Sell {position.middle_strike}: ${prices['middle_bid']:.2f}")
        logger.info(f"   │   Net debit: ${leg1_debit:.2f}")
        logger.info(f"   │")
        logger.info(f"   │ LEG 2 (Credit Spread) - PENDING:")
        logger.info(f"   │   Sell {position.middle_strike}: ~${prices['middle_bid']:.2f}")
        logger.info(f"   │   Buy  {position.upper_strike}: ${prices['upper_ask']:.2f}")
        logger.info(f"   │   Target: ${leg1_debit * 1.10:.2f} (10% above leg 1)")
        logger.info(f"   │")
        logger.info(f"   │ Full butterfly net credit target:")
        logger.info(f"   │   2× middle (${prices['middle_bid']:.2f} × 2) = ${prices['middle_credit']:.2f}")
        logger.info(f"   │   - wings (${prices['wing_debit']:.2f})")
        logger.info(f"   │   = ${prices['middle_credit'] - prices['wing_debit']:.2f} theoretical")
        logger.info(f"   └───────────────────────────────────────────")
        
        success, order_id, fill_price = self.trader.place_debit_spread(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            opt_type,
            leg1_debit
        )
        
        if success:
            position.leg1_debit = fill_price
            position.leg1_order_id = order_id
            position.leg1_fill_time = datetime.now()
            position.status = LegStatus.LEG1_FILLED
            
            self.pending_leg2.append(position)
            self.daily_trades += 1
            
            logger.info(f"   ✓ Leg 1 filled @ ${fill_price:.2f} debit")
            logger.info(f"   ⏳ Monitoring for Leg 2 credit >= ${fill_price * 1.10:.2f}")
            
            return True
        
        logger.error(f"   ✗ Leg 1 failed")
        return False
    
    def _execute_full_butterfly(self, position: ButterflyPosition) -> bool:
        """Execute full butterfly at once"""
        
        opt_type = position.option_type
        prices = self.trader.get_butterfly_prices(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            opt_type
        )
        
        wing_debit = prices['wing_debit']
        middle_credit = prices['middle_credit']  # This is 2× the middle bid
        theoretical_net = middle_credit - wing_debit
        
        # Target: get at least 30% more than wing cost as credit
        target_credit = wing_debit * (1 + self.config.normal_credit_target_pct)
        
        logger.info(f"   ┌─ BUTTERFLY STRUCTURE ─────────────────────")
        logger.info(f"   │ Lower wing ({position.lower_strike}): ${prices['lower_ask']:.2f} debit")
        logger.info(f"   │ Middle x2  ({position.middle_strike}): ${prices['middle_bid']:.2f} × 2 = ${middle_credit:.2f} credit")
        logger.info(f"   │ Upper wing ({position.upper_strike}): ${prices['upper_ask']:.2f} debit")
        logger.info(f"   ├───────────────────────────────────────────")
        logger.info(f"   │ Wing debit:     ${wing_debit:.2f}")
        logger.info(f"   │ Middle credit:  ${middle_credit:.2f} (2× middle)")
        logger.info(f"   │ Theoretical net: ${theoretical_net:.2f}")
        logger.info(f"   │ Target credit:  ${target_credit:.2f} (30% above wings)")
        logger.info(f"   └───────────────────────────────────────────")
        
        success, order_id, fill_price = self.trader.place_butterfly(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            opt_type,
            target_credit - wing_debit  # Net credit target
        )
        
        if success:
            position.wing_debit = wing_debit
            position.middle_credit = middle_credit
            position.net_credit = fill_price  # Actual net credit received
            position.status = LegStatus.COMPLETE
            
            self.positions.append(position)
            self.daily_trades += 1
            
            # Track credits - NET CREDIT is what we stack
            # net_credit = (2 × middle) - wings
            credit_collected = position.net_credit * 100  # Convert to dollars (options are x100)
            self.credits_today += credit_collected
            self.total_credits_collected += credit_collected
            self.completed_butterflies += 1
            
            logger.info(f"   ✓ FILLED @ net credit ${position.net_credit:.2f}")
            logger.info(f"   💰 CREDIT STACKED: ${credit_collected:.2f}")
            logger.info(f"   📊 Credits today: ${self.credits_today:.2f} | Total: ${self.total_credits_collected:.2f}")
            
            # Notify butterfly fill
            opt_symbol = f"{position.lower_strike}/{position.middle_strike}/{position.upper_strike}"
            emoji = "🟢" if position.direction == Direction.LONG else "🔴"
            self.notifier.send(
                message=f"{position.direction.value} {opt_symbol} {position.option_type}\nCredit: ${position.net_credit:.2f}\nStacked: ${credit_collected:.2f}",
                title=f"🦋 {emoji} Butterfly Filled",
                sound="bugle"
            )
            
            return True
        
        # Notify rejection
        self.notifier.send(
            message=f"Butterfly order failed\n{position.symbol} {position.lower_strike}/{position.middle_strike}/{position.upper_strike}",
            title="🦋 ❌ Trade Rejected",
            sound="pushover"
        )
        
        logger.error(f"   ✗ Butterfly failed")
        return False
    
    def monitor_leg2_opportunities(self) -> None:
        """Monitor pending leg 1 positions for leg 2 opportunities"""
        
        still_pending = []
        
        for position in self.pending_leg2:
            opt_type = position.option_type
            prices = self.trader.get_butterfly_prices(
                position.symbol,
                position.lower_strike,
                position.middle_strike,
                position.upper_strike,
                opt_type
            )
            
            # Credit spread: sell middle at bid, buy upper at ask
            potential_credit = prices['credit_spread_credit']
            target_credit = position.leg1_debit * (1 + self.config.leg_in_credit_target_pct)
            
            if potential_credit >= target_credit:
                # Execute leg 2!
                logger.info(f"")
                logger.info(f"   🎯 LEG 2 TARGET HIT for {position.id}")
                logger.info(f"   ┌─ LEG 2 (Credit Spread) ────────────────────")
                logger.info(f"   │ Sell {position.middle_strike}: ${prices['middle_bid']:.2f}")
                logger.info(f"   │ Buy  {position.upper_strike}: ${prices['upper_ask']:.2f}")
                logger.info(f"   │ Credit: ${potential_credit:.2f} >= target ${target_credit:.2f}")
                logger.info(f"   └───────────────────────────────────────────")
                
                success, order_id, fill_price = self.trader.place_credit_spread(
                    position.symbol,
                    position.middle_strike,
                    position.upper_strike,
                    opt_type,
                    potential_credit
                )
                
                if success:
                    position.leg2_credit = fill_price
                    position.leg2_order_id = order_id
                    position.leg2_fill_time = datetime.now()
                    position.status = LegStatus.COMPLETE
                    
                    # NET CREDIT = Leg2 credit - Leg1 debit
                    # Which equals: (2 × middle_bid) - (lower_ask + upper_ask)
                    position.net_credit = fill_price - position.leg1_debit
                    position.middle_credit = prices['middle_credit']  # 2× middle for tracking
                    position.wing_debit = prices['wing_debit']
                    
                    self.positions.append(position)
                    
                    # Track credits - NET CREDIT is what we stack
                    credit_collected = position.net_credit * 100  # Convert to dollars (x100 multiplier)
                    self.credits_today += credit_collected
                    self.total_credits_collected += credit_collected
                    self.completed_butterflies += 1
                    
                    logger.info(f"   ┌─ BUTTERFLY COMPLETE ───────────────────────")
                    logger.info(f"   │ Leg 1 debit:  ${position.leg1_debit:.2f}")
                    logger.info(f"   │ Leg 2 credit: ${fill_price:.2f}")
                    logger.info(f"   │ ─────────────────────────────")
                    logger.info(f"   │ NET CREDIT:   ${position.net_credit:.2f}")
                    logger.info(f"   │")
                    logger.info(f"   │ 💰 CREDIT STACKED: ${credit_collected:.2f}")
                    logger.info(f"   │ 📊 Today: ${self.credits_today:.2f} | Total: ${self.total_credits_collected:.2f}")
                    logger.info(f"   └───────────────────────────────────────────")
                    
                    # Notify butterfly complete
                    pnl_pct = ((position.net_credit / position.leg1_debit) * 100) if position.leg1_debit > 0 else 0
                    self.notifier.send(
                        message=f"Credit: +${credit_collected:.2f} ({pnl_pct:.0f}%)\nTotal today: ${self.credits_today:.2f}",
                        title="🦋 💰 Butterfly Complete",
                        sound="magic"
                    )
                else:
                    still_pending.append(position)
            else:
                still_pending.append(position)
        
        self.pending_leg2 = still_pending
    
    def check_eod(self) -> None:
        """End of day processing"""
        now = datetime.now()
        close_time = datetime.combine(now.date(), self.config.market_close)
        minutes_to_close = (close_time - now).total_seconds() / 60
        
        if minutes_to_close > self.config.eod_cutoff_minutes:
            return
        
        # Complete any pending leg 2s at market
        for position in self.pending_leg2:
            logger.warning(f"EOD: Force completing {position.id}")
            
            prices = self.trader.get_butterfly_prices(
                position.symbol,
                position.lower_strike,
                position.middle_strike,
                position.upper_strike,
                position.option_type
            )
            
            # Take whatever credit we can get
            success, order_id, fill_price = self.trader.place_credit_spread(
                position.symbol,
                position.middle_strike,
                position.upper_strike,
                position.option_type,
                prices['credit_spread_credit']
            )
            
            if success:
                position.leg2_credit = fill_price
                position.status = LegStatus.COMPLETE
                position.net_credit = fill_price - position.leg1_debit
                self.positions.append(position)
                
                # Track credits even for EOD force-completes
                credit_collected = position.net_credit * 100
                self.credits_today += credit_collected
                self.total_credits_collected += credit_collected
                self.completed_butterflies += 1
                
                logger.info(f"   EOD Leg 2 filled @ ${fill_price:.2f}")
                logger.info(f"   Net credit: ${position.net_credit:.2f}")
                logger.info(f"   💰 Credit collected: ${credit_collected:.2f}")
        
        self.pending_leg2 = []
    
    def _end_of_day_summary(self) -> None:
        """Send end of day summary"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 END OF DAY SUMMARY")
        logger.info(f"   Signals: {self.signals_today}")
        logger.info(f"   Trades: {self.trades_today}")
        logger.info(f"   Credits collected today: ${self.credits_today:.2f}")
        logger.info(f"   Total credits (cumulative): ${self.total_credits_collected:.2f}")
        logger.info(f"   Completed butterflies: {self.completed_butterflies}")
        logger.info("=" * 60)
        
        # Calculate win count (completed butterflies with positive credit)
        wins = sum(1 for p in self.positions if p.status == LegStatus.COMPLETE and p.net_credit > 0)
        win_rate = (wins / self.trades_today * 100) if self.trades_today > 0 else 0
        emoji = "📈" if self.credits_today >= 0 else "📉"
        sign = "+" if self.credits_today >= 0 else ""
        
        self.notifier.send(
            message=f"Butterflies: {self.completed_butterflies}\nWins: {wins} ({win_rate:.0f}%)\nCredits: {sign}${self.credits_today:.2f}",
            title=f"🦋 {emoji} Daily Summary",
            sound="classical"
        )
    
    def run(self) -> None:
        """Main run loop"""
        self.running = True
        logger.info("")
        logger.info("=" * 60)
        logger.info("🦋 BUTTERFLY BOT STARTING")
        logger.info("=" * 60)
        
        equity = self.trader.get_equity()
        self.current_symbol, _ = self.select_instrument()
        mode = "PAPER" if self.paper_mode else "LIVE"
        
        logger.info(f"Equity: ${equity:,.2f}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Instrument: {self.current_symbol}")
        
        # Send startup notification
        self.notifier.send(
            message=f"Mode: {mode}\nSymbol: {self.current_symbol}\nEquity: ${equity:,.2f}",
            title="🦋 Butterfly Bot Started",
            sound="bike"
        )
        
        # Load historical bars to initialize detector
        self.load_historical_bars()
        
        # Track state for logging
        last_bar_count = len(self.detector.bars)
        last_state_log = datetime.now()
        
        while self.running:
            try:
                now = datetime.now()
                
                # Check market hours
                market_open = datetime.combine(now.date(), self.config.market_open)
                market_close = datetime.combine(now.date(), self.config.market_close)
                
                if now < market_open:
                    # Before market open - wait
                    wait_seconds = (market_open - now).total_seconds()
                    if wait_seconds > 60:
                        logger.info(f"Market opens in {wait_seconds/60:.1f} minutes, waiting...")
                    time.sleep(min(60, max(1, wait_seconds)))
                    continue
                
                if now > market_close:
                    # After market close
                    if self.trades_today > 0 or self.signals_today > 0:
                        self._end_of_day_summary()
                        # Reset for next day
                        self.signals_today = 0
                        self.trades_today = 0
                        self.pnl_today = 0.0
                        self.credits_today = 0.0  # Reset daily credits
                        self.daily_trades = 0     # Reset daily trade counter
                    logger.info("Market closed, waiting for next session...")
                    time.sleep(60)
                    continue
                
                # ============== FETCH AND PROCESS BARS ==============
                
                current_bar_boundary = self.get_current_bar_boundary()
                
                # Fetch latest bars
                bars = self.fetch_es_bars(count=10)
                
                if bars:
                    # Process any completed bars we haven't seen
                    for bar in bars:
                        bar_end_time = bar.timestamp + timedelta(minutes=5)
                        is_complete = bar_end_time <= current_bar_boundary
                        
                        # Only process completed bars we haven't seen
                        if is_complete and (self.last_processed_bar_time is None or 
                                            bar.timestamp > self.last_processed_bar_time):
                            
                            logger.info(f"📊 Bar: {bar.timestamp.strftime('%H:%M')} | "
                                       f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
                                       f"C={bar.close:.2f} V={bar.volume:,}")
                            
                            # Feed to detector
                            signal = self.detector.add_bar(bar)
                            self.last_processed_bar_time = bar.timestamp
                            
                            if signal:
                                logger.info("")
                                logger.info(f"🚨 SIGNAL DETECTED!")
                                logger.info(f"   Type: {signal.signal_type.value}")
                                logger.info(f"   Direction: {signal.direction.value}")
                                logger.info(f"   Reason: {signal.reason}")
                                logger.info(f"   Price: {signal.price:.2f}")
                                logger.info(f"   OR Bias: {signal.or_bias}")
                                
                                # Process the signal (create butterfly position)
                                position = self.process_signal(signal, bar.close)
                                
                                if position:
                                    logger.info(f"   ✓ Position created: {position.id}")
                            else:
                                # Log detector state periodically (every 5 bars or 5 minutes)
                                if (len(self.detector.bars) > last_bar_count or 
                                    (now - last_state_log).seconds > 300):
                                    
                                    state = self.detector.get_state_summary()
                                    logger.debug(f"State: pos={state['position']}, "
                                               f"or_bias={state['or_bias']}, "
                                               f"bars_since={state['bars_since_signal']}, "
                                               f"cooldown={state['cooldown_clear']}")
                                    last_bar_count = len(self.detector.bars)
                                    last_state_log = now
                
                # ============== MONITOR LEG 2 OPPORTUNITIES ==============
                
                if self.pending_leg2:
                    self.monitor_leg2_opportunities()
                
                # ============== CHECK EOD ==============
                
                self.check_eod()
                
                # ============== SLEEP ==============
                
                time.sleep(self.config.poll_interval_seconds)
                
            except KeyboardInterrupt:
                print(">>> KeyboardInterrupt caught")  # Debug
                logger.info("Received interrupt, shutting down...")
                self.running = False
                break  # Exit the loop immediately
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.notifier.send(
                    message=str(e)[:500],
                    title="🦋 ⚠️ Bot Error",
                    sound="siren"
                )
                time.sleep(30)
        
        # Always send shutdown notification
        print(">>> Exited main loop, calling _shutdown()")  # Debug
        self._shutdown()
    
    def _shutdown(self) -> None:
        """Clean shutdown with notifications - idempotent, safe to call multiple times"""
        print(">>> _shutdown() called")  # Debug print
        
        if self._shutdown_complete:
            print(">>> Shutdown already complete, skipping")
            logger.debug("Shutdown already complete, skipping")
            return
        
        self._shutdown_complete = True
        print(">>> Sending shutdown notification...")  # Debug print
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🦋 BOT SHUTDOWN")
        logger.info("=" * 60)
        logger.info(f"   Signals today: {self.signals_today}")
        logger.info(f"   Trades today: {self.trades_today}")
        logger.info(f"   Credits today: ${self.credits_today:.2f}")
        logger.info(f"   Total credits collected: ${self.total_credits_collected:.2f}")
        logger.info(f"   Completed butterflies: {self.completed_butterflies}")
        logger.info("=" * 60)
        
        # Send notification directly
        try:
            sign = "+" if self.credits_today >= 0 else ""
            result = self.notifier.send(
                message=f"Trades: {self.trades_today}\nCredits: {sign}${self.credits_today:.2f}\nButterflies: {self.completed_butterflies}",
                title="🦋 Butterfly Bot Stopped",
                sound="gamelan"
            )
            print(f">>> Notification result: {result}")  # Debug print
            logger.info(f"Shutdown notification sent: {result}")
        except Exception as e:
            print(f">>> Notification error: {e}")  # Debug print
            logger.error(f"Failed to send shutdown notification: {e}")
        
        print(">>> _shutdown() complete")  # Debug print
        logger.info("Shutdown complete.")
    
    def stop(self) -> None:
        """Stop the bot gracefully"""
        print(">>> stop() called")  # Debug
        logger.info("Stop requested...")
        self.running = False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Butterfly Credit Stacker Bot')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--equity', type=float, default=10000, help='Starting equity for paper')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🦋 SPX/XSP BUTTERFLY CREDIT STACKER")
    print("=" * 60)
    
    if args.paper:
        print(f"Mode: PAPER TRADING")
        print(f"Starting equity: ${args.equity:,.2f}")
        
        # Create mock client for paper mode
        class MockClient:
            pass
        
        trader = SchwabOptionTrader(MockClient(), paper_mode=True)
        trader._paper_equity = args.equity
    else:
        print("Mode: LIVE TRADING")
        
        from schwab_auth import SchwabAuth
        from schwab_client import SchwabClient
        
        auth = SchwabAuth(
            app_key=config.schwab.app_key,
            app_secret=config.schwab.app_secret,
            redirect_uri=config.schwab.redirect_uri,
            token_file=config.schwab.token_file
        )
        
        if not auth.is_authenticated:
            print("❌ Not authenticated. Run schwab_auth.py first.")
            return
        
        auth.refresh_access_token()
        client = SchwabClient(auth)
        trader = SchwabOptionTrader(client, paper_mode=False)
    
    cfg = ButterflyConfig()
    bot = ButterflyBot(cfg, trader, paper_mode=args.paper)
    
    # Register atexit handler as backup - _shutdown is idempotent so safe to call multiple times
    print(">>> Registering atexit handler")  # Debug
    atexit.register(lambda: bot._shutdown())
    
    # For SIGTERM (kill command), use a handler
    def sigterm_handler(sig, frame):
        print(f"\n>>> SIGTERM received")
        bot.running = False
        bot._shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, sigterm_handler)
    # Don't override SIGINT - let KeyboardInterrupt propagate naturally
    
    # Show configuration
    equity = trader.get_equity()
    print(f"\nEquity: ${equity:,.2f}")
    
    if equity < cfg.leg_in_threshold:
        print(f"Strategy: LEGGING (equity < ${cfg.leg_in_threshold})")
        print(f"Target: 10% credit on leg 2")
    elif equity < cfg.spx_min_equity:
        print("Instrument: XSP")  
        print(f"Target: 30% credit")
    else:
        print("Instrument: SPX")
        print(f"Target: 30% credit")
    
    print(f"\nSignal cooldown: {cfg.signal_cooldown_bars} bars")
    print(f"OR bias filter: {cfg.use_or_bias}")
    print(f"Max daily trades: {cfg.max_daily_trades}")
    
    print("\n" + "-" * 60)
    print("Starting bot... (Ctrl+C to stop)")
    print("-" * 60 + "\n")
    
    bot.run()
    print(">>> bot.run() returned")  # Debug


if __name__ == "__main__":
    main()