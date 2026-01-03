#!/usr/bin/env python3
"""
SPX/XSP Butterfly Credit Stacker - Production Bot

Features:
- Auto-selects SPX or XSP based on equity
- Small account mode (<$200): Legs into butterflies with debit spread first
- Pushover notifications for all trades
- Runs as background daemon
- Real option pricing from Schwab API

Usage:
    python butterfly_bot_prod.py                    # Normal mode
    python butterfly_bot_prod.py --paper            # Paper trading
    ./run_butterfly_bot.sh start                    # Background daemon
    ./run_butterfly_bot.sh stop                     # Stop daemon
    ./run_butterfly_bot.sh status                   # Check status
"""

import os
import sys
import time
import signal
import logging
import argparse
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
    
    def get_equity(self) -> float:
        """Get account equity/buying power"""
        if self.paper_mode:
            return getattr(self, '_paper_equity', 10000.0)
        return self.client.get_buying_power()
    
    def get_spx_price(self) -> float:
        """Get current SPX price"""
        quote = self.client.get_quote('$SPX')
        return quote.last_price
    
    def get_option_chain(self, symbol: str = '$SPX') -> Dict:
        """Get option chain with caching"""
        now = datetime.now()
        
        if (self._cache_time and 
            (now - self._cache_time).seconds < 30):
            return self._chain_cache.get(symbol, {})
        
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
            order_id = f"PAPER-{datetime.now().strftime('%H%M%S')}"
            # Simulate fill at limit
            self._paper_equity = getattr(self, '_paper_equity', 10000) - (limit_price * 100)
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
            order_id = f"PAPER-{datetime.now().strftime('%H%M%S')}"
            # Simulate fill at limit
            self._paper_equity = getattr(self, '_paper_equity', 10000) + (limit_price * 100)
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
            order_id = f"PAPER-{datetime.now().strftime('%H%M%S')}"
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
        logger.info(f"   Mode: {'LEGGING' if use_legging else 'FULL'}")
        logger.info("=" * 60)
        
        self.notifier.send_notification(
            title=f"🦋 New {signal.direction.value} Signal",
            message=f"{position}\nMode: {'Legging' if use_legging else 'Full'}",
            priority=0
        )
        
        # Execute
        if use_legging:
            success = self._execute_leg1(position)
        else:
            success = self._execute_full_butterfly(position)
        
        if success:
            self.signals_today += 1
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
        
        # Debit spread: buy lower at ask, sell middle at bid
        debit = prices['debit_spread_cost']
        
        logger.info(f"   LEG 1: Buy {position.lower_strike}, Sell {position.middle_strike}")
        logger.info(f"   Debit: ${debit:.2f}")
        
        success, order_id, fill_price = self.trader.place_debit_spread(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            opt_type,
            debit
        )
        
        if success:
            position.leg1_debit = fill_price
            position.leg1_order_id = order_id
            position.leg1_fill_time = datetime.now()
            position.status = LegStatus.LEG1_FILLED
            
            self.pending_leg2.append(position)
            self.daily_trades += 1
            
            logger.info(f"   ✓ Leg 1 filled @ ${fill_price:.2f}")
            
            # Calculate target for leg 2
            target_credit = fill_price * (1 + self.config.leg_in_credit_target_pct)
            logger.info(f"   Target Leg 2 credit: ${target_credit:.2f} (10% above debit)")
            
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
        
        wing_cost = prices['wing_debit']
        target_credit = wing_cost * (1 + self.config.normal_credit_target_pct)
        
        logger.info(f"   Wings: ${wing_cost:.2f}")
        logger.info(f"   Target credit: ${target_credit:.2f} (30%)")
        
        success, order_id, fill_price = self.trader.place_butterfly(
            position.symbol,
            position.lower_strike,
            position.middle_strike,
            position.upper_strike,
            opt_type,
            target_credit - wing_cost  # Net credit target
        )
        
        if success:
            position.wing_debit = wing_cost
            position.middle_credit = wing_cost + fill_price  # Derived from net
            position.net_credit = fill_price
            position.status = LegStatus.COMPLETE
            
            self.positions.append(position)
            self.daily_trades += 1
            
            logger.info(f"   ✓ Butterfly filled @ net ${fill_price:.2f}")
            return True
        
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
                logger.info(f"   🎯 LEG 2 TARGET HIT for {position.id}")
                logger.info(f"   Credit: ${potential_credit:.2f} >= target ${target_credit:.2f}")
                
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
                    position.net_credit = fill_price - position.leg1_debit
                    
                    self.positions.append(position)
                    
                    logger.info(f"   ✓ Leg 2 filled @ ${fill_price:.2f}")
                    logger.info(f"   Net credit: ${position.net_credit:.2f}")
                    
                    self.notifier.send_notification(
                        title="🦋 Butterfly Complete!",
                        message=f"{position}\nNet: ${position.net_credit:.2f}",
                        priority=0
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
        
        self.pending_leg2 = []
    
    def run(self) -> None:
        """Main run loop"""
        self.running = True
        logger.info("🦋 Butterfly Bot starting...")
        
        self.notifier.send_notification(
            title="🦋 Bot Started",
            message=f"Butterfly bot running\nEquity: ${self.trader.get_equity():,.2f}",
            priority=-1
        )
        
        while self.running:
            try:
                now = datetime.now()
                
                # Check market hours
                market_open = datetime.combine(now.date(), self.config.market_open)
                market_close = datetime.combine(now.date(), self.config.market_close)
                
                if now < market_open or now > market_close:
                    time.sleep(60)
                    continue
                
                # Monitor leg 2 opportunities
                if self.pending_leg2:
                    self.monitor_leg2_opportunities()
                
                # Check EOD
                self.check_eod()
                
                time.sleep(self.config.poll_interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.notifier.send_notification("Bot Error", str(e), priority=1)
                time.sleep(30)
        
        self.notifier.send_notification(
            title="🦋 Bot Stopped",
            message=f"Trades today: {self.trades_today}\nP&L: ${self.pnl_today:,.2f}",
            priority=0
        )
    
    def stop(self) -> None:
        """Stop the bot"""
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
        print(f"Paper trading mode - Starting equity: ${args.equity:,.2f}")
        
        # Create mock trader for paper mode
        class MockClient:
            def get_buying_power(self): return args.equity
            def get_quote(self, sym): 
                class Q: last_price = 6000.0
                return Q()
            def get_option_chain(self, **kwargs): return {'callExpDateMap': {}, 'putExpDateMap': {}}
        
        trader = SchwabOptionTrader(MockClient(), paper_mode=True)
        trader._paper_equity = args.equity
    else:
        print("Live trading mode")
        
        from schwab_auth import SchwabAuth
        from schwab_client import SchwabClient
        
        auth = SchwabAuth(
            app_key=config.schwab.app_key,
            app_secret=config.schwab.app_secret,
            redirect_uri=config.schwab.redirect_uri,
            token_file=config.schwab.token_file
        )
        
        if not auth.is_authenticated:
            print("Not authenticated. Run schwab_auth.py first.")
            return
        
        auth.refresh_access_token()
        client = SchwabClient(auth)
        trader = SchwabOptionTrader(client, paper_mode=False)
    
    cfg = ButterflyConfig()
    bot = ButterflyBot(cfg, trader, paper_mode=args.paper)
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Show configuration
    equity = trader.get_equity()
    print(f"\nEquity: ${equity:,.2f}")
    
    if equity < cfg.leg_in_threshold:
        print(f"Mode: LEGGING (equity < ${cfg.leg_in_threshold})")
        print(f"Target: 10% credit")
    elif equity < cfg.xsp_min_equity:
        print("Mode: XSP")
        print(f"Target: 30% credit")
    elif equity < cfg.spx_min_equity:
        print("Mode: XSP")  
        print(f"Target: 30% credit")
    else:
        print("Mode: SPX")
        print(f"Target: 30% credit")
    
    print("\nStarting bot...")
    bot.run()


if __name__ == "__main__":
    main()
