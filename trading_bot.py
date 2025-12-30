"""
ToS Signal Trading Bot - Main Application

This bot monitors SPY for signals matching your ThinkOrSwim AMT indicator
and automatically trades the nearest OTM SPY options via the Schwab API.

Strategy: Hold until opposite signal
- LONG signal -> Buy nearest OTM CALL
- SHORT signal -> Close CALL, Buy nearest OTM PUT
"""
import os
import sys
import time
import signal as sig
import logging
from datetime import datetime, time as dt_time, date, timedelta, timezone
from typing import Optional, Dict, Any
import json

from config import BotConfig, config
from schwab_auth import SchwabAuth
from schwab_client import SchwabClient
from signal_detector import SignalDetector, Signal, Bar, Direction
from position_manager import PositionManager
from notifications import get_notifier, PushoverNotifier
from analytics import get_tracker, PerformanceTracker

# Eastern timezone
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    # Python < 3.9 fallback
    ET = timezone(timedelta(hours=-5))  # EST (doesn't handle DST)

def get_et_now():
    """Get current time in Eastern"""
    return datetime.now(ET)

def format_et_time(dt=None):
    """Format datetime as Eastern time string"""
    if dt is None:
        dt = get_et_now()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.strftime('%Y-%m-%d %H:%M:%S ET')

# Setup logging with Eastern time
class ETFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ET)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S ET')

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
# Apply ET formatter to all handlers
et_formatter = ETFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in logging.root.handlers:
    handler.setFormatter(et_formatter)

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Monitors market data, detects signals, and executes trades.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False
        
        # Initialize components
        self.auth: Optional[SchwabAuth] = None
        self.client: Optional[SchwabClient] = None
        self.detector: Optional[SignalDetector] = None
        self.position_manager: Optional[PositionManager] = None
        self.notifier: PushoverNotifier = get_notifier()
        self.tracker: PerformanceTracker = get_tracker()
        
        # State
        self.last_bar_time: Optional[datetime] = None
        self._last_processed_bar_time: Optional[datetime] = None  # Track actual bar timestamps
        self.last_price: float = 0
        self.bar_interval_seconds = 300  # 5-minute bars
        
        # Intra-bar tracking
        self._current_bar_period: Optional[int] = None  # Which 5-min period we're in
        self._live_bar_open: Optional[float] = None
        self._live_bar_high: Optional[float] = None
        self._live_bar_low: Optional[float] = None
        self._live_bar_volume: int = 0
        self._live_bar_start: Optional[datetime] = None
        self._signal_fired_this_bar: bool = False  # Prevent duplicate signals
        self._last_intra_bar_check: Optional[datetime] = None
        
        # Metrics
        self.bars_processed = 0
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time: Optional[datetime] = None
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Trading Bot...")
        
        try:
            # Validate config
            self.config.validate()
            
            # Setup authentication
            self.auth = SchwabAuth(
                app_key=self.config.schwab.app_key,
                app_secret=self.config.schwab.app_secret,
                redirect_uri=self.config.schwab.redirect_uri,
                token_file=self.config.schwab.token_file
            )
            
            # Check if we need to authenticate
            if not self.auth.is_authenticated:
                logger.info("No existing authentication - starting OAuth flow...")
                if not self.auth.authorize_interactive():
                    logger.error("Authentication failed")
                    return False
            else:
                # Verify token is still valid
                if not self.auth.refresh_access_token():
                    logger.info("Token refresh failed - re-authenticating...")
                    if not self.auth.authorize_interactive():
                        logger.error("Re-authentication failed")
                        return False
            
            # Initialize API client
            self.client = SchwabClient(self.auth)
            
            # Verify account access
            account_hash = self.client.get_account_hash()
            logger.info(f"Connected to account: {account_hash[:8]}...")
            
            # Initialize signal detector
            self.detector = SignalDetector(
                length_period=self.config.signal.length_period,
                value_area_percent=self.config.signal.value_area_percent,
                volume_threshold=self.config.signal.volume_threshold,
                use_relaxed_volume=self.config.signal.use_relaxed_volume,
                min_confirmation_bars=self.config.signal.min_confirmation_bars,
                sustained_bars_required=self.config.signal.sustained_bars_required,
                signal_cooldown_bars=self.config.signal.signal_cooldown_bars,
                use_or_bias_filter=self.config.signal.use_or_bias_filter,
                or_buffer_points=self.config.signal.or_buffer_points,
                use_time_filter=self.config.time.use_time_filter,
                rth_only=self.config.time.rth_only
            )
            
            # Initialize position manager
            self.position_manager = PositionManager(
                client=self.client,
                symbol=self.config.trading.symbol,
                contracts=self.config.trading.contracts,
                max_daily_trades=self.config.trading.max_daily_trades,
                min_dte=self.config.trading.min_days_to_expiry,
                max_dte=self.config.trading.max_days_to_expiry,
                paper_trading=self.config.paper_trading,
                # Delta targeting
                use_delta_targeting=self.config.trading.use_delta_targeting,
                target_delta=self.config.trading.target_delta,
                afternoon_delta=self.config.trading.afternoon_delta,
                afternoon_start_hour=self.config.trading.afternoon_start_hour,
                # Risk management settings
                enable_stop_loss=self.config.trading.enable_stop_loss,
                stop_loss_percent=self.config.trading.stop_loss_percent,
                stop_loss_dollars=self.config.trading.stop_loss_dollars,
                enable_take_profit=self.config.trading.enable_take_profit,
                take_profit_percent=self.config.trading.take_profit_percent,
                take_profit_dollars=self.config.trading.take_profit_dollars,
                enable_trailing_stop=self.config.trading.enable_trailing_stop,
                trailing_stop_percent=self.config.trading.trailing_stop_percent,
                trailing_stop_activation=self.config.trading.trailing_stop_activation,
                # Fixed fractional position sizing
                use_fixed_fractional=self.config.trading.use_fixed_fractional,
                risk_percent_per_trade=self.config.trading.risk_percent_per_trade,
                max_position_size=self.config.trading.max_position_size,
                min_position_size=self.config.trading.min_position_size,
                # Daily loss limit
                enable_daily_loss_limit=self.config.trading.enable_daily_loss_limit,
                max_daily_loss_dollars=self.config.trading.max_daily_loss_dollars,
                max_daily_loss_percent=self.config.trading.max_daily_loss_percent,
                # Correlation / delta exposure
                enable_correlation_check=self.config.trading.enable_correlation_check,
                max_delta_exposure=self.config.trading.max_delta_exposure
            )
            
            # Sync with broker if live trading
            if not self.config.paper_trading:
                self.position_manager.sync_with_broker()
            
            # Initialize performance tracker with settings
            from datetime import time as dt_time
            self.tracker.enable_daily_summary = self.config.analytics.enable_daily_summary
            self.tracker.enable_weekly_report = self.config.analytics.enable_weekly_report
            self.tracker.enable_drawdown_alerts = self.config.analytics.enable_drawdown_alerts
            self.tracker.drawdown_alert_threshold = self.config.analytics.drawdown_alert_threshold
            self.tracker.daily_summary_time = dt_time(
                self.config.analytics.daily_summary_hour,
                self.config.analytics.daily_summary_minute
            )
            self.tracker.data_file = self.config.analytics.performance_data_file
            
            # Set starting balance
            try:
                starting_balance = self.client.get_buying_power()
                self.tracker.set_starting_balance(starting_balance)
                self.tracker.new_trading_day()
            except Exception as e:
                logger.warning(f"Could not initialize tracker balance: {e}")
            
            logger.info("Bot initialized successfully")
            logger.info(f"Mode: {'PAPER TRADING' if self.config.paper_trading else 'LIVE TRADING'}")
            logger.info(f"Symbol: {self.config.trading.symbol}")
            logger.info(f"Max daily trades: {self.config.trading.max_daily_trades}")
            
            # Load historical bars to initialize detector
            self._load_historical_bars()
            
            # Log initial market state
            self._log_initial_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _load_historical_bars(self):
        """Load historical bars to initialize the signal detector"""
        try:
            # Load full day of bars to capture Opening Range (9:30-10:00 AM)
            # A full RTH day is about 78 bars (6.5 hours * 12 bars/hour)
            # We load more to ensure we get OR period
            num_bars = max(self.config.signal.length_period + 10, 100)  # At least 100 bars
            logger.info(f"Loading last {num_bars} bars to initialize detector...")
            
            bars = self.client.get_recent_bars(
                symbol=self.config.trading.symbol,
                num_bars=num_bars,
                bar_minutes=5,
                extended_hours=False  # RTH only for cleaner data
            )
            
            if not bars:
                logger.warning("No historical bars available - starting fresh")
                return
            
            logger.info(f"Loaded {len(bars)} historical bars")
            
            # Check if we have OR period bars (9:30 - 10:00 AM)
            today = date.today()
            or_bars = [b for b in bars if b['datetime'].date() == today 
                       and b['datetime'].hour == 9 and b['datetime'].minute >= 30]
            or_bars += [b for b in bars if b['datetime'].date() == today 
                        and b['datetime'].hour == 10 and b['datetime'].minute == 0]
            
            if or_bars:
                logger.info(f"Found {len(or_bars)} Opening Range bars from today")
            else:
                logger.warning("No Opening Range bars found - OR bias may be incorrect")
            
            # Feed bars to detector WITH suppress_signals=True
            # This builds up the value area context without triggering trades
            for bar_data in bars:
                bar = Bar(
                    timestamp=bar_data['datetime'],
                    open=bar_data['open'],
                    high=bar_data['high'],
                    low=bar_data['low'],
                    close=bar_data['close'],
                    volume=bar_data['volume']
                )
                self.detector.add_bar(bar, suppress_signals=True)
            
            # Get state after initialization
            state = self.detector.get_state_summary()
            
            logger.info("")
            logger.info("╔════════════════════════════════════════════════════════════╗")
            logger.info("║          DETECTOR INITIALIZED WITH HISTORICAL DATA         ║")
            logger.info("╠════════════════════════════════════════════════════════════╣")
            logger.info(f"║  Bars Loaded:  {len(bars):<43}║")
            logger.info(f"║  Time Range:   {bars[0]['datetime'].strftime('%H:%M')} - {bars[-1]['datetime'].strftime('%H:%M')} ET{' ' * 32}║")
            
            if state['vah'] > 0:
                logger.info("╠════════════════════════════════════════════════════════════╣")
                logger.info(f"║  VAH:  ${state['vah']:<9.2f}                                      ║")
                logger.info(f"║  POC:  ${state['poc']:<9.2f}                                      ║")
                logger.info(f"║  VAL:  ${state['val']:<9.2f}                                      ║")
            else:
                logger.info("║  Value Area:   Still building...                          ║")
            
            if state['or_complete']:
                import math
                if state['or_low'] > 0 and not math.isinf(state['or_high']):
                    logger.info("╠════════════════════════════════════════════════════════════╣")
                    logger.info(f"║  OR Low:   ${state['or_low']:<9.2f}                                  ║")
                    logger.info(f"║  OR High:  ${state['or_high']:<9.2f}                                  ║")
                    logger.info(f"║  OR Bias:  {state['or_bias']:<47}║")
            else:
                logger.info("║  Opening Range: Not yet complete                          ║")
            
            logger.info("╠════════════════════════════════════════════════════════════╣")
            logger.info("║  ✓ Signals suppressed during historical load               ║")
            logger.info("║  ✓ Will only trade on NEW bars going forward               ║")
            logger.info("╚════════════════════════════════════════════════════════════╝")
            logger.info("")
            
            # Set last processed bar time to the last historical bar
            # This prevents re-processing the same bar
            self._last_processed_bar_time = bars[-1]['datetime']
            logger.info(f"Last historical bar: {self._last_processed_bar_time.strftime('%Y-%m-%d %H:%M ET')}")
            logger.info("Will only process bars AFTER this timestamp")
            logger.info("")
            
        except Exception as e:
            logger.warning(f"Could not load historical bars: {e}")
            logger.info("Starting fresh - detector will build state from live data")
    
    def _log_initial_state(self):
        """Log initial market state on startup"""
        try:
            # Get the most recent bar from price history
            bars = self.client.get_recent_bars(
                symbol=self.config.trading.symbol,
                num_bars=1,
                bar_minutes=5,
                extended_hours=True
            )
            
            if not bars:
                logger.warning("No bars available for initial state")
                return
            
            bar = bars[-1]
            
            logger.info("")
            logger.info("╔════════════════════════════════════════════════════════════╗")
            logger.info("║                   INITIAL MARKET STATE                     ║")
            logger.info("╠════════════════════════════════════════════════════════════╣")
            logger.info(f"║  Symbol:     {self.config.trading.symbol:<45}║")
            logger.info(f"║  Last Bar:   {bar['datetime'].strftime('%Y-%m-%d %H:%M ET'):<45}║")
            logger.info("╠════════════════════════════════════════════════════════════╣")
            logger.info(f"║  Open:       ${bar['open']:<44.2f}║")
            logger.info(f"║  High:       ${bar['high']:<44.2f}║")
            logger.info(f"║  Low:        ${bar['low']:<44.2f}║")
            logger.info(f"║  Close:      ${bar['close']:<44.2f}║")
            logger.info(f"║  Volume:     {bar['volume']:>12,}{' ' * 32}║")
            logger.info("╠════════════════════════════════════════════════════════════╣")
            logger.info(f"║  Time Now:   {format_et_time():<45}║")
            logger.info("╚════════════════════════════════════════════════════════════╝")
            logger.info("")
            
            self.last_price = bar['close']
            
        except Exception as e:
            logger.warning(f"Could not fetch initial state: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        current_time = now.time()
        
        # Check RTH hours
        market_open = self.config.time.market_open
        market_close = self.config.time.market_close
        
        if current_time < market_open or current_time >= market_close:
            return False
        
        # Check weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        return True
    
    def get_current_bar(self) -> Optional[Bar]:
        """
        Get the current 5-minute bar data from price history.
        """
        try:
            # Calculate current 5-min period
            now = get_et_now()
            current_period_start = now.replace(
                minute=(now.minute // 5) * 5,
                second=0,
                microsecond=0,
                tzinfo=None
            )
            
            # Get recent bars from price history
            bars = self.client.get_recent_bars(
                symbol=self.config.trading.symbol,
                num_bars=5,
                bar_minutes=5,
                extended_hours=True
            )
            
            if not bars:
                logger.warning("No bars returned from price history")
                return None
            
            last_bar = bars[-1]
            bar_time = last_bar['datetime']
            
            # Calculate the bar's 5-min period start
            bar_period_start = bar_time.replace(
                minute=(bar_time.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            
            # Create bar object
            bar = Bar(
                timestamp=bar_time,
                open=last_bar['open'],
                high=last_bar['high'],
                low=last_bar['low'],
                close=last_bar['close'],
                volume=last_bar['volume']
            )
            
            self.last_price = bar.close
            return bar
            
        except Exception as e:
            logger.error(f"Error getting bar: {e}")
            return None
    
    def should_process_new_bar(self) -> bool:
        """Check if we should process a new bar based on time"""
        now = datetime.now()
        
        if self.last_bar_time is None:
            return True
        
        # Check if we've crossed into a new 5-minute period
        current_period = now.minute // 5
        last_period = self.last_bar_time.minute // 5
        
        if now.hour != self.last_bar_time.hour:
            return True
        
        if current_period != last_period:
            return True
        
        return False
    
    def _is_new_bar(self, bar: Bar) -> bool:
        """Check if this bar is different from the last processed bar"""
        if self._last_processed_bar_time is None:
            logger.debug(f"First bar - timestamp: {bar.timestamp}")
            return True
        
        # Compare bar timestamps - only process if it's a new bar
        is_new = bar.timestamp > self._last_processed_bar_time
        
        if not is_new:
            logger.debug(f"Same bar as before ({bar.timestamp}) - skipping")
        else:
            logger.debug(f"New bar: {bar.timestamp} > {self._last_processed_bar_time}")
        
        return is_new
    
    def process_bar(self, bar: Bar) -> Optional[Signal]:
        """Process a bar and check for signals"""
        self.bars_processed += 1
        self.last_bar_time = bar.timestamp
        
        # Update position tracking
        self.position_manager.update_bars_held()
        
        # Add bar to detector and check for signals
        signal = self.detector.add_bar(bar)
        
        # Get current state from detector
        state = self.detector.get_state_summary()
        
        # Log bar info with ET time
        et_time = get_et_now().strftime('%H:%M:%S ET')
        is_synthetic = bar.volume == 0
        bar_type = " (synthetic - no trades)" if is_synthetic else ""
        
        logger.info("")
        logger.info(f"─── Bar #{self.bars_processed} | {et_time}{bar_type} ───")
        logger.info(f"  Price: ${bar.close:.2f} (O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f})")
        if not is_synthetic:
            logger.info(f"  Volume: {bar.volume:,}")
        
        # Only show levels if we have enough data
        vah, poc, val = state['vah'], state['poc'], state['val']
        if vah > 0 and poc > 0 and val > 0:
            logger.info(f"  VAH: ${vah:.2f} | POC: ${poc:.2f} | VAL: ${val:.2f}")
            logger.info(f"  Position vs Levels: {state['position']}")
            logger.info(f"  Bars Above VAH: {state['bars_above_vah']} | Bars Below VAL: {state['bars_below_val']}")
        else:
            logger.info(f"  Value Area: Building... (need {self.config.signal.length_period} bars, have {self.bars_processed})")
        
        # OR status
        or_high, or_low = state['or_high'], state['or_low']
        if state['or_complete']:
            # Check for valid OR values
            import math
            if or_low > 0 and not math.isinf(or_high) and or_high > or_low:
                logger.info(f"  OR Range: ${or_low:.2f} - ${or_high:.2f} | Bias: {state['or_bias']}")
            else:
                logger.info(f"  OR: Complete | Bias: {state['or_bias']}")
        else:
            logger.info(f"  OR: Building... (completes at 10:00 ET)")
        
        # Position status
        pos = self.position_manager.get_position_summary()
        if pos['status'] != 'FLAT':
            logger.info(f"  Current Position: {pos['status']} | P&L: ${pos.get('unrealized_pnl', 0):.2f}")
        
        # Cooldown status
        cooldown = state.get('bars_since_signal', 999)
        cooldown_bars = self.config.signal.signal_cooldown_bars
        if cooldown < cooldown_bars:
            logger.info(f"  Cooldown: {cooldown}/{cooldown_bars} bars")
        
        if signal:
            self.signals_generated += 1
            logger.info("")
            logger.info("🚨 " + "=" * 50)
            logger.info(f"🚨 SIGNAL: {signal.signal_type.value} - {signal.direction.value}")
            logger.info(f"🚨 Price: ${signal.price:.2f}")
            logger.info(f"🚨 VAH: ${signal.vah:.2f}, POC: ${signal.poc:.2f}, VAL: ${signal.val:.2f}")
            logger.info(f"🚨 OR Bias: {'BULL' if signal.or_bias == 1 else 'BEAR' if signal.or_bias == -1 else 'NEUTRAL'}")
            logger.info(f"🚨 Reason: {signal.reason}")
            logger.info("🚨 " + "=" * 50)
            logger.info("")
        
        return signal
    
    def execute_signal(self, signal: Signal) -> bool:
        """Execute a trade based on signal"""
        # Notify on signal detection
        self.notifier.signal_alert(
            signal_type=signal.signal_type.value,
            direction=signal.direction.value,
            price=signal.price,
            symbol=self.config.trading.symbol
        )
        
        trade = self.position_manager.process_signal(signal)
        
        if trade:
            self.trades_executed += 1
            logger.info(f"Trade executed: {trade.id}")
            logger.info(f"  Direction: {trade.direction.value}")
            logger.info(f"  Option: {trade.option_symbol}")
            logger.info(f"  Strike: {trade.option_strike}")
            logger.info(f"  Expiry: {trade.option_expiry}")
            
            # Notify on trade execution
            self.notifier.trade_executed(
                direction=trade.direction.value,
                symbol=self.config.trading.symbol,
                option_symbol=trade.option_symbol,
                price=trade.entry_price,
                quantity=trade.quantity
            )
            return True
        
        return False
    
    def _get_current_bar_period(self) -> int:
        """Get the current 5-minute bar period (0-11 within each hour)"""
        now = get_et_now()
        return now.hour * 12 + now.minute // 5
    
    def _reset_live_bar(self) -> None:
        """Reset live bar tracking for a new period"""
        self._live_bar_open = None
        self._live_bar_high = None
        self._live_bar_low = None
        self._live_bar_volume = 0
        self._live_bar_start = None
        self._signal_fired_this_bar = False
    
    def _update_live_bar(self, quote: Any) -> Optional[Bar]:
        """
        Update the live bar with current quote data.
        Returns a Bar object representing the current in-progress bar.
        """
        try:
            price = quote.last_price
            now = get_et_now()
            
            if self._live_bar_open is None:
                # First tick of this bar
                self._live_bar_open = price
                self._live_bar_high = price
                self._live_bar_low = price
                self._live_bar_start = now
            else:
                # Update high/low
                if price > self._live_bar_high:
                    self._live_bar_high = price
                if price < self._live_bar_low:
                    self._live_bar_low = price
            
            # Build live bar
            live_bar = Bar(
                timestamp=self._live_bar_start or now,
                open=self._live_bar_open,
                high=self._live_bar_high,
                low=self._live_bar_low,
                close=price,  # Current price is the "close" of the live bar
                volume=self._live_bar_volume  # Volume is approximate
            )
            
            return live_bar
            
        except Exception as e:
            logger.debug(f"Error updating live bar: {e}")
            return None
    
    def _check_intra_bar_signal(self) -> None:
        """Check for signals on the live (in-progress) bar"""
        if not self.config.enable_intra_bar_signals:
            return
        
        # Don't check if we already fired a signal this bar
        if self._signal_fired_this_bar:
            return
        
        # Check if enough time has passed since last intra-bar check
        now = datetime.now()
        if self._last_intra_bar_check:
            elapsed = (now - self._last_intra_bar_check).total_seconds()
            if elapsed < self.config.intra_bar_check_interval:
                return
        
        self._last_intra_bar_check = now
        
        try:
            # Get current quote
            quote = self.client.get_quote(self.config.trading.symbol)
            
            # Update live bar
            live_bar = self._update_live_bar(quote)
            if not live_bar:
                return
            
            # Check for signal on live bar
            signal = self.detector.check_live_bar(live_bar)
            
            if signal:
                logger.info("")
                logger.info("╔════════════════════════════════════════════════════════════╗")
                logger.info("║            🔔 INTRA-BAR SIGNAL DETECTED                    ║")
                logger.info("╠════════════════════════════════════════════════════════════╣")
                logger.info(f"║  Signal: {signal.signal_type.value:<47}║")
                logger.info(f"║  Direction: {signal.direction.value:<44}║")
                logger.info(f"║  Price: ${live_bar.close:<46.2f}║")
                logger.info("╚════════════════════════════════════════════════════════════╝")
                logger.info("")
                
                # Mark that we fired a signal this bar
                self._signal_fired_this_bar = True
                self.signals_generated += 1
                
                # Execute if not locked out
                if not self.position_manager.is_locked_out():
                    self.execute_signal(signal)
                else:
                    logger.warning("Signal ignored - daily trade limit reached")
                    
        except Exception as e:
            logger.debug(f"Error in intra-bar check: {e}")
    
    def run_loop(self) -> None:
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        if self.config.enable_intra_bar_signals:
            logger.info(f"Intra-bar signal checking ENABLED (every {self.config.intra_bar_check_interval}s)")
        else:
            logger.info("Intra-bar signals disabled - will only check on bar close")
        logger.info("Waiting for new 5-minute bar...")
        
        while self.running:
            try:
                # Check for new trading day
                self.tracker.new_trading_day()
                
                # Check for scheduled reports (daily summary, weekly report)
                self.tracker.check_scheduled_reports()
                
                # Check market hours
                if not self.is_market_open():
                    if self.config.time.rth_only:
                        logger.debug("Market closed - waiting...")
                        time.sleep(60)
                        continue
                
                # Check if we've moved to a new bar period
                current_period = self._get_current_bar_period()
                if self._current_bar_period != current_period:
                    # New bar period - reset live bar tracking
                    self._reset_live_bar()
                    self._current_bar_period = current_period
                
                # Check risk management on existing position
                if self.position_manager.has_position():
                    exit_reason = self.position_manager.update_position_price()
                    if exit_reason:
                        logger.warning(f"Risk management triggered: {exit_reason}")
                        closed_trades = self.position_manager.force_close_all(exit_reason)
                        if closed_trades:
                            self.trades_executed += len(closed_trades)
                            logger.info(f"Position closed due to: {exit_reason}")
                            
                            # Record trade P&L in tracker
                            for trade in closed_trades:
                                if hasattr(trade, 'pnl') and trade.pnl is not None:
                                    self.tracker.record_trade(trade.pnl, trade.pnl > 0)
                
                # INTRA-BAR SIGNAL CHECK
                # Check for signals on live bar (if not already fired this bar)
                if not self._signal_fired_this_bar:
                    self._check_intra_bar_signal()
                
                # Check if we should process a completed bar
                if self.should_process_new_bar():
                    bar = self.get_current_bar()
                    
                    if bar:
                        if self._is_new_bar(bar):
                            # This is a genuinely new completed bar
                            # Skip signal check if we already fired intra-bar
                            if self._signal_fired_this_bar:
                                logger.info(f"Bar #{self.bars_processed + 1} closed - signal already fired intra-bar")
                                # Still process bar to update state, but suppress signal
                                self.detector.add_bar(bar, suppress_signals=False)
                                self.bars_processed += 1
                                self._last_processed_bar_time = bar.timestamp
                            else:
                                # Process normally
                                signal = self.process_bar(bar)
                                self._last_processed_bar_time = bar.timestamp
                                
                                if signal:
                                    if not self.position_manager.is_locked_out():
                                        self.execute_signal(signal)
                                    else:
                                        logger.warning("Signal ignored - daily trade limit reached")
                
                # Update balance periodically for drawdown tracking
                if self.bars_processed > 0 and self.bars_processed % 6 == 0:  # Every 30 min
                    try:
                        current_balance = self.client.get_buying_power()
                        self.tracker.update_balance(current_balance)
                    except:
                        pass
                
                # Log status periodically
                if self.bars_processed > 0 and self.bars_processed % 12 == 0:  # Every hour on 5-min bars
                    self.log_status()
                
                # Sleep until next check
                time.sleep(self.config.data_poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def log_status(self) -> None:
        """Log current bot status"""
        detector_state = self.detector.get_state_summary()
        position_summary = self.position_manager.get_position_summary()
        daily_stats = self.position_manager.get_daily_stats()
        
        logger.info("=" * 50)
        logger.info("BOT STATUS")
        logger.info(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"  Price: {self.last_price:.2f}")
        logger.info(f"  Position: {position_summary.get('status', 'FLAT')}")
        logger.info(f"  OR Bias: {detector_state.get('or_bias', 'N/A')}")
        logger.info(f"  VAH: {detector_state.get('vah', 0):.2f}")
        logger.info(f"  POC: {detector_state.get('poc', 0):.2f}")
        logger.info(f"  VAL: {detector_state.get('val', 0):.2f}")
        logger.info(f"  Trades today: {daily_stats.get('trades_taken', 0)}/{self.config.trading.max_daily_trades}")
        logger.info(f"  Bars processed: {self.bars_processed}")
        logger.info(f"  Signals: {self.signals_generated}")
        logger.info("=" * 50)
    
    def start(self) -> None:
        """Start the bot"""
        if not self.initialize():
            logger.error("Failed to initialize bot")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        # Setup signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)
        
        # Send start notification
        mode = "PAPER" if self.config.paper_trading else "LIVE"
        self.notifier.bot_started(mode, self.config.trading.symbol)
        
        logger.info("Bot started - entering main loop")
        self.run_loop()
    
    def stop(self) -> None:
        """Stop the bot gracefully"""
        logger.info("Stopping bot...")
        self.running = False
        
        # Log final stats
        self.log_final_stats()
        
        # Send stop notification
        daily_stats = self.position_manager.get_daily_stats()
        self.notifier.bot_stopped(
            trades=daily_stats.get('total_trades', 0),
            pnl=daily_stats.get('total_pnl', 0)
        )
        
        logger.info("Bot stopped")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def log_final_stats(self) -> None:
        """Log final session statistics"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        daily_stats = self.position_manager.get_daily_stats()
        
        logger.info("=" * 50)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Runtime: {runtime}")
        logger.info(f"  Bars processed: {self.bars_processed}")
        logger.info(f"  Signals generated: {self.signals_generated}")
        logger.info(f"  Trades executed: {self.trades_executed}")
        logger.info(f"  Today's P&L: ${daily_stats.get('total_pnl', 0):.2f}")
        logger.info(f"  Win rate: {daily_stats.get('win_rate', 0):.1f}%")
        logger.info("=" * 50)


def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ToS Signal Trading Bot - SPY Options                ║
    ║         Based on Auction Market Theory Indicator            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required environment variables
    if not os.getenv("SCHWAB_APP_KEY") or not os.getenv("SCHWAB_APP_SECRET"):
        print("\n⚠️  Missing required environment variables!")
        print("\nPlease set the following:")
        print("  export SCHWAB_APP_KEY='your-app-key'")
        print("  export SCHWAB_APP_SECRET='your-app-secret'")
        print("\nOptionally:")
        print("  export SCHWAB_REDIRECT_URI='https://127.0.0.1:8182/callback'")
        print("\nGet your API credentials from:")
        print("  https://developer.schwab.com/")
        return
    
    # Check for --no-confirm flag (for background/daemon mode)
    no_confirm = '--no-confirm' in sys.argv or '-y' in sys.argv
    
    # Load configuration
    bot_config = BotConfig()
    
    # Display configuration
    print(f"\n📊 Configuration:")
    print(f"  Symbol: {bot_config.trading.symbol}")
    print(f"  Contracts: {bot_config.trading.contracts}")
    print(f"  Max daily trades: {bot_config.trading.max_daily_trades}")
    print(f"  Paper trading: {bot_config.paper_trading}")
    print(f"  RTH only: {bot_config.time.rth_only}")
    print(f"  OR bias filter: {bot_config.signal.use_or_bias_filter}")
    
    if bot_config.paper_trading:
        print("\n📝 PAPER TRADING MODE - No real orders will be placed")
    else:
        print("\n⚠️  LIVE TRADING MODE - Real orders will be placed!")
        if no_confirm:
            print("   (--no-confirm flag set, skipping confirmation)")
        else:
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Aborted.")
                return
    
    # Create and start bot
    bot = TradingBot(bot_config)
    
    print("\n🚀 Starting bot...")
    bot.start()


if __name__ == "__main__":
    main()