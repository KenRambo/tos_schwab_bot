"""
Signal Detector - Replicates ToS AMT Indicator Logic

This module implements the same signal detection logic as your ThinkOrSwim
indicator, translated to Python for use with live market data.

FIXED 2026-01-05:
- VA calculation now uses only last length_period bars for StdDev (matches ToS)
- reset_session() clears bars to prevent cross-day contamination
- Added VIX regime support
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta
from typing import Optional, List, Deque
from collections import deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types matching ToS indicator"""
    NONE = "NONE"
    
    # Long signals
    VAL_BOUNCE = "VAL_BOUNCE"
    POC_RECLAIM = "POC_RECLAIM"
    BREAKOUT = "BREAKOUT"
    SUSTAINED_BREAKOUT = "SUSTAINED_BREAKOUT"
    PRIOR_VAL_BOUNCE = "PRIOR_VAL_BOUNCE"
    PRIOR_POC_RECLAIM = "PRIOR_POC_RECLAIM"
    
    # Short signals
    VAH_REJECTION = "VAH_REJECTION"
    POC_BREAKDOWN = "POC_BREAKDOWN"
    BREAKDOWN = "BREAKDOWN"
    SUSTAINED_BREAKDOWN = "SUSTAINED_BREAKDOWN"
    PRIOR_VAH_REJECTION = "PRIOR_VAH_REJECTION"
    PRIOR_POC_BREAKDOWN = "PRIOR_POC_BREAKDOWN"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Bar:
    """OHLCV bar data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Signal:
    """Trading signal"""
    signal_type: SignalType
    direction: Direction
    timestamp: datetime
    price: float
    vah: float
    val: float
    poc: float
    or_bias: int  # 1 = bullish, -1 = bearish, 0 = neutral
    confidence: float = 1.0
    reason: str = ""


@dataclass
class ValueArea:
    """Value Area levels"""
    vah: float  # Value Area High
    val: float  # Value Area Low
    poc: float  # Point of Control
    vwap: float


class SignalDetector:
    """
    Detects trading signals based on Auction Market Theory.
    Replicates the logic from your ToS indicator.
    """
    
    def __init__(
        self,
        length_period: int = 20,
        value_area_percent: float = 70.0,
        volume_threshold: float = 1.478,
        use_relaxed_volume: bool = True,
        min_confirmation_bars: int = 2,
        sustained_bars_required: int = 5,
        signal_cooldown_bars: int = 17,
        use_or_bias_filter: bool = True,
        or_buffer_points: float = 1.0,
        opening_range_minutes: int = 30,
        use_time_filter: bool = False,
        rth_only: bool = True,
        # VIX Regime settings
        use_vix_regime: bool = True,
        vix_high_threshold: int = 25,
        vix_low_threshold: int = 15,
        high_vol_cooldown_mult: float = 1.43,
        low_vol_cooldown_mult: float = 0.88,
        # Signal enable/disable flags - OPTIMIZED DEFAULTS
        enable_val_bounce: bool = True,
        enable_poc_reclaim: bool = False,
        enable_breakout: bool = False,
        enable_sustained_breakout: bool = False,
        enable_prior_val_bounce: bool = True,
        enable_prior_poc_reclaim: bool = False,
        enable_vah_rejection: bool = True,
        enable_poc_breakdown: bool = False,
        enable_breakdown: bool = True,
        enable_sustained_breakdown: bool = False,
        enable_prior_vah_rejection: bool = True,
        enable_prior_poc_breakdown: bool = False
    ):
        # Configuration
        self.length_period = length_period
        self.value_area_percent = value_area_percent
        self.volume_threshold = volume_threshold
        self.use_relaxed_volume = use_relaxed_volume
        self.min_confirmation_bars = min_confirmation_bars
        self.sustained_bars_required = sustained_bars_required
        self.signal_cooldown_bars = signal_cooldown_bars
        self.use_or_bias_filter = use_or_bias_filter
        self.or_buffer_points = or_buffer_points
        self.opening_range_minutes = opening_range_minutes
        self.use_time_filter = use_time_filter
        self.rth_only = rth_only
        
        # VIX Regime settings
        self.use_vix_regime = use_vix_regime
        self.vix_high_threshold = vix_high_threshold
        self.vix_low_threshold = vix_low_threshold
        self.high_vol_cooldown_mult = high_vol_cooldown_mult
        self.low_vol_cooldown_mult = low_vol_cooldown_mult
        self.current_vix: float = 20.0  # Default VIX value
        
        # Signal enable flags
        self.enable_val_bounce = enable_val_bounce
        self.enable_poc_reclaim = enable_poc_reclaim
        self.enable_breakout = enable_breakout
        self.enable_sustained_breakout = enable_sustained_breakout
        self.enable_prior_val_bounce = enable_prior_val_bounce
        self.enable_prior_poc_reclaim = enable_prior_poc_reclaim
        self.enable_vah_rejection = enable_vah_rejection
        self.enable_poc_breakdown = enable_poc_breakdown
        self.enable_breakdown = enable_breakdown
        self.enable_sustained_breakdown = enable_sustained_breakdown
        self.enable_prior_vah_rejection = enable_prior_vah_rejection
        self.enable_prior_poc_breakdown = enable_prior_poc_breakdown
        
        # State
        self.bars: Deque[Bar] = deque(maxlen=length_period * 2)
        self.current_position: Direction = Direction.FLAT
        self.bars_since_signal: int = 999
        self.daily_trade_count: int = 0
        self.last_signal_date: Optional[date] = None
        
        # Session state
        self.session_high: float = 0
        self.session_low: float = float('inf')
        self.session_volume: int = 0
        self.session_vwap_sum: float = 0
        self.session_vol_weighted: float = 0
        
        # Opening range state
        self.or_high: float = 0
        self.or_low: float = float('inf')
        self.or_complete: bool = False
        self.or_bias: int = 0  # 1 = bullish, -1 = bearish, 0 = neutral
        self.first_breakout_dir: int = 0
        
        # Prior day state
        self.prior_day_vah: float = 0
        self.prior_day_val: float = 0
        self.prior_day_poc: float = 0
        
        # Tracking for signal conditions
        self.bars_above_vah: int = 0
        self.bars_below_val: int = 0
        
        # Last signal tracking for cooldown
        self.last_signal_bar: int = 0
        self.total_bars: int = 0
        
        # Log configuration
        logger.info(f"SignalDetector initialized:")
        logger.info(f"  length_period={length_period}, volume_threshold={volume_threshold}")
        logger.info(f"  signal_cooldown_bars={signal_cooldown_bars}, sustained_bars={sustained_bars_required}")
        logger.info(f"  use_vix_regime={use_vix_regime}")
        logger.info(f"  Enabled signals: VAL_BOUNCE={enable_val_bounce}, VAH_REJECTION={enable_vah_rejection}, "
                   f"BREAKOUT={enable_breakout}, BREAKDOWN={enable_breakdown}, "
                   f"SUSTAINED_BREAKOUT={enable_sustained_breakout}, SUSTAINED_BREAKDOWN={enable_sustained_breakdown}")
    
    def set_vix(self, vix_value: float) -> None:
        """Update the current VIX value for regime-based cooldown adjustment"""
        self.current_vix = vix_value
    
    def _get_effective_cooldown(self) -> int:
        """Get the effective cooldown based on VIX regime"""
        if not self.use_vix_regime:
            return self.signal_cooldown_bars
        
        # Treat VIX=0 or negative as unavailable, use default cooldown
        if self.current_vix <= 0:
            return self.signal_cooldown_bars
        
        if self.current_vix >= self.vix_high_threshold:
            # High volatility - longer cooldown
            return round(self.signal_cooldown_bars * self.high_vol_cooldown_mult)
        elif self.current_vix <= self.vix_low_threshold:
            # Low volatility - shorter cooldown (but minimum 3)
            return max(3, round(self.signal_cooldown_bars * self.low_vol_cooldown_mult))
        else:
            # Normal volatility
            return self.signal_cooldown_bars
    
    def reset_session(self) -> None:
        """Reset session state for new trading day"""
        # Save prior day values
        if self.session_high > 0:
            va = self._calculate_value_area()
            self.prior_day_vah = va.vah
            self.prior_day_val = va.val
            self.prior_day_poc = va.poc
        
        # Reset session
        self.session_high = 0
        self.session_low = float('inf')
        self.session_volume = 0
        self.session_vwap_sum = 0
        self.session_vol_weighted = 0
        
        # FIX: Clear bars from previous session to prevent cross-day pollution
        self.bars.clear()
        
        # Reset opening range
        self.or_high = 0
        self.or_low = float('inf')
        self.or_complete = False
        self.or_bias = 0
        self.first_breakout_dir = 0
        
        # Reset daily trade count
        self.daily_trade_count = 0
        
        # FIX: Reset position tracking counters
        self.bars_above_vah = 0
        self.bars_below_val = 0
        
        # FIX: Reset cooldown tracking
        self.bars_since_signal = 999
        
        logger.info("Session reset for new trading day")
    
    def add_bar(self, bar: Bar, suppress_signals: bool = False) -> Optional[Signal]:
        """
        Add a new bar and check for signals.
        
        Args:
            bar: The OHLCV bar to add
            suppress_signals: If True, update state but don't generate signals
                             (used for loading historical bars)
        
        Returns Signal if one is generated, None otherwise.
        """
        # Check for new day
        if self.bars and bar.timestamp.date() != self.bars[-1].timestamp.date():
            self.reset_session()
        
        # Reset cooldown at RTH open (9:30 AM) so globex signals don't suppress morning signals
        bar_time = bar.timestamp.time()
        prev_bar_time = self.bars[-1].timestamp.time() if self.bars else None
        rth_open = time(9, 30)
        
        if prev_bar_time and prev_bar_time < rth_open <= bar_time:
            self.bars_since_signal = 999
            logger.info("RTH open - cooldown reset")
        
        # Update session state
        self._update_session(bar)
        
        # Update opening range
        self._update_opening_range(bar)
        
        # Add bar to history
        self.bars.append(bar)
        self.total_bars += 1
        self.bars_since_signal += 1
        
        # Need enough history
        if len(self.bars) < self.length_period:
            return None
        
        # Calculate value area
        va = self._calculate_value_area()
        
        # Update position tracking
        self._update_position_tracking(bar, va)
        
        # If suppressing signals (historical load), stop here
        if suppress_signals:
            return None
        
        # Check time filters
        if not self._check_time_filters(bar):
            return None
        
        # Check for signals
        signal = self._check_signals(bar, va)
        
        if signal:
            self.bars_since_signal = 0
            self.last_signal_bar = self.total_bars
            self.daily_trade_count += 1
            
            # Update position
            if signal.direction == Direction.LONG:
                self.current_position = Direction.LONG
            elif signal.direction == Direction.SHORT:
                self.current_position = Direction.SHORT
        
        return signal
    
    def check_live_bar(self, live_bar: Bar) -> Optional[Signal]:
        """
        Check for signals on a live (in-progress) bar WITHOUT committing it to history.
        
        This allows the bot to fire signals mid-bar like ToS does, rather than
        waiting for bar close.
        
        Args:
            live_bar: The current in-progress bar (OHLCV from bar open to now)
        
        Returns Signal if conditions are met, None otherwise.
        """
        # Need enough history
        if len(self.bars) < self.length_period:
            return None
        
        # Temporarily add the live bar to calculate value area
        # We'll remove it after checking
        temp_bars = self.bars.copy()
        temp_bars.append(live_bar)
        
        # Calculate value area with live bar included
        # We need to temporarily update session state
        temp_session_high = max(self.session_high, live_bar.high)
        temp_session_low = min(self.session_low, live_bar.low)
        temp_session_volume = self.session_volume + live_bar.volume
        
        typical_price = (live_bar.high + live_bar.low + live_bar.close) / 3
        temp_vwap_sum = self.session_vwap_sum + live_bar.volume * typical_price
        temp_vol_weighted = self.session_vol_weighted + live_bar.volume * ((live_bar.high + live_bar.low) / 2)
        
        # Calculate VA using temp values
        if temp_session_volume == 0:
            return None
        
        vwap = temp_vwap_sum / temp_session_volume
        poc = temp_vol_weighted / temp_session_volume
        
        # FIX: Use only the last length_period bars for std dev (matching ToS)
        recent_bars = list(temp_bars)[-self.length_period:]
        if len(recent_bars) >= 2:
            closes = [b.close for b in recent_bars]
            std_dev = statistics.stdev(closes)
        else:
            std_dev = 0
        
        vah = vwap + (std_dev * 0.5)
        val = vwap - (std_dev * 0.5)
        
        va = ValueArea(vah=vah, val=val, poc=poc, vwap=vwap)
        
        # Check time filters
        if not self._check_time_filters(live_bar):
            return None
        
        # Check for signals using live bar
        signal = self._check_signals(live_bar, va)
        
        # Note: We don't update any state here - that happens when the completed bar comes in
        
        if signal:
            logger.info(f"[LIVE BAR] Signal detected mid-bar: {signal.signal_type.value}")
        
        return signal
    
    def _update_session(self, bar: Bar) -> None:
        """Update session-level tracking"""
        if bar.high > self.session_high:
            self.session_high = bar.high
        if bar.low < self.session_low:
            self.session_low = bar.low
        
        self.session_volume += bar.volume
        
        typical_price = (bar.high + bar.low + bar.close) / 3
        self.session_vwap_sum += bar.volume * typical_price
        self.session_vol_weighted += bar.volume * ((bar.high + bar.low) / 2)
    
    def _update_opening_range(self, bar: Bar) -> None:
        """Update opening range tracking"""
        bar_time = bar.timestamp.time()
        or_start = time(9, 30)
        or_end = time(10, 0)
        
        if bar_time >= or_start and bar_time < or_end:
            # Within opening range period
            if bar.high > self.or_high:
                self.or_high = bar.high
            if bar.low < self.or_low:
                self.or_low = bar.low
        elif bar_time >= or_end and not self.or_complete:
            # OR period just ended or we started after OR period
            self.or_complete = True
            
            # Check if we actually captured OR data
            if self.or_high > 0 and self.or_low < float('inf'):
                logger.info(f"Opening Range complete: High={self.or_high:.2f}, Low={self.or_low:.2f}")
            else:
                # We missed the OR period - use current value area as fallback
                logger.warning("Opening Range not captured (started after 10:00 AM)")
                logger.warning("Using Value Area as OR proxy - consider disabling OR bias filter")
                # Set OR to value area levels as a reasonable fallback
                va = self._calculate_value_area()
                if va.vah > 0:
                    self.or_high = va.vah
                    self.or_low = va.val
                    logger.info(f"OR proxy from VA: High={self.or_high:.2f}, Low={self.or_low:.2f}")
        
        # Update OR bias if complete and valid
        # Must match ToS logic: only BULLISH if clearly above OR+buffer, BEARISH if below OR-buffer
        # Everything else (including between OR high and OR high+buffer) is NEUTRAL
        if self.or_complete and self.or_high > 0 and self.or_low < float('inf'):
            if bar.close > self.or_high + self.or_buffer_points:
                self.or_bias = 1  # Bullish
                if self.first_breakout_dir == 0:
                    self.first_breakout_dir = 1
            elif bar.close < self.or_low - self.or_buffer_points:
                self.or_bias = -1  # Bearish
                if self.first_breakout_dir == 0:
                    self.first_breakout_dir = -1
            else:
                self.or_bias = 0  # Neutral - includes buffer zones and inside OR
    
    def _calculate_value_area(self) -> ValueArea:
        """
        Calculate current value area levels.
        
        FIXED: Now matches ToS calculation:
        - VAH = Session VWAP + (0.5 × StdDev of last N closes)
        - VAL = Session VWAP - (0.5 × StdDev of last N closes)
        - POC = Volume-weighted mid-price (session)
        
        The key fix is using only the last `length_period` bars for StdDev,
        matching ToS: StDev(close, lengthPeriod)
        """
        if self.session_volume == 0:
            return ValueArea(vah=0, val=0, poc=0, vwap=0)
        
        # VWAP (session-based)
        vwap = self.session_vwap_sum / self.session_volume
        
        # POC (volume-weighted price, session-based)
        poc = self.session_vol_weighted / self.session_volume
        
        # FIX: Use only the last length_period bars for std dev calculation
        # This matches ToS: StDev(close, lengthPeriod)
        recent_bars = list(self.bars)[-self.length_period:]
        
        if len(recent_bars) >= 2:
            closes = [bar.close for bar in recent_bars]
            std_dev = statistics.stdev(closes)
        else:
            std_dev = 0
        
        # Value Area = VWAP ± (0.5 × StdDev)
        vah = vwap + (std_dev * 0.5)
        val = vwap - (std_dev * 0.5)
        
        return ValueArea(vah=vah, val=val, poc=poc, vwap=vwap)
    
    def _update_position_tracking(self, bar: Bar, va: ValueArea) -> None:
        """Update bars above/below VAH/VAL tracking"""
        if bar.close > va.vah:
            self.bars_above_vah += 1
            self.bars_below_val = 0
        elif bar.close < va.val:
            self.bars_below_val += 1
            self.bars_above_vah = 0
        else:
            self.bars_above_vah = 0
            self.bars_below_val = 0
    
    def _check_time_filters(self, bar: Bar) -> bool:
        """Check if current time is valid for trading"""
        bar_time = bar.timestamp.time()
        
        # RTH check
        if self.rth_only:
            if bar_time < time(9, 30) or bar_time >= time(16, 0):
                return False
        
        # Optional time filter (avoid open, lunch, close)
        if self.use_time_filter:
            # Avoid first 25 minutes
            if bar_time < time(9, 55):
                return False
            # Avoid lunch
            if time(12, 0) <= bar_time < time(13, 30):
                return False
            # Avoid last 15 minutes
            if bar_time >= time(15, 45):
                return False
        
        return True
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed (with VIX adjustment)"""
        effective_cooldown = self._get_effective_cooldown()
        return self.bars_since_signal >= effective_cooldown
    
    def _check_volume_condition(self) -> bool:
        """Check volume conditions"""
        if len(self.bars) < self.length_period:
            return False
        
        current_volume = self.bars[-1].volume
        avg_volume = statistics.mean([b.volume for b in list(self.bars)[-self.length_period:]])
        
        if self.use_relaxed_volume:
            return current_volume > avg_volume
        else:
            return current_volume > (avg_volume * self.volume_threshold)
    
    def _check_volume_increasing(self) -> bool:
        """Check if volume is increasing"""
        if len(self.bars) < 2:
            return False
        
        if self.use_relaxed_volume:
            return self.bars[-1].volume > self.bars[-2].volume
        else:
            return (self.bars[-1].volume > self.bars[-2].volume and 
                    self.bars[-2].volume > self.bars[-3].volume if len(self.bars) >= 3 else False)
    
    def _check_or_allows(self, direction: Direction) -> bool:
        """Check if Opening Range bias allows this direction"""
        if not self.use_or_bias_filter:
            return True
        
        if not self.or_complete:
            return True  # Before OR complete, allow all
        
        if self.or_bias == 0:
            return True  # Neutral, allow all
        
        if direction == Direction.LONG and self.or_bias == 1:
            return True
        if direction == Direction.SHORT and self.or_bias == -1:
            return True
        
        return False
    
    def _check_signals(self, bar: Bar, va: ValueArea) -> Optional[Signal]:
        """Check for all signal types"""
        if not self._check_cooldown():
            return None
        
        prev_bar = self.bars[-2] if len(self.bars) >= 2 else None
        if not prev_bar:
            return None
        
        vol_condition = self._check_volume_condition()
        vol_increasing = self._check_volume_increasing()
        
        # Position filters
        allow_long = bar.close >= va.val
        allow_short = bar.close <= va.vah
        
        # Previous bar positions
        prev_above_vah = prev_bar.close > va.vah
        prev_below_val = prev_bar.close < va.val
        prev_above_poc = prev_bar.close > va.poc
        prev_below_poc = prev_bar.close < va.poc
        
        # Current positions
        price_above_vah = bar.close > va.vah
        price_below_val = bar.close < va.val
        price_above_poc = bar.close > va.poc
        price_below_poc = bar.close < va.poc
        
        # ==================== LONG SIGNALS ====================
        
        # 1. VAL Bounce
        val_touch = bar.low <= va.val and bar.close > va.val
        val_bounce = val_touch and vol_condition and bar.close > bar.open
        
        if self.enable_val_bounce and val_bounce and allow_long and self._check_or_allows(Direction.LONG):
            return Signal(
                signal_type=SignalType.VAL_BOUNCE,
                direction=Direction.LONG,
                timestamp=bar.timestamp,
                price=bar.close,
                vah=va.vah,
                val=va.val,
                poc=va.poc,
                or_bias=self.or_bias,
                reason="VAL Bounce - Price touched VAL and bounced with volume"
            )
        
        # 2. POC Reclaim
        poc_reclaim = prev_below_poc and price_above_poc and vol_increasing
        
        if self.enable_poc_reclaim and poc_reclaim and allow_long and self._check_or_allows(Direction.LONG):
            return Signal(
                signal_type=SignalType.POC_RECLAIM,
                direction=Direction.LONG,
                timestamp=bar.timestamp,
                price=bar.close,
                vah=va.vah,
                val=va.val,
                poc=va.poc,
                or_bias=self.or_bias,
                reason="POC Reclaim - Price crossed above POC with increasing volume"
            )
        
        # 3. Breakout (accepted above VAH)
        if self.enable_breakout and self.bars_above_vah >= self.min_confirmation_bars:
            breakout = not prev_above_vah and price_above_vah and vol_condition
            if breakout and allow_long and self._check_or_allows(Direction.LONG):
                return Signal(
                    signal_type=SignalType.BREAKOUT,
                    direction=Direction.LONG,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    vah=va.vah,
                    val=va.val,
                    poc=va.poc,
                    or_bias=self.or_bias,
                    reason=f"Breakout - Accepted above VAH for {self.bars_above_vah} bars"
                )
        
        # 4. Sustained Breakout
        if self.enable_sustained_breakout and self.bars_above_vah >= self.sustained_bars_required:
            prev_bars_above = self.bars_above_vah - 1
            if prev_bars_above < self.sustained_bars_required:
                if allow_long and self._check_or_allows(Direction.LONG):
                    return Signal(
                        signal_type=SignalType.SUSTAINED_BREAKOUT,
                        direction=Direction.LONG,
                        timestamp=bar.timestamp,
                        price=bar.close,
                        vah=va.vah,
                        val=va.val,
                        poc=va.poc,
                        or_bias=self.or_bias,
                        reason=f"Sustained Breakout - Above VAH for {self.bars_above_vah} bars"
                    )
        
        # ==================== SHORT SIGNALS ====================
        
        # 1. VAH Rejection
        vah_touch = bar.high >= va.vah and bar.close < va.vah
        vah_rejection = vah_touch and vol_condition and bar.close < bar.open
        
        if self.enable_vah_rejection and vah_rejection and allow_short and self._check_or_allows(Direction.SHORT):
            return Signal(
                signal_type=SignalType.VAH_REJECTION,
                direction=Direction.SHORT,
                timestamp=bar.timestamp,
                price=bar.close,
                vah=va.vah,
                val=va.val,
                poc=va.poc,
                or_bias=self.or_bias,
                reason="VAH Rejection - Price touched VAH and rejected with volume"
            )
        
        # 2. POC Breakdown
        poc_breakdown = prev_above_poc and price_below_poc and vol_increasing
        
        if self.enable_poc_breakdown and poc_breakdown and allow_short and self._check_or_allows(Direction.SHORT):
            return Signal(
                signal_type=SignalType.POC_BREAKDOWN,
                direction=Direction.SHORT,
                timestamp=bar.timestamp,
                price=bar.close,
                vah=va.vah,
                val=va.val,
                poc=va.poc,
                or_bias=self.or_bias,
                reason="POC Breakdown - Price crossed below POC with increasing volume"
            )
        
        # 3. Breakdown (accepted below VAL)
        if self.enable_breakdown and self.bars_below_val >= self.min_confirmation_bars:
            breakdown = not prev_below_val and price_below_val and vol_condition
            if breakdown and allow_short and self._check_or_allows(Direction.SHORT):
                return Signal(
                    signal_type=SignalType.BREAKDOWN,
                    direction=Direction.SHORT,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    vah=va.vah,
                    val=va.val,
                    poc=va.poc,
                    or_bias=self.or_bias,
                    reason=f"Breakdown - Accepted below VAL for {self.bars_below_val} bars"
                )
        
        # 4. Sustained Breakdown
        if self.enable_sustained_breakdown and self.bars_below_val >= self.sustained_bars_required:
            prev_bars_below = self.bars_below_val - 1
            if prev_bars_below < self.sustained_bars_required:
                if allow_short and self._check_or_allows(Direction.SHORT):
                    return Signal(
                        signal_type=SignalType.SUSTAINED_BREAKDOWN,
                        direction=Direction.SHORT,
                        timestamp=bar.timestamp,
                        price=bar.close,
                        vah=va.vah,
                        val=va.val,
                        poc=va.poc,
                        or_bias=self.or_bias,
                        reason=f"Sustained Breakdown - Below VAL for {self.bars_below_val} bars"
                    )
        
        # ==================== PRIOR DAY SIGNALS ====================
        
        if self.prior_day_vah > 0:
            # Prior VAL Bounce
            prior_val_touch = bar.low <= self.prior_day_val and bar.close > self.prior_day_val
            prior_val_bounce = prior_val_touch and vol_condition and bar.close > bar.open
            
            if self.enable_prior_val_bounce and prior_val_bounce and allow_long and self._check_or_allows(Direction.LONG):
                return Signal(
                    signal_type=SignalType.PRIOR_VAL_BOUNCE,
                    direction=Direction.LONG,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    vah=va.vah,
                    val=va.val,
                    poc=va.poc,
                    or_bias=self.or_bias,
                    reason="Prior Day VAL Bounce"
                )
            
            # Prior VAH Rejection
            prior_vah_touch = bar.high >= self.prior_day_vah and bar.close < self.prior_day_vah
            prior_vah_rejection = prior_vah_touch and vol_condition and bar.close < bar.open
            
            if self.enable_prior_vah_rejection and prior_vah_rejection and allow_short and self._check_or_allows(Direction.SHORT):
                return Signal(
                    signal_type=SignalType.PRIOR_VAH_REJECTION,
                    direction=Direction.SHORT,
                    timestamp=bar.timestamp,
                    price=bar.close,
                    vah=va.vah,
                    val=va.val,
                    poc=va.poc,
                    or_bias=self.or_bias,
                    reason="Prior Day VAH Rejection"
                )
        
        return None
    
    def get_current_levels(self) -> Optional[ValueArea]:
        """Get current value area levels"""
        if len(self.bars) < self.length_period:
            return None
        return self._calculate_value_area()
    
    def get_state_summary(self) -> dict:
        """Get current detector state summary"""
        va = self._calculate_value_area() if len(self.bars) >= self.length_period else None
        effective_cooldown = self._get_effective_cooldown()
        
        return {
            "position": self.current_position.value,
            "or_bias": "BULLISH" if self.or_bias == 1 else "BEARISH" if self.or_bias == -1 else "NEUTRAL",
            "or_complete": self.or_complete,
            "or_high": self.or_high,
            "or_low": self.or_low,
            "daily_trades": self.daily_trade_count,
            "bars_since_signal": self.bars_since_signal,
            "cooldown_clear": self._check_cooldown(),
            "effective_cooldown": effective_cooldown,
            "vix": self.current_vix,
            "vah": va.vah if va else 0,
            "val": va.val if va else 0,
            "poc": va.poc if va else 0,
            "bars_above_vah": self.bars_above_vah,
            "bars_below_val": self.bars_below_val
        }