"""
Apply Optimized Configuration
==============================

Reads optimization results (JSON) and:
1. Updates config.py with new parameters
2. Generates updated ThinkScript (.ts) file

Usage:
    python apply_config.py --from multirun_results/recommended_config.json
    python apply_config.py --from best_params.json --preview  # Preview only, don't write
    python apply_config.py --from multirun_results/analysis.json  # Also works with analysis.json
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any


def load_params(filepath: str) -> Dict[str, Any]:
    """Load parameters from JSON file (handles multiple formats)"""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "recommended_config" in data:
        # From analysis.json
        return data["recommended_config"]
    elif "params" in data:
        # From best_params.json or individual run
        return data["params"]
    else:
        # Direct params dict
        return data


def generate_config_py(params: Dict[str, Any]) -> str:
    """Generate updated config.py content matching BotConfig structure for trading_bot.py"""
    
    # Extract params with defaults
    p = params
    
    config_content = f'''"""
Configuration settings for the ToS Signal Trading Bot
======================================================
Auto-generated from optimization results on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

To regenerate: python apply_config.py --from <results.json>
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import time

@dataclass
class SchwabConfig:
    """Schwab API configuration"""
    app_key: str = os.getenv("SCHWAB_APP_KEY", "")
    app_secret: str = os.getenv("SCHWAB_APP_SECRET", "")
    redirect_uri: str = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback")
    token_file: str = "schwab_tokens.json"
    account_hash: Optional[str] = None  # Will be fetched on first run


@dataclass
class TradingConfig:
    """Trading parameters - OPTIMIZED"""
    symbol: str = "SPY"
    contracts: int = 1
    max_daily_trades: int = {p.get('max_daily_trades', 3)}  # OPTIMIZED
    
    # Option selection - OPTIMIZED
    use_delta_targeting: bool = True
    target_delta: float = {p.get('target_delta', 0.30):.2f}  # OPTIMIZED
    afternoon_delta: float = {p.get('afternoon_delta', 0.40):.2f}  # OPTIMIZED
    afternoon_start_hour: int = {p.get('afternoon_hour', 12)}  # OPTIMIZED
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    # Position management
    hold_until_opposite_signal: bool = True
    
    # Position Sizing - Kelly - OPTIMIZED
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = {p.get('max_equity_risk', 0.10) * 100:.1f}  # OPTIMIZED (converted to %)
    max_position_size: int = {p.get('hard_max_contracts', 100)}  # OPTIMIZED
    min_position_size: int = 1
    
    # Kelly-specific params
    kelly_fraction: float = {p.get('kelly_fraction', 0.0):.3f}  # OPTIMIZED
    max_kelly_pct_cap: float = {p.get('max_kelly_pct_cap', 0.35):.3f}  # OPTIMIZED
    kelly_lookback: int = {p.get('kelly_lookback', 20)}  # OPTIMIZED
    
    # Risk management - Daily Loss Limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    # Risk management - Stop Loss - OPTIMIZED
    enable_stop_loss: bool = {p.get('enable_stop_loss', False)}  # OPTIMIZED
    stop_loss_percent: float = {p.get('stop_loss_percent', 50.0):.1f}  # OPTIMIZED
    stop_loss_dollars: float = 500.0
    
    # Risk management - Take Profit - OPTIMIZED
    enable_take_profit: bool = {p.get('enable_take_profit', False)}  # OPTIMIZED
    take_profit_percent: float = {p.get('take_profit_percent', 100.0):.1f}  # OPTIMIZED
    take_profit_dollars: float = 500.0
    
    # Risk management - Trailing Stop - OPTIMIZED
    enable_trailing_stop: bool = {p.get('enable_trailing_stop', False)}  # OPTIMIZED
    trailing_stop_percent: float = {p.get('trailing_stop_percent', 25.0):.1f}  # OPTIMIZED
    trailing_stop_activation: float = {p.get('trailing_stop_activation', 50.0):.1f}  # OPTIMIZED
    
    # Min hold time - OPTIMIZED
    min_hold_bars: int = {p.get('min_hold_bars', 0)}  # OPTIMIZED
    
    # Correlation / Exposure Check
    enable_correlation_check: bool = True
    max_delta_exposure: float = 100.0


@dataclass
class TimeConfig:
    """Trading hours configuration (Eastern Time)"""
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    
    # Opening range period
    or_start: time = field(default_factory=lambda: time(9, 30))
    or_end: time = field(default_factory=lambda: time(10, 0))
    
    # Avoid trading during these periods
    avoid_first_minutes: int = 25
    avoid_lunch_start: time = field(default_factory=lambda: time(12, 0))
    avoid_lunch_end: time = field(default_factory=lambda: time(13, 30))
    avoid_last_minutes: int = 15
    
    use_time_filter: bool = {p.get('use_time_filter', False)}  # OPTIMIZED
    rth_only: bool = True


@dataclass 
class SignalConfig:
    """Signal detection configuration - OPTIMIZED"""
    # Value area settings
    length_period: int = 20
    value_area_percent: float = 70.0
    
    # Volume thresholds - OPTIMIZED
    volume_threshold: float = {p.get('volume_threshold', 1.3):.3f}  # OPTIMIZED
    use_relaxed_volume: bool = True
    
    # Confirmation - OPTIMIZED
    min_confirmation_bars: int = {p.get('min_confirmation_bars', 2)}  # OPTIMIZED
    sustained_bars_required: int = {p.get('sustained_bars_required', 3)}  # OPTIMIZED
    signal_cooldown_bars: int = {p.get('signal_cooldown_bars', 8)}  # OPTIMIZED
    
    # Opening range bias - OPTIMIZED
    use_or_bias_filter: bool = {p.get('use_or_bias_filter', True)}  # OPTIMIZED
    or_buffer_points: float = 1.0
    
    # Minimum move for credit
    min_es_point_move: float = 6.0
    
    # VIX Regime Settings - OPTIMIZED
    use_vix_regime: bool = {p.get('use_vix_regime', False)}  # OPTIMIZED
    vix_high_threshold: int = {p.get('vix_high_threshold', 25)}  # OPTIMIZED
    vix_low_threshold: int = {p.get('vix_low_threshold', 15)}  # OPTIMIZED
    
    # High Vol Adjustments - OPTIMIZED
    high_vol_cooldown_mult: float = {p.get('high_vol_cooldown_mult', 1.5):.2f}  # OPTIMIZED
    high_vol_confirmation_mult: float = {p.get('high_vol_confirmation_mult', 1.5):.2f}  # OPTIMIZED
    high_vol_sustained_mult: float = {p.get('high_vol_sustained_mult', 1.5):.2f}  # OPTIMIZED
    high_vol_volume_add: float = {p.get('high_vol_volume_add', 0.2):.2f}  # OPTIMIZED
    high_vol_delta_adj: float = {p.get('high_vol_delta_adj', 0.05):.3f}  # OPTIMIZED
    high_vol_min_hold_mult: float = {p.get('high_vol_min_hold_mult', 1.5):.2f}  # OPTIMIZED
    
    # Low Vol Adjustments - OPTIMIZED
    low_vol_cooldown_mult: float = {p.get('low_vol_cooldown_mult', 0.7):.2f}  # OPTIMIZED
    low_vol_confirmation_mult: float = {p.get('low_vol_confirmation_mult', 0.8):.2f}  # OPTIMIZED
    low_vol_sustained_mult: float = {p.get('low_vol_sustained_mult', 0.8):.2f}  # OPTIMIZED
    low_vol_volume_add: float = {p.get('low_vol_volume_add', -0.1):.2f}  # OPTIMIZED
    low_vol_delta_adj: float = {p.get('low_vol_delta_adj', -0.05):.3f}  # OPTIMIZED
    
    # VWAP Filter - OPTIMIZED (institutional anchor)
    use_vwap_filter: bool = {p.get('use_vwap_filter', False)}  # OPTIMIZED
    vwap_filter_mode: str = "{p.get('vwap_filter_mode', 'strict')}"  # OPTIMIZED - "strict" or "confirm"
    
    # NYSE TICK Filter - OPTIMIZED (breadth confirmation)
    use_tick_filter: bool = {p.get('use_tick_filter', False)}  # OPTIMIZED
    tick_extreme_threshold: int = {p.get('tick_extreme_threshold', 500)}  # OPTIMIZED
    
    # Time Window Settings - OPTIMIZED (market mechanics)
    signal_start_minutes: int = {p.get('signal_start_minutes', 0)}  # OPTIMIZED - Minutes after 9:30 to start
    signal_end_minutes: int = {p.get('signal_end_minutes', 0)}  # OPTIMIZED - Minutes before 16:00 to stop
    
    # ATR-Based Stops - OPTIMIZED
    use_atr_stops: bool = {p.get('use_atr_stops', False)}  # OPTIMIZED
    atr_stop_mult: float = {p.get('atr_stop_mult', 2.0):.2f}  # OPTIMIZED
    atr_target_mult: float = {p.get('atr_target_mult', 3.0):.2f}  # OPTIMIZED
    
    # Min Premium Filter - OPTIMIZED (avoid illiquid strikes)
    min_option_premium: float = {p.get('min_option_premium', 0.25):.2f}  # OPTIMIZED
    
    # Signal enables - LONG signals - OPTIMIZED
    enable_val_bounce: bool = {p.get('enable_val_bounce', True)}  # OPTIMIZED
    enable_poc_reclaim: bool = {p.get('enable_poc_reclaim', True)}  # OPTIMIZED
    enable_breakout: bool = {p.get('enable_breakout', True)}  # OPTIMIZED
    enable_sustained_breakout: bool = {p.get('enable_sustained_breakout', True)}  # OPTIMIZED
    
    # Signal enables - SHORT signals - OPTIMIZED
    enable_vah_rejection: bool = {p.get('enable_vah_rejection', True)}  # OPTIMIZED
    enable_poc_breakdown: bool = {p.get('enable_poc_breakdown', True)}  # OPTIMIZED
    enable_breakdown: bool = {p.get('enable_breakdown', True)}  # OPTIMIZED
    enable_sustained_breakdown: bool = {p.get('enable_sustained_breakdown', True)}  # OPTIMIZED


@dataclass
class AlertConfig:
    """Alert/notification settings"""
    enable_discord: bool = False
    discord_webhook: str = os.getenv("DISCORD_WEBHOOK", "")
    
    enable_telegram: bool = False
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    log_to_file: bool = True
    log_file: str = "trading_bot.log"


@dataclass
class AnalyticsConfig:
    """Analytics and reporting settings"""
    enable_daily_summary: bool = True
    daily_summary_hour: int = 16
    daily_summary_minute: int = 15
    
    enable_weekly_report: bool = True
    
    enable_drawdown_alerts: bool = True
    drawdown_alert_threshold: float = 10.0
    
    performance_data_file: str = "performance_data.json"


@dataclass
class BotConfig:
    """Main bot configuration"""
    schwab: SchwabConfig = field(default_factory=SchwabConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    
    # Data source
    data_poll_interval: int = 5
    use_streaming: bool = True
    
    # Intra-bar signal checking
    enable_intra_bar_signals: bool = True
    intra_bar_check_interval: int = 15
    
    # Paper trading mode
    paper_trading: bool = True  # Set to False for live trading
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.schwab.app_key or not self.schwab.app_secret:
            raise ValueError("Schwab API credentials not configured")
        return True


# Default configuration instance
config = BotConfig()
'''
    
    return config_content


def generate_thinkscript(params: Dict[str, Any]) -> str:
    """Generate updated ThinkScript with optimized signal parameters - V3.5 format"""
    
    p = params
    
    # Convert Python bools to ThinkScript yes/no
    def ts_yn(val, default=True):
        v = val if val is not None else default
        return "yes" if v else "no"
    
    thinkscript = f'''# Auction Market Theory Indicator with Trade Signals - V3.5 OPTIMIZED
# Based on Volume Profile, Value Area, Point of Control, and Price Action
# OPTIMIZED: Parameters tuned by Bayesian optimizer on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# To regenerate: python apply_config.py --from <results.json>

declare upper;

# ==================== INPUTS - OPTIMIZED VALUES ====================
input lengthPeriod = 20;
input valueAreaPercent = 70;
input showValueArea = yes;
input showPOC = yes;
input showSignals = yes;
input volumeThreshold = {p.get('volume_threshold', 1.3):.2f}; # OPTIMIZED
input enableAlerts = yes;
input minConfirmationBars = {p.get('min_confirmation_bars', 2)}; # OPTIMIZED
input useSessionVA = yes;

# Credit Optimization Settings
input minESPointMove = 6;
input wingEntryDelay = 1;
input bodyEntryDelay = 3;
input showCreditZones = yes;

# Take Profit Settings
input showTargets = no;
input target1Multiplier = 1.5; 
input target2Multiplier = 3.0; 
input useVATargets = no;

# Prior Day Settings
input showPriorDayVA = no;
input tradePriorDayLevels = yes; 

# Win Rate & EV Optimization Settings
input useTrendFilter = no;
input useTimeFilter = {ts_yn(p.get('use_time_filter', False))}; # OPTIMIZED
input rthOnlySignals = yes;
input showStopLoss = no;
input riskRewardRatio = 2.0;
input tickSize = 0.25; 

# Time Window Settings - OPTIMIZED (market mechanics)
input signalStartMinutes = {p.get('signal_start_minutes', 0)}; # OPTIMIZED - Minutes after 9:30 to start
input signalEndMinutes = {p.get('signal_end_minutes', 0)}; # OPTIMIZED - Minutes before 16:00 to stop

# ATR-Based Stops - OPTIMIZED
input useATRStops = {ts_yn(p.get('use_atr_stops', False))}; # OPTIMIZED
input atrStopMult = {p.get('atr_stop_mult', 2.0):.2f}; # OPTIMIZED
input atrTargetMult = {p.get('atr_target_mult', 3.0):.2f}; # OPTIMIZED

# Min Premium Filter - OPTIMIZED
input minOptionPremium = {p.get('min_option_premium', 0.25):.2f}; # OPTIMIZED - Skip illiquid strikes

# VWAP Filter - OPTIMIZED (institutional anchor)
input useVWAPFilter = {ts_yn(p.get('use_vwap_filter', False))}; # OPTIMIZED
input vwapFilterMode = {{"strict", "confirm"}}[{0 if p.get('vwap_filter_mode', 'strict') == 'strict' else 1}]; # OPTIMIZED

# NYSE TICK Filter - OPTIMIZED (breadth confirmation)
input useTICKFilter = {ts_yn(p.get('use_tick_filter', False))}; # OPTIMIZED
input tickExtremeThreshold = {p.get('tick_extreme_threshold', 500)}; # OPTIMIZED - Block signals against extreme TICK

# Display Simplification
input showOnlyActiveSignals = yes; 
input hideValueAreaCloud = yes;

# Signal Sensitivity Settings - OPTIMIZED
input useRelaxedVolume = yes;
input sustainedBarsRequired = {p.get('sustained_bars_required', 3)}; # OPTIMIZED (base value)
input showDebugLabels = no;
input signalCooldownBars = {p.get('signal_cooldown_bars', 8)}; # OPTIMIZED (base cooldown)

# VIX Regime Settings - OPTIMIZED
input useVixRegime = {ts_yn(p.get('use_vix_regime', False))}; # OPTIMIZED
input vixHighThreshold = {p.get('vix_high_threshold', 25)}; # OPTIMIZED - VIX above this = HIGH vol
input vixLowThreshold = {p.get('vix_low_threshold', 15)}; # OPTIMIZED - VIX below this = LOW vol

# High Vol Adjustments - OPTIMIZED
input highVolCooldownMult = {p.get('high_vol_cooldown_mult', 1.5):.2f}; # OPTIMIZED
input highVolConfirmationMult = {p.get('high_vol_confirmation_mult', 1.5):.2f}; # OPTIMIZED
input highVolSustainedMult = {p.get('high_vol_sustained_mult', 1.5):.2f}; # OPTIMIZED
input highVolVolumeAdd = {p.get('high_vol_volume_add', 0.2):.2f}; # OPTIMIZED

# Low Vol Adjustments - OPTIMIZED
input lowVolCooldownMult = {p.get('low_vol_cooldown_mult', 0.7):.2f}; # OPTIMIZED
input lowVolConfirmationMult = {p.get('low_vol_confirmation_mult', 0.8):.2f}; # OPTIMIZED
input lowVolSustainedMult = {p.get('low_vol_sustained_mult', 0.8):.2f}; # OPTIMIZED
input lowVolVolumeAdd = {p.get('low_vol_volume_add', -0.1):.2f}; # OPTIMIZED

# Exit Signal Settings
input showExitSignals = yes;
input showPositionStatus = yes;
input showHoldDuration = yes;
input exitAlertSound = yes;
input showSignalBubbles = no;

# Daily Trade Lockout System
input maxDailyTrades = {p.get('max_daily_trades', 3)}; # OPTIMIZED
input enableLockout = yes;
input showTradeCounter = yes;
input lockoutWarningAt = 2;

# Opening Range Bias - OPTIMIZED
input openingRangeMinutes = 30;
input showOpeningRange = yes;
input useORBiasFilter = {ts_yn(p.get('use_or_bias_filter', True))}; # OPTIMIZED
input orBufferPoints = 1.0;

# Put/Call Zone Display
input showPutCallZones = yes;

# ==================== SIGNAL ENABLE TOGGLES - OPTIMIZED ====================
# These control which signals are active
input enableVALBounce = {ts_yn(p.get('enable_val_bounce', True))}; # OPTIMIZED
input enableVAHRejection = {ts_yn(p.get('enable_vah_rejection', True))}; # OPTIMIZED
input enablePOCReclaim = {ts_yn(p.get('enable_poc_reclaim', True))}; # OPTIMIZED
input enablePOCBreakdown = {ts_yn(p.get('enable_poc_breakdown', True))}; # OPTIMIZED
input enableBreakout = {ts_yn(p.get('enable_breakout', True))}; # OPTIMIZED
input enableBreakdown = {ts_yn(p.get('enable_breakdown', True))}; # OPTIMIZED
input enableSustainedBreakout = {ts_yn(p.get('enable_sustained_breakout', True))}; # OPTIMIZED
input enableSustainedBreakdown = {ts_yn(p.get('enable_sustained_breakdown', True))}; # OPTIMIZED

# ==================== VALUE AREA CALCULATIONS ====================

def isNewDay = GetDay() != GetDay()[1];
def sessionHigh = if isNewDay then high else if high > sessionHigh[1] then high else sessionHigh[1];
def sessionLow = if isNewDay then low else if low < sessionLow[1] then low else sessionLow[1];
def sessionVolume = if isNewDay then volume else sessionVolume[1] + volume;
def sessionVWAPSum = if isNewDay then (volume * (high + low + close) / 3) 
                     else sessionVWAPSum[1] + (volume * (high + low + close) / 3);
def sessionVolWeighted = if isNewDay then (volume * ((high + low) / 2))
                         else sessionVolWeighted[1] + (volume * ((high + low) / 2));

# ==================== OPENING RANGE CALCULATION ====================

def orStartTime = 0930;
def isInOpeningRange = SecondsFromTime(orStartTime) >= 0 and SecondsFromTime(1000) < 0;
def orComplete = SecondsFromTime(1000) >= 0;

rec openingRangeHigh = if isNewDay then high 
                       else if isInOpeningRange and high > openingRangeHigh[1] then high
                       else openingRangeHigh[1];
             
rec openingRangeLow = if isNewDay then low
                      else if isInOpeningRange and low < openingRangeLow[1] then low
                      else openingRangeLow[1];

def orMid = (openingRangeHigh + openingRangeLow) / 2;
def orSize = openingRangeHigh - openingRangeLow;

def aboveORHigh = close > (openingRangeHigh + orBufferPoints);
def belowORLow = close < (openingRangeLow - orBufferPoints);

def orBias = if !orComplete then 0
             else if aboveORHigh then 1
             else if belowORLow then -1
             else 0;

rec firstBreakoutDir = if isNewDay then 0
                       else if firstBreakoutDir[1] != 0 then firstBreakoutDir[1]
                       else if orComplete and aboveORHigh then 1
                       else if orComplete and belowORLow then -1
                       else 0;

# Rolling calculations
def lengthPeriodVal = lengthPeriod;
def rollingHigh = highest(high, lengthPeriodVal);
def rollingLow = lowest(low, lengthPeriodVal);
def rollingVolume = sum(volume, lengthPeriodVal);
def rollingVWAPSum = sum(volume * (high + low + close) / 3, lengthPeriodVal);
def rollingVolWeighted = sum(volume * ((high + low) / 2), lengthPeriodVal);

def priceHigh = if useSessionVA then sessionHigh else rollingHigh;
def priceLow = if useSessionVA then sessionLow else rollingLow;
def priceRange = priceHigh - priceLow;

def volumeSum = if useSessionVA then sessionVolume else rollingVolume;
def vwapSum = if useSessionVA then sessionVWAPSum else rollingVWAPSum;
def vwapValue = vwapSum / volumeSum;

def volWeightedSum = if useSessionVA then sessionVolWeighted else rollingVolWeighted;
def pocValue = volWeightedSum / volumeSum;

def stdDev = StDev(close, lengthPeriodVal);
def valueAreaHigh = vwapValue + (stdDev * 0.5);
def valueAreaLow = vwapValue - (stdDev * 0.5);

# ==================== PRIOR DAY VALUE AREA ====================

def priorDayVAH = if isNewDay then valueAreaHigh[1] else priorDayVAH[1];
def priorDayVAL = if isNewDay then valueAreaLow[1] else priorDayVAL[1];
def priorDayPOC = if isNewDay then pocValue[1] else priorDayPOC[1];
def hasPriorDay = !IsNaN(priorDayVAH) and priorDayVAH != 0;

# ==================== PRICE POSITION ====================

def priceAboveVAH = close > valueAreaHigh;
def priceBelowVAL = close < valueAreaLow;
def priceInsideVA = close >= valueAreaLow and close <= valueAreaHigh;
def priceAbovePOC = close > pocValue;
def priceBelowPOC = close < pocValue;

def prevAboveVAH = close[1] > valueAreaHigh[1];
def prevBelowVAL = close[1] < valueAreaLow[1];
def prevAbovePOC = close[1] > pocValue[1];
def prevBelowPOC = close[1] < pocValue[1];

def priceAbovePriorVAH = close > priorDayVAH;
def priceBelowPriorVAL = close < priorDayVAL;
def priceAbovePriorPOC = close > priorDayPOC;
def priceBelowPriorPOC = close < priorDayPOC;

def prevAbovePriorVAH = close[1] > priorDayVAH[1];
def prevBelowPriorVAL = close[1] > priorDayVAL[1];
def prevAbovePriorPOC = close[1] > priorDayPOC[1];
def prevBelowPriorPOC = close[1] < priorDayPOC[1];

# ==================== VOLUME ANALYSIS ====================

def avgVolume = Average(volume, lengthPeriodVal);
def volumeSpike = volume > (avgVolume * volumeThreshold);
def volumeElevated = volume > avgVolume;
def increasingVolume = volume > volume[1];
def increasingVolume2Bar = volume > volume[1] and volume[1] > volume[2];

def volCondition = if useRelaxedVolume then volumeElevated else volumeSpike;
def volIncreasing = if useRelaxedVolume then increasingVolume else increasingVolume2Bar;

# ==================== FILTERS ====================

def trendFilterLong = yes;
def trendFilterShort = yes;

def timeNow = SecondsFromTime(0930);
def marketClose = SecondsTillTime(1600);
def isRTH = timeNow >= 0 and marketClose >= 0;
def avoidLunch = timeNow >= 9000 and timeNow <= 12600;
def avoidClose = marketClose < 900;
def goodTradingTime = isRTH and !avoidLunch and !avoidClose;

# Time Window Filter - OPTIMIZED (market mechanics)
def minutesSinceOpen = timeNow / 60;  # Convert seconds to minutes
def minutesUntilClose = marketClose / 60;
def inTimeWindow = minutesSinceOpen >= signalStartMinutes and minutesUntilClose >= signalEndMinutes;

def timeFilter = if useTimeFilter then goodTradingTime else yes;
def rthFilter = if rthOnlySignals then isRTH else yes;
def timeWindowFilter = if signalStartMinutes > 0 or signalEndMinutes > 0 then inTimeWindow else yes;

# VWAP Filter - Institutional anchor
def vwapValue = vwap;
def priceAboveVWAP = close >= vwapValue;
def priceBelowVWAP = close <= vwapValue;

# VWAP allows based on mode
def vwapAllowsLong = if !useVWAPFilter then yes
                     else priceAboveVWAP;
def vwapAllowsShort = if !useVWAPFilter then yes
                      else priceBelowVWAP;

# NYSE TICK Filter - Breadth confirmation
def tickValue = close("$TICK");
def tickAllowsLong = if !useTICKFilter then yes
                     else tickValue > -tickExtremeThreshold;  # Don't long in extreme negative breadth
def tickAllowsShort = if !useTICKFilter then yes
                      else tickValue < tickExtremeThreshold;   # Don't short in extreme positive breadth

# ==================== SIGNAL LOGIC - WITH ENABLE TOGGLES ====================

# --- RAW SIGNAL CONDITIONS ---

# LONG RAW 1: VAL Bounce
def valTouch = low <= valueAreaLow and close > valueAreaLow;
def valBounce = valTouch and volCondition and close > open;
def valBounceRaw = enableVALBounce and valBounce and !valBounce[1] and timeFilter and rthFilter and timeWindowFilter;

# LONG RAW 2: POC Reclaim
def pocReclaim = priceBelowPOC[1] and priceAbovePOC and volIncreasing;
def pocReclaimRaw = enablePOCReclaim and pocReclaim and !pocReclaim[1] and timeFilter and rthFilter and timeWindowFilter;

# LONG RAW 3: Breakout above VAH
def breakoutBar = !prevAboveVAH and priceAboveVAH and volCondition;
def barsAboveVAH = if priceAboveVAH then barsAboveVAH[1] + 1 else 0;
def breakoutAcceptance = breakoutBar[minConfirmationBars] and barsAboveVAH >= minConfirmationBars;
def breakoutRaw = enableBreakout and breakoutAcceptance and !breakoutAcceptance[1] and timeFilter and rthFilter and timeWindowFilter;

# LONG RAW 4: Sustained breakout
def sustainedAboveVAH = barsAboveVAH >= sustainedBarsRequired;
def sustainedBreakoutRaw = enableSustainedBreakout and sustainedAboveVAH and !sustainedAboveVAH[1] and timeFilter and rthFilter and timeWindowFilter;

# SHORT RAW 1: VAH Rejection
def vahTouch = high >= valueAreaHigh and close < valueAreaHigh;
def vahRejection = vahTouch and volCondition and close < open;
def vahRejectionRaw = enableVAHRejection and vahRejection and !vahRejection[1] and timeFilter and rthFilter and timeWindowFilter;

# SHORT RAW 2: POC Breakdown
def pocBreakdown = prevAbovePOC and priceBelowPOC and volIncreasing;
def pocBreakdownRaw = enablePOCBreakdown and pocBreakdown and !pocBreakdown[1] and timeFilter and rthFilter and timeWindowFilter;

# SHORT RAW 3: Breakdown below VAL
def breakdownBar = !prevBelowVAL and priceBelowVAL and volCondition;
def barsBelowVAL = if priceBelowVAL then barsBelowVAL[1] + 1 else 0;
def breakdownAcceptance = breakdownBar[minConfirmationBars] and barsBelowVAL >= minConfirmationBars;
def breakdownRaw = enableBreakdown and breakdownAcceptance and !breakdownAcceptance[1] and timeFilter and rthFilter and timeWindowFilter;

# SHORT RAW 4: Sustained breakdown
def sustainedBelowVAL = barsBelowVAL >= sustainedBarsRequired;
def sustainedBreakdownRaw = enableSustainedBreakdown and sustainedBelowVAL and !sustainedBelowVAL[1] and timeFilter and rthFilter and timeWindowFilter;

# --- COOLDOWN LOGIC ---
def anyRawLong = valBounceRaw or pocReclaimRaw or breakoutRaw or sustainedBreakoutRaw;
def anyRawShort = vahRejectionRaw or pocBreakdownRaw or breakdownRaw or sustainedBreakdownRaw;
def anyRawSignal = anyRawLong or anyRawShort;

# VIX Regime Detection
def vix = close("VIX");
def vixRegime = if !useVixRegime then 0
                else if vix >= vixHighThreshold then 1   # HIGH vol
                else if vix <= vixLowThreshold then -1   # LOW vol  
                else 0;                                  # NORMAL

# Dynamic parameters based on VIX regime
# Cooldown
def effectiveCooldown = if !useVixRegime then signalCooldownBars
                        else if vixRegime == 1 then Round(signalCooldownBars * highVolCooldownMult, 0)
                        else if vixRegime == -1 then Max(3, Round(signalCooldownBars * lowVolCooldownMult, 0))
                        else signalCooldownBars;

# Confirmation bars (for breakout/breakdown acceptance)
def effectiveConfirmation = if !useVixRegime then minConfirmationBars
                            else if vixRegime == 1 then Round(minConfirmationBars * highVolConfirmationMult, 0)
                            else if vixRegime == -1 then Max(1, Round(minConfirmationBars * lowVolConfirmationMult, 0))
                            else minConfirmationBars;

# Sustained bars
def effectiveSustained = if !useVixRegime then sustainedBarsRequired
                         else if vixRegime == 1 then Round(sustainedBarsRequired * highVolSustainedMult, 0)
                         else if vixRegime == -1 then Max(1, Round(sustainedBarsRequired * lowVolSustainedMult, 0))
                         else sustainedBarsRequired;

# Volume threshold
def effectiveVolume = if !useVixRegime then volumeThreshold
                      else if vixRegime == 1 then volumeThreshold + highVolVolumeAdd
                      else if vixRegime == -1 then Max(1.0, volumeThreshold + lowVolVolumeAdd)
                      else volumeThreshold;

# --- VIX-ADJUSTED SIGNAL CHECKS ---
# Override the raw signals with VIX-adjusted versions
# These re-check the bar counts against effective thresholds

# Breakout: needs effectiveConfirmation bars above VAH
def breakoutAcceptanceVix = breakoutBar[effectiveConfirmation] and barsAboveVAH >= effectiveConfirmation;
def breakoutRawVix = enableBreakout and breakoutAcceptanceVix and !breakoutAcceptanceVix[1] and timeFilter and rthFilter;

# Sustained breakout: needs effectiveSustained bars above VAH  
def sustainedAboveVAHVix = barsAboveVAH >= effectiveSustained;
def sustainedBreakoutRawVix = enableSustainedBreakout and sustainedAboveVAHVix and !sustainedAboveVAHVix[1] and timeFilter and rthFilter;

# Breakdown: needs effectiveConfirmation bars below VAL
def breakdownAcceptanceVix = breakdownBar[effectiveConfirmation] and barsBelowVAL >= effectiveConfirmation;
def breakdownRawVix = enableBreakdown and breakdownAcceptanceVix and !breakdownAcceptanceVix[1] and timeFilter and rthFilter;

# Sustained breakdown: needs effectiveSustained bars below VAL
def sustainedBelowVALVix = barsBelowVAL >= effectiveSustained;
def sustainedBreakdownRawVix = enableSustainedBreakdown and sustainedBelowVALVix and !sustainedBelowVALVix[1] and timeFilter and rthFilter;

# Volume condition with VIX adjustment
def volConditionVix = if useRelaxedVolume 
                      then (volRatio > effectiveVolume * 0.8) or (close > open and volRatio > effectiveVolume * 0.5)
                      else volRatio > effectiveVolume;

# Use VIX-adjusted signals when VIX regime is enabled
def breakoutFinal = if useVixRegime then breakoutRawVix else breakoutRaw;
def sustainedBreakoutFinal = if useVixRegime then sustainedBreakoutRawVix else sustainedBreakoutRaw;
def breakdownFinal = if useVixRegime then breakdownRawVix else breakdownRaw;
def sustainedBreakdownFinal = if useVixRegime then sustainedBreakdownRawVix else sustainedBreakdownRaw;

rec barsSinceLastSignal = if barsSinceLastSignal[1] >= effectiveCooldown and anyRawSignal then 0 
                          else barsSinceLastSignal[1] + 1;

def cooldownClear = barsSinceLastSignal[1] >= effectiveCooldown;

# --- POSITION FILTER ---
def allowLongSignals = close >= valueAreaLow;
def allowShortSignals = close <= valueAreaHigh;

# --- OR BIAS FILTER ---
def orAllowLong = if !useORBiasFilter then yes
                  else if !orComplete then yes
                  else if orBias == 1 then yes
                  else if orBias == 0 then yes
                  else no;

def orAllowShort = if !useORBiasFilter then yes
                   else if !orComplete then yes
                   else if orBias == -1 then yes
                   else if orBias == 0 then yes
                   else no;

# --- FINAL SIGNALS ---
# Use VIX-adjusted signals for breakout/breakdown when VIX regime is enabled
# Include VWAP and TICK filters for institutional alignment and breadth confirmation
def valBounceLong = valBounceRaw and cooldownClear and allowLongSignals and orAllowLong and vwapAllowsLong and tickAllowsLong;
def pocReclaimLong = pocReclaimRaw and cooldownClear and allowLongSignals and orAllowLong and vwapAllowsLong and tickAllowsLong;
def breakoutLong = breakoutFinal and cooldownClear and allowLongSignals and orAllowLong and vwapAllowsLong and tickAllowsLong;
def sustainedBreakoutNew = sustainedBreakoutFinal and cooldownClear and allowLongSignals and orAllowLong and vwapAllowsLong and tickAllowsLong;

def vahRejectionShort = vahRejectionRaw and cooldownClear and allowShortSignals and orAllowShort and vwapAllowsShort and tickAllowsShort;
def pocBreakdownShort = pocBreakdownRaw and cooldownClear and allowShortSignals and orAllowShort and vwapAllowsShort and tickAllowsShort;
def breakdownShort = breakdownFinal and cooldownClear and allowShortSignals and orAllowShort and vwapAllowsShort and tickAllowsShort;
def sustainedBreakdownNew = sustainedBreakdownFinal and cooldownClear and allowShortSignals and orAllowShort and vwapAllowsShort and tickAllowsShort;

# ==================== PRIOR DAY VA SIGNALS ====================

def priorValTouch = hasPriorDay and low <= priorDayVAL and close > priorDayVAL;
def priorValBounce = priorValTouch and volCondition and close > open;
def priorValBounceRaw = tradePriorDayLevels and enableVALBounce and priorValBounce and !priorValBounce[1] and timeFilter and rthFilter;
def priorValBounceLong = priorValBounceRaw and cooldownClear and allowLongSignals and orAllowLong;

def priorPocReclaim = hasPriorDay and prevBelowPriorPOC and priceAbovePriorPOC and volIncreasing;
def priorPocReclaimRaw = tradePriorDayLevels and enablePOCReclaim and priorPocReclaim and !priorPocReclaim[1] and timeFilter and rthFilter;
def priorPocReclaimLong = priorPocReclaimRaw and cooldownClear and allowLongSignals and orAllowLong;

def priorVahTouch = hasPriorDay and high >= priorDayVAH and close < priorDayVAH;
def priorVahRejection = priorVahTouch and volCondition and close < open;
def priorVahRejectionRaw = tradePriorDayLevels and enableVAHRejection and priorVahRejection and !priorVahRejection[1] and timeFilter and rthFilter;
def priorVahRejectionShort = priorVahRejectionRaw and cooldownClear and allowShortSignals and orAllowShort;

def priorPocBreakdown = hasPriorDay and prevAbovePriorPOC and priceBelowPriorPOC and volIncreasing;
def priorPocBreakdownRaw = tradePriorDayLevels and enablePOCBreakdown and priorPocBreakdown and !priorPocBreakdown[1] and timeFilter and rthFilter;
def priorPocBreakdownShort = priorPocBreakdownRaw and cooldownClear and allowShortSignals and orAllowShort;

# Combined signals (with time window filter)
def longSignalRaw = valBounceLong or pocReclaimLong or breakoutLong or sustainedBreakoutNew or
                 priorValBounceLong or priorPocReclaimLong;
def shortSignalRaw = vahRejectionShort or pocBreakdownShort or breakdownShort or sustainedBreakdownNew or
                  priorVahRejectionShort or priorPocBreakdownShort;

# Apply time window filter to final signals
def longSignal = longSignalRaw and timeWindowFilter;
def shortSignal = shortSignalRaw and timeWindowFilter;

# ==================== POSITION TRACKING ====================

rec currentPosition = if longSignal then 1 
                      else if shortSignal then -1 
                      else currentPosition[1];

def previousPosition = currentPosition[1];
def hasPosition = currentPosition != 0;
def wasLong = previousPosition == 1;
def wasShort = previousPosition == -1;
def isLong = currentPosition == 1;
def isShort = currentPosition == -1;

# ==================== EXIT SIGNALS ====================

def exitLongSignal = wasLong and shortSignal;
def exitShortSignal = wasShort and longSignal;
def exitSignal = exitLongSignal or exitShortSignal;

# ==================== DAILY TRADE COUNTER ====================

rec dailyTradeCount = if isNewDay then 0
                      else if (longSignal or shortSignal) and !isNewDay then dailyTradeCount[1] + 1
                      else dailyTradeCount[1];

def isLockedOut = enableLockout and dailyTradeCount >= maxDailyTrades;
def isWarning = enableLockout and dailyTradeCount >= lockoutWarningAt and !isLockedOut;
def tradesRemaining = maxDailyTrades - dailyTradeCount;
def justLockedOut = isLockedOut and !isLockedOut[1];

# ==================== HOLD DURATION ====================

rec barsInPosition = if longSignal or shortSignal then 1
                     else if hasPosition then barsInPosition[1] + 1
                     else 0;

rec positionEntryPrice = if longSignal or shortSignal then close
                         else positionEntryPrice[1];

def unrealizedPL = if isLong then close - positionEntryPrice
                   else if isShort then positionEntryPrice - close
                   else 0;

def exitPrice = if exitSignal then close else Double.NaN;
def realizedPL = if exitLongSignal then close - positionEntryPrice[1]
                 else if exitShortSignal then positionEntryPrice[1] - close
                 else Double.NaN;

# ==================== CREDIT OPPORTUNITY ====================

def barsSinceSignal = if longSignal or shortSignal then 0 else barsSinceSignal[1] + 1;
def priceAtSignal = if barsSinceSignal == 0 then close else priceAtSignal[1];
def pointsMovedFromSignal = AbsValue(close - priceAtSignal);

def strongMomentum = pointsMovedFromSignal >= minESPointMove;
def volumeExpansion = volume > (avgVolume * 2.0);
def highCreditZone = strongMomentum and volumeExpansion and barsSinceSignal > 0 and barsSinceSignal <= 6;

# ==================== PLOTS ====================

plot VWAP = vwapValue;
VWAP.SetDefaultColor(Color.YELLOW);
VWAP.SetLineWeight(2);
VWAP.SetStyle(Curve.FIRM);

plot VAH = if showValueArea then valueAreaHigh else Double.NaN;
VAH.SetDefaultColor(Color.CYAN);
VAH.SetLineWeight(2);
VAH.SetStyle(Curve.FIRM);

plot VAL = if showValueArea then valueAreaLow else Double.NaN;
VAL.SetDefaultColor(Color.CYAN);
VAL.SetLineWeight(2);
VAL.SetStyle(Curve.FIRM);

plot POC = if showPOC then pocValue else Double.NaN;
POC.SetDefaultColor(Color.MAGENTA);
POC.SetLineWeight(2);
POC.SetStyle(Curve.SHORT_DASH);

AddCloud(if hideValueAreaCloud then Double.NaN else VAH, 
         if hideValueAreaCloud then Double.NaN else VAL, 
         Color.DARK_GRAY, Color.DARK_GRAY);

# Prior Day Lines
plot PriorVAH = if showPriorDayVA and hasPriorDay then priorDayVAH else Double.NaN;
PriorVAH.SetDefaultColor(Color.CYAN);
PriorVAH.SetLineWeight(1);
PriorVAH.SetStyle(Curve.SHORT_DASH);

plot PriorVAL = if showPriorDayVA and hasPriorDay then priorDayVAL else Double.NaN;
PriorVAL.SetDefaultColor(Color.CYAN);
PriorVAL.SetLineWeight(1);
PriorVAL.SetStyle(Curve.SHORT_DASH);

plot PriorPOC = if showPriorDayVA and hasPriorDay then priorDayPOC else Double.NaN;
PriorPOC.SetDefaultColor(Color.MAGENTA);
PriorPOC.SetLineWeight(1);
PriorPOC.SetStyle(Curve.SHORT_DASH);

# Opening Range Lines
plot ORHighLine = if showOpeningRange and orComplete then openingRangeHigh else Double.NaN;
ORHighLine.SetDefaultColor(Color.LIME);
ORHighLine.SetLineWeight(2);
ORHighLine.SetStyle(Curve.LONG_DASH);

plot ORLowLine = if showOpeningRange and orComplete then openingRangeLow else Double.NaN;
ORLowLine.SetDefaultColor(Color.PINK);
ORLowLine.SetLineWeight(2);
ORLowLine.SetStyle(Curve.LONG_DASH);

plot ORMidline = if showOpeningRange and orComplete then orMid else Double.NaN;
ORMidline.SetDefaultColor(Color.GRAY);
ORMidline.SetLineWeight(1);
ORMidline.SetStyle(Curve.SHORT_DASH);

# VWAP Plot (when filter enabled)
plot VWAPLine = if useVWAPFilter then vwapValue else Double.NaN;
VWAPLine.SetDefaultColor(Color.YELLOW);
VWAPLine.SetLineWeight(2);
VWAPLine.SetStyle(Curve.FIRM);

# Signal Arrows
plot LongArrow = if showSignals and longSignal and !isLockedOut then low - (ATR(14) * 0.5) else Double.NaN;
LongArrow.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
LongArrow.SetLineWeight(3);
LongArrow.SetDefaultColor(Color.GREEN);

plot ShortArrow = if showSignals and shortSignal and !isLockedOut then high + (ATR(14) * 0.5) else Double.NaN;
ShortArrow.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
ShortArrow.SetLineWeight(3);
ShortArrow.SetDefaultColor(Color.RED);

# Exit Markers
plot ExitLongMarker = if showExitSignals and exitLongSignal then high + (ATR(14) * 0.8) else Double.NaN;
ExitLongMarker.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
ExitLongMarker.SetLineWeight(4);
ExitLongMarker.SetDefaultColor(Color.ORANGE);

plot ExitShortMarker = if showExitSignals and exitShortSignal then low - (ATR(14) * 0.8) else Double.NaN;
ExitShortMarker.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
ExitShortMarker.SetLineWeight(4);
ExitShortMarker.SetDefaultColor(Color.ORANGE);

plot EntryLine = if hasPosition and showExitSignals then positionEntryPrice else Double.NaN;
EntryLine.SetDefaultColor(Color.WHITE);
EntryLine.SetLineWeight(1);
EntryLine.SetStyle(Curve.SHORT_DASH);

# ==================== LABELS ====================

AddLabel(yes, "VAH: " + Round(valueAreaHigh, 2), Color.CYAN);
AddLabel(yes, "POC: " + Round(pocValue, 2), Color.MAGENTA);
AddLabel(yes, "VAL: " + Round(valueAreaLow, 2), Color.CYAN);

AddLabel(showOpeningRange and !orComplete, "OR: Building...", Color.GRAY);
AddLabel(showOpeningRange and orComplete and orBias == 1, "Bias: BULL", Color.LIME);
AddLabel(showOpeningRange and orComplete and orBias == -1, "Bias: BEAR", Color.PINK);
AddLabel(showOpeningRange and orComplete and orBias == 0, "Bias: NEUTRAL", Color.GRAY);

# VIX Regime Labels - shows all adjusted parameters
AddLabel(useVixRegime, "VIX: " + Round(vix, 1), 
         if vixRegime == 1 then Color.RED 
         else if vixRegime == -1 then Color.GREEN 
         else Color.GRAY);
AddLabel(useVixRegime and vixRegime == 1, 
         "HIGH VOL | CD:" + effectiveCooldown + " Conf:" + effectiveConfirmation + " Sust:" + effectiveSustained, 
         Color.RED);
AddLabel(useVixRegime and vixRegime == -1, 
         "LOW VOL | CD:" + effectiveCooldown + " Conf:" + effectiveConfirmation + " Sust:" + effectiveSustained, 
         Color.GREEN);
AddLabel(useVixRegime and vixRegime == 0, 
         "NORMAL | CD:" + effectiveCooldown + " Conf:" + effectiveConfirmation + " Sust:" + effectiveSustained, 
         Color.GRAY);

AddLabel(showPositionStatus and isLong, "LONG @ " + Round(positionEntryPrice, 2), Color.GREEN);
AddLabel(showPositionStatus and isShort, "SHORT @ " + Round(positionEntryPrice, 2), Color.RED);
AddLabel(showPositionStatus and !hasPosition and !isLockedOut, "FLAT", Color.GRAY);

AddLabel(showTradeCounter and !isLockedOut, 
         "Trades: " + dailyTradeCount + "/" + maxDailyTrades,
         if isWarning then Color.YELLOW else Color.WHITE);

AddLabel(isWarning, "⚠️ LAST TRADE", Color.YELLOW);
AddLabel(isLockedOut, "🛑 LOCKED OUT - STOP TRADING", Color.RED);
AddLabel(rthOnlySignals and !isRTH and !isLockedOut, "OUTSIDE RTH", Color.GRAY);

# Time Window Labels
AddLabel(signalStartMinutes > 0 and minutesSinceOpen < signalStartMinutes, 
         "WAITING " + (signalStartMinutes - minutesSinceOpen) + "min", Color.YELLOW);
AddLabel(signalEndMinutes > 0 and minutesUntilClose < signalEndMinutes, 
         "NO SIGNALS - EOD", Color.GRAY);

# ATR Label (for ATR-based stops)
def atr14 = ATR(14);
AddLabel(useATRStops, "ATR: " + Round(atr14, 2) + " | Stop: " + Round(atr14 * atrStopMult, 2) + " | Tgt: " + Round(atr14 * atrTargetMult, 2), Color.CYAN);

# VWAP Filter Labels
AddLabel(useVWAPFilter, "VWAP: " + Round(vwapValue, 2), 
         if priceAboveVWAP then Color.GREEN else Color.RED);
AddLabel(useVWAPFilter and !vwapAllowsLong, "⛔ VWAP: No Longs", Color.RED);
AddLabel(useVWAPFilter and !vwapAllowsShort, "⛔ VWAP: No Shorts", Color.GREEN);

# TICK Filter Labels
AddLabel(useTICKFilter, "TICK: " + Round(tickValue, 0), 
         if tickValue > 500 then Color.GREEN 
         else if tickValue < -500 then Color.RED 
         else Color.GRAY);
AddLabel(useTICKFilter and !tickAllowsLong, "⛔ TICK: No Longs (extreme -)", Color.RED);
AddLabel(useTICKFilter and !tickAllowsShort, "⛔ TICK: No Shorts (extreme +)", Color.GREEN);

AddLabel(longSignal and !wasShort and !isLockedOut, "🔔 LONG SIGNAL", Color.GREEN);
AddLabel(shortSignal and !wasLong and !isLockedOut, "🔔 SHORT SIGNAL", Color.RED);
AddLabel(showExitSignals and exitLongSignal, "🔄 EXIT LONG → SHORT", Color.ORANGE);
AddLabel(showExitSignals and exitShortSignal, "🔄 EXIT SHORT → LONG", Color.ORANGE);

# ==================== ALERTS ====================

def alertValBounceLong = if enableAlerts then valBounceLong else no;
def alertPocReclaimLong = if enableAlerts then pocReclaimLong else no;
def alertBreakoutLong = if enableAlerts then breakoutLong else no;
def alertSustainedBreakout = if enableAlerts then sustainedBreakoutNew else no;
def alertVahRejectionShort = if enableAlerts then vahRejectionShort else no;
def alertPocBreakdownShort = if enableAlerts then pocBreakdownShort else no;
def alertBreakdownShort = if enableAlerts then breakdownShort else no;
def alertSustainedBreakdown = if enableAlerts then sustainedBreakdownNew else no;
def alertExitLong = if enableAlerts and exitAlertSound then exitLongSignal else no;
def alertExitShort = if enableAlerts and exitAlertSound then exitShortSignal else no;

Alert(alertValBounceLong, "LONG SIGNAL: VAL Bounce", Alert.BAR, Sound.Ding);
Alert(alertPocReclaimLong, "LONG SIGNAL: POC Reclaim", Alert.BAR, Sound.Ding);
Alert(alertBreakoutLong, "LONG BREAKOUT!", Alert.BAR, Sound.Bell);
Alert(alertSustainedBreakout, "LONG: Sustained Breakout!", Alert.BAR, Sound.Bell);
Alert(alertVahRejectionShort, "SHORT SIGNAL: VAH Rejection", Alert.BAR, Sound.Ding);
Alert(alertPocBreakdownShort, "SHORT SIGNAL: POC Breakdown", Alert.BAR, Sound.Ding);
Alert(alertBreakdownShort, "SHORT BREAKDOWN!", Alert.BAR, Sound.Bell);
Alert(alertSustainedBreakdown, "SHORT: Sustained Breakdown!", Alert.BAR, Sound.Bell);
Alert(alertExitLong, "EXIT LONG - Flip to Short!", Alert.BAR, Sound.Chimes);
Alert(alertExitShort, "EXIT SHORT - Flip to Long!", Alert.BAR, Sound.Chimes);
Alert(enableLockout and justLockedOut, "DAILY LIMIT REACHED!", Alert.BAR, Sound.Ring);

# ==================== TIME LINES ====================

input showTimeLines = yes;

def isMarketOpen = SecondsFromTime(0930) >= 0 and SecondsFromTime(0930) < 300;
def isEuroClose = SecondsFromTime(1130) >= 0 and SecondsFromTime(1130) < 300;
def isLunchStart = SecondsFromTime(1200) >= 0 and SecondsFromTime(1200) < 300;
def isLunchEnd = SecondsFromTime(1330) >= 0 and SecondsFromTime(1330) < 300;
def isPowerHour = SecondsFromTime(1500) >= 0 and SecondsFromTime(1500) < 300;
def isMarketClose = SecondsFromTime(1600) >= 0 and SecondsFromTime(1600) < 300;

AddVerticalLine(showTimeLines and isMarketOpen, "OPEN 9:30", Color.WHITE, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isEuroClose, "EU Close 11:30", Color.ORANGE, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isLunchStart, "Lunch 12:00", Color.GRAY, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isLunchEnd, "Lunch End 1:30", Color.GRAY, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isPowerHour, "Power Hour 3:00", Color.CYAN, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isMarketClose, "CLOSE 4:00", Color.WHITE, Curve.SHORT_DASH);

# ==================== ZONE CLOUDS ====================

def upperBound = Highest(high, 50);
def lowerBound = Lowest(low, 50);

def inCallZone = close > valueAreaHigh;
def inPutZone = close < valueAreaLow;

def callZoneUpper = if showPutCallZones and inCallZone then upperBound else Double.NaN;
def callZoneLower = if showPutCallZones and inCallZone then lowerBound else Double.NaN;
def putZoneUpper = if showPutCallZones and inPutZone then upperBound else Double.NaN;
def putZoneLower = if showPutCallZones and inPutZone then lowerBound else Double.NaN;
def creditUpper = if showCreditZones and highCreditZone then upperBound else Double.NaN;
def creditLower = if showCreditZones and highCreditZone then lowerBound else Double.NaN;

AddCloud(callZoneUpper, callZoneLower, Color.GREEN);
AddCloud(putZoneUpper, putZoneLower, Color.RED);
AddCloud(creditUpper, creditLower, Color.ORANGE);

# ==================== SIGNAL BUBBLES ====================

AddChartBubble(showSignals and showSignalBubbles and valBounceLong, low - (ATR(14) * 0.5), "VAL", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and pocReclaimLong, low - (ATR(14) * 0.5), "POC+", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and breakoutLong, low - (ATR(14) * 0.5), "BO", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and sustainedBreakoutNew, low - (ATR(14) * 0.5), "SUS-BO", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and vahRejectionShort, high + (ATR(14) * 0.5), "VAH", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and pocBreakdownShort, high + (ATR(14) * 0.5), "POC-", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and breakdownShort, high + (ATR(14) * 0.5), "BD", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and sustainedBreakdownNew, high + (ATR(14) * 0.5), "SUS-BD", Color.RED, yes);

AddChartBubble(showExitSignals and showSignalBubbles and exitLongSignal, high + (ATR(14) * 1.2), "EXIT\\nLONG", Color.ORANGE, yes);
AddChartBubble(showExitSignals and showSignalBubbles and exitShortSignal, low - (ATR(14) * 1.2), "EXIT\\nSHORT", Color.ORANGE, no);
'''
    
    return thinkscript


def preview_changes(params: Dict[str, Any]):
    """Preview the changes without writing files"""
    
    print("\n" + "=" * 70)
    print("                    CONFIGURATION PREVIEW")
    print("=" * 70)
    
    print("\n  Signal Parameters:")
    print(f"    signal_cooldown_bars: {params.get('signal_cooldown_bars', 8)}")
    print(f"    min_confirmation_bars: {params.get('min_confirmation_bars', 2)}")
    print(f"    sustained_bars_required: {params.get('sustained_bars_required', 3)}")
    print(f"    volume_threshold: {params.get('volume_threshold', 1.3):.3f}")
    
    print("\n  Filters:")
    print(f"    use_or_bias_filter: {params.get('use_or_bias_filter', True)}")
    print(f"    use_time_filter: {params.get('use_time_filter', False)}")
    
    print("\n  Signal Enables (LONG):")
    print(f"    enable_val_bounce: {params.get('enable_val_bounce', True)}")
    print(f"    enable_poc_reclaim: {params.get('enable_poc_reclaim', True)}")
    print(f"    enable_breakout: {params.get('enable_breakout', True)}")
    print(f"    enable_sustained_breakout: {params.get('enable_sustained_breakout', True)}")
    
    print("\n  Signal Enables (SHORT):")
    print(f"    enable_vah_rejection: {params.get('enable_vah_rejection', True)}")
    print(f"    enable_poc_breakdown: {params.get('enable_poc_breakdown', True)}")
    print(f"    enable_breakdown: {params.get('enable_breakdown', True)}")
    print(f"    enable_sustained_breakdown: {params.get('enable_sustained_breakdown', True)}")
    
    print("\n  Delta Targeting:")
    print(f"    target_delta: {params.get('target_delta', 0.30):.2f}")
    print(f"    afternoon_delta: {params.get('afternoon_delta', 0.40):.2f}")
    print(f"    afternoon_hour: {params.get('afternoon_hour', 12)}")
    
    print("\n  Kelly Sizing:")
    print(f"    kelly_fraction: {params.get('kelly_fraction', 0.0):.3f}")
    print(f"    max_equity_risk: {params.get('max_equity_risk', 0.10):.3f}")
    print(f"    max_kelly_pct_cap: {params.get('max_kelly_pct_cap', 0.35):.3f}")
    print(f"    hard_max_contracts: {params.get('hard_max_contracts', 100)}")
    print(f"    kelly_lookback: {params.get('kelly_lookback', 20)}")
    
    print("\n  Risk Management:")
    print(f"    max_daily_trades: {params.get('max_daily_trades', 3)}")
    print(f"    enable_stop_loss: {params.get('enable_stop_loss', False)}")
    print(f"    stop_loss_percent: {params.get('stop_loss_percent', 50)}")
    print(f"    enable_take_profit: {params.get('enable_take_profit', False)}")
    print(f"    take_profit_percent: {params.get('take_profit_percent', 100)}")
    print(f"    enable_trailing_stop: {params.get('enable_trailing_stop', False)}")
    print(f"    trailing_stop_percent: {params.get('trailing_stop_percent', 25)}")
    print(f"    trailing_stop_activation: {params.get('trailing_stop_activation', 50)}")
    print(f"    min_hold_bars: {params.get('min_hold_bars', 0)}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Apply optimized configuration")
    parser.add_argument("--from", dest="from_file", required=True, help="JSON file with optimized params")
    parser.add_argument("--preview", action="store_true", help="Preview only, don't write files")
    parser.add_argument("--config-out", default="config.py", help="Output config.py path")
    parser.add_argument("--thinkscript-out", default="AMT_Optimized.ts", help="Output ThinkScript path")
    parser.add_argument("--backup", action="store_true", help="Backup existing files before overwriting")
    
    args = parser.parse_args()
    
    # Load params
    if not os.path.exists(args.from_file):
        print(f"ERROR: File not found: {args.from_file}")
        sys.exit(1)
    
    print(f"\n📂 Loading params from: {args.from_file}")
    params = load_params(args.from_file)
    print(f"   Found {len(params)} parameters")
    
    # Preview
    preview_changes(params)
    
    if args.preview:
        print("\n  (Preview mode - no files written)")
        return
    
    # Backup existing files
    if args.backup:
        for f in [args.config_out, args.thinkscript_out]:
            if os.path.exists(f):
                backup = f + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(f, backup)
                print(f"\n  📦 Backed up {f} → {backup}")
    
    # Generate and write config.py
    config_content = generate_config_py(params)
    with open(args.config_out, "w") as f:
        f.write(config_content)
    print(f"\n  ✅ Written: {args.config_out}")
    
    # Generate and write ThinkScript
    thinkscript_content = generate_thinkscript(params)
    with open(args.thinkscript_out, "w") as f:
        f.write(thinkscript_content)
    print(f"  ✅ Written: {args.thinkscript_out}")
    
    print("\n" + "=" * 70)
    print("  NEXT STEPS:")
    print("=" * 70)
    print(f"""
  1. Review the generated files:
     - {args.config_out} (Python bot config)
     - {args.thinkscript_out} (ThinkOrSwim study)

  2. To use the ThinkScript in ToS:
     a. Open ThinkOrSwim
     b. Go to Charts → Studies → Edit Studies
     c. Create New Study or Edit existing
     d. Paste contents of {args.thinkscript_out}
     e. Save and apply to chart

  3. Test the bot with new config:
     python trading_bot.py --paper  # If you have paper trading
     python trading_bot.py          # Live trading

  4. Monitor for signal alignment between ToS and bot
""")


if __name__ == "__main__":
    main()