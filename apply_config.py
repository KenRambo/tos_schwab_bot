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
    """Generate updated config.py content"""
    
    # Extract params with defaults
    p = params
    
    config_content = f'''"""
Trading Bot Configuration
=========================
Auto-generated from optimization results on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

To regenerate: python apply_config.py --from <results.json>
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class SchwabConfig:
    """Schwab API configuration"""
    app_key: str = os.getenv("SCHWAB_APP_KEY", "")
    app_secret: str = os.getenv("SCHWAB_APP_SECRET", "")
    redirect_uri: str = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback")
    token_file: str = "schwab_tokens.json"
    account_number: str = os.getenv("SCHWAB_ACCOUNT_NUMBER", "")


@dataclass
class SignalConfig:
    """Signal detection parameters - OPTIMIZED"""
    # Core signal params
    signal_cooldown_bars: int = {p.get('signal_cooldown_bars', 8)}
    min_confirmation_bars: int = {p.get('min_confirmation_bars', 2)}
    sustained_bars_required: int = {p.get('sustained_bars_required', 3)}
    volume_threshold: float = {p.get('volume_threshold', 1.3):.3f}
    
    # Filters
    use_or_bias_filter: bool = {p.get('use_or_bias_filter', True)}
    use_time_filter: bool = {p.get('use_time_filter', False)}
    
    # Signal enables - LONG signals
    enable_val_bounce: bool = {p.get('enable_val_bounce', True)}
    enable_poc_reclaim: bool = {p.get('enable_poc_reclaim', True)}
    enable_breakout: bool = {p.get('enable_breakout', True)}
    enable_sustained_breakout: bool = {p.get('enable_sustained_breakout', True)}
    
    # Signal enables - SHORT signals
    enable_vah_rejection: bool = {p.get('enable_vah_rejection', True)}
    enable_poc_breakdown: bool = {p.get('enable_poc_breakdown', True)}
    enable_breakdown: bool = {p.get('enable_breakdown', True)}
    enable_sustained_breakdown: bool = {p.get('enable_sustained_breakdown', True)}


@dataclass
class DeltaConfig:
    """Delta targeting parameters - OPTIMIZED"""
    target_delta: float = {p.get('target_delta', 0.30):.2f}
    afternoon_delta: float = {p.get('afternoon_delta', 0.40):.2f}
    afternoon_hour: int = {p.get('afternoon_hour', 12)}


@dataclass
class KellyConfig:
    """Kelly position sizing parameters - OPTIMIZED"""
    kelly_fraction: float = {p.get('kelly_fraction', 0.0):.3f}
    max_equity_risk: float = {p.get('max_equity_risk', 0.10):.3f}
    max_kelly_pct_cap: float = {p.get('max_kelly_pct_cap', 0.35):.3f}
    hard_max_contracts: int = {p.get('hard_max_contracts', 100)}
    kelly_lookback: int = {p.get('kelly_lookback', 20)}


@dataclass
class RiskConfig:
    """Risk management parameters - OPTIMIZED"""
    max_daily_trades: int = {p.get('max_daily_trades', 3)}
    
    # Stop loss
    enable_stop_loss: bool = {p.get('enable_stop_loss', False)}
    stop_loss_percent: float = {p.get('stop_loss_percent', 50):.1f}
    
    # Take profit
    enable_take_profit: bool = {p.get('enable_take_profit', False)}
    take_profit_percent: float = {p.get('take_profit_percent', 100):.1f}
    
    # Trailing stop
    enable_trailing_stop: bool = {p.get('enable_trailing_stop', False)}
    trailing_stop_percent: float = {p.get('trailing_stop_percent', 25):.1f}
    trailing_stop_activation: float = {p.get('trailing_stop_activation', 50):.1f}
    
    # Hold time
    min_hold_bars: int = {p.get('min_hold_bars', 0)}


@dataclass
class TradingConfig:
    """Main trading configuration"""
    symbol: str = "SPY"
    
    # Sub-configs
    schwab: SchwabConfig = None
    signal: SignalConfig = None
    delta: DeltaConfig = None
    kelly: KellyConfig = None
    risk: RiskConfig = None
    
    def __post_init__(self):
        if self.schwab is None:
            self.schwab = SchwabConfig()
        if self.signal is None:
            self.signal = SignalConfig()
        if self.delta is None:
            self.delta = DeltaConfig()
        if self.kelly is None:
            self.kelly = KellyConfig()
        if self.risk is None:
            self.risk = RiskConfig()


# Global config instance
config = TradingConfig()
'''
    
    return config_content


def generate_thinkscript(params: Dict[str, Any]) -> str:
    """Generate updated ThinkScript with optimized signal parameters"""
    
    p = params
    
    # Convert Python bools to ThinkScript
    def ts_bool(val, default=True):
        return "yes" if val else "no"
    
    def ts_yn(val, default=True):
        v = val if val is not None else default
        return "yes" if v else "no"
    
    thinkscript = f'''#
# AMT Signal Detector - OPTIMIZED
# ================================
# Auto-generated from optimization results on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# To regenerate: python apply_config.py --from <results.json>
#

declare lower;

#==================== INPUTS ====================

# Profile Settings
input length = 20;
input valueAreaPercent = 70;
input showPOC = yes;
input showVA = yes;

# Signal Parameters - OPTIMIZED
input signalCooldownBars = {p.get('signal_cooldown_bars', 8)};
input minConfirmationBars = {p.get('min_confirmation_bars', 2)};
input sustainedBarsRequired = {p.get('sustained_bars_required', 3)};
input volumeThreshold = {p.get('volume_threshold', 1.3):.2f};

# Filters - OPTIMIZED
input useORBiasFilter = {ts_yn(p.get('use_or_bias_filter', True))};
input useTimeFilter = {ts_yn(p.get('use_time_filter', False))};
input tradingStartTime = 0930;
input tradingEndTime = 1200;

# Signal Enables - LONG - OPTIMIZED
input enableVALBounce = {ts_yn(p.get('enable_val_bounce', True))};
input enablePOCReclaim = {ts_yn(p.get('enable_poc_reclaim', True))};
input enableBreakout = {ts_yn(p.get('enable_breakout', True))};
input enableSustainedBreakout = {ts_yn(p.get('enable_sustained_breakout', True))};

# Signal Enables - SHORT - OPTIMIZED
input enableVAHRejection = {ts_yn(p.get('enable_vah_rejection', True))};
input enablePOCBreakdown = {ts_yn(p.get('enable_poc_breakdown', True))};
input enableBreakdown = {ts_yn(p.get('enable_breakdown', True))};
input enableSustainedBreakdown = {ts_yn(p.get('enable_sustained_breakdown', True))};

# Visual Settings
input showSignalArrows = yes;
input showSignalLabels = yes;
input showValueAreaCloud = yes;

#==================== TIME FUNCTIONS ====================

def RTH = GetTime() >= RegularTradingStart(GetYYYYMMDD()) and 
          GetTime() <= RegularTradingEnd(GetYYYYMMDD());

def timeOK = if useTimeFilter 
             then SecondsFromTime(tradingStartTime) >= 0 and SecondsTillTime(tradingEndTime) > 0
             else RTH;

def newDay = GetDay() != GetDay()[1];

#==================== OPENING RANGE ====================

def ORActive = SecondsFromTime(0930) >= 0 and SecondsTillTime(1000) > 0;
def ORHigh = if newDay then high else if ORActive then Max(ORHigh[1], high) else ORHigh[1];
def ORLow = if newDay then low else if ORActive then Min(ORLow[1], low) else ORLow[1];
def ORComplete = SecondsFromTime(1000) >= 0;

def aboveOR = close > ORHigh;
def belowOR = close < ORLow;

#==================== VOLUME PROFILE CALCULATIONS ====================

def priceRange = Highest(high, length) - Lowest(low, length);
def POC = reference VolumeProfile("time per profile" = "DAY", "on expansion" = no, "price per row height mode" = "TICKSIZE").POC;
def VAHigh = reference VolumeProfile("time per profile" = "DAY", "on expansion" = no, "price per row height mode" = "TICKSIZE").VAHigh;
def VALow = reference VolumeProfile("time per profile" = "DAY", "on expansion" = no, "price per row height mode" = "TICKSIZE").VALow;

# Prior day values
def priorPOC = if newDay then POC[1] else priorPOC[1];
def priorVAH = if newDay then VAHigh[1] else priorVAH[1];
def priorVAL = if newDay then VALow[1] else priorVAL[1];

#==================== VOLUME ANALYSIS ====================

def avgVolume = Average(volume, length);
def highVolume = volume > avgVolume * volumeThreshold;
def relativeVolume = volume / avgVolume;

#==================== PRICE POSITION ====================

def aboveVAH = close > VAHigh;
def belowVAL = close < VALow;
def abovePOC = close > POC;
def belowPOC = close < POC;
def inValueArea = close >= VALow and close <= VAHigh;

# Price position history
def wasAboveVAH = aboveVAH[1];
def wasBelowVAL = belowVAL[1];
def wasAbovePOC = abovePOC[1];
def wasBelowPOC = belowPOC[1];

#==================== SIGNAL COOLDOWN ====================

def barsSinceSignal = if barsSinceSignal[1] >= signalCooldownBars then signalCooldownBars else barsSinceSignal[1] + 1;
def cooldownComplete = barsSinceSignal >= signalCooldownBars;

#==================== CONFIRMATION TRACKING ====================

def barsAboveVAL = if close > VALow then barsAboveVAL[1] + 1 else 0;
def barsBelowVAH = if close < VAHigh then barsBelowVAH[1] + 1 else 0;
def barsAbovePOC = if close > POC then barsAbovePOC[1] + 1 else 0;
def barsBelowPOC = if close < POC then barsBelowPOC[1] + 1 else 0;
def barsAboveVAH = if close > VAHigh then barsAboveVAH[1] + 1 else 0;
def barsBelowVAL = if close < VALow then barsBelowVAL[1] + 1 else 0;

#==================== OR BIAS FILTER ====================

def longBiasOK = if useORBiasFilter then (ORComplete and aboveOR) or !ORComplete else yes;
def shortBiasOK = if useORBiasFilter then (ORComplete and belowOR) or !ORComplete else yes;

#==================== SIGNAL DETECTION ====================

# VAL Bounce (Long)
def valBounceRaw = enableVALBounce and 
                   wasBelowVAL and !belowVAL and 
                   close > VALow and 
                   barsAboveVAL >= minConfirmationBars;
def valBounceSignal = valBounceRaw and cooldownComplete and timeOK and longBiasOK;

# POC Reclaim (Long)
def pocReclaimRaw = enablePOCReclaim and 
                    wasBelowPOC and abovePOC and 
                    barsAbovePOC >= minConfirmationBars;
def pocReclaimSignal = pocReclaimRaw and cooldownComplete and timeOK and longBiasOK;

# Breakout (Long)
def breakoutRaw = enableBreakout and 
                  !wasAboveVAH and aboveVAH and 
                  highVolume;
def breakoutSignal = breakoutRaw and cooldownComplete and timeOK and longBiasOK;

# Sustained Breakout (Long)
def sustainedBreakoutRaw = enableSustainedBreakout and 
                           aboveVAH and 
                           barsAboveVAH >= sustainedBarsRequired;
def sustainedBreakoutSignal = sustainedBreakoutRaw and cooldownComplete and timeOK and longBiasOK;

# VAH Rejection (Short)
def vahRejectionRaw = enableVAHRejection and 
                      wasAboveVAH and !aboveVAH and 
                      close < VAHigh and 
                      barsBelowVAH >= minConfirmationBars;
def vahRejectionSignal = vahRejectionRaw and cooldownComplete and timeOK and shortBiasOK;

# POC Breakdown (Short)
def pocBreakdownRaw = enablePOCBreakdown and 
                      wasAbovePOC and belowPOC and 
                      barsBelowPOC >= minConfirmationBars;
def pocBreakdownSignal = pocBreakdownRaw and cooldownComplete and timeOK and shortBiasOK;

# Breakdown (Short)
def breakdownRaw = enableBreakdown and 
                   !wasBelowVAL and belowVAL and 
                   highVolume;
def breakdownSignal = breakdownRaw and cooldownComplete and timeOK and shortBiasOK;

# Sustained Breakdown (Short)
def sustainedBreakdownRaw = enableSustainedBreakdown and 
                            belowVAL and 
                            barsBelowVAL >= sustainedBarsRequired;
def sustainedBreakdownSignal = sustainedBreakdownRaw and cooldownComplete and timeOK and shortBiasOK;

#==================== COMBINED SIGNALS ====================

def longSignal = valBounceSignal or pocReclaimSignal or breakoutSignal or sustainedBreakoutSignal;
def shortSignal = vahRejectionSignal or pocBreakdownSignal or breakdownSignal or sustainedBreakdownSignal;

# Reset cooldown on signal
def cooldownReset = if longSignal or shortSignal then 0 else barsSinceSignal;

#==================== PLOTS ====================

# Value Area
plot pPOC = if showPOC then POC else Double.NaN;
plot pVAH = if showVA then VAHigh else Double.NaN;
plot pVAL = if showVA then VALow else Double.NaN;

pPOC.SetDefaultColor(Color.YELLOW);
pPOC.SetLineWeight(2);
pVAH.SetDefaultColor(Color.RED);
pVAL.SetDefaultColor(Color.GREEN);

# Value Area Cloud
AddCloud(if showValueAreaCloud then VAHigh else Double.NaN, VALow, Color.LIGHT_GRAY);

# Signal Arrows
plot longArrow = if showSignalArrows and longSignal then low - 0.5 else Double.NaN;
plot shortArrow = if showSignalArrows and shortSignal then high + 0.5 else Double.NaN;

longArrow.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
longArrow.SetDefaultColor(Color.GREEN);
longArrow.SetLineWeight(3);

shortArrow.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
shortArrow.SetDefaultColor(Color.RED);
shortArrow.SetLineWeight(3);

#==================== LABELS ====================

AddLabel(showSignalLabels and valBounceSignal, "VAL BOUNCE", Color.GREEN);
AddLabel(showSignalLabels and pocReclaimSignal, "POC RECLAIM", Color.GREEN);
AddLabel(showSignalLabels and breakoutSignal, "BREAKOUT", Color.GREEN);
AddLabel(showSignalLabels and sustainedBreakoutSignal, "SUSTAINED BREAKOUT", Color.GREEN);

AddLabel(showSignalLabels and vahRejectionSignal, "VAH REJECTION", Color.RED);
AddLabel(showSignalLabels and pocBreakdownSignal, "POC BREAKDOWN", Color.RED);
AddLabel(showSignalLabels and breakdownSignal, "BREAKDOWN", Color.RED);
AddLabel(showSignalLabels and sustainedBreakdownSignal, "SUSTAINED BREAKDOWN", Color.RED);

# Status Labels
AddLabel(yes, "Cooldown: " + barsSinceSignal + "/" + signalCooldownBars, 
         if cooldownComplete then Color.GREEN else Color.ORANGE);
AddLabel(useORBiasFilter, "OR Bias: " + (if aboveOR then "LONG" else if belowOR then "SHORT" else "NEUTRAL"),
         if aboveOR then Color.GREEN else if belowOR then Color.RED else Color.GRAY);
AddLabel(yes, "Vol: " + Round(relativeVolume, 1) + "x", 
         if highVolume then Color.GREEN else Color.GRAY);

#==================== ALERTS ====================

Alert(longSignal, "LONG Signal", Alert.BAR, Sound.Ding);
Alert(shortSignal, "SHORT Signal", Alert.BAR, Sound.Ding);
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