"""
Apply Optimized Config to Trading Bot and ThinkScript V3.5
===========================================================

Reads optimization results and generates:
1. config.py - Bot configuration with optimized parameters
2. Prints the ThinkScript INPUT values to manually update in ToS

Usage:
    python apply_config_simple.py --from recommended_config_variant_a.json
"""
import json
import argparse
from datetime import datetime
from typing import Dict, Any


def load_config(filepath: str) -> Dict[str, Any]:
    """Load config from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    if "params" in data:
        return data["params"]
    elif "recommended_config" in data:
        return data["recommended_config"]
    else:
        return data


def generate_config_py(params: Dict[str, Any]) -> str:
    """Generate config.py for the trading bot."""
    
    p = params
    
    return f'''"""
Configuration settings for the ToS Signal Trading Bot
======================================================
Auto-generated from optimization results on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OPTIMIZED VERSION
To regenerate: python apply_config_simple.py --from <results.json>
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
    account_hash: Optional[str] = None


@dataclass
class TradingConfig:
    """Trading parameters - OPTIMIZED"""
    symbol: str = "SPY"
    contracts: int = 1
    max_daily_trades: int = {p.get('max_daily_trades', 3)}
    
    use_delta_targeting: bool = True
    target_delta: float = {p.get('target_delta', 0.70):.2f}
    afternoon_delta: float = {p.get('afternoon_delta', 0.75):.2f}
    afternoon_start_hour: int = {p.get('afternoon_hour', 12)}
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    hold_until_opposite_signal: bool = True
    
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = {p.get('max_equity_risk', 0.15) * 100:.1f}
    max_position_size: int = {p.get('hard_max_contracts', 50)}
    min_position_size: int = 1
    
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    enable_stop_loss: bool = {p.get('enable_stop_loss', False)}
    stop_loss_percent: float = {p.get('stop_loss_percent', 50.0):.1f}
    stop_loss_dollars: float = 500.0
    
    enable_take_profit: bool = False
    take_profit_percent: float = 100.0
    take_profit_dollars: float = 500.0
    
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = {p.get('trailing_stop_percent', 30.0):.1f}
    trailing_stop_activation: float = {p.get('trailing_stop_activation', 40.0):.1f}
    
    enable_correlation_check: bool = True
    max_delta_exposure: float = 100.0


@dataclass
class TimeConfig:
    """Trading hours configuration (Eastern Time)"""
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    or_start: time = field(default_factory=lambda: time(9, 30))
    or_end: time = field(default_factory=lambda: time(10, 0))
    avoid_first_minutes: int = 25
    avoid_lunch_start: time = field(default_factory=lambda: time(12, 0))
    avoid_lunch_end: time = field(default_factory=lambda: time(13, 30))
    avoid_last_minutes: int = 15
    use_time_filter: bool = False
    rth_only: bool = True


@dataclass 
class SignalConfig:
    """Signal detection configuration - OPTIMIZED"""
    length_period: int = 20
    value_area_percent: float = 70.0
    
    volume_threshold: float = {p.get('volume_threshold', 1.3):.2f}
    use_relaxed_volume: bool = True
    
    min_confirmation_bars: int = {p.get('min_confirmation_bars', 2)}
    sustained_bars_required: int = {p.get('sustained_bars_required', 3)}
    signal_cooldown_bars: int = {p.get('signal_cooldown_bars', 8)}
    
    use_or_bias_filter: bool = {p.get('use_or_bias_filter', True)}
    or_buffer_points: float = 1.0
    
    use_vix_regime: bool = {p.get('use_vix_regime', False)}
    vix_high_threshold: int = {p.get('vix_high_threshold', 25)}
    vix_low_threshold: int = {p.get('vix_low_threshold', 15)}
    high_vol_cooldown_mult: float = {p.get('high_vol_cooldown_mult', 1.5):.2f}
    low_vol_cooldown_mult: float = {p.get('low_vol_cooldown_mult', 0.7):.2f}
    
    enable_val_bounce: bool = {p.get('enable_val_bounce', True)}
    enable_poc_reclaim: bool = {p.get('enable_poc_reclaim', False)}
    enable_breakout: bool = {p.get('enable_breakout', True)}
    enable_sustained_breakout: bool = {p.get('enable_sustained_breakout', False)}
    
    enable_vah_rejection: bool = {p.get('enable_vah_rejection', True)}
    enable_poc_breakdown: bool = {p.get('enable_poc_breakdown', False)}
    enable_breakdown: bool = {p.get('enable_breakdown', True)}
    enable_sustained_breakdown: bool = {p.get('enable_sustained_breakdown', False)}
    
    min_es_point_move: float = 6.0


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
    
    data_poll_interval: int = 5
    use_streaming: bool = True
    enable_intra_bar_signals: bool = True
    intra_bar_check_interval: int = 15
    paper_trading: bool = True
    
    def validate(self) -> bool:
        if not self.schwab.app_key or not self.schwab.app_secret:
            raise ValueError("Schwab API credentials not configured")
        return True


config = BotConfig()
'''


def print_tos_inputs(params: Dict[str, Any]):
    """Print the ThinkScript input values to update manually."""
    
    p = params
    
    def ts_yn(val, default=True):
        v = val if val is not None else default
        return "yes" if v else "no"
    
    print("\n" + "=" * 60)
    print("  THINKSCRIPT V3.5 - UPDATE THESE INPUTS")
    print("=" * 60)
    print("""
In ThinkOrSwim, edit your AMT V3.5 study and change these inputs:

# Signal Parameters
input volumeThreshold = {:.2f};
input minConfirmationBars = {};
input sustainedBarsRequired = {};
input signalCooldownBars = {};

# Daily Trade Lockout
input maxDailyTrades = {};

# Signal Enables
input enableVALBounce = {};
input enableVAHRejection = {};
input enableBreakout = {};
input enableBreakdown = {};
input enablePOCReclaim = {};
input enablePOCBreakdown = {};
input enableSustainedBreakout = {};
input enableSustainedBreakdown = {};

# Opening Range Bias
input useORBiasFilter = {};

# VIX Regime
input useVixRegime = {};
input vixHighThreshold = {};
input vixLowThreshold = {};
input highVolCooldownMult = {:.2f};
input lowVolCooldownMult = {:.2f};
""".format(
        p.get('volume_threshold', 1.3),
        p.get('min_confirmation_bars', 2),
        p.get('sustained_bars_required', 3),
        p.get('signal_cooldown_bars', 8),
        p.get('max_daily_trades', 3),
        ts_yn(p.get('enable_val_bounce', True)),
        ts_yn(p.get('enable_vah_rejection', True)),
        ts_yn(p.get('enable_breakout', True)),
        ts_yn(p.get('enable_breakdown', True)),
        ts_yn(p.get('enable_poc_reclaim', False)),
        ts_yn(p.get('enable_poc_breakdown', False)),
        ts_yn(p.get('enable_sustained_breakout', False)),
        ts_yn(p.get('enable_sustained_breakdown', False)),
        ts_yn(p.get('use_or_bias_filter', True)),
        ts_yn(p.get('use_vix_regime', False)),
        p.get('vix_high_threshold', 25),
        p.get('vix_low_threshold', 15),
        p.get('high_vol_cooldown_mult', 1.5),
        p.get('low_vol_cooldown_mult', 0.7)
    ))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Apply optimized config")
    parser.add_argument("--from", dest="from_file", required=True, help="JSON file with optimized params")
    parser.add_argument("--config-out", default="config.py", help="Output config.py path")
    parser.add_argument("--preview", action="store_true", help="Preview without writing files")
    
    args = parser.parse_args()
    
    print(f"\n📂 Loading config from: {args.from_file}")
    params = load_config(args.from_file)
    
    print(f"\n📊 Optimized Parameters:")
    print(f"   Signal Cooldown: {params.get('signal_cooldown_bars', 8)} bars")
    print(f"   Volume Threshold: {params.get('volume_threshold', 1.3):.2f}")
    print(f"   Max Daily Trades: {params.get('max_daily_trades', 3)}")
    print(f"   Target Delta: {params.get('target_delta', 0.70):.2f}")
    print(f"   Afternoon Delta: {params.get('afternoon_delta', 0.75):.2f}")
    print(f"\n   Signals Enabled:")
    print(f"     VAL Bounce: {params.get('enable_val_bounce', True)}")
    print(f"     VAH Rejection: {params.get('enable_vah_rejection', True)}")
    print(f"     Breakout: {params.get('enable_breakout', True)}")
    print(f"     Breakdown: {params.get('enable_breakdown', True)}")
    print(f"\n   VIX Regime: {params.get('use_vix_regime', False)}")
    print(f"   OR Bias Filter: {params.get('use_or_bias_filter', True)}")
    
    if args.preview:
        print("\n[Preview mode - no files written]")
        print_tos_inputs(params)
        return
    
    # Generate config.py
    config_content = generate_config_py(params)
    with open(args.config_out, "w") as f:
        f.write(config_content)
    print(f"\n✅ Generated: {args.config_out}")
    
    # Print ToS input values
    print_tos_inputs(params)
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Update the input values above in your ToS AMT V3.5 study")
    print(f"   2. Restart trading bot to use new config.py")
    print()


if __name__ == "__main__":
    main()