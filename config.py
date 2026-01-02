"""
Configuration settings for the ToS Signal Trading Bot
======================================================
Auto-generated from optimization results on 2026-01-02 12:50:26

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
    max_daily_trades: int = 6
    
    use_delta_targeting: bool = True
    target_delta: float = 0.67
    afternoon_delta: float = 0.83
    afternoon_start_hour: int = 12
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    hold_until_opposite_signal: bool = True
    
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 23.0
    max_position_size: int = 99
    min_position_size: int = 1
    
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    enable_stop_loss: bool = False
    stop_loss_percent: float = 65.0
    stop_loss_dollars: float = 500.0
    
    enable_take_profit: bool = False
    take_profit_percent: float = 100.0
    take_profit_dollars: float = 500.0
    
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 21.0
    trailing_stop_activation: float = 32.0
    
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
    
    volume_threshold: float = 1.48
    use_relaxed_volume: bool = True
    
    min_confirmation_bars: int = 2
    sustained_bars_required: int = 5
    signal_cooldown_bars: int = 17
    
    use_or_bias_filter: bool = True
    or_buffer_points: float = 1.0
    
    use_vix_regime: bool = True
    vix_high_threshold: int = 25
    vix_low_threshold: int = 15
    high_vol_cooldown_mult: float = 1.43
    low_vol_cooldown_mult: float = 0.88
    
    enable_val_bounce: bool = True
    enable_poc_reclaim: bool = False
    enable_breakout: bool = False
    enable_sustained_breakout: bool = False
    
    enable_vah_rejection: bool = True
    enable_poc_breakdown: bool = False
    enable_breakdown: bool = True
    enable_sustained_breakdown: bool = False
    
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
