"""
Configuration settings for the ToS Signal Trading Bot
======================================================
Auto-generated from optimization results on 2025-12-31 12:54:29

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
    max_daily_trades: int = 5.0  # OPTIMIZED
    
    # Option selection - OPTIMIZED
    use_delta_targeting: bool = True
    target_delta: float = 0.28  # OPTIMIZED
    afternoon_delta: float = 0.37  # OPTIMIZED
    afternoon_start_hour: int = 12.5  # OPTIMIZED
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    # Position management
    hold_until_opposite_signal: bool = True
    
    # Position Sizing - Kelly - OPTIMIZED
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 13.5  # OPTIMIZED (converted to %)
    max_position_size: int = 62  # OPTIMIZED
    min_position_size: int = 1
    
    # Kelly-specific params
    kelly_fraction: float = 0.032  # OPTIMIZED
    max_kelly_pct_cap: float = 0.230  # OPTIMIZED
    kelly_lookback: int = 20.5  # OPTIMIZED
    
    # Risk management - Daily Loss Limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    # Risk management - Stop Loss - OPTIMIZED
    enable_stop_loss: bool = False  # OPTIMIZED
    stop_loss_percent: float = 50.0  # OPTIMIZED
    stop_loss_dollars: float = 500.0
    
    # Risk management - Take Profit - OPTIMIZED
    enable_take_profit: bool = False  # OPTIMIZED
    take_profit_percent: float = 100.0  # OPTIMIZED
    take_profit_dollars: float = 500.0
    
    # Risk management - Trailing Stop - OPTIMIZED
    enable_trailing_stop: bool = True  # OPTIMIZED
    trailing_stop_percent: float = 14.0  # OPTIMIZED
    trailing_stop_activation: float = 32.0  # OPTIMIZED
    
    # Min hold time - OPTIMIZED
    min_hold_bars: int = 4  # OPTIMIZED
    
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
    
    use_time_filter: bool = True  # OPTIMIZED
    rth_only: bool = True


@dataclass 
class SignalConfig:
    """Signal detection configuration - OPTIMIZED"""
    # Value area settings
    length_period: int = 20
    value_area_percent: float = 70.0
    
    # Volume thresholds - OPTIMIZED
    volume_threshold: float = 1.297  # OPTIMIZED
    use_relaxed_volume: bool = True
    
    # Confirmation - OPTIMIZED
    min_confirmation_bars: int = 4  # OPTIMIZED
    sustained_bars_required: int = 5  # OPTIMIZED
    signal_cooldown_bars: int = 21  # OPTIMIZED
    
    # Opening range bias - OPTIMIZED
    use_or_bias_filter: bool = True  # OPTIMIZED
    or_buffer_points: float = 1.0
    
    # Minimum move for credit
    min_es_point_move: float = 6.0
    
    # Signal enables - LONG signals - OPTIMIZED
    enable_val_bounce: bool = True  # OPTIMIZED
    enable_poc_reclaim: bool = False  # OPTIMIZED
    enable_breakout: bool = True  # OPTIMIZED
    enable_sustained_breakout: bool = False  # OPTIMIZED
    
    # Signal enables - SHORT signals - OPTIMIZED
    enable_vah_rejection: bool = True  # OPTIMIZED
    enable_poc_breakdown: bool = False  # OPTIMIZED
    enable_breakdown: bool = False  # OPTIMIZED
    enable_sustained_breakdown: bool = False  # OPTIMIZED


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
