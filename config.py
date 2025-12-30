"""
Configuration settings for the ToS Signal Trading Bot
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
    """Trading parameters"""
    symbol: str = "SPY"
    contracts: int = 1
    max_daily_trades: int = 10
    
    # Option selection
    use_delta_targeting: bool = True  # True = select by delta, False = nearest OTM
    target_delta: float = 0.30  # Morning delta (0.30 = 30 delta)
    afternoon_delta: float = 0.40  # Afternoon delta (higher for 0DTE gamma)
    afternoon_start_hour: int = 12  # When to switch to afternoon delta (12 = noon ET)
    min_days_to_expiry: int = 0  # 0DTE allowed
    max_days_to_expiry: int = 0  # 0 = 0DTE only, 1 = include tomorrow, etc.
    prefer_weekly: bool = True
    
    # Position management
    hold_until_opposite_signal: bool = True  # Match your ToS strategy
    
    # Position Sizing - Fixed Fractional
    use_fixed_fractional: bool = True  # Size based on account balance
    risk_percent_per_trade: float = 30.0  # Risk X% of account per trade
    max_position_size: int = 10  # Never exceed X contracts regardless of account size
    min_position_size: int = 1  # Always trade at least X contracts
    
    # Risk management - Daily Loss Limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0  # Stop trading if daily loss exceeds $X
    max_daily_loss_percent: float = 10.0  # OR stop if daily loss exceeds X% of starting balance
    
    # Risk management - Stop Loss
    enable_stop_loss: bool = False  # Set True to enable stop loss
    stop_loss_percent: float = 50.0  # Exit if option loses X% of value
    stop_loss_dollars: float = 500.0  # OR exit if loss exceeds $X (whichever hits first)
    
    # Risk management - Take Profit
    enable_take_profit: bool = False  # Set True to enable take profit
    take_profit_percent: float = 100.0  # Exit if option gains X% (100% = double)
    take_profit_dollars: float = 500.0  # OR exit if profit exceeds $X (whichever hits first)
    
    # Risk management - Trailing Stop
    enable_trailing_stop: bool = True  # Set True to enable trailing stop
    trailing_stop_percent: float = 25.0  # Trail by X% from high water mark
    trailing_stop_activation: float = 50.0  # Only activate after X% gain
    
    # Correlation / Exposure Check
    enable_correlation_check: bool = True  # Don't add if already exposed
    max_delta_exposure: float = 100.0  # Max total delta (+ or -) across all positions
    

@dataclass
class TimeConfig:
    """Trading hours configuration (Eastern Time)"""
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))
    
    # Opening range period
    or_start: time = field(default_factory=lambda: time(9, 30))
    or_end: time = field(default_factory=lambda: time(10, 0))
    
    # Avoid trading during these periods (matching your ToS script)
    avoid_first_minutes: int = 10  # First 10 min after open
    avoid_lunch_start: time = field(default_factory=lambda: time(12, 0))
    avoid_lunch_end: time = field(default_factory=lambda: time(13, 30))
    avoid_last_minutes: int = 15  # Last 15 min before close
    
    use_time_filter: bool = True  # Match your ToS setting
    rth_only: bool = True  # Only trade during RTH


@dataclass 
class SignalConfig:
    """Signal detection configuration (matching your ToS script)"""
    # Value area settings
    length_period: int = 20
    value_area_percent: float = 70.0
    
    # Volume thresholds
    volume_threshold: float = 1.3
    use_relaxed_volume: bool = True
    
    # Confirmation
    min_confirmation_bars: int = 2
    sustained_bars_required: int = 3
    signal_cooldown_bars: int = 0
    
    # Opening range bias
    use_or_bias_filter: bool = True
    or_buffer_points: float = 1.0
    
    # Minimum move for credit
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
    # Daily summary notification at market close
    enable_daily_summary: bool = True
    daily_summary_hour: int = 16  # 4 PM ET
    daily_summary_minute: int = 15  # 4:15 PM ET
    
    # Weekly performance report
    enable_weekly_report: bool = True
    
    # Drawdown alerts
    enable_drawdown_alerts: bool = True
    drawdown_alert_threshold: float = 10.0  # Alert when down 10% from peak
    
    # Performance data file
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
    data_poll_interval: int = 5  # Seconds between price checks
    use_streaming: bool = True  # Use WebSocket streaming if available
    
    # Intra-bar signal checking (to match ToS behavior)
    enable_intra_bar_signals: bool = True  # Check signals during bar formation
    intra_bar_check_interval: int = 60  # Seconds between intra-bar checks
    
    # Paper trading mode
    paper_trading: bool = True  # Set to False for live trading
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.schwab.app_key or not self.schwab.app_secret:
            raise ValueError("Schwab API credentials not configured")
        return True


# Default configuration instance
config = BotConfig()