"""
OPTIMIZED CONFIGURATION - Based on 30-Day Backtest Analysis
============================================================

Key findings from backtest:
- Overall: 151 trades, 31.8% win rate, -$630.50 P&L
- Quick trades (≤3 bars) had only 6.7% win rate (-$1,231)
- Long holds (>15 bars) had 75% win rate (+$2,239)
- Morning trades: +$139, Afternoon trades: -$770
- VAL_BOUNCE and VAH_REJECTION were biggest losers

CHANGES MADE:
1. Disabled VAL_BOUNCE and VAH_REJECTION signals (negative expectancy)
2. Increased signal cooldown from 8 to 15 bars (reduce whipsaws)
3. Added morning-only trading (stop at 12:00 PM)
4. Increased confirmation bars from 2 to 3
5. Increased sustained bars from 3 to 4

To use this config:
    cp config_optimized.py config.py
    python backtest.py --days 30 --output trades_optimized.csv
"""
import os
from dataclasses import dataclass, field
from datetime import time
from typing import Optional


@dataclass
class SchwabConfig:
    """Schwab API credentials"""
    app_key: str = os.getenv("SCHWAB_APP_KEY", "")
    app_secret: str = os.getenv("SCHWAB_APP_SECRET", "")
    redirect_uri: str = "https://127.0.0.1:8182/callback"
    token_file: str = "schwab_tokens.json"


@dataclass
class TradingConfig:
    """Trading parameters"""
    symbol: str = "SPY"
    contracts: int = 1
    max_daily_trades: int = 3
    
    # Option selection
    use_delta_targeting: bool = True
    target_delta: float = 0.30  # Morning: 30 delta
    afternoon_delta: float = 0.40  # Afternoon: 40 delta (if trading)
    afternoon_start_hour: int = 12
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    # Position management
    hold_until_opposite_signal: bool = True
    
    # Position Sizing - Fixed Fractional
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 2.0
    max_position_size: int = 10
    min_position_size: int = 1
    
    # Risk management - Daily Loss Limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    # Risk management - Stop Loss
    enable_stop_loss: bool = False
    stop_loss_percent: float = 50.0
    stop_loss_dollars: float = 500.0
    
    # Risk management - Take Profit
    enable_take_profit: bool = False
    take_profit_percent: float = 100.0
    take_profit_dollars: float = 500.0
    
    # Risk management - Trailing Stop
    enable_trailing_stop: bool = False
    trailing_stop_percent: float = 25.0
    trailing_stop_activation: float = 50.0
    
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
    
    # OPTIMIZED: Stop trading at noon (afternoon was -$770)
    avoid_first_minutes: int = 25
    avoid_lunch_start: time = field(default_factory=lambda: time(12, 0))  # Stop here
    avoid_lunch_end: time = field(default_factory=lambda: time(16, 0))    # Don't resume
    avoid_last_minutes: int = 15
    
    # OPTIMIZED: Enable time filter to enforce morning-only trading
    use_time_filter: bool = True  # Changed from False
    rth_only: bool = True


@dataclass 
class SignalConfig:
    """Signal detection configuration - OPTIMIZED"""
    # Value area settings
    length_period: int = 20
    value_area_percent: float = 70.0
    
    # Volume thresholds
    volume_threshold: float = 1.3
    use_relaxed_volume: bool = True
    
    # OPTIMIZED: Increased confirmation to reduce false signals
    min_confirmation_bars: int = 3  # Was 2
    sustained_bars_required: int = 4  # Was 3
    
    # OPTIMIZED: Increased cooldown to reduce whipsaws
    signal_cooldown_bars: int = 15  # Was 8 (now 75 min vs 40 min)
    
    # Opening range bias
    use_or_bias_filter: bool = True
    or_buffer_points: float = 1.0
    
    # Minimum move for credit
    min_es_point_move: float = 6.0
    
    # ==========================================
    # SIGNAL ENABLE/DISABLE - OPTIMIZED
    # ==========================================
    # Based on backtest P&L by signal type:
    #   SUSTAINED_BREAKOUT:  +$255 ✅
    #   SUSTAINED_BREAKDOWN: +$172 ✅
    #   POC_RECLAIM:         +$79  ✅
    #   POC_BREAKDOWN:       -$125 ⚠️ (keep but monitor)
    #   VAH_REJECTION:       -$389 ❌
    #   VAL_BOUNCE:          -$431 ❌
    
    # Long signals
    enable_val_bounce: bool = False          # ❌ DISABLED - biggest loser
    enable_poc_reclaim: bool = True          # ✅ Profitable
    enable_breakout: bool = True             # ✅ Keep
    enable_sustained_breakout: bool = True   # ✅ Most profitable
    enable_prior_val_bounce: bool = False    # ❌ Disable with VAL_BOUNCE
    enable_prior_poc_reclaim: bool = True    # ✅ Keep
    
    # Short signals
    enable_vah_rejection: bool = False       # ❌ DISABLED - big loser
    enable_poc_breakdown: bool = True        # ⚠️ Keep but monitor
    enable_breakdown: bool = True            # Keep (low sample size)
    enable_sustained_breakdown: bool = True  # ✅ Profitable
    enable_prior_vah_rejection: bool = False # ❌ Disable with VAH_REJECTION
    enable_prior_poc_breakdown: bool = True  # ✅ Keep


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
    alerts: AlertConfig = field(default_factory=AlertConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    
    # Data source
    data_poll_interval: int = 5
    use_streaming: bool = True
    
    # Intra-bar signal checking
    enable_intra_bar_signals: bool = True
    intra_bar_check_interval: int = 15
    
    # Paper trading mode
    paper_trading: bool = True
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.schwab.app_key or not self.schwab.app_secret:
            raise ValueError("Schwab API credentials not configured")
        return True


# Default configuration instance
config = BotConfig()