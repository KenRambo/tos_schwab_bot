"""
Configuration settings for the ToS Signal Trading Bot
======================================================
OPTIMIZED from variant_a_seed_789_best_pnl on 2026-01-05

Optimization Results:
- Win Rate: 72.58%
- Total P&L: $274,048.37
- Total Trades: 62
- Profit Factor: 3.25
- Sharpe Ratio: 5.58
- Max Drawdown: $32,715.08

UPDATED 2026-01-08:
- Added Kelly Criterion position sizing for butterfly trades
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
    """Trading parameters - OPTIMIZED from variant_a_seed_789"""
    symbol: str = "SPY"
    execution_symbol: Optional[str] = None  # If set, trade options on this symbol instead of signal symbol
    contracts: int = 1
    max_daily_trades: int = 6  # OPTIMIZED
    
    # Butterfly mode (for credit spread stacking)
    butterfly_mode: bool = False
    butterfly_wing_width: int = 5  # Points between strikes
    butterfly_credit_target_pct: float = 0.30  # Target 30% credit over wing cost
    
    # ==========================================================================
    # KELLY CRITERION POSITION SIZING FOR BUTTERFLIES
    # ==========================================================================
    # Kelly formula: f* = (bp - q) / b
    # Where: b = win/loss ratio, p = win probability, q = loss probability
    #
    # For butterflies:
    # - Win = credit received (if expires OTM or between wings)
    # - Loss = max loss = wing_width * 100 - credit
    # - Historical win rate determines p
    #
    # IMPORTANT: Start with conservative win rate (0.50-0.65) and update
    # based on YOUR actual paper trading results after 20-30 trades.
    # ==========================================================================
    use_kelly_sizing: bool = True
    kelly_win_rate: float = 0.65  # Start conservative, update from paper trading
    kelly_avg_win: float = 1.0  # Average win as multiple of credit (1.0 = keep full credit)
    kelly_avg_loss: float = 2.5  # Average loss as multiple of credit (risk/reward)
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (0.25 = quarter Kelly, safer)
    kelly_max_contracts: int = 10  # Hard cap on contracts regardless of Kelly
    kelly_min_contracts: int = 1  # Minimum contracts to trade
    
    # Delta targeting
    use_delta_targeting: bool = True
    target_delta: float = 0.67  # OPTIMIZED
    afternoon_delta: float = 0.83  # OPTIMIZED
    afternoon_start_hour: int = 12  # OPTIMIZED
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 0
    prefer_weekly: bool = True
    
    hold_until_opposite_signal: bool = True
    
    # Position sizing - OPTIMIZED
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 23.0  # OPTIMIZED (max_equity_risk from JSON)
    max_position_size: int = 99  # OPTIMIZED (hard_max_contracts)
    min_position_size: int = 1
    
    # Daily loss limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    # Stop loss - OPTIMIZED
    enable_stop_loss: bool = False  # OPTIMIZED (disabled)
    stop_loss_percent: float = 65.0  # OPTIMIZED
    stop_loss_dollars: float = 500.0
    
    # Take profit
    enable_take_profit: bool = False
    take_profit_percent: float = 100.0
    take_profit_dollars: float = 500.0
    
    # Trailing stop - OPTIMIZED
    enable_trailing_stop: bool = True  # OPTIMIZED
    trailing_stop_percent: float = 21.0  # OPTIMIZED
    trailing_stop_activation: float = 32.0  # OPTIMIZED
    
    # Correlation / exposure
    enable_correlation_check: bool = True
    max_delta_exposure: float = 100.0
    
    @property
    def option_symbol(self) -> str:
        """Returns the symbol to use for option trading (execution_symbol if set, else signal symbol)"""
        return self.execution_symbol or self.symbol
    
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly Criterion bet fraction.
        
        Kelly formula: f* = (bp - q) / b
        Where:
            b = odds (avg_win / avg_loss)
            p = probability of winning
            q = probability of losing (1 - p)
        
        Returns: Optimal fraction of bankroll to risk (before applying kelly_fraction multiplier)
        """
        p = self.kelly_win_rate
        q = 1 - p
        b = self.kelly_avg_win / self.kelly_avg_loss if self.kelly_avg_loss > 0 else 0
        
        if b <= 0:
            return 0
        
        kelly = (b * p - q) / b
        
        # Kelly can be negative if edge is negative - don't bet
        return max(0, kelly)


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
    """Signal detection configuration - OPTIMIZED from variant_a_seed_789"""
    length_period: int = 20
    value_area_percent: float = 70.0
    
    # Volume - OPTIMIZED
    volume_threshold: float = 1.478  # OPTIMIZED
    use_relaxed_volume: bool = True
    
    # Confirmation - OPTIMIZED
    min_confirmation_bars: int = 2  # OPTIMIZED
    sustained_bars_required: int = 5  # OPTIMIZED
    signal_cooldown_bars: int = 17  # OPTIMIZED
    
    # Opening Range - OPTIMIZED
    use_or_bias_filter: bool = False  # OPTIMIZED
    or_buffer_points: float = 1.0
    
    # VIX Regime - OPTIMIZED
    use_vix_regime: bool = True  # OPTIMIZED
    vix_high_threshold: int = 25  # OPTIMIZED
    vix_low_threshold: int = 15  # OPTIMIZED
    high_vol_cooldown_mult: float = 1.43  # OPTIMIZED
    low_vol_cooldown_mult: float = 0.88  # OPTIMIZED
    
    # Gamma Exposure Filter
    use_gamma_filter: bool = True  # Filter signals based on gamma regime
    gamma_neutral_zone: float = 5.0  # Points from ZG to consider neutral
    gamma_strike_width: int = 5  # Strike rounding for zero gamma calc
    
    # Signal Enables - OPTIMIZED
    # Long signals
    enable_val_bounce: bool = True  # OPTIMIZED - ENABLED
    enable_poc_reclaim: bool = False  # OPTIMIZED - DISABLED
    enable_breakout: bool = True  # OPTIMIZED - DISABLED --> manually enabled w/GEX filter
    enable_sustained_breakout: bool = False  # OPTIMIZED - DISABLED
    
    # Short signals
    enable_vah_rejection: bool = True  # OPTIMIZED - ENABLED
    enable_poc_breakdown: bool = False  # OPTIMIZED - DISABLED
    enable_breakdown: bool = True  # OPTIMIZED - ENABLED
    enable_sustained_breakdown: bool = False  # OPTIMIZED - DISABLED
    
    # Prior day signals
    enable_prior_val_bounce: bool = True
    enable_prior_vah_rejection: bool = True
    enable_prior_poc_reclaim: bool = False
    enable_prior_poc_breakdown: bool = False
    
    # Misc
    min_es_point_move: float = 6.0


@dataclass
class AlertConfig:
    """Alert/notification settings"""
    enable_discord: bool = True
    discord_webhook: str = os.getenv("DISCORD_WEBHOOK", "")
    enable_pushover: bool = True
    pushover_user_key: str = os.getenv("PUSHOVER_USER_KEY", "")
    pushover_api_token: str = os.getenv("PUSHOVER_API_TOKEN", "")
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
    intra_bar_check_interval: int = 60
    paper_trading: bool = True
    
    def validate(self) -> bool:
        if not self.schwab.app_key or not self.schwab.app_secret:
            raise ValueError("Schwab API credentials not configured")
        return True


config = BotConfig()