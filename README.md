# ToS Signal Trading Bot

A Python trading bot that replicates the signal logic from your ThinkOrSwim Auction Market Theory (AMT) indicator and automatically trades SPY options via the Schwab API.

## Overview

This bot monitors SPY price action, detects the same signals your ToS indicator generates, and automatically executes trades:

- **LONG Signal** → Buy SPY CALL at target delta (default 30Δ)
- **SHORT Signal** → Close any CALL position, Buy SPY PUT at target delta

The bot implements your "hold until opposite signal" strategy - positions are held until an opposite direction signal fires.

## Features

### Signal Detection (Matching Your ToS Indicator)

**Long Signals:**
- VAL Bounce - Price touches Value Area Low and bounces with volume
- POC Reclaim - Price crosses above Point of Control with increasing volume
- Breakout - Accepted above VAH for confirmation bars
- Sustained Breakout - Extended time above VAH
- Prior Day VAL Bounce / POC Reclaim

**Short Signals:**
- VAH Rejection - Price touches Value Area High and rejects with volume
- POC Breakdown - Price crosses below POC with increasing volume
- Breakdown - Accepted below VAL for confirmation bars
- Sustained Breakdown - Extended time below VAL
- Prior Day VAH Rejection / POC Breakdown

### Filters (Matching Your ToS Settings)

- **Opening Range Bias Filter** - Only takes signals aligned with first 30-minute direction
- **RTH Only** - Signals only during regular trading hours (9:30 AM - 4:00 PM ET)
- **Time Filter** (optional) - Avoids first 25 min, lunch hour, last 15 min
- **Signal Cooldown** - 8 bars (40 min on 5-min chart) between signals
- **Daily Trade Limit** - Maximum 3 trades per day (configurable)

### Option Selection

- **Delta Targeting** - Select options by delta (default: 30Δ) instead of just nearest OTM
- **0DTE Support** - Automatically handles after-hours option selection for next trading day
- Configurable DTE range (0-7 days)

### Position Sizing & Risk Management

- **Fixed Fractional Sizing** - Size positions based on account balance (default: 2% risk per trade)
- **Daily Loss Limit** - Stop trading if daily loss exceeds $500 or 5% of account
- **Delta Exposure Check** - Prevents overexposure by limiting total delta across positions
- **Max Position Size** - Hard cap on contracts (default: 10)
- Optional stop loss, take profit, and trailing stop

### Monitoring & Analytics

- **Daily P&L Summary** - Push notification at 4:15 PM ET with day's performance
- **Weekly Report** - Win rate, avg win/loss, Sharpe ratio, best/worst day
- **Drawdown Alerts** - Notification when down 10%+ from peak balance
- Performance data persisted to JSON for historical tracking

### Push Notifications (Pushover)

Get mobile alerts for:
- 🤖 Bot started/stopped
- 📈📉 Signal detected
- 🟢🔴 Trade executed
- 💰💸 Position closed (with P&L)
- 💳 Insufficient buying power
- 🔻 Drawdown alert
- 📊 Daily/weekly summaries

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KenRambo/tos_schwab_bot.git
cd tos_schwab_bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Get Schwab API Credentials

1. Go to [Schwab Developer Portal](https://developer.schwab.com/)
2. Create a new application
3. Set callback URL to: `https://127.0.0.1:8182/callback`
4. Note your App Key and App Secret

### 4. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit with your credentials
nano .env
```

```bash
# Required
SCHWAB_APP_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret

# Optional - Push notifications
PUSHOVER_USER_KEY=your_user_key
PUSHOVER_API_TOKEN=your_api_token
```

### 5. Setup Pushover (Optional)

1. Download Pushover app on iOS/Android ($5 one-time)
2. Create account at [pushover.net](https://pushover.net)
3. Get User Key from dashboard
4. Create Application to get API Token
5. Add to `.env` file

## Usage

### Interactive Mode

```bash
python trading_bot.py
```

### Background Mode (survives terminal close)

```bash
# Start bot in background
./bot.sh start

# Check status
./bot.sh status

# View live logs
./bot.sh logs

# Stop bot
./bot.sh stop
```

On first run, the bot will:
1. Open your browser for Schwab OAuth authentication
2. After you log in, capture the callback token
3. Save tokens for future sessions (auto-refresh)

## Configuration

Edit `config.py` to customize:

### Trading Settings

```python
@dataclass
class TradingConfig:
    symbol: str = "SPY"
    contracts: int = 1              # Base contracts (overridden by fixed fractional)
    max_daily_trades: int = 3
    
    # Option Selection
    use_delta_targeting: bool = True
    target_delta: float = 0.30      # 30 delta
    min_days_to_expiry: int = 0     # 0DTE allowed
    max_days_to_expiry: int = 0     # 0 = 0DTE only
    
    # Position Sizing
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 2.0   # Risk 2% per trade
    max_position_size: int = 10           # Never exceed 10 contracts
    min_position_size: int = 1
    
    # Daily Loss Limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_dollars: float = 500.0
    max_daily_loss_percent: float = 5.0
    
    # Delta Exposure
    enable_correlation_check: bool = True
    max_delta_exposure: float = 100.0     # Max total delta
```

### Signal Settings

```python
@dataclass
class SignalConfig:
    length_period: int = 20
    value_area_percent: float = 70.0
    volume_threshold: float = 1.3
    use_relaxed_volume: bool = True
    min_confirmation_bars: int = 2
    sustained_bars_required: int = 3
    signal_cooldown_bars: int = 8
    use_or_bias_filter: bool = True
```

### Analytics Settings

```python
@dataclass
class AnalyticsConfig:
    enable_daily_summary: bool = True
    daily_summary_hour: int = 16          # 4 PM ET
    daily_summary_minute: int = 15        # 4:15 PM ET
    enable_weekly_report: bool = True
    enable_drawdown_alerts: bool = True
    drawdown_alert_threshold: float = 10.0  # Alert at 10% drawdown
```

### Paper Trading vs Live Trading

The bot starts in **paper trading mode** by default. To switch to live:

1. Edit `config.py`: Set `paper_trading = False`
2. Restart the bot
3. Type `CONFIRM` when prompted (or use `--no-confirm` flag)

## File Structure

```
tos_schwab_bot/
├── trading_bot.py       # Main bot application
├── config.py            # Configuration settings
├── schwab_auth.py       # OAuth2 authentication
├── schwab_client.py     # Schwab API client
├── signal_detector.py   # Signal detection logic
├── position_manager.py  # Position/trade management
├── notifications.py     # Pushover push notifications
├── analytics.py         # Performance tracking & reporting
├── bot.sh               # Background process management
├── requirements.txt     # Python dependencies
├── .env.template        # Environment template
└── README.md            # This file
```

## How It Works

### Startup Flow

```
1. Load configuration
2. Authenticate with Schwab API
3. Initialize signal detector
4. Load 30 historical bars to calculate VAH/POC/VAL
5. Set starting balance for position sizing
6. Enter main trading loop
```

### Signal Detection Flow

```
1. Fetch 5-minute bars from Schwab price history
2. Calculate Value Area (VAH, POC, VAL) over 20 bars
3. Track Opening Range (first 30 min)
4. Check all signal conditions
5. Apply filters (OR bias, time, cooldown, daily limits)
6. Generate signal if conditions met
```

### Trade Execution Flow

```
1. Signal detected (e.g., VAL_BOUNCE = LONG)
2. Check daily loss limit
3. Check delta exposure limit
4. Calculate position size (fixed fractional)
5. Find option at target delta (30Δ)
6. Verify buying power
7. Place market order via Schwab API
8. Track position until opposite signal
9. Record P&L, update analytics
```

### Monitoring Flow

```
1. Every 30 min: Update balance, check drawdown
2. At 4:15 PM ET: Send daily summary notification
3. Friday close: Send weekly report
4. On position close: Record P&L, notify
```

## Matching Your ToS Indicator

The Python signal detection mirrors your ToS script logic:

| ToS Parameter | Python Equivalent |
|---------------|-------------------|
| `lengthPeriod = 20` | `length_period = 20` |
| `volumeThreshold = 1.3` | `volume_threshold = 1.3` |
| `useRelaxedVolume = yes` | `use_relaxed_volume = True` |
| `minConfirmationBars = 2` | `min_confirmation_bars = 2` |
| `sustainedBarsRequired = 3` | `sustained_bars_required = 3` |
| `signalCooldownBars = 8` | `signal_cooldown_bars = 8` |
| `useORBiasFilter = yes` | `use_or_bias_filter = True` |
| `orBufferPoints = 1.0` | `or_buffer_points = 1.0` |
| `maxDailyTrades = 3` | `max_daily_trades = 3` |
| `rthOnlySignals = yes` | `rth_only = True` |

## Sample Log Output

```
2025-01-02 10:30:05 ET - INFO - ─── Bar #42 | 10:30:05 ET ───
2025-01-02 10:30:05 ET - INFO -   Price: $591.25 (O:590.80 H:591.50 L:590.75)
2025-01-02 10:30:05 ET - INFO -   Volume: 1,245,678
2025-01-02 10:30:05 ET - INFO -   VAH: $592.10 | POC: $591.50 | VAL: $590.25
2025-01-02 10:30:05 ET - INFO -   OR Range: $589.50 - $591.75 | Bias: BULLISH
2025-01-02 10:30:05 ET - INFO - 
2025-01-02 10:30:05 ET - INFO - 🚨 SIGNAL: VAL_BOUNCE - LONG
2025-01-02 10:30:05 ET - INFO - Position sizing: $10,000.00 balance, 2.0% risk = 3 contracts
2025-01-02 10:30:05 ET - INFO - Delta exposure OK: 0 + 30 = 30 (max: 100)
2025-01-02 10:30:05 ET - INFO - Looking for CALL option, SPY @ $591.25, target delta: 30%
2025-01-02 10:30:05 ET - INFO - Selected: SPY250102C00593000 | Strike: 593.0 | Delta: 0.31
2025-01-02 10:30:05 ET - INFO - Trade executed: T00001
```

## Safety Features

1. **Paper Trading Default** - Won't place real orders until explicitly enabled
2. **Daily Trade Limit** - Prevents overtrading (default: 3/day)
3. **Daily Loss Limit** - Stops trading when down $500 or 5%
4. **Delta Exposure Limit** - Prevents over-leveraging
5. **RTH Only** - No trades outside market hours
6. **Confirmation Required** - Must type "CONFIRM" for live mode
7. **Buying Power Check** - Validates funds before each trade
8. **Drawdown Alerts** - Notifies when down 10%+ from peak

## Troubleshooting

### "Authentication failed"
- Verify your App Key and App Secret are correct
- Ensure redirect URI matches exactly: `https://127.0.0.1:8182/callback`
- Check that your Schwab app is approved and active

### "No suitable option found"
- Market may be closed (options only trade RTH)
- Verify `max_days_to_expiry` setting
- If after hours, bot automatically uses next trading day

### "Locked out - daily limit reached"
- Working as designed - prevents overtrading
- Could be trade count limit OR daily loss limit
- Resets automatically at midnight

### "Delta exposure limit"
- You have existing positions that would exceed max delta
- Close existing positions or increase `max_delta_exposure`

### No signals firing
- Check if within RTH hours (9:30 AM - 4:00 PM ET)
- Verify OR bias filter isn't blocking (signals must align with opening range direction)
- Check signal cooldown (8 bars = 40 min between signals)
- Enable debug logging: Change `level=logging.INFO` to `level=logging.DEBUG`

### No push notifications
- Verify `PUSHOVER_USER_KEY` and `PUSHOVER_API_TOKEN` in `.env`
- Test with: `python -c "from notifications import get_notifier; get_notifier().send('Test', 'Test')"`

## Disclaimer

⚠️ **This software is for educational purposes only.**

- Trading options involves substantial risk of loss
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses
- Always test thoroughly in paper trading mode first
- Never trade with money you cannot afford to lose

## License

MIT License - Use at your own risk.