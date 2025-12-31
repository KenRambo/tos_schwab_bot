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
├── signal_detector.py   # Signal detection logic
├── position_manager.py  # Position/trade management
├── schwab_auth.py       # OAuth2 authentication
├── schwab_client.py     # Schwab API client
├── notifications.py     # Pushover push notifications
├── analytics.py         # Performance tracking & reporting
│
├── # Optimization Tools
├── optimizer.py         # Grid search optimizer
├── optimizer_smart.py   # Bayesian optimizer (Optuna)
├── optimizer_multirun.py # Multi-run statistical analysis
├── apply_config.py      # Apply optimized params to config + ThinkScript
├── backtest.py          # Backtesting module
│
├── # ThinkScript Studies
├── AMT_Complete_V2.ts   # Full AMT study (mean reversion + trend)
├── AMT_Complete.ts      # Previous version
├── AMT_Experimental.ts  # Experimental signals
├── AMT_TrendDay.ts      # Trend day detection
│
├── # Utilities
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

## Strategy Optimization

The bot includes powerful optimization tools to find the best parameter combinations for your trading strategy.

### Optimizer Options

| Tool | Purpose | Best For |
|------|---------|----------|
| `optimizer.py` | Grid search (exhaustive) | Small parameter spaces, exact testing |
| `optimizer_smart.py` | Bayesian optimization (Optuna) | Large parameter spaces, fast convergence |
| `optimizer_multirun.py` | Statistical confidence testing | Validating results across multiple runs |

### Quick Start

```bash
# Install Optuna (required for smart optimizer)
pip install optuna

# Run smart optimizer (recommended)
python optimizer_smart.py --days 90 --trials 1000 --turbo

# Apply optimized config
python apply_config.py --from best_params.json
```

### Smart Optimizer (Recommended)

The Bayesian optimizer uses Optuna's Tree-structured Parzen Estimator (TPE) to intelligently search the parameter space. It finds near-optimal parameters in 500-2000 trials instead of millions.

```bash
# Basic usage (90 days, 1000 trials)
python optimizer_smart.py --days 90 --trials 1000 --turbo

# Staged optimization (signal params first, then risk params)
python optimizer_smart.py --days 90 --phase signal --trials 500 --save-best signal_params.json
python optimizer_smart.py --days 90 --phase risk --lock signal_params.json --trials 500 --save-best final_params.json

# Deep search with longer history
python optimizer_smart.py --days 120 --trials 2000 --turbo
```

**Options:**
- `--days N` - Days of historical data (default: 90)
- `--trials N` - Number of optimization trials (default: 500)
- `--turbo` - Use more CPU cores for parallel processing
- `--phase [all|signal|risk]` - Optimize all params, signals only, or risk only
- `--lock FILE` - Lock params from a previous run
- `--save-best FILE` - Save best params to JSON
- `--min-trades N` - Minimum trades for valid result (default: 30)

### Multi-Run Optimizer (Statistical Confidence)

Run the optimizer multiple times with different random seeds to build confidence in your parameter choices.

```bash
# Default: 5 runs with different seeds
python optimizer_multirun.py --days 90 --trials 500 --turbo

# More confidence: 10 runs
python optimizer_multirun.py --runs 10 --days 90 --trials 500 --turbo

# Custom seeds
python optimizer_multirun.py --runs 3 --seeds "42,123,999"
```

**Output:**
- `multirun_results/analysis.json` - Statistical analysis across all runs
- `multirun_results/recommended_config.json` - Consensus parameters
- Individual run results in `multirun_results/run_N_seed_X.json`

**What It Shows:**
```
PARAMETER RECOMMENDATIONS

Signal Enables:
  enable_val_bounce........................ False  (1T/4F) [HIGH]
  enable_vah_rejection..................... False  (0T/5F) [HIGH]
  enable_poc_reclaim....................... True   (4T/1F) [HIGH]

Numeric Parameters:
  signal_cooldown_bars..................... 15     (range: 12-18) [HIGH]
  kelly_fraction........................... 0.234  (range: 0.1-0.4) [MED]
```

Confidence levels:
- **HIGH (≥80%)** - Parameter is stable, trust it
- **MED (60-80%)** - Somewhat stable
- **LOW (<60%)** - Unstable, may not matter much

### Grid Search Optimizer (Exhaustive)

For smaller parameter spaces or when you want to test every combination.

```bash
# Fast mode (~500 combinations, ~20 sec)
python optimizer.py --fast

# Risk-focused mode (~2,880 combinations, ~2 min)
python optimizer.py --risk

# Full mode (~10,000 combinations, ~5 min)
python optimizer.py --full

# Custom settings
python optimizer.py --days 60 --capital 25000
```

### Parameters Being Optimized

**Signal Parameters:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `signal_cooldown_bars` | 5-25 | Bars between signals |
| `min_confirmation_bars` | 1-5 | Bars to confirm signal |
| `sustained_bars_required` | 2-6 | Bars for sustained breakout |
| `volume_threshold` | 1.0-2.0 | Volume multiplier |

**Signal Enables (Boolean):**
- `enable_val_bounce`, `enable_vah_rejection`
- `enable_poc_reclaim`, `enable_poc_breakdown`
- `enable_breakout`, `enable_breakdown`
- `enable_sustained_breakout`, `enable_sustained_breakdown`

**Filters:**
- `use_time_filter` - Morning-only trading
- `use_or_bias_filter` - Opening range bias filter

**Delta Targeting:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `target_delta` | 0.20-0.40 | Morning delta target |
| `afternoon_delta` | 0.30-0.50 | Afternoon delta target |
| `afternoon_hour` | 11-14 | Hour to switch deltas |

**Kelly Position Sizing:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `kelly_fraction` | 0.0-3.0 | Kelly multiplier (0=fixed, 1=full Kelly) |
| `max_equity_risk` | 0.05-0.30 | Max % equity per trade |
| `max_kelly_pct_cap` | 0.15-0.40 | Cap on raw Kelly % |
| `hard_max_contracts` | 20-150 | Absolute contract limit |
| `kelly_lookback` | 10-30 | Rolling window for Kelly stats |

**Risk Management:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `max_daily_trades` | 1-10 | Max trades per day |
| `enable_stop_loss` | T/F | Enable stop loss |
| `stop_loss_percent` | 20-70 | Stop loss % |
| `enable_take_profit` | T/F | Enable take profit |
| `take_profit_percent` | 40-200 | Take profit % |
| `enable_trailing_stop` | T/F | Enable trailing stop |
| `min_hold_bars` | 0-10 | Min bars before exit |

### Applying Optimized Config

After optimization, apply the results to your config and ThinkScript:

```bash
# Preview changes (no files written)
python apply_config.py --from multirun_results/recommended_config.json --preview

# Apply config (writes config.py + AMT_Optimized.ts)
python apply_config.py --from multirun_results/recommended_config.json

# With backup of existing files
python apply_config.py --from best_params.json --backup
```

This generates:
- `config.py` - Updated Python bot configuration
- `AMT_Optimized.ts` - Updated ThinkScript with matching signal parameters

### Running Long Optimizations

For long-running optimizations (overnight, closing laptop):

```bash
# Using nohup (simplest)
nohup python optimizer_multirun.py --runs 10 --days 90 --trials 1000 --turbo > multirun.log 2>&1 &
tail -f multirun.log

# Using screen (recommended)
screen -S optimizer
caffeinate -i python optimizer_multirun.py --runs 10 --days 90 --trials 1000 --turbo
# Press Ctrl+A, then D to detach
# Later: screen -r optimizer

# To cancel
pkill -f optimizer
```

### Time Estimates (M1 Max with --turbo)

| Mode | Combinations | Time |
|------|--------------|------|
| Grid --fast | 512 | ~20 sec |
| Grid --risk | 2,880 | ~2 min |
| Grid --full | 10,000 | ~5 min |
| Smart (500 trials) | 500 | ~2 min |
| Smart (1000 trials) | 1,000 | ~3 min |
| Smart (2000 trials) | 2,000 | ~5 min |
| Multirun (5x500) | 2,500 | ~10 min |

### Recommended Workflow

1. **Initial exploration** - Smart optimizer with 500 trials
   ```bash
   python optimizer_smart.py --days 90 --trials 500 --turbo --save-best initial.json
   ```

2. **Build confidence** - Multi-run with 5-10 seeds
   ```bash
   python optimizer_multirun.py --runs 5 --days 90 --trials 500 --turbo
   ```

3. **Validate on different period** - Test on 120 days
   ```bash
   python optimizer_smart.py --days 120 --trials 1000 --turbo
   ```

4. **Apply results**
   ```bash
   python apply_config.py --from multirun_results/recommended_config.json --backup
   ```

5. **Update ThinkScript** - Copy `AMT_Optimized.ts` contents to ToS

---

## Backtesting

The backtesting module replays historical data through the signal detector to validate strategy performance.

### Usage

```bash
# Default: Last 30 trading days
python backtest.py

# Last 60 trading days
python backtest.py --days 60

# From specific start date
python backtest.py --start 2024-12-01

# Export trades to CSV
python backtest.py --output trades.csv

# Verbose mode (shows each trade)
python backtest.py --verbose

# Custom settings
python backtest.py --days 30 --capital 25000 --contracts 2 --output results.csv
```

### Sample Output

```
======================================================================
                        BACKTEST RESULTS
======================================================================
  Period: 2024-11-15 to 2024-12-30
  Trading Days: 30
  Total Bars: 2,340

----------------------------------------------------------------------
  TRADE STATISTICS
----------------------------------------------------------------------
  Total Trades:      47
  Winning Trades:    28
  Losing Trades:     19
  Win Rate:          59.6%

  Avg Win:           $156.32
  Avg Loss:          $-89.45
  Avg Trade:         $56.78
  Avg Bars Held:     8.3

----------------------------------------------------------------------
  PERFORMANCE
----------------------------------------------------------------------
  Total P&L:         $2,668.66
  Profit Factor:     2.58
  Expectancy:        $56.78 per trade

  Best Day:          $425.00
  Worst Day:         $-312.50
  Sharpe Ratio:      1.45

----------------------------------------------------------------------
  RISK METRICS
----------------------------------------------------------------------
  Starting Capital:  $10,000.00
  Ending Capital:    $12,668.66
  Return:            26.7%
  Max Drawdown:      $845.00 (7.2%)

----------------------------------------------------------------------
  SIGNALS BY TYPE
----------------------------------------------------------------------
  VAL_BOUNCE                  12 trades    $    892.50
  POC_RECLAIM                  9 trades    $    645.00
  VAH_REJECTION                8 trades    $    412.30
  BREAKOUT                     7 trades    $    318.86
  ...
======================================================================
```

### What It Measures

- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross wins / gross losses (>1 is profitable)
- **Expectancy** - Average P&L per trade
- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Max Drawdown** - Largest peak-to-trough decline
- **Signal Breakdown** - P&L by signal type to identify best performers

### Option P&L Estimation

The backtester estimates option P&L using:
```
Option P&L ≈ Underlying Move × Delta × 100 × Contracts
```

This is a simplification - real 0DTE options are affected by IV crush, theta decay, and gamma. Use results as directional guidance, not exact predictions.

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