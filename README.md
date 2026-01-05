# ToS Signal Trading Bot

A Python trading bot that replicates the signal logic from your ThinkOrSwim Auction Market Theory (AMT) indicator and automatically trades SPY options via the Schwab API.

## 🎯 Latest Optimization Results (January 2026)

**Variant A (Aggressive) - Best Single Run:**
| Metric | Value |
|--------|-------|
| Win Rate | 72.6% |
| Total P&L | $274,048 |
| Trades | 62 |
| Profit Factor | 3.25 |
| Sharpe Ratio | 5.58 |
| Max Drawdown | $32,715 |

**Key Finding:** Enabling short signals (VAH rejection + breakdown) increased P&L by 2.5x compared to long-only strategy.

## 🔧 Recent Bug Fixes (January 5, 2026)

Fixed critical bugs causing false signals and Value Area mismatch vs ToS:

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| **VA off by ~$6** | Bot VAH: $654.70, ToS VAH: $660.35 | StdDev calculated on all 40 bars instead of last 20 | Use `list(self.bars)[-self.length_period:]` |
| **Wild VA early session** | VAH: $707 at 12:30 (impossible) | Previous day's bars still in deque | Add `self.bars.clear()` in `reset_session()` |
| **SUSTAINED_BREAKOUT firing when disabled** | Signal fired despite `enable_sustained_breakout=False` in config | Signal enable flags not passed from config to SignalDetector | Pass all `enable_*` flags in SignalDetector initialization |

**Files Changed:**
- `signal_detector.py` - Fixed `_calculate_value_area()` and `reset_session()`
- `trading_bot.py` - Added all signal enable flags to SignalDetector init

## Overview

This bot monitors SPY price action, detects the same signals your ToS indicator generates, and automatically executes trades:

- **LONG Signal** → Buy SPY CALL at target delta (default 0.67Δ ATM)
- **SHORT Signal** → Buy SPY PUT at target delta

The bot implements your "hold until opposite signal" strategy - positions are held until an opposite direction signal fires.

## Features

### Signal Detection (Matching Your ToS Indicator)

**Long Signals:**
- ✅ VAL Bounce - Price touches Value Area Low and bounces with volume
- ❌ POC Reclaim - Disabled (noisy)
- ❌ Breakout - Disabled by optimizer
- ❌ Sustained Breakout - Disabled

**Short Signals:**
- ✅ VAH Rejection - Price touches Value Area High and rejects with volume
- ❌ POC Breakdown - Disabled (noisy)
- ✅ Breakdown - Accepted below VAL for confirmation bars
- ❌ Sustained Breakdown - Disabled

### Filters (Matching Your ToS Settings)

- **Opening Range Bias Filter** - Only takes signals aligned with first 30-minute direction
- **RTH Only** - Signals only during regular trading hours (9:30 AM - 4:00 PM ET)
- **Signal Cooldown** - 17 bars (~85 min on 5-min chart) between signals
- **Daily Trade Limit** - Maximum 6 trades per day
- **VIX Regime** - Adjusts cooldown based on volatility (optional)

### Option Selection

- **High Delta Targeting** - Select ATM options (0.67Δ morning, 0.83Δ afternoon)
- **0DTE Support** - Automatically handles same-day expiration
- Configurable DTE range (0-7 days)

### Position Sizing & Risk Management

- **Fixed Fractional Sizing** - Size positions based on account balance
- **Trailing Stop** - 21% trail with 32% activation threshold
- **Daily Loss Limit** - Stop trading if daily loss exceeds $500 or 5% of account
- **Delta Exposure Check** - Prevents overexposure by limiting total delta
- **Max Position Size** - Hard cap on contracts (default: 99)

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

Edit `config.py` to customize, or use the optimizer to generate optimized settings:

```bash
# Apply optimized config from Variant A results
python apply_config_simple.py --from recommended_config_variant_a.json
```

### Optimized Trading Settings (Variant A)

```python
@dataclass
class TradingConfig:
    symbol: str = "SPY"
    max_daily_trades: int = 6              # OPTIMIZED (was 3)
    
    # Option Selection - HIGH DELTA ATM
    use_delta_targeting: bool = True
    target_delta: float = 0.67             # OPTIMIZED - ATM calls/puts
    afternoon_delta: float = 0.83          # OPTIMIZED - higher after noon
    min_days_to_expiry: int = 0            # 0DTE
    max_days_to_expiry: int = 0
    
    # Position Sizing
    use_fixed_fractional: bool = True
    risk_percent_per_trade: float = 23.0   # OPTIMIZED (aggressive)
    max_position_size: int = 99            # OPTIMIZED
    
    # Trailing Stop (always enabled)
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 21.0    # OPTIMIZED
    trailing_stop_activation: float = 32.0 # OPTIMIZED
    
    # Stop Loss / Take Profit
    enable_stop_loss: bool = False         # OPTIMIZED - use trailing instead
    enable_take_profit: bool = False
```

### Optimized Signal Settings

```python
@dataclass
class SignalConfig:
    length_period: int = 20
    volume_threshold: float = 1.478        # OPTIMIZED (was 1.3)
    use_relaxed_volume: bool = True
    min_confirmation_bars: int = 2
    sustained_bars_required: int = 5       # OPTIMIZED (was 3)
    signal_cooldown_bars: int = 17         # OPTIMIZED (was 8) - ~85 min
    use_or_bias_filter: bool = True
    
    # Signal Enables - OPTIMIZED
    enable_val_bounce: bool = True         # Long dips
    enable_vah_rejection: bool = True      # Short rallies
    enable_breakdown: bool = True          # Short breakdowns
    enable_breakout: bool = False          # DISABLED by optimizer
    enable_poc_reclaim: bool = False       # Disabled - noisy
    enable_poc_breakdown: bool = False     # Disabled - noisy
    enable_sustained_breakout: bool = False  # DISABLED
    enable_sustained_breakdown: bool = False # DISABLED
    
    # VIX Regime
    use_vix_regime: bool = True            # OPTIMIZED
    vix_high_threshold: int = 25
    vix_low_threshold: int = 15
    high_vol_cooldown_mult: float = 1.43   # Longer cooldown in high VIX
    low_vol_cooldown_mult: float = 0.88    # Shorter cooldown in low VIX
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
├── config.py            # Configuration settings (auto-generated)
├── signal_detector.py   # Signal detection logic
├── position_manager.py  # Position/trade management
├── schwab_auth.py       # OAuth2 authentication
├── schwab_client.py     # Schwab API client
├── notifications.py     # Pushover push notifications
├── analytics.py         # Performance tracking & reporting
│
├── # Optimization Tools
├── optimizer_simple.py      # Simplified optimizer (high delta focus)
├── optimizer_variant_a.py   # Variant A - Force shorts enabled
├── optimizer_variant_b.py   # Variant B - Force VIX regime
├── optimizer_variant_d.py   # Variant D - Short cooldown
├── optimizer_multirun_simple.py  # Multi-run statistical analysis
├── run_all_variants.py      # Run all variants with shared data cache
├── apply_config_simple.py   # Apply optimized params to config.py
├── backtest.py              # Backtesting module
│
├── # Optimized Configs
├── recommended_config_variant_a.json  # Best aggressive config ($274k)
├── recommended_config_consensus.json  # Conservative consensus config
│
├── # ThinkScript Studies
├── AMT_V35_Optimized.ts     # Latest optimized study (use this)
├── AMT_Complete_V2.ts       # Full AMT study (legacy)
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
3. Initialize signal detector with ALL config params (including signal enables)
4. Load historical bars (today only) to calculate VAH/POC/VAL
5. Set starting balance for position sizing
6. Enter main trading loop
```

### Signal Detection Flow

```
1. Fetch 5-minute bars from Schwab price history
2. Calculate Value Area:
   - VWAP = session volume-weighted average price
   - StdDev = standard deviation of last 20 closes (not all bars!)
   - VAH = VWAP + (StdDev × 0.5)
   - VAL = VWAP - (StdDev × 0.5)
3. Track Opening Range (first 30 min)
4. Check all signal conditions
5. Apply filters (OR bias, time, cooldown, daily limits)
6. Generate signal if conditions met AND signal type is enabled
```

### Trade Execution Flow

```
1. Signal detected (e.g., VAL_BOUNCE = LONG)
2. Check daily loss limit
3. Check delta exposure limit
4. Calculate position size (fixed fractional)
5. Find option at target delta (67Δ morning, 83Δ afternoon)
6. Verify buying power
7. Place market order via Schwab API
8. Track position until opposite signal
9. Record P&L, update analytics
```

### Session Reset Flow (New Trading Day)

```
1. Save prior day VAH/VAL/POC
2. Clear session accumulators (high, low, volume, VWAP)
3. Clear bars deque (prevents cross-day contamination)
4. Reset Opening Range tracking
5. Reset daily trade count
6. Reset bars_above_vah / bars_below_val counters
7. Reset cooldown tracking
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

# Apply config (writes config.py)
python apply_config_simple.py --from recommended_config_variant_a.json
```

This generates `config.py` with optimized parameters. The ThinkScript (`AMT_V35_Optimized.ts`) is already pre-configured.

### Variant Testing Results

| Variant | Win Rate | Total P&L | Trades | Description |
|---------|----------|-----------|--------|-------------|
| Baseline | 84.8% | $57,416 | 33 | Long-only, original params |
| **A: Force Shorts** | **71.8%** | **$143,530** | **60** | **VAH rejection + breakdown enabled** |
| B: Force VIX | 82.1% | $53,784 | 37 | VIX regime adjustment |
| D: Short Cooldown | 71.7% | $43,703 | 65 | More frequent signals |

**Key Finding:** Variant A (forcing shorts enabled) produced 2.5x more profit despite lower win rate. Lower win rate but bigger wins when right.

### Running Your Own Optimization

```bash
# Quick test (500 trials, ~2 min)
python optimizer_simple.py --days 90 --trials 500 --turbo

# Full multi-run with statistical confidence (5 runs)
python optimizer_multirun_simple.py --runs 5 --days 365 --trials 500 --turbo

# Run all variants with shared data cache
python run_all_variants.py --runs 5 --days 365 --trials 500 --turbo
```

### Recommended Configs

**Aggressive (Variant A best run):**
```bash
python apply_config_simple.py --from recommended_config_variant_a.json
```
- $274k P&L, 72.6% WR, $32k max drawdown
- Best for: Larger accounts, higher risk tolerance

**Conservative (Consensus):**
```bash
python apply_config_simple.py --from recommended_config_consensus.json
```
- $143k avg P&L, 71-74% WR, $18k max drawdown  
- Best for: Smaller accounts, lower risk tolerance

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

| ToS Parameter | Python Equivalent | Optimized Value |
|---------------|-------------------|-----------------|
| `volumeThreshold` | `volume_threshold` | **1.478** |
| `minConfirmationBars` | `min_confirmation_bars` | 2 |
| `sustainedBarsRequired` | `sustained_bars_required` | **5** |
| `signalCooldownBars` | `signal_cooldown_bars` | **17** |
| `useORBiasFilter` | `use_or_bias_filter` | True |
| `maxDailyTrades` | `max_daily_trades` | **6** |
| `enableVALBounce` | `enable_val_bounce` | True |
| `enableVAHRejection` | `enable_vah_rejection` | **True** |
| `enableBreakout` | `enable_breakout` | **False** |
| `enableBreakdown` | `enable_breakdown` | True |
| `enableSustainedBreakout` | `enable_sustained_breakout` | **False** |
| `enableSustainedBreakdown` | `enable_sustained_breakdown` | **False** |
| `useVixRegime` | `use_vix_regime` | **True** |
| `vixHighThreshold` | `vix_high_threshold` | 25 |
| `vixLowThreshold` | `vix_low_threshold` | 15 |

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
2025-01-02 10:30:05 ET - INFO - Looking for CALL option, SPY @ $591.25, target delta: 67%
2025-01-02 10:30:05 ET - INFO - Selected: SPY250102C00593000 | Strike: 593.0 | Delta: 0.67
2025-01-02 10:30:05 ET - INFO - Trade executed: T00001
```

## Safety Features

1. **Paper Trading Default** - Won't place real orders until explicitly enabled
2. **Daily Trade Limit** - Prevents overtrading (default: 6/day)
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
- Check signal cooldown (17 bars = ~85 min between signals)
- Enable debug logging: Change `level=logging.INFO` to `level=logging.DEBUG`

### Value Area doesn't match ToS
- **Fixed in v1.1** - StdDev now uses last 20 bars only (was using all 40)
- **Fixed in v1.1** - Session reset now clears bars deque
- Restart bot after updating `signal_detector.py`

### Signals firing when they should be disabled
- **Fixed in v1.1** - All `enable_*` flags now passed to SignalDetector
- Update both `signal_detector.py` and `trading_bot.py`
- Verify your `config.py` has the correct enable flags set

### No push notifications
- Verify `PUSHOVER_USER_KEY` and `PUSHOVER_API_TOKEN` in `.env`
- Test with: `python -c "from notifications import get_notifier; get_notifier().send('Test', 'Test')"`

## Changelog

### v1.1 (January 5, 2026)
- **FIXED:** Value Area calculation now matches ToS exactly (StdDev uses last N bars only)
- **FIXED:** Session reset clears bars deque to prevent cross-day contamination  
- **FIXED:** All signal enable flags now passed from config to SignalDetector
- **ADDED:** VIX regime support with dynamic cooldown adjustment
- **ADDED:** `bars_above_vah` / `bars_below_val` counters in state summary

### v1.0 (December 2025)
- Initial release with full signal detection matching ToS AMT indicator
- Schwab API integration for automated options trading
- Opening Range bias filter
- Daily trade lockout system
- Pushover notifications
- Backtesting and optimization tools

## Disclaimer

⚠️ **This software is for educational purposes only.**

- Trading options involves substantial risk of loss
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses
- Always test thoroughly in paper trading mode first
- Never trade with money you cannot afford to lose

## License

MIT License - Use at your own risk.