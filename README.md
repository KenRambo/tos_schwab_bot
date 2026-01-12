# ToS Signal Trading Bot

A Python trading bot that replicates the signal logic from your ThinkOrSwim Auction Market Theory (AMT) indicator and automatically trades options via the Schwab API.

**Supports:**
- SPY/QQQ single-leg options (calls/puts)
- **ES futures signals → SPX butterfly credit spreads** (new!)
- **Kelly Criterion position sizing for butterflies** (new!)
- Multi-symbol daemon mode

## 🎯 Latest Optimization Results (January 2026)

*Note: These results are from backtesting the **single-leg options strategy** (SPY calls/puts). Butterfly mode is new and should be paper traded to establish your own win rate before live trading.*

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

## 📅 Economic Calendar Alerts (New!)

The bot sends daily alerts about high-impact economic events (FOMC, CPI, NFP, etc.) at market open.

### Setup

**Option 1: With FMP API (more events)**
1. Get a free API key from [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
2. Set the environment variable:
   ```bash
   export FMP_API_KEY="your_api_key_here"
   ```

**Option 2: Without API (static data)**
No setup required! The bot includes static calendars for major events (FOMC, CPI, NFP, GDP, PCE) through 2026.

### Usage

**Standalone (for testing):**
```bash
# Daily alert
python economic_calendar.py --daily

# Weekly summary
python economic_calendar.py --weekly

# Check specific date
python economic_calendar.py --date 2026-01-28

# Check if high-impact events today
python economic_calendar.py --check
```

**Integrated with bot:**
The bot automatically sends an alert at 9:30 AM ET with today's high-impact events:
```
📅 Thursday, Jan 09, 2026
⚠️ 1 High-Impact Event Today
──────────────────────────────
🔴 08:30 ET - Non-Farm Payrolls
──────────────────────────────
💡 Consider adjusting position size near event times.
```

**Cron job (alternative):**
```bash
# Add to crontab for 9:30 AM ET weekdays
30 9 * * 1-5 cd /path/to/bot && python economic_calendar.py --daily

# Weekly summary Sunday evening
0 20 * * 0 cd /path/to/bot && python economic_calendar.py --weekly
```

### High-Impact Events Tracked

**With API (FMP):** All US economic events marked as high-impact

**Static Calendar (no API):**
- **Fed:** FOMC Interest Rate Decisions (8 per year)
- **Inflation:** CPI (monthly), Core PCE (monthly)
- **Employment:** Non-Farm Payrolls (monthly)
- **Growth:** GDP (quarterly)

---

## 🦋 Butterfly Credit Spread Mode (New!)

Trade SPX/XSP butterfly credit spreads based on ES futures signals:

```bash
# Quick start - ES signals → SPX butterflies (paper mode)
./bot.sh butterfly

# Live trading
./bot.sh butterfly /ES SPX --live

# Manual flags
python trading_bot.py --symbol /ES --execution-symbol SPX --butterfly
```

**How it works:**
- Watches /ES futures for AMT signals (VAL bounce, VAH rejection, etc.)
- On LONG signal → Opens call butterfly on SPX
- On SHORT signal → Opens put butterfly on SPX
- Collects credit on each butterfly for "credit stacking" strategy
- **Kelly Criterion** automatically sizes position based on win rate and bankroll

### Butterfly Execution Logic

When a signal fires, the bot:

1. **Calculates strikes** based on underlying price and symbol-specific width:
   - SPX: $5 strike width
   - XSP: $1 strike width
   - SPY: $1 strike width

2. **Builds option symbols** in OCC 21-character format:
   ```
   SPXW  260109P05910000
   └──┘  └────┘└┘└──────┘
    6      6   1    8     = 21 chars
   root   date P  strike×1000
   ```
   - Root: 6 chars padded (SPXW + 2 spaces)
   - Date: YYMMDD
   - Type: C or P
   - Strike: 8 digits (strike × 1000, zero-padded)

3. **Prices the butterfly**:
   - Wing debit = lower_ask + upper_ask
   - Middle credit = middle_bid × 2
   - Target credit = wing_debit × credit_target_pct (default 30%)

4. **Sizes position via Kelly Criterion** (if enabled):
   - Calculates optimal bet fraction based on win rate
   - Applies fractional Kelly (default 25% = quarter Kelly)
   - Respects min/max contract limits

5. **Places order** using one of two methods:

   **TRIGGER (OTO) - Default:**
   ```
   Primary:   BUY lower + BUY upper @ MARKET
   Triggered: SELL 2x middle @ LIMIT (fires when wings fill)
   ```

   **SEQUENTIAL - Fallback:**
   ```
   Step 1: BUY lower + upper @ MARKET
   Step 2: Wait for fill (up to 30s)
   Step 3: SELL 2x middle @ LIMIT
   ```

   The bot tries TRIGGER first and automatically falls back to SEQUENTIAL if the broker doesn't support OTO orders.

6. **Sends notification** with full breakdown:
   ```
   🦋 🟢 SPX ×3 Butterfly ✓
   LONG CALL ×3
   Strikes: 5900/5905/5910
   ─────────────────
   Wings: $20.40 debit
   Middle: 2×$13.50=$27.00
   ─────────────────
   Net Credit: $6.60 (32.4%)
   Target: $6.12 (30%) ✓
   ─────────────────
   Qty: 3 | 💰 $1,980.00
   ```

## 📊 Kelly Criterion Position Sizing (New!)

The bot uses Kelly Criterion to optimally size butterfly positions based on your historical win rate and risk/reward ratio.

### How Kelly Works

The Kelly formula determines what fraction of your bankroll to bet:

```
f* = (bp - q) / b

Where:
  b = win/loss ratio (avg_win / avg_loss)
  p = probability of winning (win rate)
  q = probability of losing (1 - p)
```

For a 65% win rate with 1:2.5 risk/reward:
```
b = 1.0 / 2.5 = 0.4
p = 0.65
q = 0.35

f* = (0.4 × 0.65 - 0.35) / 0.4
f* = 0.10 = 10% of bankroll
```

### Fractional Kelly (Safer)

Full Kelly is aggressive and assumes perfect knowledge. We use **fractional Kelly** (default 25%) for safety:

```
Adjusted Kelly = Full Kelly × kelly_fraction
               = 10% × 0.25
               = 2.5% of bankroll
```

### Position Sizing Example

With a $50,000 account trading 5-point SPX butterflies at $3 net credit:

```
Max loss per butterfly = ($5 × 100) - ($3 × 100) = $200

Risk amount = $50,000 × 2.5% = $1,250
Contracts = $1,250 ÷ $200 = 6.25 → 6 contracts
```

### Kelly Configuration

```python
# In config.py - TradingConfig
use_kelly_sizing: bool = True           # Enable Kelly position sizing
kelly_win_rate: float = 0.65            # Your win rate (start conservative, update from paper trading)
kelly_avg_win: float = 1.0              # Avg win as multiple of credit (keep full credit)
kelly_avg_loss: float = 2.5             # Avg loss as multiple of credit
kelly_fraction: float = 0.25            # Fraction of Kelly (0.25 = quarter Kelly)
kelly_max_contracts: int = 10           # Hard cap on contracts
kelly_min_contracts: int = 1            # Minimum contracts to trade
```

**Important:** The `kelly_win_rate` should be based on YOUR actual butterfly trading results. Start with a conservative estimate (0.50-0.65) and update it after paper trading for at least 20-30 trades.

### Kelly Settings by Risk Profile

| Profile | kelly_fraction | Max Contracts | Description |
|---------|----------------|---------------|-------------|
| Conservative | 0.10 | 3 | ~2% of bankroll per trade |
| Moderate | 0.25 | 5 | ~5% of bankroll per trade |
| Aggressive | 0.50 | 10 | ~10% of bankroll per trade |
| Full Kelly | 1.00 | 20 | ~19% of bankroll (risky!) |

### Disabling Kelly Sizing

To use fixed contract size instead:

```python
use_kelly_sizing: bool = False
contracts: int = 2  # Fixed 2 contracts per trade
```

## 📊 Gamma Exposure Filter (New!)

Filters signals based on dealer gamma positioning to avoid trading against market makers:

| Gamma Regime | Dealer Position | Market Behavior | Signal Filtering |
|--------------|-----------------|-----------------|------------------|
| **Positive** (price ABOVE ZG) | Long gamma | Mean reversion | ✅ VAL/VAH signals, ❌ Breakouts blocked |
| **Neutral** (near ZG) | Mixed | Choppy | ✅ All signals allowed |
| **Negative** (price BELOW ZG) | Short gamma | Trending/momentum | ✅ All signals, ⚠️ Fades risky |

**How it works:**
- Zero Gamma (ZG) is the price level where dealer gamma exposure flips
- Price ABOVE ZG = Positive gamma = dealers fade moves (mean reversion)
- Price BELOW ZG = Negative gamma = dealers chase moves (momentum)

**Automatic mode (default):**
The bot automatically fetches the daily open from Schwab (including globex session for futures) and uses it to approximate ZG. This matches how ToS calculates it.

```bash
# Just run - gamma filter uses automatic approximation
./bot.sh butterfly /ES SPX
```

**Manual override (most accurate):**
For best accuracy, get real ZG from SpotGamma/Tradytics:
```bash
./bot.sh butterfly /ES SPX --zg 7005
```

**Disable entirely:**
```bash
./bot.sh butterfly /ES SPX --no-gamma
```

**Log output:**
```
  Gamma: Daily open (incl. globex) set to $7002.50
  Gamma: 🔴 NEGATIVE | ZG: $7005 | Bias: RIDE MOMENTUM
  Levels: PUT₁ $6980 | ZG $7005 | CALL₁ $7030
  Price vs ZG: $6990.75 (-14.25)
```

## 🔧 Recent Bug Fixes (January 8, 2026)

| Bug | Symptom | Fix |
|-----|---------|-----|
| **Target credit not used** | Orders placed at theoretical price | Now uses target_net_credit for limit orders |
| **Butterfly quantity hardcoded** | Always traded 1 contract | Now supports Kelly sizing with multiple contracts |
| **Paper trading unrealistic** | Hardcoded prices | Now simulates delta decay by strike distance |
| **No fill verification** | Assumed fill at limit | Now polls order status for actual fill price |
| **Symbol-specific width missing** | SPX/XSP used same width | Now uses 5pt for SPX, 1pt for XSP |
| **No VA at market open** | "need 20 bars" until 11:10 AM | Load overnight bars for VA, fixed session reset logic |
| **Drawdown alert spam** | Alert every 5 seconds | Throttled to max 1 per 30 minutes |
| **VA shows 0.00 after restart** | "need 20 bars, have 67" but VA=0 | Filter for RTH-only bars, reset session before load |
| **VA off by ~$6** | Bot VAH didn't match ToS | StdDev now uses last 20 bars only |
| **Cross-session pollution** | Wild VA early session | `reset_session()` clears bars deque |
| **Signals firing when disabled** | SUSTAINED_BREAKOUT despite config | Signal enable flags passed to detector |
| **OR bias buffer zone** | Wrong bias in buffer zone | Added explicit `else` clause for NEUTRAL |
| **Globex suppressing RTH** | Morning signals blocked | RTH cooldown reset at 9:30 AM |
| **VIX=0 during globex** | Wrong cooldown (15 vs 17) | Treat VIX≤0 as unavailable |
| **Price $0.00 on signals** | Invalid quote price | Validate price before live bar update |

## Overview

This bot monitors SPY price action, detects the same signals your ToS indicator generates, and automatically executes trades:

- **LONG Signal** → Buy SPY CALL at target delta (default 0.67Δ ATM)
- **SHORT Signal** → Buy SPY PUT at target delta

The bot implements your "hold until opposite signal" strategy - positions are held until an opposite direction signal fires.

## Features

### Trading Modes

**Single-Leg Options (Default):**
- LONG Signal → Buy CALL at target delta (default 0.67Δ)
- SHORT Signal → Buy PUT at target delta
- Hold until opposite signal fires

**Butterfly Credit Spreads (`--butterfly`):**
- LONG Signal → Call butterfly (buy lower/upper, sell 2× middle)
- SHORT Signal → Put butterfly (buy lower/upper, sell 2× middle)
- Collect net credit on each trade ("credit stacking")
- Supports SPX ($5 strikes) and XSP ($1 strikes)
- **Kelly Criterion** position sizing based on win rate

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
- **RTH Cooldown Reset** - Globex signals don't suppress morning signals
- **Signal Cooldown** - 17 bars (~85 min on 5-min chart) between signals
- **Daily Trade Limit** - Maximum 6 trades per day
- **VIX Regime** - Adjusts cooldown based on volatility (handles VIX=0 gracefully)
- **Gamma Exposure Filter** - Blocks breakouts in positive gamma, warns on fades in negative gamma

### Missed Signal Scanning (New!)

When the bot restarts mid-session, it scans historical bars for signals that occurred while offline:

```
╔════════════════════════════════════════════════════════════╗
║          SCANNING FOR MISSED SIGNALS...                    ║
╚════════════════════════════════════════════════════════════╝

⚠️  FOUND 1 SIGNAL(S) WHILE OFFLINE:

  1. 📈 VAL_BOUNCE - LONG @ $6955.75 (10:45:00 ET)

============================================================
🚨 MOST RECENT SIGNAL (MISSED):
   VAL_BOUNCE - LONG @ $6955.75
   Time: 10:45:00 ET
   VAH: $6962.47 | POC: $6958.44 | VAL: $6954.70
   ⚠️  Consider manual entry if still valid!
============================================================
```

**Features:**
- Replays all historical bars with signals enabled
- Applies gamma filter to each signal
- Lists all signals that fired while offline
- Sends push notification for missed signals
- Shows if signal would have been blocked by gamma

### Symbol Mapping

Trade options on a different symbol than your signal source:

```bash
# Watch /ES futures, trade SPX options
python trading_bot.py --symbol /ES --execution-symbol SPX

# Watch /NQ futures, trade QQQ options  
python trading_bot.py --symbol /NQ --execution-symbol QQQ
```

### Option Selection

- **High Delta Targeting** - Select ATM options (0.67Δ morning, 0.83Δ afternoon)
- **0DTE Support** - Automatically handles same-day expiration
- Configurable DTE range (0-7 days)

### Position Sizing & Risk Management

- **Kelly Criterion Sizing** - Optimal position size based on win rate (butterflies)
- **Fixed Fractional Sizing** - Size positions based on account balance (single-leg)
- **Trailing Stop** - 21% trail with 32% activation threshold
- **Daily Loss Limit** - Stop trading if daily loss exceeds $500 or 5% of account
- **Delta Exposure Check** - Prevents overexposure by limiting total delta
- **Max Position Size** - Hard cap on contracts (Kelly: 10, single-leg: 99)

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
- 🦋 Butterfly filled (with credit collected, quantity, Kelly sizing)
- 🦋 Butterfly rejected
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

### Quick Start

```bash
# SPY paper trading (default)
python trading_bot.py

# SPY live trading
python trading_bot.py --live

# ES signals → SPX butterflies
python trading_bot.py --symbol /ES --execution-symbol SPX --butterfly
```

### CLI Options

```bash
python trading_bot.py [OPTIONS]

Options:
  --symbol, -s SYMBOL          Signal symbol to watch (default: SPY)
  --execution-symbol, -e SYM   Trade options on this symbol instead
  --butterfly                  Enable butterfly credit spread mode
  --paper, -p                  Paper trading (default)
  --live                       Live trading (requires confirmation)
  --contracts, -c N            Contracts per trade
  --max-trades, -m N           Max trades per day
  --cooldown N                 Signal cooldown in bars
  --zg PRICE                   Manual Zero Gamma level (from SpotGamma/Tradytics)
  --no-gamma                   Disable gamma exposure filter
  --no-confirm, -y             Skip confirmation (for daemons)
  --log-file PATH              Custom log file
```

### Daemon Mode (Background)

```bash
# Standard trading
./bot.sh start SPY                    # SPY paper trading
./bot.sh start SPY --live             # SPY live trading
./bot.sh start QQQ                    # QQQ paper trading

# Butterfly mode (ES → SPX)
./bot.sh butterfly                    # Paper mode (default)
./bot.sh butterfly /ES SPX --live     # Live mode

# Control
./bot.sh stop SPY                     # Stop one bot
./bot.sh stop-all                     # Stop all bots
./bot.sh restart SPY                  # Restart
./bot.sh status                       # Show running bots
./bot.sh logs SPY                     # Follow logs
```

### Symbol Mapping Examples

```bash
# ES futures signals → SPX options
python trading_bot.py -s /ES -e SPX

# ES futures signals → SPX butterflies  
python trading_bot.py -s /ES -e SPX --butterfly

# NQ futures signals → QQQ options
python trading_bot.py -s /NQ -e QQQ
```

### Testing Butterfly Orders

Use the test script to verify API connectivity without risking fills:

```bash
# Paper mode (default) - logs what would be sent
python test_butterfly_order.py

# Live TRIGGER mode - places OTO order that won't fill (low credit)
python test_butterfly_order.py --live

# Live SEQUENTIAL mode - places separate wing/middle orders
python test_butterfly_order.py --live --sequential

# Live mode with auto-cancel
python test_butterfly_order.py --live --cancel

# Custom settings
python test_butterfly_order.py --live --symbol SPX --width 5 --credit 0.05
```

**Order Modes:**
- **TRIGGER (default)**: One-Triggers-Other order - wings trigger middle sell
- **SEQUENTIAL**: Place wings first, wait for fill, then place middle sell

If TRIGGER mode returns an error, try `--sequential` to see if your broker supports that format instead.

### Interactive Mode

```bash
python trading_bot.py
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
    execution_symbol: str = None           # NEW: Trade options on different symbol
    max_daily_trades: int = 6              # OPTIMIZED (was 3)
    
    # Butterfly Mode (for SPX/XSP credit stacking)
    butterfly_mode: bool = False           # Enable with --butterfly
    butterfly_wing_width: int = 5          # Points between strikes
    butterfly_credit_target_pct: float = 0.30  # Target 30% credit
    
    # Kelly Criterion Position Sizing (for butterflies)
    use_kelly_sizing: bool = True          # Enable Kelly position sizing
    kelly_win_rate: float = 0.65           # Start conservative, update from paper trading
    kelly_avg_win: float = 1.0             # Avg win = keep full credit
    kelly_avg_loss: float = 2.5            # Avg loss = 2.5× credit
    kelly_fraction: float = 0.25           # Quarter Kelly (safer)
    kelly_max_contracts: int = 10          # Hard cap
    kelly_min_contracts: int = 1           # Minimum
    
    # Option Selection - HIGH DELTA ATM
    use_delta_targeting: bool = True
    target_delta: float = 0.67             # OPTIMIZED - ATM calls/puts
    afternoon_delta: float = 0.83          # OPTIMIZED - higher after noon
    min_days_to_expiry: int = 0            # 0DTE
    max_days_to_expiry: int = 0
    
    # Position Sizing (single-leg)
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
    
    # Gamma Exposure Filter
    use_gamma_filter: bool = True          # Filter based on dealer gamma
    gamma_neutral_zone: float = 5.0        # Points from ZG to be neutral
    gamma_strike_width: int = 5            # Strike rounding ($5 for SPX)
```

### Paper Trading vs Live Trading

**Paper trading is the DEFAULT.** The bot won't place real orders unless you explicitly enable live mode.

To trade live:

```bash
# CLI flag
python trading_bot.py --live

# Or edit config.py
paper_trading = False

# Daemon mode
./bot.sh start SPY --live
./bot.sh butterfly /ES SPX --live
```

When switching to live, you must type `CONFIRM` when prompted (or use `--no-confirm` for daemons).

## File Structure

```
tos_schwab_bot/
├── trading_bot.py       # Main bot application (supports --butterfly, --execution-symbol)
├── config.py            # Configuration settings (includes Kelly params)
├── signal_detector.py   # Signal detection logic
├── position_manager.py  # Position/trade management (butterflies + Kelly sizing)
├── gamma_context.py     # Gamma exposure calculations and filtering
├── schwab_auth.py       # OAuth2 authentication
├── schwab_client.py     # Schwab API client
├── notifications.py     # Pushover push notifications
├── analytics.py         # Performance tracking & reporting
│
├── # Daemon Management
├── bot.sh               # Multi-symbol daemon manager
│
├── # Testing
├── test_butterfly_order.py  # Test live butterfly orders (won't fill)
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
├── AMT_V37_OPTIMIZED.ts     # Latest (gamma filter, completed bar filter)
├── AMT_V36_OPTIMIZED.ts     # Previous version
│
├── # Utilities
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

### Butterfly Trade Execution Flow

```
1. Signal detected (e.g., VAL_BOUNCE = LONG)
2. Check daily loss limit
3. Get underlying price (SPX for SPX/XSP)
4. Calculate strikes with symbol-specific width (5pt SPX, 1pt XSP)
5. Get butterfly prices (wings + middle)
6. Calculate target credit = wing_debit × credit_target_pct
7. Calculate quantity via Kelly Criterion:
   - Full Kelly = (bp - q) / b
   - Adjusted Kelly = Full Kelly × kelly_fraction
   - Risk amount = account × Adjusted Kelly
   - Contracts = Risk amount ÷ max_loss_per_contract
8. Place limit order at target credit
9. Verify fill (poll order status)
10. Track credits, send notification
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

### Single-Leg Mode (Default)

```
2025-01-02 10:30:05 ET - INFO - ─── Bar #42 | 10:30:05 ET ───
2025-01-02 10:30:05 ET - INFO -   Price: $591.25 (O:590.80 H:591.50 L:590.75)
2025-01-02 10:30:05 ET - INFO -   Volume: 1,245,678
2025-01-02 10:30:05 ET - INFO -   VAH: $592.10 | POC: $591.50 | VAL: $590.25
2025-01-02 10:30:05 ET - INFO -   OR Range: $589.50 - $591.75 | Bias: BULLISH
2025-01-02 10:30:05 ET - INFO -   Gamma: 🟢 POSITIVE | ZG: $590 | Bias: FADE MOVES
2025-01-02 10:30:05 ET - INFO -   Levels: PUT₁ $585 | ZG $590 | CALL₁ $595
2025-01-02 10:30:05 ET - INFO -   Price vs ZG: $591.25 (+1.25)
2025-01-02 10:30:05 ET - INFO - 
2025-01-02 10:30:05 ET - INFO - 🚨 SIGNAL: VAL_BOUNCE - LONG
2025-01-02 10:30:05 ET - INFO - 🚨 Gamma: POSITIVE | ZG: $590
2025-01-02 10:30:05 ET - INFO - Position sizing: $10,000.00 balance, 2.0% risk = 3 contracts
2025-01-02 10:30:05 ET - INFO - Delta exposure OK: 0 + 30 = 30 (max: 100)
2025-01-02 10:30:05 ET - INFO - Looking for CALL option, SPY @ $591.25, target delta: 67%
2025-01-02 10:30:05 ET - INFO - Selected: SPY250102C00593000 | Strike: 593.0 | Delta: 0.67
2025-01-02 10:30:05 ET - INFO - Trade executed: T00001
```

### Signal Blocked by Gamma Filter

```
2025-01-02 11:45:05 ET - INFO - ─── Bar #57 | 11:45:05 ET ───
2025-01-02 11:45:05 ET - INFO -   Price: $593.50 (O:593.00 H:593.75 L:592.80)
2025-01-02 11:45:05 ET - INFO -   Gamma: 🟢 POSITIVE | ZG: $591 | Bias: FADE MOVES
2025-01-02 11:45:05 ET - INFO - 
2025-01-02 11:45:05 ET - INFO - ⚠️ Signal blocked by gamma filter: ✗ BREAKOUT blocked in +gamma (breakouts get faded)
```

### Butterfly Mode with Kelly Sizing (--butterfly)

```
2026-01-08 10:35:02 ET - INFO - ============================================================
2026-01-08 10:35:02 ET - INFO - 🦋 BUTTERFLY SIGNAL: SHORT PUT
2026-01-08 10:35:02 ET - INFO -    Signal: VAH_REJECTION
2026-01-08 10:35:02 ET - INFO -    Price: $5925.50
2026-01-08 10:35:02 ET - INFO - ============================================================
2026-01-08 10:35:02 ET - INFO -    Underlying: $5925.50
2026-01-08 10:35:02 ET - INFO -    Strikes: 5910/5915/5920 (width: 5)
2026-01-08 10:35:02 ET - INFO -    ┌─ BUTTERFLY STRUCTURE ─────────────────────
2026-01-08 10:35:02 ET - INFO -    │ Lower wing (5910): $8.40 debit
2026-01-08 10:35:02 ET - INFO -    │ Middle x2  (5915): $11.20 × 2 = $22.40 credit
2026-01-08 10:35:02 ET - INFO -    │ Upper wing (5920): $9.60 debit
2026-01-08 10:35:02 ET - INFO -    ├───────────────────────────────────────────
2026-01-08 10:35:02 ET - INFO -    │ Wing debit:       $18.00
2026-01-08 10:35:02 ET - INFO -    │ Middle credit:    $22.40
2026-01-08 10:35:02 ET - INFO -    │ Theoretical net:  $4.40
2026-01-08 10:35:02 ET - INFO -    ├───────────────────────────────────────────
2026-01-08 10:35:02 ET - INFO -    │ Target credit %:  30%
2026-01-08 10:35:02 ET - INFO -    │ Target net:       $5.40
2026-01-08 10:35:02 ET - INFO -    │ Required middle:  $11.70 each ($23.40 total)
2026-01-08 10:35:02 ET - INFO -    └───────────────────────────────────────────
2026-01-08 10:35:02 ET - INFO -    ┌─ POSITION SIZING ──────────────────────────
2026-01-08 10:35:02 ET - INFO -    │ Kelly sizing: ENABLED
2026-01-08 10:35:02 ET - INFO -    │ Win rate: 65.0%
2026-01-08 10:35:02 ET - INFO -    │ Full Kelly: 10.0%
2026-01-08 10:35:02 ET - INFO -    │ Fractional (25%): 2.5%
2026-01-08 10:35:02 ET - INFO -    │ Contracts: 2
2026-01-08 10:35:02 ET - INFO -    └───────────────────────────────────────────
2026-01-08 10:35:02 ET - INFO - Kelly sizing: $50,000 × 2.50% = $1,250 risk
2026-01-08 10:35:02 ET - INFO - Kelly sizing: $1,250 ÷ $126 max loss = 2 contracts
2026-01-08 10:35:02 ET - INFO - [PAPER] 🦋 Butterfly order (×2):
2026-01-08 10:35:02 ET - INFO - [PAPER]   BUY  2x 5910 PUT
2026-01-08 10:35:02 ET - INFO - [PAPER]   SELL 4x 5915 PUT
2026-01-08 10:35:02 ET - INFO - [PAPER]   BUY  2x 5920 PUT
2026-01-08 10:35:02 ET - INFO - [PAPER]   Limit Credit: $5.40 per spread
2026-01-08 10:35:02 ET - INFO - [PAPER]   ✓ Filled @ $5.40
2026-01-08 10:35:02 ET - INFO -    ✓ FILLED 2x @ net credit $5.40
2026-01-08 10:35:02 ET - INFO -    📊 Credit achieved: 30.0% (target: 30%)
2026-01-08 10:35:02 ET - INFO -    💰 CREDIT STACKED: $1,080.00 (2 contracts)
2026-01-08 10:35:02 ET - INFO -    📊 Credits today: $1,080.00 | Total: $3,240.00
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
9. **Gamma Filter** - Blocks momentum signals in mean-reversion environments
10. **Missed Signal Alerts** - Notifies of signals that occurred while bot was offline
11. **Kelly Position Limits** - Hard cap on contracts regardless of Kelly sizing

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

### Value Area shows 0.00 or "Building..."
- **Fixed in v1.4** - Now loads overnight/globex bars so VA is ready at market open
- Bot loads ~200 bars including overnight session (6pm previous day to now)
- Session reset logic updated to not wipe overnight data at RTH transition
- If you still see "Building...", check that `extended_hours=True` in the API call

### Value Area doesn't match ToS
- **By design** - Bot uses VWAP ± StdDev, ToS uses true volume profile
- On trending days, bot's VA will be narrower than ToS
- This is intentional - the VWAP approach was backtested and validated
- Narrower VA on trend days = more conservative (avoids fading strong moves)
- **Fixed in v1.1** - StdDev now uses last 20 bars only (was using all 40)
- **Fixed in v1.1** - Session reset now clears bars deque

### Signals firing when they should be disabled
- **Fixed in v1.1** - All `enable_*` flags now passed to SignalDetector
- Update both `signal_detector.py` and `trading_bot.py`
- Verify your `config.py` has the correct enable flags set

### Kelly sizing contracts too high/low
- Adjust `kelly_fraction` (0.25 = quarter Kelly is conservative)
- Adjust `kelly_max_contracts` hard cap
- Check `kelly_win_rate` matches your historical performance
- Verify account balance is being read correctly

### No push notifications
- Verify `PUSHOVER_USER_KEY` and `PUSHOVER_API_TOKEN` in `.env`
- Test with: `python -c "from notifications import get_notifier; get_notifier().send('Test', 'Test')"`

## Changelog

### v1.5 (January 8, 2026)
- **NEW:** Economic calendar alerts via FMP API (FOMC, CPI, NFP, etc.)
- **NEW:** Kelly Criterion position sizing for butterflies
- **NEW:** `test_butterfly_order.py` script for testing live orders
- **NEW:** TRIGGER (OTO) and SEQUENTIAL order methods with auto-fallback
- **FIXED:** Option symbol format now uses correct 21-char OCC format (`SPXW  260109P05910000`)
- **FIXED:** Butterfly target credit now used in order placement (was using theoretical)
- **FIXED:** Butterfly quantity now dynamic via Kelly (was hardcoded to 1)
- **FIXED:** Paper trading simulation now uses realistic delta decay
- **FIXED:** Symbol-specific wing widths (5pt SPX, 1pt XSP/SPY)
- **FIXED:** Fill verification polls order status for actual fill price
- **IMPROVED:** Notifications show quantity, Kelly breakdown, and target comparison
- **IMPROVED:** Logging shows full Kelly calculation and position sizing rationale

### v1.4 (January 8, 2026)
- **FIXED:** VA now ready at market open - overnight/globex bars included for VA calculation
- **FIXED:** Session reset no longer wipes overnight data when transitioning to RTH
- **IMPROVED:** Extended hours data loaded (200 bars covering overnight session)

### v1.3 (January 7, 2026)
- **NEW:** Gamma exposure filter (`use_gamma_filter`) - filters signals based on dealer gamma positioning
- **NEW:** `gamma_context.py` module for zero gamma calculation and regime detection
- **NEW:** `--no-gamma` CLI flag to disable gamma filtering
- **NEW:** Gamma levels plotted on ToS chart (Zero Gamma, Call γ1, Put γ1)
- **NEW:** `onlyCompletedBars` in ToS study - prevents false signals that disappear before bar closes
- **NEW:** Missed signal scanning on restart - shows signals that occurred while bot was offline
- **FIXED:** VA showing 0.00 after mid-session restart (now filters for RTH-only bars, resets session before load)
- **FIXED:** Log now shows actual detector bar count instead of total bars processed
- **FIXED:** Drawdown alerts spamming every 5 seconds (now throttled to max 1 per 30 minutes)
- **UPDATED:** ToS study to V3.7 with gamma filter integration
- **UPDATED:** Bot logs gamma regime and levels each bar

### v1.2 (January 6, 2026)
- **NEW:** Butterfly credit spread mode (`--butterfly` flag)
- **NEW:** Symbol mapping (`--execution-symbol`) for ES→SPX trading
- **NEW:** `./bot.sh butterfly` convenience command
- **FIXED:** OR bias buffer zone now defaults to NEUTRAL (was keeping stale value)
- **FIXED:** RTH cooldown reset at 9:30 AM (globex signals no longer suppress morning)
- **FIXED:** VIX=0 during extended hours uses default cooldown (was triggering low-vol multiplier)
- **FIXED:** Price $0.00 on intra-bar signals (now validates quote price)
- **UPDATED:** bot.sh with butterfly examples and improved help text

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