# tos_schwab_bot
Python Auto-trading bot for ToS
=======
# ToS Signal Trading Bot

A Python trading bot that replicates the signal logic from your ThinkOrSwim Auction Market Theory (AMT) indicator and automatically trades SPY options via the Schwab API.

## Overview

This bot monitors SPY price action, detects the same signals your ToS indicator generates, and automatically executes trades:

- **LONG Signal** → Buy nearest OTM SPY CALL (1 contract)
- **SHORT Signal** → Close any CALL position, Buy nearest OTM SPY PUT (1 contract)

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

### Trading Features

- Paper trading mode for testing
- Automatic nearest OTM option selection
- Position tracking and P&L calculation
- Daily trade lockout system
- Comprehensive logging

## Installation

### 1. Clone/Download the Bot

```bash
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

Or export directly:

```bash
export SCHWAB_APP_KEY='your-app-key'
export SCHWAB_APP_SECRET='your-app-secret'
```

## Usage

### Start the Bot

```bash
python trading_bot.py
```

On first run, the bot will:
1. Open your browser for Schwab OAuth authentication
2. After you log in, capture the callback token
3. Save tokens for future sessions (auto-refresh)

### Configuration

Edit `config.py` to customize:

```python
# Trading settings
symbol = "SPY"           # Underlying to trade
contracts = 1            # Contracts per trade
max_daily_trades = 3     # Daily trade limit

# Option selection
min_days_to_expiry = 0   # 0DTE allowed
max_days_to_expiry = 7   # Max 1 week out

# Signal settings (match your ToS indicator)
length_period = 20
signal_cooldown_bars = 8
use_or_bias_filter = True

# Mode
paper_trading = True     # Set False for live trading
```

### Paper Trading vs Live Trading

The bot starts in **paper trading mode** by default. To switch to live:

1. Edit `config.py`: Set `paper_trading = False`
2. Restart the bot
3. Type `CONFIRM` when prompted

## File Structure

```
tos_schwab_bot/
├── trading_bot.py      # Main bot application
├── config.py           # Configuration settings
├── schwab_auth.py      # OAuth2 authentication
├── schwab_client.py    # Schwab API client
├── signal_detector.py  # Signal detection logic
├── position_manager.py # Position/trade management
├── requirements.txt    # Python dependencies
├── .env.template       # Environment template
└── README.md           # This file
```

## How It Works

### Signal Detection Flow

```
1. Fetch SPY quote every 5 seconds
2. Build 5-minute bars from quotes
3. Calculate Value Area (VAH, POC, VAL)
4. Track Opening Range (first 30 min)
5. Check all signal conditions
6. Apply filters (OR bias, time, cooldown)
7. Generate signal if conditions met
```

### Trade Execution Flow

```
1. Signal detected (e.g., VAL_BOUNCE = LONG)
2. Check daily trade limit
3. If opposite position exists, close it
4. Find nearest OTM option (CALL for LONG)
5. Place market order via Schwab API
6. Track position until opposite signal
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

## Logging

The bot logs to both console and `trading_bot.log`:

```
2024-01-15 10:30:05 - INFO - SIGNAL: VAL_BOUNCE - LONG
2024-01-15 10:30:05 - INFO -   Price: 475.50
2024-01-15 10:30:05 - INFO -   VAH: 476.20, POC: 475.80, VAL: 475.40
2024-01-15 10:30:05 - INFO -   OR Bias: BULL
2024-01-15 10:30:05 - INFO - Trade executed: T00001
2024-01-15 10:30:05 - INFO -   Option: SPY240115C00476000
```

## Safety Features

1. **Paper Trading Default** - Won't place real orders until explicitly enabled
2. **Daily Trade Limit** - Prevents overtrading (default: 3/day)
3. **RTH Only** - No trades outside market hours
4. **Confirmation Required** - Must type "CONFIRM" for live mode
5. **Token Security** - OAuth tokens stored locally, auto-refresh

## Troubleshooting

### "Authentication failed"
- Verify your App Key and App Secret are correct
- Ensure redirect URI matches exactly: `https://127.0.0.1:8182/callback`
- Check that your Schwab app is approved and active

### "No suitable option found"
- Market may be closed
- Check that SPY options are trading
- Verify `max_days_to_expiry` allows current expirations

### "Locked out - daily limit reached"
- Working as designed - prevents overtrading
- Resets automatically at midnight

### No signals firing
- Check if within RTH hours
- Verify OR bias filter isn't blocking
- Check signal cooldown (8 bars = 40 min)
- Enable debug logging in config

## Disclaimer

⚠️ **This software is for educational purposes only.**

- Trading options involves substantial risk of loss
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses
- Always test thoroughly in paper trading mode first
- Never trade with money you cannot afford to lose

## License

MIT License - Use at your own risk.
>>>>>>> 1f07689 (initial commit)
