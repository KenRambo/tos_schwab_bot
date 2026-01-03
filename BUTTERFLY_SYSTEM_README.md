# 🦋 SPX/XSP Butterfly Credit Stacker

## Overview

A trading bot that stacks 0DTE butterfly credit spreads based on **ES futures** AMT signals, trading **SPX or XSP options**. Designed to:
- Collect theta throughout the day
- Avoid day trades (always completes butterfly)
- Auto-select SPX or XSP based on account size
- Let positions expire (cash settled)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       ES FUTURES (/ES)                          │
│                    (Signal Detection)                           │
│                                                                 │
│   Your AMT indicator runs on ES 5-min bars                     │
│   Signals: VAL_BOUNCE, VAH_REJECTION, BREAKOUT, etc.           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Signal fires!
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FETCH CURRENT $SPX PRICE                       │
│                                                                 │
│   Don't convert from ES - just get actual SPX quote            │
│   Use SPX price for ATM strike calculation                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPX or XSP OPTIONS                           │
│                     (Trade Execution)                           │
│                                                                 │
│   Account < $25k  →  Trade $XSP (SPX/10)                       │
│   Account ≥ $25k  →  Trade $SPX                                │
│                                                                 │
│   Strikes based on ACTUAL SPX price at signal time             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point:** When a signal fires on ES, we fetch the **actual $SPX price** to calculate strikes. No ES→SPX conversion needed.

## Strategy

```
ES Signal (from AMT indicator)
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. BUY WINGS (ATM + ATM±10)            │
│    - Calls for bullish                  │
│    - Puts for bearish                   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 2. MONITOR middle strike               │
│    Wait for: bid × 2 ≥ debit × 1.30    │
│                                         │
│    Target: 30% net credit              │
│    Abort if: Wings down 30%            │
│    Abort if: EOD approaching           │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 3. SELL 2x MIDDLE                       │
│    Complete butterfly regardless        │
│    (even at net debit)                  │
│    NO DAY TRADE!                        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 4. LET EXPIRE (Cash Settled)           │
│    - Keep net credit                    │
│    - Potential kicker if pins middle    │
│    - Max loss = wing width - credit     │
└─────────────────────────────────────────┘
```

## Instrument Selection

| Account Equity | Instrument | Index Price | Max Risk/Trade |
|----------------|------------|-------------|----------------|
| < $2,500       | Can't trade | -          | -              |
| $2,500-$25,000 | **XSP**    | ~$600       | ~$50           |
| > $25,000      | **SPX**    | ~$6,000     | ~$500          |

XSP is 1/10th the size of SPX, perfect for smaller accounts.

## Configuration

```python
ButterflyConfig(
    # Instrument Selection
    spx_min_equity=25000.0,      # Min equity for SPX
    xsp_min_equity=2500.0,       # Min equity for XSP
    
    # Structure
    wing_width=5,                 # Points between strikes
    credit_target_pct=0.30,       # 30% credit target
    abort_loss_pct=0.30,          # Early exit if wings down 30%
    
    # Position Sizing
    max_risk_per_trade_pct=0.02,  # 2% of equity per trade
    max_concurrent_butterflies=10,
    max_daily_butterflies=20,
    
    # Timing
    eod_cutoff_minutes=15,        # Complete all 15 min before close
    
    # Paper Trading
    paper_trading=True,
    paper_starting_equity=10000.0
)
```

## Usage

### Paper Trading (Testing)
```bash
python butterfly_trader.py --paper --equity 10000
```

### Backtest (Demo Data)
```bash
python butterfly_trader.py --demo --days 30
```

### Backtest (Real ES Data from Schwab)
```bash
python butterfly_trader.py --backtest --days 60
# Fetches /ES futures data, runs signals, simulates SPX butterflies
```

### Live Trading
```bash
python butterfly_trader.py
```

## Live Trading Flow

```python
# When ES signal fires:
def on_signal(signal):
    # 1. Get actual SPX price (not converted from ES)
    spx_price = client.get_quote('$SPX').last_price  # e.g., 6000
    
    # 2. Select instrument based on equity
    if equity >= 25000:
        symbol = '$SPX'
        price = spx_price        # 6000
    else:
        symbol = '$XSP'  
        price = spx_price / 10   # 600
    
    # 3. Calculate ATM strikes
    atm = round(price / 5) * 5   # Round to nearest 5
    
    # 4. Build butterfly
    # ...
```

## Files

| File | Description |
|------|-------------|
| `butterfly_trader.py` | Main production bot (all-in-one) |
| `butterfly_bot.py` | Core butterfly logic (standalone) |
| `backtest_butterfly.py` | Standalone backtester |
| `butterfly_stacker.py` | Signal-integrated version |

## Key Features

### 1. Auto SPX/XSP Selection
```python
selector = InstrumentSelector(config)
symbol, price, mult = selector.select_instrument(equity, spx_price)
# Returns: ('$XSP', 600.0, 100) for small accounts
# Returns: ('$SPX', 6000.0, 100) for large accounts
```

### 2. Realistic Option Pricing
```python
pricer = RealisticOptionPricer()
# Uses Black-Scholes with:
# - IV smile/skew
# - Realistic bid-ask spreads
# - 0DTE rapid time decay
```

### 3. Paper Trading Simulator
```python
simulator = PaperTradingSimulator(starting_equity=10000, pricer=pricer)
# Simulates fills at bid/ask
# Tracks P&L in real-time
# Perfect for testing
```

### 4. Never Incomplete Butterflies
Even on "abort" conditions, we **always complete the butterfly** by selling 2x middle:
- Avoids day trades
- Defined risk
- Preserves kicker potential

## Risk Management

| Scenario | Max Loss |
|----------|----------|
| Net Credit Butterfly | $0 (worst case: expire worthless, keep credit) |
| Net Debit Butterfly | Debit paid (e.g., $200 if paid $2.00 net debit) |
| Pin at Middle | **MAX PROFIT!** (wing_width + net_credit) |

## Example Day

```
10:15 AM - Bullish Signal @ SPX 6050
           Buy 6050/6060 calls for $7.20
           Target: Sell 2x 6055 calls for $9.36 (30% credit)

10:45 AM - Target HIT! Middle bid = $4.80
           Sell 2x 6055 calls for $9.60
           Net Credit: $2.40 ($240 per contract)
           Status: Riding to expiration 🎢

11:30 AM - Bearish Signal @ SPX 6065
           Buy 6055/6045 puts for $6.80
           Target: Sell 2x 6050 puts for $8.84

12:15 PM - Wings down 25%... still waiting

12:45 PM - Wings down 30%! EARLY EXIT
           Sell 2x 6050 puts at current bid ($2.90)
           Net: $5.80 credit - $6.80 debit = -$1.00 (net debit)
           Status: Still riding (defined risk) 🎢

4:00 PM - SETTLEMENT
           SPX closes at 6055

           Butterfly 1 (Calls): Settles at $5.00 intrinsic
                               P&L = $240 + $500 = $740 ✅

           Butterfly 2 (Puts): Settles at $0 (OTM)
                              P&L = -$100 ❌

           Daily Total: +$640
```

## Integration with Existing Bot

To integrate with your `trading_bot.py`:

```python
from butterfly_trader import ButterflyTrader, ButterflyConfig

# In your main bot
config = ButterflyConfig(paper_trading=False)
butterfly_bot = ButterflyTrader(config, schwab_client=self.client)

# When signal fires
def on_signal(self, signal):
    if signal.direction in [Direction.LONG, Direction.SHORT]:
        butterfly_bot.process_signal(signal)

# In monitoring loop
def monitor(self):
    spx_price = self.get_spx_price()
    butterfly_bot.monitor_pending(spx_price)
```

## Next Steps

1. **Run backtest with real data** to validate strategy
2. **Paper trade for 1 week** to verify execution
3. **Start live with XSP** (smaller size)
4. **Scale to SPX** once profitable

## Notes

- SPX/XSP are **cash settled** - no assignment risk
- 0DTE options have rapid time decay - **timing matters**
- The 30% credit target may not always be hit - that's OK
- Completing at net debit is still **defined risk**
- Multiple butterflies can be profitable if SPX pins near ANY middle strike