"""
Simplified Strategy Optimizer - High Delta ATM Focus
=====================================================

Stripped back to essentials:
- Core AMT signals only
- VIX regime adjustments (optional)
- Black-Scholes pricing with HIGH DELTA (0.60-0.80) ATM contracts
- No TICK filter, no time windows, no ATR stops

High delta = reliable directional exposure, not gamma lottery tickets.

Usage:
    python optimizer_simple.py                         # Default 90 days
    python optimizer_simple.py --days 365 --trials 1000 --turbo
"""
import os
import sys
import logging
import argparse
import json
import warnings
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from multiprocessing import cpu_count
import time as time_module

# Suppress noise
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import FrozenTrial
except ImportError:
    print("ERROR: pip install optuna")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.ERROR)

from signal_detector import SignalDetector, Bar

# -----------------------------
# Disk cache (SPY 5m + VIX close)
# -----------------------------
CACHE_DIR = Path(os.environ.get("BARS_CACHE_DIR", "./data_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPY_CACHE_FILE = CACHE_DIR / "spy_5m.pkl"
VIX_CACHE_FILE = CACHE_DIR / "vix_5m.pkl"

# Globals
GLOBAL_BARS = None
BEST_RESULT = None


@dataclass
class TrialResult:
    params: Dict[str, Any]
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    score: float = 0.0


# -----------------------------
# Black-Scholes
# -----------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes European option price."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 1e-8:
        return max(0.0, S - K) if is_call else max(0.0, K - S)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 1e-8:
        return (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def solve_strike_for_delta(
    S: float,
    target_delta: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool,
    iters: int = 40,
) -> float:
    """Find strike K for target delta (binary search)."""
    lo, hi = S * 0.8, S * 1.2  # Tighter range for ATM

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = bs_delta(S, mid, T, r, sigma, is_call)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vix_to_iv(vix: float) -> float:
    """Convert VIX to IV decimal with bounds."""
    return max(0.08, min(1.0, vix / 100.0))


def years_to_expiry(now: datetime, expiry: datetime) -> float:
    """Time to expiry in years, min 5 minutes."""
    dt = max((expiry - now).total_seconds(), 300)
    return dt / (365.0 * 24.0 * 3600.0)


# -----------------------------
# Scoring - bias toward PROFIT
# -----------------------------
def score_result(result: TrialResult, starting_capital: float = 10000.0) -> float:
    """Score that emphasizes profit while maintaining reasonable risk."""
    if result.total_trades < 20:
        return 0.0

    # Profit is king - 40% weight
    pnl_score = max(0.0, min(1.0, result.total_pnl / 10000.0))

    # Win rate still matters - 25% weight
    wr_score = max(0.0, min(1.0, (result.win_rate - 40.0) / 30.0))

    # Profit factor - 15% weight
    pf_score = max(0.0, min(1.0, result.profit_factor / 2.5))

    # Drawdown control - 20% weight
    dd_frac = result.max_drawdown / starting_capital
    dd_score = max(0.0, 1.0 - min(1.0, dd_frac / 0.25))

    score = (pnl_score * 0.40 + wr_score * 0.25 + pf_score * 0.15 + dd_score * 0.20)

    # Bonus for high profit
    if result.total_pnl > 5000:
        score *= 1.2
    if result.total_pnl > 10000:
        score *= 1.3

    return score


# -----------------------------
# Backtest
# -----------------------------
def run_backtest(params: Dict[str, Any]) -> TrialResult:
    global GLOBAL_BARS

    bars = GLOBAL_BARS
    if not bars:
        return TrialResult(params=params)

    # Constants
    starting_capital = 10000.0
    commission = 0.65
    risk_free_rate = 0.05

    # Delta targeting - HIGH DELTA for ATM exposure
    target_delta = float(params.get("target_delta", 0.70))  # Default 0.70 not 0.30!
    afternoon_delta = float(params.get("afternoon_delta", 0.75))
    afternoon_hour = int(params.get("afternoon_hour", 12))

    max_daily_trades = int(params.get("max_daily_trades", 3))

    # VIX regime (simplified)
    use_vix_regime = params.get("use_vix_regime", False)
    vix_high_threshold = params.get("vix_high_threshold", 25)
    vix_low_threshold = params.get("vix_low_threshold", 15)
    high_vol_cooldown_mult = params.get("high_vol_cooldown_mult", 1.5)
    low_vol_cooldown_mult = params.get("low_vol_cooldown_mult", 0.7)

    # Base signal params
    base_cooldown = params.get("signal_cooldown_bars", 8)

    # Risk management
    enable_trailing_stop = params.get("enable_trailing_stop", True)
    trailing_stop_percent = params.get("trailing_stop_percent", 30)
    trailing_stop_activation = params.get("trailing_stop_activation", 40)

    enable_stop_loss = params.get("enable_stop_loss", False)
    stop_loss_percent = params.get("stop_loss_percent", 50)

    # Position sizing
    kelly_fraction = float(params.get("kelly_fraction", 0.5))
    max_equity_risk = float(params.get("max_equity_risk", 0.15))
    HARD_MAX_CONTRACTS = int(params.get("hard_max_contracts", 50))

    try:
        detector = SignalDetector(
            length_period=20,
            value_area_percent=70.0,
            volume_threshold=params.get("volume_threshold", 1.3),
            use_relaxed_volume=True,
            min_confirmation_bars=params.get("min_confirmation_bars", 2),
            sustained_bars_required=params.get("sustained_bars_required", 3),
            signal_cooldown_bars=base_cooldown,
            use_or_bias_filter=params.get("use_or_bias_filter", True),
            or_buffer_points=1.0,
            rth_only=True,
            use_time_filter=False,
            # Signal enables
            enable_val_bounce=params.get("enable_val_bounce", True),
            enable_poc_reclaim=params.get("enable_poc_reclaim", False),  # Often noisy
            enable_breakout=params.get("enable_breakout", True),
            enable_sustained_breakout=params.get("enable_sustained_breakout", False),
            enable_prior_val_bounce=params.get("enable_val_bounce", True),
            enable_prior_poc_reclaim=False,
            enable_vah_rejection=params.get("enable_vah_rejection", True),
            enable_poc_breakdown=params.get("enable_poc_breakdown", False),
            enable_breakdown=params.get("enable_breakdown", True),
            enable_sustained_breakdown=params.get("enable_sustained_breakdown", False),
            enable_prior_vah_rejection=params.get("enable_vah_rejection", True),
            enable_prior_poc_breakdown=False,
        )

        trades = []
        current_trade = None
        trade_counter = 0
        current_date = None
        daily_trade_count = 0
        equity = starting_capital
        equity_curve = [starting_capital]
        daily_pnl: Dict[Any, float] = {}

        # Kelly rolling window
        rolling_window = 20
        recent_wins: List[float] = []
        recent_losses: List[float] = []

        for bar_data in bars:
            bar = Bar(
                timestamp=bar_data["datetime"],
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
            )

            current_vix = bar_data.get("vix", 18.0)
            bar_date = bar.timestamp.date()

            # New day
            if current_date != bar_date:
                current_date = bar_date
                daily_trade_count = 0
                daily_pnl.setdefault(bar_date, 0.0)

                # VIX regime cooldown adjustment
                if use_vix_regime:
                    if current_vix >= vix_high_threshold:
                        detector.signal_cooldown_bars = int(base_cooldown * high_vol_cooldown_mult)
                    elif current_vix <= vix_low_threshold:
                        detector.signal_cooldown_bars = max(3, int(base_cooldown * low_vol_cooldown_mult))
                    else:
                        detector.signal_cooldown_bars = base_cooldown

            # Update existing trade
            if current_trade:
                current_trade["bars_held"] += 1

                S = bar.close
                K = current_trade["strike"]
                expiry_dt = current_trade["expiry_dt"]
                is_call = current_trade["is_call"]
                sigma = vix_to_iv(current_vix)
                T = years_to_expiry(bar.timestamp, expiry_dt)

                option_entry = current_trade["option_entry"]
                current_price = max(0.01, bs_price(S, K, T, risk_free_rate, sigma, is_call))
                current_pnl_pct = ((current_price - option_entry) / option_entry) * 100.0

                current_trade["high_water_mark"] = max(
                    current_trade.get("high_water_mark", current_pnl_pct),
                    current_pnl_pct,
                )

                exit_reason = None

                # Stop loss
                if enable_stop_loss and current_pnl_pct <= -stop_loss_percent:
                    exit_reason = "Stop Loss"

                # Trailing stop
                if enable_trailing_stop and not exit_reason:
                    hwm = current_trade["high_water_mark"]
                    if hwm >= trailing_stop_activation:
                        if current_pnl_pct <= (hwm - trailing_stop_percent):
                            exit_reason = "Trailing Stop"

                if exit_reason:
                    contracts = current_trade["contracts"]
                    pnl = (current_price - option_entry) * 100.0 * contracts - commission * 2.0 * contracts
                    current_trade["pnl"] = pnl
                    trades.append(current_trade)

                    # Update Kelly stats
                    per_contract = pnl / max(1, contracts)
                    if pnl > 0:
                        recent_wins.append(per_contract)
                        if len(recent_wins) > rolling_window:
                            recent_wins.pop(0)
                    else:
                        recent_losses.append(abs(per_contract))
                        if len(recent_losses) > rolling_window:
                            recent_losses.pop(0)

                    equity += pnl
                    equity_curve.append(equity)
                    daily_pnl[bar_date] += pnl
                    current_trade = None
                    continue

            # Get signal
            signal = detector.add_bar(bar)

            if signal and daily_trade_count < max_daily_trades:
                # Close on opposite signal
                if current_trade and current_trade["direction"] != signal.direction.value:
                    S = bar.close
                    K = current_trade["strike"]
                    expiry_dt = current_trade["expiry_dt"]
                    is_call = current_trade["is_call"]
                    sigma = vix_to_iv(current_vix)
                    T = years_to_expiry(bar.timestamp, expiry_dt)

                    option_entry = current_trade["option_entry"]
                    option_exit = max(0.01, bs_price(S, K, T, risk_free_rate, sigma, is_call))

                    contracts = current_trade["contracts"]
                    pnl = (option_exit - option_entry) * 100.0 * contracts - commission * 2.0 * contracts
                    current_trade["pnl"] = pnl
                    trades.append(current_trade)

                    per_contract = pnl / max(1, contracts)
                    if pnl > 0:
                        recent_wins.append(per_contract)
                        if len(recent_wins) > rolling_window:
                            recent_wins.pop(0)
                    else:
                        recent_losses.append(abs(per_contract))
                        if len(recent_losses) > rolling_window:
                            recent_losses.pop(0)

                    equity += pnl
                    equity_curve.append(equity)
                    daily_pnl[bar_date] += pnl
                    current_trade = None

                # Open new trade
                if not current_trade:
                    trade_counter += 1
                    daily_trade_count += 1

                    # HIGH DELTA targeting
                    delta_target = afternoon_delta if bar.timestamp.hour >= afternoon_hour else target_delta

                    S0 = bar.close
                    sigma0 = vix_to_iv(current_vix)
                    expiry_dt = datetime.combine(bar.timestamp.date(), dt_time(16, 0))
                    T0 = years_to_expiry(bar.timestamp, expiry_dt)

                    is_call = (signal.direction.value == "LONG")
                    target_d = delta_target if is_call else -delta_target

                    K0 = solve_strike_for_delta(S0, target_d, T0, risk_free_rate, sigma0, is_call)
                    option_entry = max(0.01, bs_price(S0, K0, T0, risk_free_rate, sigma0, is_call))

                    # Skip if premium too low (illiquid)
                    if option_entry < 0.30:
                        daily_trade_count -= 1
                        trade_counter -= 1
                        continue

                    option_cost = option_entry * 100.0

                    # Position sizing
                    max_by_equity = int((max_equity_risk * equity) / option_cost)
                    max_by_equity = max(1, min(max_by_equity, HARD_MAX_CONTRACTS))

                    contracts = 1
                    if kelly_fraction > 0 and len(recent_wins) >= 5 and len(recent_losses) >= 5:
                        wr = len(recent_wins) / (len(recent_wins) + len(recent_losses))
                        avg_win = sum(recent_wins) / len(recent_wins)
                        avg_loss = sum(recent_losses) / len(recent_losses)

                        if avg_loss > 0:
                            b = avg_win / avg_loss
                            kelly = wr - ((1.0 - wr) / b)
                            kelly = max(0, min(kelly, 0.25)) * kelly_fraction
                            kelly_contracts = int((kelly * equity) / option_cost)
                            contracts = max(1, min(kelly_contracts, max_by_equity))

                    current_trade = {
                        "id": trade_counter,
                        "signal": signal.signal_type.value,
                        "direction": signal.direction.value,
                        "entry_time": bar.timestamp,
                        "entry_price": S0,
                        "strike": K0,
                        "expiry_dt": expiry_dt,
                        "is_call": is_call,
                        "option_entry": option_entry,
                        "contracts": contracts,
                        "bars_held": 0,
                        "pnl": 0.0,
                        "high_water_mark": 0.0,
                    }

        # Close remaining trade
        if current_trade and bars:
            last = bars[-1]
            S = last["close"]
            K = current_trade["strike"]
            expiry_dt = current_trade["expiry_dt"]
            is_call = current_trade["is_call"]
            sigma = vix_to_iv(last.get("vix", 18.0))
            T = years_to_expiry(last["datetime"], expiry_dt)

            option_exit = max(0.01, bs_price(S, K, T, risk_free_rate, sigma, is_call))
            contracts = current_trade["contracts"]
            pnl = (option_exit - current_trade["option_entry"]) * 100.0 * contracts - commission * 2.0 * contracts
            current_trade["pnl"] = pnl
            trades.append(current_trade)
            equity += pnl
            equity_curve.append(equity)

        # Calculate metrics
        result = TrialResult(params=params)
        if not trades:
            return result

        result.total_trades = len(trades)
        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]

        result.win_rate = (len(winners) / len(trades)) * 100.0
        result.total_pnl = sum(t["pnl"] for t in trades)

        gross_profit = sum(t["pnl"] for t in winners) if winners else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0.0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown
        peak = starting_capital
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Sharpe
        if daily_pnl:
            returns = list(daily_pnl.values())
            if len(returns) > 1:
                import statistics

                mean_r = statistics.mean(returns)
                std_r = statistics.stdev(returns)
                result.sharpe_ratio = (mean_r / std_r) * (252**0.5) if std_r > 0 else 0.0

        result.score = score_result(result, starting_capital)
        return result

    except Exception:
        return TrialResult(params=params)


# -----------------------------
# Objective - SIMPLIFIED
# -----------------------------
def create_objective(min_trades: int = 30):
    def objective(trial: optuna.Trial) -> float:
        global BEST_RESULT

        params = {
            # Core signal params
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 5, 20),
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 4),
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 5),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.8),
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            # Signal enables - simplified set
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
            # POC and sustained signals default OFF (often noisy)
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            # VIX regime - optional
            "use_vix_regime": trial.suggest_categorical("use_vix_regime", [True, False]),
            "vix_high_threshold": 25,
            "vix_low_threshold": 15,
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.0),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.5, 0.9),
            # HIGH DELTA targeting - the key change
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            # Risk management
            "max_daily_trades": trial.suggest_int("max_daily_trades", 2, 6),
            "enable_trailing_stop": True,  # Always on
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            # Position sizing
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.3, 1.5),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.10, 0.25),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 20, 100),
        }

        result = run_backtest(params)

        if result.total_trades < min_trades:
            raise optuna.TrialPruned()

        if BEST_RESULT is None or result.score > BEST_RESULT.score:
            BEST_RESULT = result

        return -result.score

    return objective


# -----------------------------
# Data fetch (cache-aware)
# -----------------------------
def _load_cache(path: Path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_cache(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(path)


def _dedupe_bars(bars: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for b in sorted(bars, key=lambda x: x["datetime"]):
        k = b["datetime"].isoformat()
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out


def fetch_data(days: int = 90, start_date: str = None, end_date: str = None) -> List[Dict]:
    """
    Fetch SPY + VIX data with HARD 10-day forward chunking.
    Fixes Schwab silent truncation on intraday history.
    """
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    from config import config

    # Resolve requested range
    if start_date and end_date:
        req_start = datetime.strptime(start_date, "%Y-%m-%d")
        req_end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        req_end = datetime.now()
        req_start = req_end - timedelta(days=int(days * 1.5))

    # Load caches
    spy_cached: List[Dict] = _load_cache(SPY_CACHE_FILE) or []
    vix_cached: Dict[str, float] = _load_cache(VIX_CACHE_FILE) or {}

    spy_cached = _dedupe_bars(spy_cached)

    cached_start = spy_cached[0]["datetime"] if spy_cached else None
    cached_end = spy_cached[-1]["datetime"] if spy_cached else None

    # Determine missing range
    fetch_start = req_start if not cached_start else min(req_start, cached_start)
    fetch_end = req_end if not cached_end else max(req_end, cached_end)

    # If cache already covers request, return slice
    if cached_start and cached_end and req_start >= cached_start and req_end <= cached_end:
        for b in spy_cached:
            b["vix"] = vix_cached.get(b["datetime"].strftime("%Y-%m-%d"), 18.0)
        return [b for b in spy_cached if req_start <= b["datetime"] <= req_end]

    # Auth once
    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri,
        token_file=config.schwab.token_file,
    )

    if not auth.is_authenticated:
        if not auth.authorize_interactive():
            raise RuntimeError("Auth failed")
    else:
        auth.refresh_access_token()

    client = SchwabClient(auth)

    # ---- FIX: STRICT forward chunking (≤10 days) ----
    all_spy = spy_cached.copy()
    CHUNK_DAYS = 9  # stay under Schwab hard limit

    cursor = fetch_start
    while cursor < fetch_end:
        chunk_start = cursor
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), fetch_end)

        try:
            chunk = client.get_price_history(
                symbol="SPY",
                period_type="day",
                period=10,
                frequency_type="minute",
                frequency=5,
                extended_hours=False,
                start_date=chunk_start,
                end_date=chunk_end,
            )
            if chunk:
                all_spy.extend(chunk)
        except Exception:
            pass

        cursor = chunk_end + timedelta(minutes=5)
        time_module.sleep(0.15)

    all_spy = _dedupe_bars(all_spy)
    _save_cache(SPY_CACHE_FILE, all_spy)

    # ---- VIX (same forward chunking) ----
    cursor = fetch_start
    while cursor < fetch_end:
        chunk_start = cursor
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), fetch_end)

        try:
            vix = client.get_price_history(
                symbol="$VIX",
                period_type="day",
                period=10,
                frequency_type="minute",
                frequency=5,
                extended_hours=False,
                start_date=chunk_start,
                end_date=chunk_end,
            )
            if vix:
                for v in vix:
                    vix_cached[v["datetime"].strftime("%Y-%m-%d")] = v["close"]
        except Exception:
            pass

        cursor = chunk_end + timedelta(minutes=5)
        time_module.sleep(0.1)

    _save_cache(VIX_CACHE_FILE, vix_cached)

    # Attach VIX + slice
    for b in all_spy:
        b["vix"] = vix_cached.get(b["datetime"].strftime("%Y-%m-%d"), 18.0)

    return [b for b in all_spy if req_start <= b["datetime"] <= req_end]



# -----------------------------
# Progress
# -----------------------------
class Progress:
    def __init__(self, n: int):
        self.n = n
        self.start = time_module.time()
        self.best_wr = 0.0
        self.best_pnl = 0.0

    def __call__(self, study, trial):
        global BEST_RESULT

        done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        pct = done / self.n if self.n else 0

        elapsed = time_module.time() - self.start
        eta = (elapsed / pct - elapsed) if pct > 0 else 0

        bar = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))

        if BEST_RESULT:
            self.best_wr = max(self.best_wr, BEST_RESULT.win_rate)
            self.best_pnl = max(self.best_pnl, BEST_RESULT.total_pnl)

        print(
            f"\r[{bar}] {pct*100:5.1f}% | ETA {eta:5.0f}s | Best: {self.best_wr:.1f}% WR, ${self.best_pnl:,.0f}",
            end="",
            flush=True,
        )


# -----------------------------
# Main
# -----------------------------
def main():
    global GLOBAL_BARS, BEST_RESULT

    parser = argparse.ArgumentParser(description="Simplified Optimizer - High Delta Focus")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="best_simple.json")

    args = parser.parse_args()

    n_jobs = int(cpu_count() * 1.5) if args.turbo else max(1, cpu_count() - 1)

    print("=" * 60)
    print("  SIMPLIFIED OPTIMIZER - HIGH DELTA ATM FOCUS")
    print("=" * 60)

    GLOBAL_BARS = fetch_data(args.days, args.start_date, args.end_date)

    if not GLOBAL_BARS:
        print("ERROR: No data")
        sys.exit(1)

    print(f"\n  Optimizing: {args.trials} trials, {n_jobs} workers\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed, multivariate=True),
    )

    study.optimize(
        create_objective(args.min_trades),
        n_trials=args.trials,
        n_jobs=n_jobs,
        callbacks=[Progress(args.trials)],
        show_progress_bar=False,
    )

    print()

    if BEST_RESULT:
        print("\n" + "=" * 60)
        print("  BEST RESULT")
        print("=" * 60)
        print(f"  Win Rate:      {BEST_RESULT.win_rate:.1f}%")
        print(f"  Total P&L:     ${BEST_RESULT.total_pnl:,.2f}")
        print(f"  Profit Factor: {BEST_RESULT.profit_factor:.2f}")
        print(f"  Max Drawdown:  ${BEST_RESULT.max_drawdown:,.2f}")
        print(f"  Trades:        {BEST_RESULT.total_trades}")
        print(f"  Sharpe:        {BEST_RESULT.sharpe_ratio:.2f}")

        p = BEST_RESULT.params
        print("\n  Key Params:")
        print(f"    target_delta: {p.get('target_delta', 0.70):.2f}")
        print(f"    afternoon_delta: {p.get('afternoon_delta', 0.75):.2f}")
        print(f"    signal_cooldown_bars: {p.get('signal_cooldown_bars', 8)}")
        print(f"    trailing_stop_percent: {p.get('trailing_stop_percent', 30)}")
        print(f"    enable_val_bounce: {p.get('enable_val_bounce')}")
        print(f"    enable_vah_rejection: {p.get('enable_vah_rejection')}")
        print(f"    enable_breakout: {p.get('enable_breakout')}")
        print(f"    use_vix_regime: {p.get('use_vix_regime')}")

        # Save
        with open(args.save, "w") as f:
            json.dump(
                {
                    "params": BEST_RESULT.params,
                    "metrics": {
                        "win_rate": BEST_RESULT.win_rate,
                        "total_pnl": BEST_RESULT.total_pnl,
                        "profit_factor": BEST_RESULT.profit_factor,
                        "max_drawdown": BEST_RESULT.max_drawdown,
                        "total_trades": BEST_RESULT.total_trades,
                        "sharpe_ratio": BEST_RESULT.sharpe_ratio,
                    },
                },
                f,
                indent=2,
            )
        print(f"\n  Saved to {args.save}")

    print("=" * 60)


if __name__ == "__main__":
    main()