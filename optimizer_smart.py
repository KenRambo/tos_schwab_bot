"""
Smart Strategy Optimizer using Optuna (Bayesian Optimization)
==============================================================

Console output is intentionally minimal:
- Suppresses warnings
- Suppresses Optuna chatter
- Shows only a single live progress line with:
  - progress bar, ETA
  - current best WR and P&L

Kelly sizing:
- Scales with equity via a max % equity-at-risk cap per trade (max_equity_risk)
- Includes guardrails to prevent runaway sizing / "hangs"
- No "max_contracts" optimization; uses equity-based cap + hard cap guardrail

Black–Scholes option pricing:
- Replaces constant-delta option approximation with BS pricing
- Uses VIX (bar_data["vix"]) as an IV proxy
- LONG = call, SHORT = put
- Default expiry is same-day 16:00 (0DTE)

Requirements:
    pip install optuna

Usage:
    python optimizer_smart.py                              # Full optimization (500 trials)
    python optimizer_smart.py --trials 1000 --turbo        # More trials, more workers
    python optimizer_smart.py --phase signal --trials 500  # Signal params only
    python optimizer_smart.py --phase risk --lock best_params.json  # Risk params only (uses locked signals)
    python optimizer_smart.py --days 120 --trials 2000     # Longer history, deep search
"""
import os
import sys
import logging
import argparse
import json
import warnings
import math
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from multiprocessing import cpu_count
import time as time_module

# -----------------------------
# Console + logging suppression
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import FrozenTrial
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.ERROR)

from signal_detector import SignalDetector, Bar  # noqa: E402


# -----------------------------
# Globals
# -----------------------------
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
# Param (de)serialization
# -----------------------------
def _load_locked_params(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_best_params(path: str, params: Dict[str, Any], result: "TrialResult") -> None:
    if not path:
        return
    try:
        output = {
            "params": params,
            "metrics": {
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_factor,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "score": result.score,
            },
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)
    except Exception:
        pass


# -----------------------------
# Math: Normal CDF / PDF
# -----------------------------
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    # Using erf for numerical stability
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -----------------------------
# Black–Scholes pricing
# -----------------------------
def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Black–Scholes price for European option.
    S: spot
    K: strike
    T: time to expiry in years
    r: continuously-compounded risk-free rate
    sigma: annualized volatility (decimal)
    """
    if S <= 0 or K <= 0:
        return 0.0

    # Guardrails: if T or sigma are too small, approximate with intrinsic
    if T <= 0.0 or sigma <= 1e-8:
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return intrinsic

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black–Scholes delta."""
    if S <= 0 or K <= 0 or T <= 0.0 or sigma <= 1e-8:
        # Intrinsic-ish fallback
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def solve_strike_for_delta(
    S: float,
    target_delta_signed: float,
    T: float,
    r: float,
    sigma: float,
    strike_low: float,
    strike_high: float,
    is_call: bool,
    iters: int = 40,
) -> float:
    """
    Find strike K such that BS delta ~ target_delta_signed.
    For calls, delta in (0,1). For puts, delta in (-1,0).
    """
    lo = max(0.01, strike_low)
    hi = max(lo + 0.01, strike_high)

    # Ensure bracket: delta is monotonic decreasing in K for both calls and puts
    # If bracket doesn't contain target, we still return midpoint; guardrails will handle.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = bs_delta(S, mid, T, r, sigma, is_call)
        if d > target_delta_signed:
            # delta too high => strike too low => raise strike
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _compute_iv_from_vix(vix: float, iv_mult: float, iv_floor: float, iv_cap: float) -> float:
    # VIX is approximately 30-day implied vol (annualized %) for SPX; used here as a proxy for SPY.
    iv = (max(0.0, float(vix)) / 100.0) * float(iv_mult)
    return max(float(iv_floor), min(float(iv_cap), iv))


def _years_to_expiry(now: datetime, expiry: datetime, min_minutes: int = 5) -> float:
    dt = (expiry - now).total_seconds()
    # Guardrail: never let T go to 0 during the day (avoids singularities)
    min_sec = float(min_minutes) * 60.0
    dt = max(dt, min_sec)
    return dt / (365.0 * 24.0 * 3600.0)


def _same_day_expiry_dt(ts: datetime) -> datetime:
    # 16:00 local timestamp date; assumes timestamps align to market hours in data.
    return datetime.combine(ts.date(), dt_time(16, 0))


# -----------------------------
# Scoring
# -----------------------------
def _risk_aware_score(result: TrialResult, starting_capital: float = 10000.0) -> float:
    """Higher is better. Weighs drawdown heavily."""
    if result.total_trades <= 0:
        return 0.0

    wr_score = max(0.0, min(1.0, (result.win_rate - 45.0) / 25.0))
    pnl_score = max(0.0, min(1.0, result.total_pnl / 5000.0))
    pf_score = max(0.0, min(1.0, result.profit_factor / 2.0))
    sharpe_score = max(0.0, min(1.0, result.sharpe_ratio / 2.0))

    dd_frac = (result.max_drawdown / starting_capital) if result.max_drawdown > 0 else 0.0
    dd_score = max(0.0, 1.0 - min(1.0, dd_frac / 0.30))  # 0 at 30%+ DD

    score = (
        wr_score * 0.20
        + pnl_score * 0.15
        + pf_score * 0.15
        + sharpe_score * 0.20
        + dd_score * 0.30
    )

    if result.total_trades < 40:
        score *= 0.85

    if result.win_rate >= 60.0:
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

    # Delta targets (used to choose strike via BS delta solving)
    target_delta = float(params.get("target_delta", 0.30))
    afternoon_delta = float(params.get("afternoon_delta", 0.40))
    afternoon_hour = int(params.get("afternoon_hour", 12))

    commission = 0.65
    max_daily_trades = int(params.get("max_daily_trades", 3))

    # Time window filters
    signal_start_minutes = int(params.get("signal_start_minutes", 0))
    signal_end_minutes = int(params.get("signal_end_minutes", 0))

    # ATR-based stops
    use_atr_stops = params.get("use_atr_stops", False)
    atr_stop_mult = float(params.get("atr_stop_mult", 2.0))
    atr_target_mult = float(params.get("atr_target_mult", 3.0))

    # Min premium filter
    min_option_premium = float(params.get("min_option_premium", 0.25))

    # VWAP filter
    use_vwap_filter = params.get("use_vwap_filter", False)
    vwap_filter_mode = params.get("vwap_filter_mode", "strict")

    # NYSE TICK filter
    use_tick_filter = params.get("use_tick_filter", False)
    tick_extreme_threshold = int(params.get("tick_extreme_threshold", 500))

    # VIX regime
    use_vix_regime = params.get("use_vix_regime", False)
    vix_high_threshold = params.get("vix_high_threshold", 25)
    vix_low_threshold = params.get("vix_low_threshold", 15)

    high_vol_cooldown_mult = params.get("high_vol_cooldown_mult", 1.5)
    high_vol_confirmation_mult = params.get("high_vol_confirmation_mult", 1.5)
    high_vol_sustained_mult = params.get("high_vol_sustained_mult", 1.5)
    high_vol_volume_add = params.get("high_vol_volume_add", 0.2)
    high_vol_delta_adj = params.get("high_vol_delta_adj", 0.05)
    high_vol_min_hold_mult = params.get("high_vol_min_hold_mult", 1.5)

    low_vol_cooldown_mult = params.get("low_vol_cooldown_mult", 0.7)
    low_vol_confirmation_mult = params.get("low_vol_confirmation_mult", 0.8)
    low_vol_sustained_mult = params.get("low_vol_sustained_mult", 0.8)
    low_vol_volume_add = params.get("low_vol_volume_add", -0.1)
    low_vol_delta_adj = params.get("low_vol_delta_adj", -0.05)

    # Base detector params
    base_cooldown = params.get("signal_cooldown_bars", 8)
    base_confirmation = params.get("min_confirmation_bars", 2)
    base_sustained = params.get("sustained_bars_required", 3)
    base_volume_threshold = params.get("volume_threshold", 1.3)
    base_min_hold = params.get("min_hold_bars", 0)

    # Guardrails for sizing
    HARD_MAX_CONTRACTS = int(params.get("hard_max_contracts", 100))
    MAX_KELLY_PCT_CAP = float(params.get("max_kelly_pct_cap", 0.35))
    MIN_OPTION_COST = float(params.get("min_option_cost", 25.0))

    # Black–Scholes / IV parameters (not optimized by default; can be locked or added to search if desired)
    risk_free_rate = float(params.get("risk_free_rate", 0.05))
    iv_mult = float(params.get("iv_mult", 1.0))
    iv_floor = float(params.get("iv_floor", 0.05))
    iv_cap = float(params.get("iv_cap", 1.50))
    min_T_minutes = int(params.get("min_T_minutes", 5))

    try:
        detector = SignalDetector(
            length_period=20,
            value_area_percent=70.0,
            volume_threshold=base_volume_threshold,
            use_relaxed_volume=True,
            min_confirmation_bars=base_confirmation,
            sustained_bars_required=base_sustained,
            signal_cooldown_bars=base_cooldown,
            use_or_bias_filter=params.get("use_or_bias_filter", True),
            or_buffer_points=1.0,
            rth_only=True,
            use_time_filter=params.get("use_time_filter", False),
            # Enables
            enable_val_bounce=params.get("enable_val_bounce", True),
            enable_poc_reclaim=params.get("enable_poc_reclaim", True),
            enable_breakout=params.get("enable_breakout", True),
            enable_sustained_breakout=params.get("enable_sustained_breakout", True),
            enable_prior_val_bounce=params.get("enable_val_bounce", True),
            enable_prior_poc_reclaim=True,
            enable_vah_rejection=params.get("enable_vah_rejection", True),
            enable_poc_breakdown=params.get("enable_poc_breakdown", True),
            enable_breakdown=params.get("enable_breakdown", True),
            enable_sustained_breakdown=params.get("enable_sustained_breakdown", True),
            enable_prior_vah_rejection=params.get("enable_vah_rejection", True),
            enable_prior_poc_breakdown=True,
        )

        # Risk management
        enable_stop_loss = params.get("enable_stop_loss", False)
        stop_loss_percent = params.get("stop_loss_percent", 50)
        enable_take_profit = params.get("enable_take_profit", False)
        take_profit_percent = params.get("take_profit_percent", 100)
        enable_trailing_stop = params.get("enable_trailing_stop", False)
        trailing_stop_percent = params.get("trailing_stop_percent", 25)
        trailing_stop_activation = params.get("trailing_stop_activation", 50)
        min_hold_bars = params.get("min_hold_bars", 0)

        # Kelly sizing
        kelly_fraction = float(params.get("kelly_fraction", 0.0))
        max_equity_risk = float(params.get("max_equity_risk", 0.10))
        base_contracts = 1

        trades = []
        current_trade = None
        trade_counter = 0
        current_date = None
        daily_trade_count = 0
        equity = starting_capital
        equity_curve = [starting_capital]
        daily_pnl: Dict[Any, float] = {}

        # Kelly rolling
        rolling_window = int(params.get("kelly_lookback", 20))
        recent_wins: List[float] = []
        recent_losses: List[float] = []

        # Regime
        current_vix_regime = "NORMAL"

        # ATR
        atr_period = 14
        recent_true_ranges: List[float] = []
        current_atr = 0.0
        prev_close = None

        # VWAP (daily reset)
        vwap_cum_pv = 0.0
        vwap_cum_vol = 0.0
        current_vwap = 0.0

        # VIX-adjusted values (init)
        effective_min_hold = base_min_hold
        effective_delta_adj = 0.0

        for bar_data in bars:
            bar = Bar(
                timestamp=bar_data["datetime"],
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
            )

            # ATR update
            if prev_close is not None:
                true_range = max(
                    bar.high - bar.low,
                    abs(bar.high - prev_close),
                    abs(bar.low - prev_close),
                )
                recent_true_ranges.append(true_range)
                if len(recent_true_ranges) > atr_period:
                    recent_true_ranges.pop(0)
                if len(recent_true_ranges) >= atr_period:
                    current_atr = sum(recent_true_ranges) / len(recent_true_ranges)
            prev_close = bar.close

            # VIX/TICK
            current_vix = bar_data.get("vix", 18.0)
            current_tick = bar_data.get("tick", 0.0)

            bar_date = bar.timestamp.date()

            # New day reset / regime update
            if current_date != bar_date:
                current_date = bar_date
                daily_trade_count = 0
                daily_pnl.setdefault(bar_date, 0.0)

                # Reset VWAP
                vwap_cum_pv = 0.0
                vwap_cum_vol = 0.0
                current_vwap = 0.0

                if use_vix_regime:
                    if current_vix >= vix_high_threshold:
                        current_vix_regime = "HIGH"
                        effective_cooldown = int(base_cooldown * high_vol_cooldown_mult)
                        effective_confirmation = int(base_confirmation * high_vol_confirmation_mult)
                        effective_sustained = int(base_sustained * high_vol_sustained_mult)
                        effective_volume = base_volume_threshold + high_vol_volume_add
                        effective_min_hold = int(base_min_hold * high_vol_min_hold_mult)
                        effective_delta_adj = float(high_vol_delta_adj)
                    elif current_vix <= vix_low_threshold:
                        current_vix_regime = "LOW"
                        effective_cooldown = int(base_cooldown * low_vol_cooldown_mult)
                        effective_confirmation = max(1, int(base_confirmation * low_vol_confirmation_mult))
                        effective_sustained = max(1, int(base_sustained * low_vol_sustained_mult))
                        effective_volume = max(1.0, base_volume_threshold + low_vol_volume_add)
                        effective_min_hold = base_min_hold
                        effective_delta_adj = float(low_vol_delta_adj)
                    else:
                        current_vix_regime = "NORMAL"
                        effective_cooldown = base_cooldown
                        effective_confirmation = base_confirmation
                        effective_sustained = base_sustained
                        effective_volume = base_volume_threshold
                        effective_min_hold = base_min_hold
                        effective_delta_adj = 0.0

                    detector.signal_cooldown_bars = max(3, effective_cooldown)
                    detector.min_confirmation_bars = max(1, effective_confirmation)
                    detector.sustained_bars_required = max(1, effective_sustained)
                    detector.volume_threshold = effective_volume
                else:
                    effective_min_hold = base_min_hold
                    effective_delta_adj = 0.0

            # -----------------------------
            # Mark-to-market open trade using BS
            # -----------------------------
            if current_trade:
                current_trade["bars_held"] += 1

                S = float(bar.close)
                expiry_dt = current_trade["expiry_dt"]
                K = float(current_trade["strike"])
                is_call = bool(current_trade["is_call"])

                # Current IV from VIX proxy
                sigma = _compute_iv_from_vix(current_vix, iv_mult, iv_floor, iv_cap)

                # Time to expiry
                T = _years_to_expiry(bar.timestamp, expiry_dt, min_minutes=min_T_minutes)

                option_entry = float(current_trade["option_entry"])
                current_option_price = bs_price(S, K, T, risk_free_rate, sigma, is_call)
                current_option_price = max(0.01, current_option_price)

                current_pnl_pct = ((current_option_price - option_entry) / option_entry) * 100.0

                current_trade["high_water_mark"] = max(
                    current_trade.get("high_water_mark", current_pnl_pct),
                    current_pnl_pct,
                )

                # VIX-adjusted min-hold
                current_min_hold = effective_min_hold if use_vix_regime else base_min_hold

                # Compute current delta for ATR math (optional but improves realism)
                current_delta = bs_delta(S, K, T, risk_free_rate, sigma, is_call)
                current_trade["current_delta"] = current_delta

                exit_reason = None
                if current_trade["bars_held"] >= current_min_hold:
                    # ATR stops: translate ATR move into option PnL% using delta approximation *around current state*
                    if use_atr_stops and current_atr > 0.0:
                        entry_atr = float(current_trade.get("entry_atr", current_atr))
                        # delta magnitude for PnL% estimate (avoid sign issues)
                        dmag = max(0.05, min(0.95, abs(current_delta)))
                        # approximate option move = ATR * delta
                        atr_stop_level = -(entry_atr * atr_stop_mult * dmag * 100.0) / max(0.01, option_entry)
                        atr_target_level = (entry_atr * atr_target_mult * dmag * 100.0) / max(0.01, option_entry)

                        if current_pnl_pct <= atr_stop_level:
                            exit_reason = "ATR Stop"
                        elif current_pnl_pct >= atr_target_level:
                            exit_reason = "ATR Target"

                    if not exit_reason:
                        if enable_stop_loss and current_pnl_pct <= -stop_loss_percent:
                            exit_reason = "Stop Loss"
                        elif enable_take_profit and current_pnl_pct >= take_profit_percent:
                            exit_reason = "Take Profit"
                        elif enable_trailing_stop:
                            hwm = current_trade["high_water_mark"]
                            if hwm >= trailing_stop_activation and current_pnl_pct <= (hwm - trailing_stop_percent):
                                exit_reason = "Trailing Stop"

                if exit_reason:
                    contracts = int(current_trade.get("contracts", base_contracts))
                    pnl = (current_option_price - option_entry) * 100.0 * contracts - commission * 2.0 * contracts
                    current_trade["pnl"] = pnl
                    current_trade["exit_reason"] = exit_reason
                    trades.append(current_trade)

                    per_contract_pnl = pnl / max(1, contracts)
                    if pnl > 0:
                        recent_wins.append(per_contract_pnl)
                        if len(recent_wins) > rolling_window:
                            recent_wins.pop(0)
                    else:
                        recent_losses.append(abs(per_contract_pnl))
                        if len(recent_losses) > rolling_window:
                            recent_losses.pop(0)

                    equity += pnl
                    equity_curve.append(equity)
                    daily_pnl[bar_date] += pnl
                    current_trade = None
                    continue

            # -----------------------------
            # Signal processing + filters
            # -----------------------------
            signal = detector.add_bar(bar)

            # VWAP update
            typical_price = (bar.high + bar.low + bar.close) / 3.0
            vwap_cum_pv += typical_price * bar.volume
            vwap_cum_vol += bar.volume
            if vwap_cum_vol > 0:
                current_vwap = vwap_cum_pv / vwap_cum_vol

            # Time window
            bar_hour = bar.timestamp.hour
            bar_minute = bar.timestamp.minute
            minutes_since_open = (bar_hour - 9) * 60 + (bar_minute - 30)
            minutes_until_close = (16 - bar_hour) * 60 - bar_minute
            in_time_window = (minutes_since_open >= signal_start_minutes) and (minutes_until_close >= signal_end_minutes)

            # VWAP filter
            vwap_allows_long = True
            vwap_allows_short = True
            if use_vwap_filter and current_vwap > 0:
                vwap_allows_long = bar.close >= current_vwap
                vwap_allows_short = bar.close <= current_vwap

            # TICK filter
            tick_allows_long = True
            tick_allows_short = True
            if use_tick_filter:
                tick_allows_short = current_tick < tick_extreme_threshold
                tick_allows_long = current_tick > -tick_extreme_threshold

            if signal and daily_trade_count < max_daily_trades and in_time_window:
                sd = signal.direction.value
                if sd == "LONG" and (not vwap_allows_long or not tick_allows_long):
                    signal = None
                elif sd == "SHORT" and (not vwap_allows_short or not tick_allows_short):
                    signal = None

            if signal and daily_trade_count < max_daily_trades and in_time_window:
                # Close existing on opposite signal (BS repricing)
                if current_trade and current_trade["direction"] != signal.direction.value:
                    if current_trade["bars_held"] >= min_hold_bars:
                        S = float(bar.close)
                        expiry_dt = current_trade["expiry_dt"]
                        K = float(current_trade["strike"])
                        is_call = bool(current_trade["is_call"])
                        sigma = _compute_iv_from_vix(current_vix, iv_mult, iv_floor, iv_cap)
                        T = _years_to_expiry(bar.timestamp, expiry_dt, min_minutes=min_T_minutes)
                        option_entry = float(current_trade["option_entry"])
                        option_exit = max(0.01, bs_price(S, K, T, risk_free_rate, sigma, is_call))

                        contracts = int(current_trade.get("contracts", base_contracts))
                        pnl = (option_exit - option_entry) * 100.0 * contracts - commission * 2.0 * contracts
                        current_trade["pnl"] = pnl
                        current_trade["exit_reason"] = "Opposite Signal"
                        trades.append(current_trade)

                        per_contract_pnl = pnl / max(1, contracts)
                        if pnl > 0:
                            recent_wins.append(per_contract_pnl)
                            if len(recent_wins) > rolling_window:
                                recent_wins.pop(0)
                        else:
                            recent_losses.append(abs(per_contract_pnl))
                            if len(recent_losses) > rolling_window:
                                recent_losses.pop(0)

                        equity += pnl
                        equity_curve.append(equity)
                        daily_pnl[bar_date] += pnl
                        current_trade = None

                # Open new trade (BS-based)
                if not current_trade:
                    trade_counter += 1
                    daily_trade_count += 1

                    # Select target delta by time of day
                    delta_abs = afternoon_delta if bar.timestamp.hour >= afternoon_hour else target_delta
                    # VIX regime delta adjustment
                    if use_vix_regime:
                        delta_abs = max(0.15, min(0.50, delta_abs + effective_delta_adj))

                    S0 = float(bar.close)
                    expiry_dt = _same_day_expiry_dt(bar.timestamp)

                    # IV proxy
                    sigma0 = _compute_iv_from_vix(current_vix, iv_mult, iv_floor, iv_cap)
                    T0 = _years_to_expiry(bar.timestamp, expiry_dt, min_minutes=min_T_minutes)

                    is_call = (signal.direction.value == "LONG")
                    # Signed target delta for strike solver
                    target_delta_signed = float(delta_abs) if is_call else -float(delta_abs)

                    # Strike solve bracket
                    K_low = S0 * 0.5
                    K_high = S0 * 1.5
                    K0 = solve_strike_for_delta(
                        S=S0,
                        target_delta_signed=target_delta_signed,
                        T=T0,
                        r=risk_free_rate,
                        sigma=sigma0,
                        strike_low=K_low,
                        strike_high=K_high,
                        is_call=is_call,
                        iters=40,
                    )

                    option_entry = bs_price(S0, K0, T0, risk_free_rate, sigma0, is_call)
                    option_entry = max(0.01, option_entry)

                    # Min premium filter (avoid ultra-cheap / illiquid)
                    if option_entry < min_option_premium:
                        daily_trade_count -= 1
                        trade_counter -= 1
                        continue

                    option_cost = max(option_entry * 100.0, MIN_OPTION_COST)

                    # Equity cap
                    max_contracts_by_equity = int((max_equity_risk * max(0.0, equity)) / option_cost)
                    max_contracts_by_equity = max(base_contracts, min(max_contracts_by_equity, HARD_MAX_CONTRACTS))

                    # Kelly sizing
                    contracts = base_contracts
                    if kelly_fraction > 0.0 and len(recent_wins) >= 5 and len(recent_losses) >= 5:
                        total_recent = len(recent_wins) + len(recent_losses)
                        wr = len(recent_wins) / total_recent if total_recent > 0 else 0.0
                        avg_win = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
                        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 1.0

                        if avg_loss > 0.0 and avg_win > 0.0:
                            b = avg_win / avg_loss
                            kelly_raw = wr - ((1.0 - wr) / b)
                            kelly_raw = max(0.0, min(kelly_raw, MAX_KELLY_PCT_CAP))

                            kelly_pct = kelly_raw * kelly_fraction
                            kelly_pct = max(0.0, min(kelly_pct, max_equity_risk))

                            kelly_contracts = int((kelly_pct * max(0.0, equity)) / option_cost)
                            contracts = max(base_contracts, min(kelly_contracts, max_contracts_by_equity))

                    entry_delta = bs_delta(S0, K0, T0, risk_free_rate, sigma0, is_call)

                    current_trade = {
                        "id": trade_counter,
                        "signal": signal.signal_type.value,
                        "direction": signal.direction.value,
                        "entry_time": bar.timestamp,
                        "entry_price": S0,
                        "expiry_dt": expiry_dt,
                        "strike": float(K0),
                        "is_call": bool(is_call),
                        "iv_entry": float(sigma0),
                        "option_entry": float(option_entry),
                        "delta_target_abs": float(delta_abs),
                        "delta_entry": float(entry_delta),
                        "contracts": int(contracts),
                        "bars_held": 0,
                        "pnl": 0.0,
                        "high_water_mark": 0.0,
                        "entry_atr": float(current_atr),
                    }

        # Close remaining trade at end (BS repricing)
        if current_trade and bars:
            last_bar = bars[-1]
            now = last_bar["datetime"]
            S = float(last_bar["close"])
            expiry_dt = current_trade["expiry_dt"]
            K = float(current_trade["strike"])
            is_call = bool(current_trade["is_call"])
            sigma = _compute_iv_from_vix(last_bar.get("vix", 18.0), iv_mult, iv_floor, iv_cap)
            T = _years_to_expiry(now, expiry_dt, min_minutes=min_T_minutes)

            option_entry = float(current_trade["option_entry"])
            option_exit = max(0.01, bs_price(S, K, T, risk_free_rate, sigma, is_call))

            contracts = int(current_trade.get("contracts", base_contracts))
            pnl = (option_exit - option_entry) * 100.0 * contracts - commission * 2.0 * contracts
            current_trade["pnl"] = pnl
            current_trade["exit_reason"] = "End of Data"
            trades.append(current_trade)
            equity += pnl
            equity_curve.append(equity)

        # Metrics
        result = TrialResult(params=params)
        if not trades:
            return result

        result.total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        result.win_rate = (len(winning_trades) / len(trades)) * 100.0
        result.total_pnl = float(sum(t["pnl"] for t in trades))

        gross_profit = float(sum(t["pnl"] for t in winning_trades)) if winning_trades else 0.0
        gross_loss = float(abs(sum(t["pnl"] for t in losing_trades))) if losing_trades else 0.0
        result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # Max drawdown
        peak = starting_capital
        max_dd = 0.0
        for eq in equity_curve:
            eq = float(eq)
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Sharpe (daily)
        if daily_pnl:
            daily_returns = list(daily_pnl.values())
            if len(daily_returns) > 1:
                import statistics

                mean_return = statistics.mean(daily_returns)
                std_return = statistics.stdev(daily_returns)
                result.sharpe_ratio = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0

        result.score = _risk_aware_score(result, starting_capital)
        return result

    except Exception:
        return TrialResult(params=params)


# -----------------------------
# Objective
# -----------------------------
def create_objective(min_trades: int = 30, phase: str = "all", locked_params: Optional[Dict[str, Any]] = None):
    locked = locked_params or {}

    def objective(trial: optuna.Trial) -> float:
        global BEST_RESULT

        params: Dict[str, Any] = {}
        if locked:
            params.update(locked)

        # Signal/filter params
        if phase in ("all", "signal"):
            params.update(
                {
                    "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 5, 25),
                    "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 5),
                    "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 6),
                    "volume_threshold": trial.suggest_float("volume_threshold", 1.0, 2.0),
                    "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
                    "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
                    "enable_poc_reclaim": trial.suggest_categorical("enable_poc_reclaim", [True, False]),
                    "enable_poc_breakdown": trial.suggest_categorical("enable_poc_breakdown", [True, False]),
                    "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
                    "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
                    "enable_sustained_breakout": trial.suggest_categorical("enable_sustained_breakout", [True, False]),
                    "enable_sustained_breakdown": trial.suggest_categorical("enable_sustained_breakdown", [True, False]),
                    "use_time_filter": trial.suggest_categorical("use_time_filter", [True, False]),
                    "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
                }
            )

            # VWAP Filter
            use_vwap_filter = trial.suggest_categorical("use_vwap_filter", [True, False])
            params["use_vwap_filter"] = use_vwap_filter
            if use_vwap_filter:
                params["vwap_filter_mode"] = trial.suggest_categorical("vwap_filter_mode", ["strict", "confirm"])
            else:
                params["vwap_filter_mode"] = "strict"

            # TICK filter
            use_tick_filter = trial.suggest_categorical("use_tick_filter", [True, False])
            params["use_tick_filter"] = use_tick_filter
            if use_tick_filter:
                params["tick_extreme_threshold"] = trial.suggest_int("tick_extreme_threshold", 400, 800)
            else:
                params["tick_extreme_threshold"] = 500

            # VIX regime params
            use_vix_regime = trial.suggest_categorical("use_vix_regime", [True, False])
            params["use_vix_regime"] = use_vix_regime

            if use_vix_regime:
                params["vix_high_threshold"] = trial.suggest_int("vix_high_threshold", 20, 35)
                params["vix_low_threshold"] = trial.suggest_int("vix_low_threshold", 12, 18)

                params["high_vol_cooldown_mult"] = trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.5)
                params["high_vol_confirmation_mult"] = trial.suggest_float("high_vol_confirmation_mult", 1.0, 2.5)
                params["high_vol_sustained_mult"] = trial.suggest_float("high_vol_sustained_mult", 1.0, 2.5)
                params["high_vol_volume_add"] = trial.suggest_float("high_vol_volume_add", 0.0, 0.5)
                params["high_vol_delta_adj"] = trial.suggest_float("high_vol_delta_adj", 0.0, 0.15)
                params["high_vol_min_hold_mult"] = trial.suggest_float("high_vol_min_hold_mult", 1.0, 2.0)

                params["low_vol_cooldown_mult"] = trial.suggest_float("low_vol_cooldown_mult", 0.4, 1.0)
                params["low_vol_confirmation_mult"] = trial.suggest_float("low_vol_confirmation_mult", 0.5, 1.0)
                params["low_vol_sustained_mult"] = trial.suggest_float("low_vol_sustained_mult", 0.5, 1.0)
                params["low_vol_volume_add"] = trial.suggest_float("low_vol_volume_add", -0.3, 0.0)
                params["low_vol_delta_adj"] = trial.suggest_float("low_vol_delta_adj", -0.10, 0.0)
            else:
                params.update(
                    {
                        "vix_high_threshold": 25,
                        "vix_low_threshold": 15,
                        "high_vol_cooldown_mult": 1.5,
                        "high_vol_confirmation_mult": 1.5,
                        "high_vol_sustained_mult": 1.5,
                        "high_vol_volume_add": 0.2,
                        "high_vol_delta_adj": 0.05,
                        "high_vol_min_hold_mult": 1.5,
                        "low_vol_cooldown_mult": 0.7,
                        "low_vol_confirmation_mult": 0.8,
                        "low_vol_sustained_mult": 0.8,
                        "low_vol_volume_add": -0.1,
                        "low_vol_delta_adj": -0.05,
                    }
                )

        # Risk/sizing params
        if phase in ("all", "risk"):
            enable_stop = trial.suggest_categorical("enable_stop_loss", [True, False])
            enable_tp = trial.suggest_categorical("enable_take_profit", [True, False])
            enable_trail = trial.suggest_categorical("enable_trailing_stop", [True, False])

            params["enable_stop_loss"] = enable_stop
            params["enable_take_profit"] = enable_tp
            params["enable_trailing_stop"] = enable_trail

            params["stop_loss_percent"] = trial.suggest_int("stop_loss_percent", 20, 70) if enable_stop else 50
            params["take_profit_percent"] = trial.suggest_int("take_profit_percent", 40, 200) if enable_tp else 100

            if enable_trail:
                params["trailing_stop_percent"] = trial.suggest_int("trailing_stop_percent", 10, 40)
                params["trailing_stop_activation"] = trial.suggest_int("trailing_stop_activation", 20, 100)
            else:
                params["trailing_stop_percent"] = 25
                params["trailing_stop_activation"] = 50

            params["min_hold_bars"] = trial.suggest_int("min_hold_bars", 0, 10)
            params["max_daily_trades"] = trial.suggest_int("max_daily_trades", 1, 10)

            params["target_delta"] = trial.suggest_float("target_delta", 0.20, 0.40)
            params["afternoon_delta"] = trial.suggest_float("afternoon_delta", 0.30, 0.50)
            params["afternoon_hour"] = trial.suggest_int("afternoon_hour", 11, 14)

            params["kelly_lookback"] = trial.suggest_int("kelly_lookback", 10, 30)

            params["kelly_fraction"] = trial.suggest_float("kelly_fraction", 0.0, 3.0)
            params["max_equity_risk"] = trial.suggest_float("max_equity_risk", 0.05, 0.25)

            params["max_kelly_pct_cap"] = trial.suggest_float("max_kelly_pct_cap", 0.15, 0.40)
            params["hard_max_contracts"] = trial.suggest_int("hard_max_contracts", 20, 150)

            params["signal_start_minutes"] = trial.suggest_int("signal_start_minutes", 0, 45)
            params["signal_end_minutes"] = trial.suggest_int("signal_end_minutes", 0, 30)

            use_atr_stops = trial.suggest_categorical("use_atr_stops", [True, False])
            params["use_atr_stops"] = use_atr_stops
            if use_atr_stops:
                params["atr_stop_mult"] = trial.suggest_float("atr_stop_mult", 1.0, 3.0)
                params["atr_target_mult"] = trial.suggest_float("atr_target_mult", 1.5, 4.0)
            else:
                params["atr_stop_mult"] = 2.0
                params["atr_target_mult"] = 3.0

            params["min_option_premium"] = trial.suggest_float("min_option_premium", 0.15, 0.75)

            # BS knobs (kept fixed unless you lock them or choose to add to search)
            params.setdefault("risk_free_rate", 0.05)
            params.setdefault("iv_mult", 1.0)
            params.setdefault("iv_floor", 0.05)
            params.setdefault("iv_cap", 1.50)
            params.setdefault("min_T_minutes", 5)

        result = run_backtest(params)

        if result.total_trades < min_trades:
            raise optuna.TrialPruned()

        if BEST_RESULT is None or result.score > BEST_RESULT.score:
            BEST_RESULT = result

        return -result.score

    return objective


# -----------------------------
# Data fetch
# -----------------------------
def fetch_historical_data(days: int = 90, start_date: str = None, end_date: str = None) -> List[Dict]:
    """
    Fetch historical data for backtesting.
    """
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    from config import config

    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri,
        token_file=config.schwab.token_file,
    )

    if not auth.is_authenticated:
        if not auth.authorize_interactive():
            raise RuntimeError("Failed to authenticate with Schwab")
    else:
        auth.refresh_access_token()

    client = SchwabClient(auth)

    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days_needed = (end - start).days
        print(f"\n  Date range: {start_date} to {end_date} ({calendar_days_needed} days)")
    else:
        calendar_days_needed = int(days * 1.5)
        end = datetime.now()
        start = end - timedelta(days=calendar_days_needed)

    all_bars: List[Dict[str, Any]] = []
    chunk_size_days = 7
    num_chunks = (calendar_days_needed // chunk_size_days) + 2

    chunk_end = end
    start_time = time_module.time()
    last_refresh_time = start_time
    bar_width = 30

    for chunk_num in range(num_chunks):
        chunk_start = chunk_end - timedelta(days=chunk_size_days)

        if chunk_start < start:
            chunk_start = start
        if chunk_end <= start:
            break

        current_time = time_module.time()
        if current_time - last_refresh_time > 300:
            try:
                auth.refresh_access_token()
                last_refresh_time = current_time
            except Exception:
                pass

        progress = (chunk_num + 1) / num_chunks
        elapsed = time_module.time() - start_time
        eta = (elapsed / progress) - elapsed if progress > 0 else 0.0
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% | ETA {eta:5.0f}s | Fetching bars...", end="", flush=True)

        try:
            chunk_bars = client.get_price_history(
                symbol="SPY",
                period_type="day",
                period=10,
                frequency_type="minute",
                frequency=5,
                extended_hours=False,
                start_date=chunk_start,
                end_date=chunk_end,
            )
            if chunk_bars:
                all_bars = chunk_bars + all_bars
        except Exception as e:
            if "401" in str(e):
                try:
                    auth.refresh_access_token()
                    last_refresh_time = time_module.time()
                except Exception:
                    pass

        chunk_end = chunk_start
        time_module.sleep(0.2)

    print(f"\r[{'█' * bar_width}] 100.0% | Fetching complete.{' ' * 20}")

    # Deduplicate + sort
    seen = set()
    unique_bars: List[Dict[str, Any]] = []
    for bar in sorted(all_bars, key=lambda x: x["datetime"]):
        key = bar["datetime"].isoformat()
        if key not in seen:
            seen.add(key)
            unique_bars.append(bar)

    trading_dates = set(bar["datetime"].date() for bar in unique_bars)
    print(f"  ✓ {len(unique_bars):,} SPY bars across {len(trading_dates)} trading days")

    # Fetch VIX data
    print("  Fetching VIX data for regime detection...")
    vix_bars: Dict[str, float] = {}

    try:
        vix_chunk_end = end
        for _ in range(num_chunks):
            vix_chunk_start = vix_chunk_end - timedelta(days=chunk_size_days)
            if vix_chunk_start < start:
                vix_chunk_start = start
            if vix_chunk_end <= start:
                break

            try:
                vix_data = client.get_price_history(
                    symbol="$VIX",  # Keep as-is from your script; verify symbol if empty
                    period_type="day",
                    period=10,
                    frequency_type="minute",
                    frequency=5,
                    extended_hours=False,
                    start_date=vix_chunk_start,
                    end_date=vix_chunk_end,
                )
                if vix_data:
                    for vb in vix_data:
                        date_key = vb["datetime"].strftime("%Y-%m-%d")
                        vix_bars[date_key] = vb["close"]
            except Exception:
                pass

            vix_chunk_end = vix_chunk_start
            time_module.sleep(0.1)

        print(f"  ✓ VIX data for {len(vix_bars)} days")
    except Exception as e:
        print(f"  ⚠ VIX fetch failed (will use default): {e}")

    # Fetch TICK
    print("  Fetching NYSE TICK data for breadth filter...")
    tick_bars: Dict[str, float] = {}

    try:
        tick_chunk_end = end
        for _ in range(num_chunks):
            tick_chunk_start = tick_chunk_end - timedelta(days=chunk_size_days)
            if tick_chunk_start < start:
                tick_chunk_start = start
            if tick_chunk_end <= start:
                break

            try:
                tick_data = client.get_price_history(
                    symbol="$TICK",
                    period_type="day",
                    period=10,
                    frequency_type="minute",
                    frequency=5,
                    extended_hours=False,
                    start_date=tick_chunk_start,
                    end_date=tick_chunk_end,
                )
                if tick_data:
                    for tb in tick_data:
                        dt_key = tb["datetime"].strftime("%Y-%m-%d %H:%M")
                        tick_bars[dt_key] = tb["close"]
            except Exception:
                pass

            tick_chunk_end = tick_chunk_start
            time_module.sleep(0.1)

        print(f"  ✓ TICK data for {len(tick_bars)} bars")
    except Exception as e:
        print(f"  ⚠ TICK fetch failed (will use default): {e}")

    # Attach VIX and TICK
    for bar in unique_bars:
        date_key = bar["datetime"].strftime("%Y-%m-%d")
        dt_key = bar["datetime"].strftime("%Y-%m-%d %H:%M")
        bar["vix"] = vix_bars.get(date_key, 18.0)
        bar["tick"] = tick_bars.get(dt_key, 0.0)

    print()
    return unique_bars


# -----------------------------
# Progress callback
# -----------------------------
class ProgressCallback:
    """Single-line progress with best WR/PnL."""

    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.start_time = time_module.time()
        self.bar_width = 30
        self.best_wr = 0.0
        self.best_pnl = 0.0

    def __call__(self, study: optuna.Study, trial: FrozenTrial):
        global BEST_RESULT

        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        progress = completed / self.n_trials if self.n_trials else 0.0

        elapsed = time_module.time() - self.start_time
        eta = (elapsed / progress) - elapsed if progress > 0 else 0.0

        filled = int(self.bar_width * progress)
        bar = "█" * filled + "░" * (self.bar_width - filled)

        if BEST_RESULT:
            self.best_wr = max(self.best_wr, BEST_RESULT.win_rate)
            self.best_pnl = max(self.best_pnl, BEST_RESULT.total_pnl)

        status = f"Best: {self.best_wr:5.1f}% WR, ${self.best_pnl:,.0f} PnL"
        print(f"\r[{bar}] {progress*100:5.1f}% | ETA {eta:5.0f}s | {status}", end="", flush=True)


# -----------------------------
# Print results
# -----------------------------
def print_best_result(result: TrialResult):
    print("\n" + "=" * 70)
    print("                    BEST CONFIGURATION FOUND")
    print("=" * 70)

    print(f"\n  Win Rate:      {result.win_rate:.1f}%")
    print(f"  Total P&L:     ${result.total_pnl:,.2f}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:  ${result.max_drawdown:,.2f}")
    print(f"  Total Trades:  {result.total_trades}")
    print(f"  Score:         {result.score:.4f}")

    p = result.params

    print(f"\n  Signal Parameters:")
    for k in [
        "signal_cooldown_bars",
        "min_confirmation_bars",
        "sustained_bars_required",
        "volume_threshold",
        "use_time_filter",
        "use_or_bias_filter",
    ]:
        if k in p:
            print(f"    {k}: {p[k]}")

    print(f"\n  Position Sizing (Kelly):")
    print(f"    kelly_fraction: {p.get('kelly_fraction', 0):.2f}")
    print(f"    max_equity_risk: {p.get('max_equity_risk', 0.10):.2f}")
    print(f"    max_kelly_pct_cap: {p.get('max_kelly_pct_cap', 0.35):.2f}")
    print(f"    hard_max_contracts: {p.get('hard_max_contracts', 100)}")
    print(f"    kelly_lookback: {p.get('kelly_lookback', 20)}")

    print(f"\n  BS Pricing:")
    print(f"    risk_free_rate: {p.get('risk_free_rate', 0.05)}")
    print(f"    iv_mult: {p.get('iv_mult', 1.0)}")
    print(f"    iv_floor: {p.get('iv_floor', 0.05)}")
    print(f"    iv_cap: {p.get('iv_cap', 1.5)}")
    print(f"    min_T_minutes: {p.get('min_T_minutes', 5)}")

    print(f"\n  Delta Targeting:")
    print(f"    target_delta: {p.get('target_delta', 0.30):.2f}")
    print(f"    afternoon_delta: {p.get('afternoon_delta', 0.40):.2f}")
    print(f"    afternoon_hour: {p.get('afternoon_hour', 12)}")

    print(f"\n  Risk Management:")
    for k in [
        "max_daily_trades",
        "enable_stop_loss",
        "stop_loss_percent",
        "enable_take_profit",
        "take_profit_percent",
        "enable_trailing_stop",
        "trailing_stop_percent",
        "trailing_stop_activation",
        "min_hold_bars",
    ]:
        if k in p:
            print(f"    {k}: {p[k]}")

    print("\n" + "=" * 70)


# -----------------------------
# Main
# -----------------------------
def main():
    global GLOBAL_BARS, BEST_RESULT

    parser = argparse.ArgumentParser(description="Smart Strategy Optimizer (Bayesian)")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--trials", type=int, default=500, help="Number of optimization trials")
    parser.add_argument("--turbo", action="store_true", help="Use more parallel workers")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades for valid result")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "signal", "risk"],
        help="Optimization phase: all, signal, or risk",
    )
    parser.add_argument("--lock", type=str, default="", help="JSON file with locked params")
    parser.add_argument("--save-best", type=str, default="best_params.json", help="Save best params to file")

    args = parser.parse_args()

    if args.turbo:
        n_jobs = int(cpu_count() * 1.5)
        print(f"🚀 TURBO MODE: {n_jobs} parallel workers\n")
    else:
        n_jobs = max(1, cpu_count() - 1)

    try:
        GLOBAL_BARS = fetch_historical_data(days=args.days)
    except Exception as e:
        print(f"ERROR: failed to fetch data: {e}")
        sys.exit(1)

    if not GLOBAL_BARS:
        print("ERROR: no data returned")
        sys.exit(1)

    locked_params = {}
    if args.lock and args.phase in ("signal", "risk"):
        locked_params = _load_locked_params(args.lock)
        if locked_params and "params" in locked_params:
            locked_params = locked_params["params"]
        if locked_params:
            print(f"📌 Loaded {len(locked_params)} locked params from {args.lock}")

    print("🧠 Starting Bayesian Optimization")
    print(f"   Phase: {args.phase} | Trials: {args.trials} | Workers: {n_jobs}\n")

    sampler = TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(),
    )

    progress = ProgressCallback(args.trials)

    start_time = time_module.time()
    study.optimize(
        create_objective(min_trades=args.min_trades, phase=args.phase, locked_params=locked_params),
        n_trials=args.trials,
        n_jobs=n_jobs,
        callbacks=[progress],
        show_progress_bar=False,
    )

    elapsed = time_module.time() - start_time
    print(f"\r[{'█' * 30}] 100.0% | Done in {elapsed:.1f}s{' ' * 30}")

    if BEST_RESULT:
        print_best_result(BEST_RESULT)
        if args.save_best:
            _save_best_params(args.save_best, BEST_RESULT.params, BEST_RESULT)
            print(f"\n💾 Saved best params to {args.save_best}")
    else:
        print("\n⚠️  No valid configurations found. Try increasing --trials or decreasing --min-trades")

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n📊 Stats: {completed} completed, {pruned} pruned, {elapsed/args.trials*1000:.1f}ms/trial")


if __name__ == "__main__":
    main()
