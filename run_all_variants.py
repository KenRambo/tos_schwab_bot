"""
Run All Variants with Shared Data Cache
=======================================

Fetches data ONCE, then runs variants A, B, D sequentially using cached bars.

Usage:
    python run_all_variants.py --start-date 2024-07-01 --end-date 2025-07-01 --runs 5 --trials 500 --turbo
"""
import os
import sys
import json
import argparse
import statistics
import pickle
from typing import List, Dict, Any, Optional
from collections import defaultdict
from multiprocessing import cpu_count
import time as time_module
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.ERROR)
for name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
    logging.getLogger(name).setLevel(logging.ERROR)

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
from optuna.samplers import TPESampler


def fetch_and_cache_data(days: int, start_date: str, end_date: str, cache_file: str = "bars_cache.pkl"):
    """Fetch data and cache to disk."""
    from optimizer_simple import fetch_data
    
    print("\n  Fetching data from API...")
    bars = fetch_data(days, start_date, end_date)
    
    if bars:
        with open(cache_file, "wb") as f:
            pickle.dump(bars, f)
        print(f"  ✓ Cached {len(bars):,} bars to {cache_file}")
    
    return bars


def load_cached_data(cache_file: str = "bars_cache.pkl"):
    """Load cached bars from disk."""
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            bars = pickle.load(f)
        print(f"  ✓ Loaded {len(bars):,} bars from cache")
        return bars
    return None


# ========== VARIANT OBJECTIVES ==========

def create_objective_baseline(min_trades: int = 30):
    """Original baseline - no forced params."""
    def objective(trial: optuna.Trial) -> float:
        import optimizer_simple
        
        params = {
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 5, 20),
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 4),
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 5),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.8),
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            
            "use_vix_regime": trial.suggest_categorical("use_vix_regime", [True, False]),
            "vix_high_threshold": 25,
            "vix_low_threshold": 15,
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.0),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.5, 0.9),
            
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            
            "max_daily_trades": trial.suggest_int("max_daily_trades", 2, 6),
            "enable_trailing_stop": True,
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.3, 1.5),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.10, 0.25),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 20, 100),
        }
        
        result = optimizer_simple.run_backtest(params)
        
        if result.total_trades < min_trades:
            raise optuna.TrialPruned()
        
        if optimizer_simple.BEST_RESULT is None or result.score > optimizer_simple.BEST_RESULT.score:
            optimizer_simple.BEST_RESULT = result
        
        return -result.score
    
    return objective


def create_objective_a(min_trades: int = 30):
    """Variant A: Force shorts enabled."""
    def objective(trial: optuna.Trial) -> float:
        import optimizer_simple
        
        params = {
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 5, 20),
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 4),
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 5),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.8),
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": True,  # FORCED
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": True,  # FORCED
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            
            "use_vix_regime": trial.suggest_categorical("use_vix_regime", [True, False]),
            "vix_high_threshold": 25,
            "vix_low_threshold": 15,
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.0),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.5, 0.9),
            
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            
            "max_daily_trades": trial.suggest_int("max_daily_trades", 2, 6),
            "enable_trailing_stop": True,
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.3, 1.5),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.10, 0.25),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 20, 100),
        }
        
        result = optimizer_simple.run_backtest(params)
        
        if result.total_trades < min_trades:
            raise optuna.TrialPruned()
        
        if optimizer_simple.BEST_RESULT is None or result.score > optimizer_simple.BEST_RESULT.score:
            optimizer_simple.BEST_RESULT = result
        
        return -result.score
    
    return objective


def create_objective_b(min_trades: int = 30):
    """Variant B: Force VIX regime enabled."""
    def objective(trial: optuna.Trial) -> float:
        import optimizer_simple
        
        params = {
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 5, 20),
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 4),
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 5),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.8),
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            
            "use_vix_regime": True,  # FORCED
            "vix_high_threshold": trial.suggest_int("vix_high_threshold", 20, 30),
            "vix_low_threshold": trial.suggest_int("vix_low_threshold", 12, 18),
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.5),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.4, 0.9),
            
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            
            "max_daily_trades": trial.suggest_int("max_daily_trades", 2, 6),
            "enable_trailing_stop": True,
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.3, 1.5),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.10, 0.25),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 20, 100),
        }
        
        result = optimizer_simple.run_backtest(params)
        
        if result.total_trades < min_trades:
            raise optuna.TrialPruned()
        
        if optimizer_simple.BEST_RESULT is None or result.score > optimizer_simple.BEST_RESULT.score:
            optimizer_simple.BEST_RESULT = result
        
        return -result.score
    
    return objective


def create_objective_d(min_trades: int = 50):
    """Variant D: Shorter cooldown, more trades."""
    def objective(trial: optuna.Trial) -> float:
        import optimizer_simple
        
        params = {
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 4, 12),  # SHORTER
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 3),
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 4),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.6),
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            
            "use_vix_regime": trial.suggest_categorical("use_vix_regime", [True, False]),
            "vix_high_threshold": 25,
            "vix_low_threshold": 15,
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.0),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.5, 0.9),
            
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            
            "max_daily_trades": trial.suggest_int("max_daily_trades", 4, 8),  # MORE TRADES
            "enable_trailing_stop": True,
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.2, 1.0),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.08, 0.20),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 15, 75),
        }
        
        result = optimizer_simple.run_backtest(params)
        
        if result.total_trades < min_trades:
            raise optuna.TrialPruned()
        
        if optimizer_simple.BEST_RESULT is None or result.score > optimizer_simple.BEST_RESULT.score:
            optimizer_simple.BEST_RESULT = result
        
        return -result.score
    
    return objective


# ========== PROGRESS CALLBACK ==========

class ProgressCallback:
    """Live progress bar for optimization."""
    
    def __init__(self, n_trials: int, variant: str, seed: int):
        self.n_trials = n_trials
        self.variant = variant
        self.seed = seed
        self.start_time = time_module.time()
        self.best_pnl = 0
        self.best_wr = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        import optimizer_simple
        
        # Count completed trials
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        total_done = completed + pruned
        
        pct = total_done / self.n_trials if self.n_trials > 0 else 0
        
        # ETA calculation
        elapsed = time_module.time() - self.start_time
        eta = (elapsed / pct - elapsed) if pct > 0.01 else 0
        
        # Update best
        if optimizer_simple.BEST_RESULT:
            self.best_pnl = max(self.best_pnl, optimizer_simple.BEST_RESULT.total_pnl)
            self.best_wr = max(self.best_wr, optimizer_simple.BEST_RESULT.win_rate)
        
        # Progress bar
        bar_width = 25
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Print progress line
        print(f"\r    [{bar}] {pct*100:5.1f}% | {total_done}/{self.n_trials} | "
              f"ETA {eta:4.0f}s | Best: {self.best_wr:.1f}% ${self.best_pnl:,.0f}   ", 
              end="", flush=True)


# ========== RUN SINGLE OPTIMIZATION ==========

def run_single(variant: str, seed: int, trials: int, turbo: bool, min_trades: int, cached_bars: List) -> Optional[Dict]:
    """Run a single optimization with given variant and seed."""
    import optimizer_simple
    
    # Reset globals
    optimizer_simple.GLOBAL_BARS = cached_bars
    optimizer_simple.BEST_RESULT = None
    
    # Select objective
    objectives = {
        "baseline": create_objective_baseline,
        "a": create_objective_a,
        "b": create_objective_b,
        "d": create_objective_d,
    }
    
    create_obj = objectives.get(variant)
    if not create_obj:
        return None
    
    # Adjust min_trades for variant D
    if variant == "d":
        min_trades = max(min_trades, 50)
    
    n_jobs = int(cpu_count() * 1.5) if turbo else max(1, cpu_count() - 1)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, multivariate=True),
    )
    
    # Create progress callback
    progress = ProgressCallback(trials, variant, seed)
    
    try:
        study.optimize(
            create_obj(min_trades=min_trades),
            n_trials=trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
            callbacks=[progress],
        )
    except Exception as e:
        print(f"\n    Error: {e}")
        return None
    
    # Clear progress line
    print()
    
    if optimizer_simple.BEST_RESULT:
        return {
            "seed": seed,
            "variant": variant,
            "params": optimizer_simple.BEST_RESULT.params,
            "metrics": {
                "win_rate": optimizer_simple.BEST_RESULT.win_rate,
                "total_pnl": optimizer_simple.BEST_RESULT.total_pnl,
                "total_trades": optimizer_simple.BEST_RESULT.total_trades,
                "profit_factor": optimizer_simple.BEST_RESULT.profit_factor,
                "sharpe_ratio": optimizer_simple.BEST_RESULT.sharpe_ratio,
                "max_drawdown": optimizer_simple.BEST_RESULT.max_drawdown,
                "score": optimizer_simple.BEST_RESULT.score,
            }
        }
    return None


# ========== ANALYZE RESULTS ==========

def analyze_variant(results: List[Dict]) -> Dict:
    """Analyze results for one variant."""
    if not results:
        return {}
    
    analysis = {"num_runs": len(results), "metrics": {}, "params": {}}
    
    for key in ["win_rate", "total_pnl", "total_trades", "profit_factor", "sharpe_ratio", "max_drawdown"]:
        values = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        if values:
            analysis["metrics"][key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }
    
    return analysis


def print_comparison(all_results: Dict[str, List[Dict]]):
    """Print comparison table of all variants."""
    
    print("\n" + "=" * 80)
    print("                         VARIANT COMPARISON")
    print("=" * 80)
    
    headers = ["Variant", "Win Rate", "Total P&L", "Trades", "Profit Factor", "Max DD"]
    print(f"\n  {'Variant':<12} {'Win Rate':>12} {'Total P&L':>14} {'Trades':>8} {'PF':>8} {'Max DD':>10}")
    print("  " + "-" * 70)
    
    variant_names = {
        "baseline": "Baseline",
        "a": "A: Shorts",
        "b": "B: VIX",
        "d": "D: Fast",
    }
    
    for variant in ["baseline", "a", "b", "d"]:
        results = all_results.get(variant, [])
        if not results:
            continue
        
        analysis = analyze_variant(results)
        m = analysis.get("metrics", {})
        
        wr = m.get("win_rate", {})
        pnl = m.get("total_pnl", {})
        trades = m.get("total_trades", {})
        pf = m.get("profit_factor", {})
        dd = m.get("max_drawdown", {})
        
        name = variant_names.get(variant, variant)
        print(f"  {name:<12} {wr.get('mean', 0):>10.1f}% {pnl.get('mean', 0):>13,.0f} {trades.get('mean', 0):>8.0f} {pf.get('mean', 0):>8.1f} {dd.get('mean', 0):>10,.0f}")
    
    print()


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="Run All Variants with Shared Cache")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="variant_results")
    parser.add_argument("--use-cache", action="store_true", help="Use existing cache if available")
    parser.add_argument("--variants", type=str, default="a,b,d", help="Comma-separated variants to run")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.output_dir, "bars_cache.pkl")
    
    seeds = [42, 123, 456, 789, 1000][:args.runs]
    variants_to_run = [v.strip() for v in args.variants.split(",")]
    
    print("\n" + "=" * 80)
    print("                    RUN ALL VARIANTS")
    print("=" * 80)
    
    if args.start_date and args.end_date:
        print(f"\n  Period: {args.start_date} to {args.end_date}")
    else:
        print(f"\n  Days: {args.days}")
    print(f"  Variants: {', '.join(variants_to_run)}")
    print(f"  Runs per variant: {args.runs}")
    print(f"  Trials per run: {args.trials}")
    print(f"  Seeds: {seeds}")
    
    # Fetch or load data
    cached_bars = None
    if args.use_cache:
        cached_bars = load_cached_data(cache_file)
    
    if not cached_bars:
        cached_bars = fetch_and_cache_data(args.days, args.start_date, args.end_date, cache_file)
    
    if not cached_bars:
        print("\n  ✗ Failed to get data!")
        return
    
    # Run all variants
    all_results = {}
    total_start = time_module.time()
    
    for variant in variants_to_run:
        variant_names = {
            "a": "A: Force Shorts",
            "b": "B: Force VIX Regime",
            "d": "D: Short Cooldown",
        }
        
        print(f"\n\n{'='*60}")
        print(f"  VARIANT {variant.upper()}: {variant_names.get(variant, variant)}")
        print("=" * 60)
        
        results = []
        
        for i, seed in enumerate(seeds):
            print(f"\n  Run {i+1}/{len(seeds)} (seed={seed}):")
            run_start = time_module.time()
            
            result = run_single(
                variant=variant,
                seed=seed,
                trials=args.trials,
                turbo=args.turbo,
                min_trades=args.min_trades,
                cached_bars=cached_bars
            )
            
            run_time = time_module.time() - run_start
            
            if result:
                results.append(result)
                wr = result["metrics"]["win_rate"]
                pnl = result["metrics"]["total_pnl"]
                trades = result["metrics"]["total_trades"]
                print(f"    ✓ Result: {wr:.1f}% WR, ${pnl:,.0f} P&L, {trades} trades ({run_time:.0f}s)")
            else:
                print(f"    ✗ Failed ({run_time:.0f}s)")
        
        all_results[variant] = results
        
        # Save variant results
        variant_file = os.path.join(args.output_dir, f"variant_{variant}_results.json")
        with open(variant_file, "w") as f:
            json.dump({
                "variant": variant,
                "analysis": analyze_variant(results),
                "runs": results
            }, f, indent=2)
        print(f"\n  💾 Saved to {variant_file}")
    
    total_time = time_module.time() - total_start
    print(f"\n\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Print comparison
    print_comparison(all_results)
    
    # Save combined results
    combined_file = os.path.join(args.output_dir, "all_variants_comparison.json")
    with open(combined_file, "w") as f:
        combined = {}
        for v, results in all_results.items():
            combined[v] = {
                "analysis": analyze_variant(results),
                "runs": results
            }
        json.dump(combined, f, indent=2)
    
    print(f"  💾 Combined results: {combined_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()