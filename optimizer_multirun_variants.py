"""
Multi-Run for Variant Tests (A, B, D)
=====================================

Runs each variant optimizer multiple times with different seeds.

Usage:
    python optimizer_multirun_variants.py --variant a --runs 5 --trials 500 --turbo
    python optimizer_multirun_variants.py --variant b --runs 5 --trials 500 --turbo
    python optimizer_multirun_variants.py --variant d --runs 5 --trials 500 --turbo
"""
import os
import sys
import json
import argparse
import statistics
from typing import List, Dict, Any, Optional
from collections import defaultdict
from multiprocessing import cpu_count
import time as time_module
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_variant_optimization(
    variant: str,
    trials: int,
    seed: int,
    turbo: bool,
    min_trades: int,
    output_file: str,
    cached_bars: List = None
) -> Optional[Dict[str, Any]]:
    """Run a single variant optimization."""
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    for name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    from optuna.samplers import TPESampler
    
    # Import base and set globals
    import optimizer_simple
    optimizer_simple.GLOBAL_BARS = cached_bars
    optimizer_simple.BEST_RESULT = None
    
    if not cached_bars:
        return None
    
    # Import variant-specific objective
    if variant == "a":
        from optimizer_variant_a import create_objective_shorts as create_obj
    elif variant == "b":
        from optimizer_variant_b import create_objective_vix as create_obj
    elif variant == "d":
        from optimizer_variant_d import create_objective_short_cooldown as create_obj
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    n_jobs = int(cpu_count() * 1.5) if turbo else max(1, cpu_count() - 1)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, multivariate=True),
    )
    
    study.optimize(
        create_obj(min_trades=min_trades),
        n_trials=trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )
    
    if optimizer_simple.BEST_RESULT:
        result = {
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
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)
        
        return result
    
    return None


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze multiple run results."""
    
    if not results:
        return {}
    
    analysis = {
        "num_runs": len(results),
        "metrics": {},
        "params": {},
    }
    
    for key in ["win_rate", "total_pnl", "total_trades", "profit_factor", "sharpe_ratio", "max_drawdown"]:
        values = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        if values:
            analysis["metrics"][key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            }
    
    all_params = defaultdict(list)
    for r in results:
        for k, v in r.get("params", {}).items():
            all_params[k].append(v)
    
    for param, values in all_params.items():
        if all(isinstance(v, bool) for v in values):
            true_count = sum(1 for v in values if v)
            false_count = len(values) - true_count
            analysis["params"][param] = {
                "type": "boolean",
                "recommended": true_count > false_count,
                "true_count": true_count,
                "false_count": false_count,
                "confidence": max(true_count, false_count) / len(values),
            }
        elif all(isinstance(v, (int, float)) for v in values):
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            analysis["params"][param] = {
                "type": "numeric",
                "mean": mean_val,
                "std": std_val,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "recommended": statistics.median(values),
            }
    
    return analysis


def print_summary(variant: str, analysis: Dict[str, Any]):
    """Print summary for variant."""
    
    variant_names = {
        "a": "FORCE SHORTS",
        "b": "FORCE VIX REGIME", 
        "d": "SHORT COOLDOWN"
    }
    
    print("\n" + "=" * 60)
    print(f"  VARIANT {variant.upper()}: {variant_names.get(variant, variant)}")
    print("=" * 60)
    
    metrics = analysis.get("metrics", {})
    
    print(f"\n  Runs: {analysis.get('num_runs', 0)}")
    
    if "win_rate" in metrics:
        m = metrics["win_rate"]
        print(f"  Win Rate:    {m['mean']:.1f}% ± {m['std']:.1f}%  [{m['min']:.1f}% - {m['max']:.1f}%]")
    
    if "total_pnl" in metrics:
        m = metrics["total_pnl"]
        print(f"  Total P&L:   ${m['mean']:,.0f} ± ${m['std']:,.0f}  [${m['min']:,.0f} - ${m['max']:,.0f}]")
    
    if "total_trades" in metrics:
        m = metrics["total_trades"]
        print(f"  Trades:      {m['mean']:.0f} ± {m['std']:.0f}  [{m['min']:.0f} - {m['max']:.0f}]")
    
    if "profit_factor" in metrics:
        m = metrics["profit_factor"]
        print(f"  Profit Fac:  {m['mean']:.1f} ± {m['std']:.1f}")
    
    if "max_drawdown" in metrics:
        m = metrics["max_drawdown"]
        print(f"  Max DD:      ${m['mean']:,.0f} ± ${m['std']:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Run Variant Optimizer")
    parser.add_argument("--variant", type=str, required=True, choices=["a", "b", "d"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set output dir based on variant
    if args.output_dir is None:
        args.output_dir = f"multirun_variant_{args.variant}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Adjust min_trades for variant D
    if args.variant == "d":
        args.min_trades = max(args.min_trades, 50)
    
    seeds = [42, 123, 456, 789, 1000][:args.runs]
    
    variant_names = {
        "a": "FORCE SHORTS (vah_rejection=True, breakdown=True)",
        "b": "FORCE VIX REGIME (use_vix_regime=True)",
        "d": "SHORT COOLDOWN (cooldown 4-12, more trades)"
    }
    
    print("\n" + "=" * 60)
    print(f"  MULTIRUN: VARIANT {args.variant.upper()}")
    print(f"  {variant_names.get(args.variant)}")
    print("=" * 60)
    
    if args.start_date and args.end_date:
        print(f"\n  Period: {args.start_date} to {args.end_date}")
    else:
        print(f"\n  Days: {args.days}")
    print(f"  Runs: {args.runs}")
    print(f"  Trials/run: {args.trials}")
    print(f"  Min trades: {args.min_trades}")
    
    # Fetch data
    print("\n  Fetching data...")
    try:
        from optimizer_simple import fetch_data
        cached_bars = fetch_data(args.days, args.start_date, args.end_date)
        print(f"  ✓ {len(cached_bars):,} bars cached")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Run optimizations
    results = []
    start_time = time_module.time()
    
    for i, seed in enumerate(seeds):
        run_start = time_module.time()
        print(f"\n  Run {i+1}/{len(seeds)} (seed={seed})...", end="", flush=True)
        
        output_file = os.path.join(args.output_dir, f"run_{i+1}_seed_{seed}.json")
        
        result = run_variant_optimization(
            variant=args.variant,
            trials=args.trials,
            seed=seed,
            turbo=args.turbo,
            min_trades=args.min_trades,
            output_file=output_file,
            cached_bars=cached_bars
        )
        
        run_time = time_module.time() - run_start
        
        if result:
            results.append(result)
            wr = result["metrics"]["win_rate"]
            pnl = result["metrics"]["total_pnl"]
            trades = result["metrics"]["total_trades"]
            print(f" ✓ {wr:.1f}% WR, ${pnl:,.0f}, {trades} trades ({run_time:.0f}s)")
        else:
            print(f" ✗ Failed ({run_time:.0f}s)")
    
    total_time = time_module.time() - start_time
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    if not results:
        print("\n  ✗ No successful runs!")
        return
    
    # Analyze
    analysis = analyze_results(results)
    print_summary(args.variant, analysis)
    
    # Save
    analysis_file = os.path.join(args.output_dir, "analysis.json")
    with open(analysis_file, "w") as f:
        json.dump({
            "variant": args.variant,
            "analysis": analysis,
            "runs": results,
        }, f, indent=2, sort_keys=True)
    
    print(f"\n  💾 {analysis_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
