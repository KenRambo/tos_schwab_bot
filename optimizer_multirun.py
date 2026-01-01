"""
Multi-Run Optimizer with Statistical Confidence Analysis
=========================================================

Runs the optimizer multiple times with different seeds and time periods
to build statistical confidence in parameter choices.

Usage:
    python optimizer_multirun.py                      # Default: 5 runs, 90 days
    python optimizer_multirun.py --runs 10            # More runs for higher confidence
    python optimizer_multirun.py --days 120           # Different time period
    python optimizer_multirun.py --trials 1000        # More trials per run
    python optimizer_multirun.py --walk-forward       # Walk-forward validation

Requirements:
    pip install optuna
"""
import os
import sys
import json
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from multiprocessing import cpu_count
import time as time_module

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_single_optimization(
    days: int,
    trials: int,
    seed: int,
    phase: str,
    turbo: bool,
    min_trades: int,
    output_file: str,
    lock_file: str = "",
    cached_bars: List = None
) -> Optional[Dict[str, Any]]:
    """Run a single optimization and return results"""
    
    # Import here to avoid loading everything at module level
    import warnings
    warnings.filterwarnings("ignore")
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    for logger_name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    from optimizer_smart import (
        create_objective,
        _load_locked_params,
        TrialResult,
    )
    from optuna.samplers import TPESampler
    
    # Set global bars (use cached if provided)
    import optimizer_smart
    
    if cached_bars:
        optimizer_smart.GLOBAL_BARS = cached_bars
    else:
        try:
            from optimizer_smart import fetch_historical_data
            optimizer_smart.GLOBAL_BARS = fetch_historical_data(days=days)
        except Exception as e:
            print(f"  ✗ Data fetch failed: {e}")
            return None
    
    optimizer_smart.BEST_RESULT = None
    
    if not optimizer_smart.GLOBAL_BARS:
        print(f"  ✗ No data")
        return None
    
    # Load locked params if provided
    locked_params = {}
    if lock_file:
        locked_params = _load_locked_params(lock_file)
        if locked_params and "params" in locked_params:
            locked_params = locked_params["params"]
    
    # Setup optimizer
    n_jobs = int(cpu_count() * 1.5) if turbo else max(1, cpu_count() - 1)
    
    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(),
    )
    
    # Run optimization (quiet)
    study.optimize(
        create_objective(min_trades=min_trades, phase=phase, locked_params=locked_params),
        n_trials=trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )
    
    # Get result
    if optimizer_smart.BEST_RESULT:
        result = {
            "seed": seed,
            "days": days,
            "params": optimizer_smart.BEST_RESULT.params,
            "metrics": {
                "win_rate": optimizer_smart.BEST_RESULT.win_rate,
                "total_pnl": optimizer_smart.BEST_RESULT.total_pnl,
                "total_trades": optimizer_smart.BEST_RESULT.total_trades,
                "profit_factor": optimizer_smart.BEST_RESULT.profit_factor,
                "sharpe_ratio": optimizer_smart.BEST_RESULT.sharpe_ratio,
                "max_drawdown": optimizer_smart.BEST_RESULT.max_drawdown,
                "score": optimizer_smart.BEST_RESULT.score,
            }
        }
        
        # Save individual result
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)
        
        return result
    
    return None


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze multiple run results for statistical confidence"""
    
    if not results:
        return {}
    
    analysis = {
        "num_runs": len(results),
        "metrics": {},
        "params": {},
        "confidence": {},
    }
    
    # Analyze metrics
    metric_keys = ["win_rate", "total_pnl", "total_trades", "profit_factor", "sharpe_ratio", "max_drawdown", "score"]
    
    for key in metric_keys:
        values = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        if values:
            analysis["metrics"][key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            }
    
    # Analyze parameters
    all_params = defaultdict(list)
    for r in results:
        for k, v in r.get("params", {}).items():
            all_params[k].append(v)
    
    for param, values in all_params.items():
        if all(isinstance(v, bool) for v in values):
            # Boolean parameter - count True/False
            true_count = sum(1 for v in values if v)
            false_count = len(values) - true_count
            majority = true_count > false_count
            confidence = max(true_count, false_count) / len(values)
            
            analysis["params"][param] = {
                "type": "boolean",
                "recommended": majority,
                "true_count": true_count,
                "false_count": false_count,
                "confidence": confidence,
            }
            
        elif all(isinstance(v, (int, float)) for v in values):
            # Numeric parameter
            analysis["params"][param] = {
                "type": "numeric",
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "recommended": statistics.median(values),
            }
            
            # Confidence: inverse of coefficient of variation
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            if mean != 0:
                cv = abs(std / mean)
                analysis["params"][param]["confidence"] = max(0, 1 - cv)
            else:
                analysis["params"][param]["confidence"] = 1.0 if std == 0 else 0.0
    
    # Overall confidence score
    confidences = [p.get("confidence", 0) for p in analysis["params"].values()]
    analysis["overall_confidence"] = statistics.mean(confidences) if confidences else 0
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results in a readable format"""
    
    print("\n" + "=" * 80)
    print("                    MULTI-RUN ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n  Runs completed: {analysis.get('num_runs', 0)}")
    print(f"  Overall confidence: {analysis.get('overall_confidence', 0):.1%}")
    
    # Metrics summary
    print("\n" + "-" * 80)
    print("  METRICS SUMMARY (across all runs)")
    print("-" * 80)
    
    metrics = analysis.get("metrics", {})
    for key in ["win_rate", "total_pnl", "profit_factor", "sharpe_ratio", "max_drawdown"]:
        if key in metrics:
            m = metrics[key]
            if key == "win_rate":
                print(f"  {key:.<25} {m['mean']:>8.1f}% ± {m['std']:.1f}%  (range: {m['min']:.1f}% - {m['max']:.1f}%)")
            elif key in ["total_pnl", "max_drawdown"]:
                print(f"  {key:.<25} ${m['mean']:>8,.0f} ± ${m['std']:,.0f}  (range: ${m['min']:,.0f} - ${m['max']:,.0f})")
            else:
                print(f"  {key:.<25} {m['mean']:>8.2f} ± {m['std']:.2f}  (range: {m['min']:.2f} - {m['max']:.2f})")
    
    # Parameter recommendations
    print("\n" + "-" * 80)
    print("  PARAMETER RECOMMENDATIONS")
    print("-" * 80)
    
    params = analysis.get("params", {})
    
    # Boolean params (signal enables)
    print("\n  Signal Enables:")
    bool_params = {k: v for k, v in params.items() if v.get("type") == "boolean" and k.startswith("enable_")}
    for param, info in sorted(bool_params.items()):
        conf = info["confidence"]
        rec = info["recommended"]
        votes = f"{info['true_count']}T/{info['false_count']}F"
        conf_str = "HIGH" if conf >= 0.8 else "MED" if conf >= 0.6 else "LOW"
        print(f"    {param:.<40} {str(rec):<6} ({votes}) [{conf_str}]")
    
    # Boolean params (filters/risk)
    print("\n  Filters & Risk Enables:")
    bool_params = {k: v for k, v in params.items() if v.get("type") == "boolean" and not k.startswith("enable_")}
    for param, info in sorted(bool_params.items()):
        conf = info["confidence"]
        rec = info["recommended"]
        votes = f"{info['true_count']}T/{info['false_count']}F"
        conf_str = "HIGH" if conf >= 0.8 else "MED" if conf >= 0.6 else "LOW"
        print(f"    {param:.<40} {str(rec):<6} ({votes}) [{conf_str}]")
    
    # Numeric params
    print("\n  Numeric Parameters:")
    num_params = {k: v for k, v in params.items() if v.get("type") == "numeric"}
    for param, info in sorted(num_params.items()):
        conf = info["confidence"]
        rec = info["recommended"]
        conf_str = "HIGH" if conf >= 0.7 else "MED" if conf >= 0.4 else "LOW"
        
        if isinstance(rec, float) and rec == int(rec):
            rec = int(rec)
        
        if isinstance(rec, float):
            print(f"    {param:.<40} {rec:<8.3f} (range: {info['min']:.3f}-{info['max']:.3f}) [{conf_str}]")
        else:
            print(f"    {param:.<40} {rec:<8} (range: {info['min']}-{info['max']}) [{conf_str}]")
    
    print("\n" + "=" * 80)


def generate_recommended_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a recommended config from analysis"""
    
    config = {}
    params = analysis.get("params", {})
    
    for param, info in params.items():
        if info.get("type") == "boolean":
            config[param] = info["recommended"]
        elif info.get("type") == "numeric":
            # Use median for recommendation
            rec = info["recommended"]
            # Round integers
            if param in ["signal_cooldown_bars", "min_confirmation_bars", "sustained_bars_required",
                        "stop_loss_percent", "take_profit_percent", "trailing_stop_percent",
                        "trailing_stop_activation", "min_hold_bars", "hard_max_contracts"]:
                config[param] = int(round(rec))
            else:
                config[param] = round(rec, 3)
    
    return config


def run_walk_forward(
    total_days: int,
    train_days: int,
    test_days: int,
    trials: int,
    turbo: bool,
    min_trades: int
) -> List[Dict[str, Any]]:
    """
    Walk-forward analysis: train on N days, test on next M days, repeat.
    This is the gold standard for avoiding overfitting.
    """
    
    print("\n" + "=" * 80)
    print("                    WALK-FORWARD ANALYSIS")
    print("=" * 80)
    print(f"\n  Total period: {total_days} days")
    print(f"  Training window: {train_days} days")
    print(f"  Testing window: {test_days} days")
    
    # This would require date-based data fetching which we don't have yet
    # For now, just note this is a TODO
    print("\n  ⚠️  Walk-forward analysis requires date-range data fetching.")
    print("     This is a TODO for future implementation.")
    print("     For now, use multi-seed runs on different day counts.\n")
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Multi-Run Optimizer with Confidence Analysis")
    parser.add_argument("--runs", type=int, default=5, help="Number of optimization runs")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD (e.g., 2025-04-01)")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (e.g., 2025-07-01)")
    parser.add_argument("--trials", type=int, default=500, help="Trials per run")
    parser.add_argument("--turbo", action="store_true", help="Use more parallel workers")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades for valid result")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "signal", "risk"])
    parser.add_argument("--output-dir", type=str, default="multirun_results", help="Output directory")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds (default: auto)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [42, 123, 456, 789, 1000, 2024, 3141, 9999, 1234, 5678][:args.runs]
    
    print("\n" + "=" * 80)
    print("                    MULTI-RUN OPTIMIZER")
    print("=" * 80)
    print(f"\n  Runs: {args.runs}")
    if args.start_date and args.end_date:
        print(f"  Date Range: {args.start_date} to {args.end_date}")
    else:
        print(f"  Days: {args.days}")
    print(f"  Trials per run: {args.trials}")
    print(f"  Phase: {args.phase}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {args.output_dir}/")
    
    if args.walk_forward:
        run_walk_forward(
            total_days=args.days,
            train_days=60,
            test_days=30,
            trials=args.trials,
            turbo=args.turbo,
            min_trades=args.min_trades
        )
        return
    
    # Run optimizations
    results = []
    start_time = time_module.time()
    
    # Pre-fetch and cache data once (avoid re-fetching for each run)
    print("\n  Fetching data (once for all runs)...")
    cached_bars = None
    try:
        from optimizer_smart import fetch_historical_data
        cached_bars = fetch_historical_data(
            days=args.days,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"  ✓ Cached {len(cached_bars):,} bars")
    except Exception as e:
        print(f"  ✗ Data fetch failed: {e}")
        print("  Try re-authenticating: python trading_bot.py")
        return
    
    for i, seed in enumerate(seeds):
        run_start = time_module.time()
        print(f"\n  Run {i+1}/{len(seeds)} (seed={seed})...", end="", flush=True)
        
        output_file = os.path.join(args.output_dir, f"run_{i+1}_seed_{seed}.json")
        
        result = run_single_optimization(
            days=args.days,
            trials=args.trials,
            seed=seed,
            phase=args.phase,
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
            print(f" ✓ {wr:.1f}% WR, ${pnl:,.0f} PnL ({run_time:.1f}s)")
        else:
            print(f" ✗ Failed ({run_time:.1f}s)")
    
    total_time = time_module.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if not results:
        print("\n  ✗ No successful runs!")
        return
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print_analysis(analysis)
    
    # Generate and save recommended config
    recommended = generate_recommended_config(analysis)
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, "analysis.json")
    with open(analysis_file, "w") as f:
        json.dump({
            "analysis": analysis,
            "recommended_config": recommended,
            "individual_runs": results,
        }, f, indent=2, sort_keys=True)
    
    print(f"\n  💾 Analysis saved to {analysis_file}")
    
    # Save recommended config separately
    config_file = os.path.join(args.output_dir, "recommended_config.json")
    with open(config_file, "w") as f:
        json.dump(recommended, f, indent=2, sort_keys=True)
    
    print(f"  💾 Recommended config saved to {config_file}")
    
    # Print recommended config
    print("\n" + "-" * 80)
    print("  RECOMMENDED CONFIG (copy to config.py)")
    print("-" * 80)
    print(f"\n{json.dumps(recommended, indent=2, sort_keys=True)}")
    print()


if __name__ == "__main__":
    main()