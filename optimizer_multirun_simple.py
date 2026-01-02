"""
Multi-Run Optimizer - Simplified Version
=========================================

Runs the simplified optimizer multiple times with different seeds
to build statistical confidence in parameter choices.

Works with optimizer_simple.py (high delta ATM focus).

Usage:
    python optimizer_multirun_simple.py                      # 5 runs, 90 days
    python optimizer_multirun_simple.py --runs 10            # More runs
    python optimizer_multirun_simple.py --days 365           # Full year
    python optimizer_multirun_simple.py --start-date 2024-07-01 --end-date 2025-07-01
"""
import os
import sys
import json
import argparse
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from multiprocessing import cpu_count
import time as time_module
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_single_optimization(
    trials: int,
    seed: int,
    turbo: bool,
    min_trades: int,
    output_file: str,
    cached_bars: List = None
) -> Optional[Dict[str, Any]]:
    """Run a single optimization and return results."""
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    for name in ["signal_detector", "schwab_auth", "schwab_client", "optuna"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    from optimizer_simple import create_objective, run_backtest
    from optuna.samplers import TPESampler
    
    # Set global bars
    import optimizer_simple
    optimizer_simple.GLOBAL_BARS = cached_bars
    optimizer_simple.BEST_RESULT = None
    
    if not cached_bars:
        return None
    
    n_jobs = int(cpu_count() * 1.5) if turbo else max(1, cpu_count() - 1)
    
    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )
    
    study.optimize(
        create_objective(min_trades=min_trades),
        n_trials=trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )
    
    if optimizer_simple.BEST_RESULT:
        result = {
            "seed": seed,
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
    """Analyze multiple run results for statistical confidence."""
    
    if not results:
        return {}
    
    analysis = {
        "num_runs": len(results),
        "metrics": {},
        "params": {},
    }
    
    # Analyze metrics
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
    
    # Analyze parameters
    all_params = defaultdict(list)
    for r in results:
        for k, v in r.get("params", {}).items():
            all_params[k].append(v)
    
    for param, values in all_params.items():
        if all(isinstance(v, bool) for v in values):
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
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Confidence: inverse of coefficient of variation
            if mean_val != 0:
                cv = abs(std_val / mean_val)
                conf = max(0, 1 - cv)
            else:
                conf = 1.0 if std_val == 0 else 0.0
            
            analysis["params"][param] = {
                "type": "numeric",
                "mean": mean_val,
                "std": std_val,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "recommended": statistics.median(values),
                "confidence": conf,
            }
    
    # Overall confidence
    confidences = [p.get("confidence", 0) for p in analysis["params"].values()]
    analysis["overall_confidence"] = statistics.mean(confidences) if confidences else 0
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results."""
    
    print("\n" + "=" * 70)
    print("                 MULTI-RUN ANALYSIS")
    print("=" * 70)
    
    print(f"\n  Runs: {analysis.get('num_runs', 0)}")
    print(f"  Overall confidence: {analysis.get('overall_confidence', 0):.1%}")
    
    # Metrics
    print("\n" + "-" * 70)
    print("  METRICS")
    print("-" * 70)
    
    metrics = analysis.get("metrics", {})
    for key in ["win_rate", "total_pnl", "profit_factor", "sharpe_ratio", "max_drawdown", "total_trades"]:
        if key in metrics:
            m = metrics[key]
            if key == "win_rate":
                print(f"  {key:.<25} {m['mean']:>7.1f}% ± {m['std']:.1f}%  [{m['min']:.1f}% - {m['max']:.1f}%]")
            elif key in ["total_pnl", "max_drawdown"]:
                print(f"  {key:.<25} ${m['mean']:>7,.0f} ± ${m['std']:,.0f}  [${m['min']:,.0f} - ${m['max']:,.0f}]")
            elif key == "total_trades":
                print(f"  {key:.<25} {m['mean']:>7.0f} ± {m['std']:.0f}  [{m['min']:.0f} - {m['max']:.0f}]")
            else:
                print(f"  {key:.<25} {m['mean']:>7.2f} ± {m['std']:.2f}  [{m['min']:.2f} - {m['max']:.2f}]")
    
    # Parameters
    print("\n" + "-" * 70)
    print("  RECOMMENDED PARAMETERS")
    print("-" * 70)
    
    params = analysis.get("params", {})
    
    # Key params first
    print("\n  Delta Targeting:")
    for p in ["target_delta", "afternoon_delta"]:
        if p in params:
            info = params[p]
            conf = "HIGH" if info["confidence"] >= 0.7 else "MED" if info["confidence"] >= 0.4 else "LOW"
            print(f"    {p:.<35} {info['recommended']:.2f}  [{info['min']:.2f}-{info['max']:.2f}] {conf}")
    
    print("\n  Signal Enables:")
    for p in sorted(params.keys()):
        if p.startswith("enable_") and params[p]["type"] == "boolean":
            info = params[p]
            conf = "HIGH" if info["confidence"] >= 0.8 else "MED" if info["confidence"] >= 0.6 else "LOW"
            votes = f"{info['true_count']}Y/{info['false_count']}N"
            print(f"    {p:.<35} {str(info['recommended']):<6} ({votes}) {conf}")
    
    print("\n  Signal Params:")
    for p in ["signal_cooldown_bars", "min_confirmation_bars", "sustained_bars_required", "volume_threshold"]:
        if p in params:
            info = params[p]
            conf = "HIGH" if info["confidence"] >= 0.7 else "MED" if info["confidence"] >= 0.4 else "LOW"
            rec = int(info['recommended']) if info['recommended'] == int(info['recommended']) else info['recommended']
            print(f"    {p:.<35} {rec}  [{info['min']}-{info['max']}] {conf}")
    
    print("\n  Risk Management:")
    for p in ["trailing_stop_percent", "trailing_stop_activation", "stop_loss_percent", "max_daily_trades"]:
        if p in params:
            info = params[p]
            conf = "HIGH" if info["confidence"] >= 0.7 else "MED" if info["confidence"] >= 0.4 else "LOW"
            rec = int(info['recommended'])
            print(f"    {p:.<35} {rec}  [{int(info['min'])}-{int(info['max'])}] {conf}")
    
    print("\n  Position Sizing:")
    for p in ["kelly_fraction", "max_equity_risk", "hard_max_contracts"]:
        if p in params:
            info = params[p]
            conf = "HIGH" if info["confidence"] >= 0.7 else "MED" if info["confidence"] >= 0.4 else "LOW"
            if p == "hard_max_contracts":
                rec = int(info['recommended'])
                print(f"    {p:.<35} {rec}  [{int(info['min'])}-{int(info['max'])}] {conf}")
            else:
                print(f"    {p:.<35} {info['recommended']:.2f}  [{info['min']:.2f}-{info['max']:.2f}] {conf}")
    
    print("\n" + "=" * 70)


def generate_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommended config from analysis."""
    
    config = {}
    params = analysis.get("params", {})
    
    for param, info in params.items():
        if info.get("type") == "boolean":
            config[param] = info["recommended"]
        elif info.get("type") == "numeric":
            rec = info["recommended"]
            # Round integers
            if param in ["signal_cooldown_bars", "min_confirmation_bars", "sustained_bars_required",
                        "stop_loss_percent", "trailing_stop_percent", "trailing_stop_activation",
                        "max_daily_trades", "hard_max_contracts", "afternoon_hour",
                        "vix_high_threshold", "vix_low_threshold"]:
                config[param] = int(round(rec))
            else:
                config[param] = round(rec, 3)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Multi-Run Simplified Optimizer")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="multirun_simple")
    parser.add_argument("--seeds", type=str, default="")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [42, 123, 456, 789, 1000, 2024, 3141, 9999, 1234, 5678][:args.runs]
    
    print("\n" + "=" * 70)
    print("       MULTI-RUN OPTIMIZER (Simplified / High Delta)")
    print("=" * 70)
    print(f"\n  Runs: {args.runs}")
    if args.start_date and args.end_date:
        print(f"  Period: {args.start_date} to {args.end_date}")
    else:
        print(f"  Days: {args.days}")
    print(f"  Trials/run: {args.trials}")
    print(f"  Seeds: {seeds[:5]}..." if len(seeds) > 5 else f"  Seeds: {seeds}")
    
    # Fetch data once
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
        
        result = run_single_optimization(
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
    print_analysis(analysis)
    
    # Generate config
    config = generate_config(analysis)
    
    # Save
    analysis_file = os.path.join(args.output_dir, "analysis.json")
    with open(analysis_file, "w") as f:
        json.dump({
            "analysis": analysis,
            "recommended_config": config,
            "runs": results,
        }, f, indent=2, sort_keys=True)
    
    config_file = os.path.join(args.output_dir, "recommended_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    
    print(f"\n  💾 {analysis_file}")
    print(f"  💾 {config_file}")
    
    # Print config
    print("\n" + "-" * 70)
    print("  RECOMMENDED CONFIG")
    print("-" * 70)
    
    key_params = ["target_delta", "afternoon_delta", "signal_cooldown_bars", 
                  "enable_val_bounce", "enable_vah_rejection", "enable_breakout",
                  "trailing_stop_percent", "use_vix_regime"]
    
    for p in key_params:
        if p in config:
            v = config[p]
            if isinstance(v, float):
                print(f"    {p}: {v:.2f}")
            else:
                print(f"    {p}: {v}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()