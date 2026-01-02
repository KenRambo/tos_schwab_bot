"""
Optimizer Variant D - Shorter Cooldown (More Trades)
====================================================
Tests if shorter cooldown maintains win rate with more opportunities.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer_simple import *

def create_objective_short_cooldown(min_trades: int = 50):  # Higher min trades
    def objective(trial: optuna.Trial) -> float:
        global BEST_RESULT
        
        params = {
            # Core signal params - SHORTER COOLDOWN RANGE
            "signal_cooldown_bars": trial.suggest_int("signal_cooldown_bars", 4, 12),  # Was 5-20
            "min_confirmation_bars": trial.suggest_int("min_confirmation_bars", 1, 3),  # Faster confirm
            "sustained_bars_required": trial.suggest_int("sustained_bars_required", 2, 4),
            "volume_threshold": trial.suggest_float("volume_threshold", 1.1, 1.6),  # Slightly tighter
            "use_or_bias_filter": trial.suggest_categorical("use_or_bias_filter", [True, False]),
            
            # Signal enables
            "enable_val_bounce": trial.suggest_categorical("enable_val_bounce", [True, False]),
            "enable_vah_rejection": trial.suggest_categorical("enable_vah_rejection", [True, False]),
            "enable_breakout": trial.suggest_categorical("enable_breakout", [True, False]),
            "enable_breakdown": trial.suggest_categorical("enable_breakdown", [True, False]),
            "enable_poc_reclaim": False,
            "enable_poc_breakdown": False,
            "enable_sustained_breakout": False,
            "enable_sustained_breakdown": False,
            
            # VIX regime
            "use_vix_regime": trial.suggest_categorical("use_vix_regime", [True, False]),
            "vix_high_threshold": 25,
            "vix_low_threshold": 15,
            "high_vol_cooldown_mult": trial.suggest_float("high_vol_cooldown_mult", 1.2, 2.0),
            "low_vol_cooldown_mult": trial.suggest_float("low_vol_cooldown_mult", 0.5, 0.9),
            
            # Delta targeting
            "target_delta": trial.suggest_float("target_delta", 0.55, 0.80),
            "afternoon_delta": trial.suggest_float("afternoon_delta", 0.60, 0.85),
            "afternoon_hour": 12,
            
            # Risk management - allow more trades
            "max_daily_trades": trial.suggest_int("max_daily_trades", 4, 8),  # More trades/day
            "enable_trailing_stop": True,
            "trailing_stop_percent": trial.suggest_int("trailing_stop_percent", 20, 40),
            "trailing_stop_activation": trial.suggest_int("trailing_stop_activation", 30, 60),
            "enable_stop_loss": trial.suggest_categorical("enable_stop_loss", [True, False]),
            "stop_loss_percent": trial.suggest_int("stop_loss_percent", 40, 70),
            
            # Position sizing - more conservative since more trades
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.2, 1.0),
            "max_equity_risk": trial.suggest_float("max_equity_risk", 0.08, 0.20),
            "hard_max_contracts": trial.suggest_int("hard_max_contracts", 15, 75),
        }
        
        result = run_backtest(params)
        
        if result.total_trades < min_trades:
            raise optuna.TrialPruned()
        
        if BEST_RESULT is None or result.score > BEST_RESULT.score:
            BEST_RESULT = result
        
        return -result.score
    
    return objective


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimizer D - Short Cooldown")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--min-trades", type=int, default=50)  # Higher requirement
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="best_short_cooldown.json")
    
    args = parser.parse_args()
    
    n_jobs = int(cpu_count() * 1.5) if args.turbo else max(1, cpu_count() - 1)
    
    print("=" * 60)
    print("  OPTIMIZER D - SHORTER COOLDOWN (MORE TRADES)")
    print("  (cooldown 4-12 bars, max_daily_trades 4-8)")
    print("=" * 60)
    
    GLOBAL_BARS = fetch_data(args.days, args.start_date, args.end_date)
    
    if not GLOBAL_BARS:
        print("ERROR: No data")
        sys.exit(1)
    
    print(f"\n🧠 Optimizing: {args.trials} trials, {n_jobs} workers\n")
    print(f"   Min trades required: {args.min_trades}\n")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed, multivariate=True),
    )
    
    study.optimize(
        create_objective_short_cooldown(args.min_trades),
        n_trials=args.trials,
        n_jobs=n_jobs,
        callbacks=[Progress(args.trials)],
        show_progress_bar=False,
    )
    
    print()
    
    if BEST_RESULT:
        print("\n" + "=" * 60)
        print("  BEST RESULT (SHORT COOLDOWN)")
        print("=" * 60)
        print(f"  Win Rate:      {BEST_RESULT.win_rate:.1f}%")
        print(f"  Total P&L:     ${BEST_RESULT.total_pnl:,.2f}")
        print(f"  Profit Factor: {BEST_RESULT.profit_factor:.2f}")
        print(f"  Max Drawdown:  ${BEST_RESULT.max_drawdown:,.2f}")
        print(f"  Trades:        {BEST_RESULT.total_trades}")
        
        p = BEST_RESULT.params
        print(f"\n  Cooldown Settings:")
        print(f"    signal_cooldown_bars: {p.get('signal_cooldown_bars')}")
        print(f"    max_daily_trades: {p.get('max_daily_trades')}")
        print(f"    min_confirmation_bars: {p.get('min_confirmation_bars')}")
        
        with open(args.save, "w") as f:
            json.dump({
                "variant": "D_short_cooldown",
                "params": BEST_RESULT.params,
                "metrics": {
                    "win_rate": BEST_RESULT.win_rate,
                    "total_pnl": BEST_RESULT.total_pnl,
                    "profit_factor": BEST_RESULT.profit_factor,
                    "max_drawdown": BEST_RESULT.max_drawdown,
                    "total_trades": BEST_RESULT.total_trades,
                    "sharpe_ratio": BEST_RESULT.sharpe_ratio,
                }
            }, f, indent=2)
        print(f"\n💾 Saved to {args.save}")
    
    print("=" * 60)
