#!/usr/bin/env python3
"""
Butterfly Backtest Runner

Run this on your local machine with Schwab API credentials.

Usage:
    python run_backtest.py                    # Last 30 days
    python run_backtest.py --days 60          # Last 60 days  
    python run_backtest.py --days 90 --output results.json
"""

import argparse
import json
from datetime import datetime, date
from butterfly_trader import ButterflyConfig, run_backtest
from backtest import fetch_historical_data
from signal_detector import Bar


def main():
    parser = argparse.ArgumentParser(description='Run Butterfly Backtest')
    parser.add_argument('--days', type=int, default=30, help='Trading days to backtest')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🦋 BUTTERFLY CREDIT STACKER - BACKTEST")
    print("=" * 60)
    
    # Fetch ES futures data
    print(f"\n📡 Fetching {args.days} days of /ES futures data...")
    print("   (This may take a minute...)")
    
    try:
        raw_bars = fetch_historical_data('/ES', args.days)
        print(f"   ✓ Loaded {len(raw_bars)} bars")
    except Exception as e:
        print(f"   ✗ Failed to fetch data: {e}")
        print("\n   Make sure you have:")
        print("   1. Valid Schwab API credentials in .env or config")
        print("   2. Run schwab_auth.py to authenticate first")
        return
    
    # Convert to Bar objects
    bars = [
        Bar(
            timestamp=b['datetime'],
            open=b['open'],
            high=b['high'],
            low=b['low'],
            close=b['close'],
            volume=b['volume']
        )
        for b in raw_bars
    ]
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    config = ButterflyConfig()
    results = run_backtest(config, bars)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n📅 Period: {results['start_date']} to {results['end_date']}")
    print(f"   Trading Days: {results['trading_days']}")
    
    print(f"\n🦋 Butterflies:")
    print(f"   Total: {results['total_butterflies']}")
    print(f"   Hit 30% Target: {results['successful_fills']} ({results['successful_fills']/max(1,results['total_butterflies'])*100:.1f}%)")
    print(f"   Early Exits: {results['early_exits']}")
    print(f"   Avg per Day: {results['total_butterflies']/max(1,results['trading_days']):.1f}")
    
    print(f"\n💰 P&L (per 1-lot):")
    print(f"   Total P&L: ${results['total_pnl']:,.2f}")
    print(f"   Avg per Butterfly: ${results['avg_pnl']:,.2f}")
    print(f"   Win Rate: {results['win_rate']*100:.1f}%")
    
    print(f"\n📈 Risk Metrics:")
    print(f"   Best Day: ${results['best_day']:,.2f}")
    print(f"   Worst Day: ${results['worst_day']:,.2f}")
    
    # Daily breakdown
    if args.verbose and results.get('daily_results'):
        print(f"\n📆 Daily Breakdown:")
        print("-" * 50)
        for day in sorted(results['daily_results'].keys())[-10:]:
            r = results['daily_results'][day]
            emoji = "✅" if r['pnl'] > 0 else "❌" if r['pnl'] < 0 else "➖"
            print(f"   {day}: {emoji} ${r['pnl']:>8,.2f} ({r['butterflies']} butterflies)")
    
    # Projections
    if results['trading_days'] > 0:
        daily_avg = results['total_pnl'] / results['trading_days']
        print(f"\n📊 Projections (if consistent):")
        print(f"   Daily Avg: ${daily_avg:,.2f}")
        print(f"   Monthly (~21 days): ${daily_avg * 21:,.2f}")
        print(f"   Annual (~252 days): ${daily_avg * 252:,.2f}")
    
    print("\n" + "=" * 60)
    
    # Save results
    if args.output:
        output_data = {
            'run_date': datetime.now().isoformat(),
            'config': {
                'wing_width': config.wing_width,
                'credit_target_pct': config.credit_target_pct,
                'abort_loss_pct': config.abort_loss_pct,
            },
            'results': {
                'start_date': str(results['start_date']),
                'end_date': str(results['end_date']),
                'trading_days': results['trading_days'],
                'total_butterflies': results['total_butterflies'],
                'successful_fills': results['successful_fills'],
                'early_exits': results['early_exits'],
                'total_pnl': results['total_pnl'],
                'avg_pnl': results['avg_pnl'],
                'win_rate': results['win_rate'],
                'best_day': results['best_day'],
                'worst_day': results['worst_day'],
            },
            'daily_results': {
                str(k): v for k, v in results.get('daily_results', {}).items()
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()