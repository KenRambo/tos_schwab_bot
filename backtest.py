"""
Backtest Utility

Test signal detection against historical data before going live.
Uses CSV files with OHLCV data.
"""
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_bars: int
    total_signals: int
    long_signals: int
    short_signals: int
    trades: List[dict]
    total_pnl: float
    win_rate: float
    avg_bars_held: float


def load_csv_data(filepath: str) -> List[Bar]:
    """
    Load OHLCV data from CSV file.
    
    Expected columns: datetime, open, high, low, close, volume
    """
    bars = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Try multiple datetime formats
                dt_str = row.get('datetime') or row.get('date') or row.get('time')
                
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M']:
                    try:
                        timestamp = datetime.strptime(dt_str, fmt)
                        break
                    except:
                        continue
                else:
                    continue
                
                bar = Bar(
                    timestamp=timestamp,
                    open=float(row.get('open', 0)),
                    high=float(row.get('high', 0)),
                    low=float(row.get('low', 0)),
                    close=float(row.get('close', 0)),
                    volume=int(float(row.get('volume', 0)))
                )
                bars.append(bar)
                
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
    
    logger.info(f"Loaded {len(bars)} bars from {filepath}")
    return bars


def run_backtest(
    bars: List[Bar],
    initial_capital: float = 10000.0,
    contracts: int = 1,
    option_premium: float = 2.00,  # Assumed avg premium
    **detector_kwargs
) -> BacktestResult:
    """
    Run backtest on historical data.
    
    Args:
        bars: List of OHLCV bars
        initial_capital: Starting capital
        contracts: Contracts per trade
        option_premium: Assumed option premium for P&L calc
        **detector_kwargs: Arguments for SignalDetector
    """
    detector = SignalDetector(**detector_kwargs)
    
    trades = []
    current_position = None
    entry_bar_idx = 0
    
    signals_by_type = {}
    
    for i, bar in enumerate(bars):
        signal = detector.add_bar(bar)
        
        if signal:
            # Track signal types
            sig_type = signal.signal_type.value
            signals_by_type[sig_type] = signals_by_type.get(sig_type, 0) + 1
            
            logger.info(f"[{bar.timestamp}] {signal.signal_type.value} - {signal.direction.value} @ {bar.close:.2f}")
            
            # Close existing position if opposite signal
            if current_position and current_position['direction'] != signal.direction:
                bars_held = i - entry_bar_idx
                
                # Simulate P&L (simplified)
                if current_position['direction'] == Direction.LONG:
                    # Long call - profits if price went up
                    price_change = bar.close - current_position['entry_price']
                else:
                    # Long put - profits if price went down
                    price_change = current_position['entry_price'] - bar.close
                
                # Simplified option P&L (delta ~0.5 for ATM-ish)
                pnl = price_change * 0.5 * 100 * contracts - (option_premium * 100 * contracts * 0.1)
                
                trade = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': bar.timestamp,
                    'direction': current_position['direction'].value,
                    'signal_type': current_position['signal_type'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': bar.close,
                    'bars_held': bars_held,
                    'pnl': pnl
                }
                trades.append(trade)
                
                logger.info(f"  Closed: {bars_held} bars, P&L: ${pnl:.2f}")
                current_position = None
            
            # Open new position
            if not current_position:
                current_position = {
                    'direction': signal.direction,
                    'signal_type': signal.signal_type.value,
                    'entry_price': bar.close,
                    'entry_time': bar.timestamp
                }
                entry_bar_idx = i
    
    # Calculate statistics
    total_signals = sum(signals_by_type.values())
    long_signals = sum(v for k, v in signals_by_type.items() 
                       if 'BOUNCE' in k or 'RECLAIM' in k or 'BREAKOUT' in k)
    short_signals = total_signals - long_signals
    
    winners = [t for t in trades if t['pnl'] > 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)
    avg_bars = sum(t['bars_held'] for t in trades) / len(trades) if trades else 0
    
    return BacktestResult(
        total_bars=len(bars),
        total_signals=total_signals,
        long_signals=long_signals,
        short_signals=short_signals,
        trades=trades,
        total_pnl=total_pnl,
        win_rate=win_rate,
        avg_bars_held=avg_bars
    )


def print_results(result: BacktestResult) -> None:
    """Print backtest results summary"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total bars processed: {result.total_bars}")
    print(f"Total signals: {result.total_signals}")
    print(f"  Long signals: {result.long_signals}")
    print(f"  Short signals: {result.short_signals}")
    print(f"\nTotal trades: {len(result.trades)}")
    print(f"Win rate: {result.win_rate:.1f}%")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Avg bars held: {result.avg_bars_held:.1f}")
    
    if result.trades:
        print("\n" + "-" * 60)
        print("TRADE LOG")
        print("-" * 60)
        for i, trade in enumerate(result.trades[:20], 1):  # Show first 20
            print(f"{i}. {trade['direction']:5} | {trade['signal_type']:20} | "
                  f"Held: {trade['bars_held']:3} bars | P&L: ${trade['pnl']:>8.2f}")
        
        if len(result.trades) > 20:
            print(f"... and {len(result.trades) - 20} more trades")
    
    print("=" * 60)


def main():
    """Run backtest from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest signal detection')
    parser.add_argument('csv_file', help='Path to CSV file with OHLCV data')
    parser.add_argument('--contracts', type=int, default=1, help='Contracts per trade')
    parser.add_argument('--cooldown', type=int, default=8, help='Signal cooldown bars')
    parser.add_argument('--or-filter', action='store_true', help='Use OR bias filter')
    
    args = parser.parse_args()
    
    # Load data
    bars = load_csv_data(args.csv_file)
    
    if not bars:
        print("No data loaded. Check CSV format.")
        return
    
    # Run backtest
    result = run_backtest(
        bars=bars,
        contracts=args.contracts,
        signal_cooldown_bars=args.cooldown,
        use_or_bias_filter=args.or_filter
    )
    
    # Print results
    print_results(result)


if __name__ == "__main__":
    main()
