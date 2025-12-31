"""
Parallelized Strategy Optimizer
================================

Runs thousands of parameter combinations in parallel to find optimal settings.
Optimizes for BOTH win rate (>60%) AND profitability.

Usage:
    python optimizer.py                    # Run with defaults (90 days, 8 workers)
    python optimizer.py --days 120         # Test on 120 days
    python optimizer.py --workers 12       # Use 12 CPU cores
    python optimizer.py --top 20           # Show top 20 results
    python optimizer.py --output results.csv  # Save all results to CSV
"""
import os
import sys
import logging
import argparse
import csv
import json
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import time as time_module

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress logging during optimization (only show errors)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also suppress other module loggers
logging.getLogger('signal_detector').setLevel(logging.ERROR)
logging.getLogger('schwab_auth').setLevel(logging.ERROR)
logging.getLogger('schwab_client').setLevel(logging.ERROR)

from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType


@dataclass
class OptimizationResult:
    """Result of a single parameter combination test"""
    # Parameters tested
    params: Dict[str, Any]
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_trade: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    
    # Scoring (composite metric)
    score: float = 0.0


# Parameter ranges to test
PARAM_GRID = {
    # Signal parameters
    'signal_cooldown_bars': [0, 2, 4, 8],
    'min_confirmation_bars': [2, 3, 4],
    'sustained_bars_required': [3, 4, 5],
    'volume_threshold': [1.2, 1.3, 1.5],
    
    # Signal enable/disable
    'enable_val_bounce': [True, False],
    'enable_vah_rejection': [True, False],
    'enable_poc_reclaim': [True],  # Keep - was profitable
    'enable_poc_breakdown': [True, False],
    'enable_sustained_breakout': [True],  # Keep - was profitable
    'enable_sustained_breakdown': [True],  # Keep - was profitable
    'enable_breakout': [True, False],
    'enable_breakdown': [True, False],
    
    # Time filters
    'use_time_filter': [True, False],
    'use_or_bias_filter': [True, False],
    
    # Risk Management - Stop Loss
    'enable_stop_loss': [True, False],
    'stop_loss_percent': [25, 50, 75],  # Exit if option loses X%
    
    # Risk Management - Take Profit
    'enable_take_profit': [True, False],
    'take_profit_percent': [50, 100, 150],  # Exit if option gains X%
    
    # Risk Management - Trailing Stop
    'enable_trailing_stop': [True, False],
    'trailing_stop_percent': [20, 30, 40],  # Trail by X% from high
    'trailing_stop_activation': [25, 50],  # Activate after X% gain
    
    # Minimum hold time (bars before allowing exit)
    'min_hold_bars': [0, 5, 10],
}

# Reduced grid for faster testing
PARAM_GRID_FAST = {
    'signal_cooldown_bars': [10, 15, 20],
    'min_confirmation_bars': [2, 3],
    'sustained_bars_required': [3, 4],
    'enable_val_bounce': [True, False],
    'enable_vah_rejection': [True, False],
    'enable_poc_breakdown': [True, False],
    'use_time_filter': [True, False],
    
    # Risk management (fast)
    'enable_stop_loss': [True, False],
    'stop_loss_percent': [50],
    'enable_take_profit': [True, False],
    'take_profit_percent': [100],
    'min_hold_bars': [0, 5],
}

# Risk-focused grid - SMART version that avoids redundant combinations
# Instead of grid search, we generate only meaningful combinations
def generate_risk_combinations() -> List[Dict]:
    """Generate risk parameter combinations without redundancy"""
    combinations = []
    
    # Base signal params (fixed from prior optimization)
    base = {
        'signal_cooldown_bars': 15,
        'min_confirmation_bars': 3,
        'sustained_bars_required': 4,
        'enable_val_bounce': False,
        'enable_vah_rejection': False,
        'enable_poc_breakdown': True,
        'enable_poc_reclaim': True,
        'enable_sustained_breakout': True,
        'enable_sustained_breakdown': True,
        'enable_breakout': True,
        'enable_breakdown': True,
        'use_time_filter': True,
        'use_or_bias_filter': True,
    }
    
    # Min hold bars options
    min_hold_options = [0, 5, 10]
    
    # Kelly fraction options
    # Kelly % = W - [(1-W) / R] where W=win rate, R=avg win/avg loss
    # Full Kelly is too aggressive, so we test fractions
    kelly_options = [
        {'kelly_fraction': 0.0},   # Fixed size (no Kelly)
        {'kelly_fraction': 0.25},  # Quarter Kelly (conservative)
        {'kelly_fraction': 0.5},   # Half Kelly (recommended)
        {'kelly_fraction': 0.75},  # Three-quarter Kelly
        {'kelly_fraction': 1.0},   # Full Kelly (aggressive)
    ]
    
    # Max contracts cap (to prevent blowup)
    max_contracts_options = [3, 5, 10]
    
    # Stop loss options
    stop_loss_options = [
        {'enable_stop_loss': False, 'stop_loss_percent': 50},  # Disabled
        {'enable_stop_loss': True, 'stop_loss_percent': 25},
        {'enable_stop_loss': True, 'stop_loss_percent': 50},
        {'enable_stop_loss': True, 'stop_loss_percent': 75},
    ]
    
    # Take profit options
    take_profit_options = [
        {'enable_take_profit': False, 'take_profit_percent': 100},  # Disabled
        {'enable_take_profit': True, 'take_profit_percent': 50},
        {'enable_take_profit': True, 'take_profit_percent': 100},
        {'enable_take_profit': True, 'take_profit_percent': 150},
    ]
    
    # Trailing stop options
    trailing_stop_options = [
        {'enable_trailing_stop': False, 'trailing_stop_percent': 25, 'trailing_stop_activation': 50},  # Disabled
        {'enable_trailing_stop': True, 'trailing_stop_percent': 20, 'trailing_stop_activation': 30},
        {'enable_trailing_stop': True, 'trailing_stop_percent': 25, 'trailing_stop_activation': 50},
        {'enable_trailing_stop': True, 'trailing_stop_percent': 30, 'trailing_stop_activation': 75},
    ]
    
    # Generate all meaningful combinations
    for min_hold in min_hold_options:
        for kelly in kelly_options:
            for max_c in max_contracts_options:
                for sl in stop_loss_options:
                    for tp in take_profit_options:
                        for ts in trailing_stop_options:
                            combo = base.copy()
                            combo['min_hold_bars'] = min_hold
                            combo.update(kelly)
                            combo['max_contracts'] = max_c
                            combo.update(sl)
                            combo.update(tp)
                            combo.update(ts)
                            combinations.append(combo)
    
    return combinations

# Keep old grid for reference but mark as slow
PARAM_GRID_RISK_SLOW = {
    'signal_cooldown_bars': [15],
    'min_confirmation_bars': [3],
    'sustained_bars_required': [4],
    'enable_val_bounce': [False],
    'enable_vah_rejection': [False],
    'enable_poc_breakdown': [True],
    'use_time_filter': [True],
    
    # Exhaustive risk management testing
    'enable_stop_loss': [True, False],
    'stop_loss_percent': [20, 30, 40, 50, 60, 75],
    'enable_take_profit': [True, False],
    'take_profit_percent': [30, 50, 75, 100, 150, 200],
    'enable_trailing_stop': [True, False],
    'trailing_stop_percent': [15, 20, 25, 30, 40],
    'trailing_stop_activation': [20, 30, 50, 75],
    'min_hold_bars': [0, 3, 5, 8, 10, 15],
}


def generate_param_combinations(param_grid: Dict) -> List[Dict]:
    """Generate all combinations of parameters"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def run_single_backtest(
    bars: List[Dict],
    params: Dict,
    starting_capital: float = 10000.0,
    target_delta: float = 0.30,
    afternoon_delta: float = 0.40,
    max_daily_trades: int = 3,
    commission: float = 0.65
) -> OptimizationResult:
    """
    Run a single backtest with given parameters.
    This function is called in parallel.
    """
    try:
        # Initialize detector with test parameters
        detector = SignalDetector(
            length_period=20,
            value_area_percent=70.0,
            volume_threshold=params.get('volume_threshold', 1.3),
            use_relaxed_volume=True,
            min_confirmation_bars=params.get('min_confirmation_bars', 2),
            sustained_bars_required=params.get('sustained_bars_required', 3),
            signal_cooldown_bars=params.get('signal_cooldown_bars', 8),
            use_or_bias_filter=params.get('use_or_bias_filter', True),
            or_buffer_points=1.0,
            rth_only=True,
            use_time_filter=params.get('use_time_filter', False),
            # Signal enables
            enable_val_bounce=params.get('enable_val_bounce', True),
            enable_poc_reclaim=params.get('enable_poc_reclaim', True),
            enable_breakout=params.get('enable_breakout', True),
            enable_sustained_breakout=params.get('enable_sustained_breakout', True),
            enable_prior_val_bounce=params.get('enable_val_bounce', True),
            enable_prior_poc_reclaim=True,
            enable_vah_rejection=params.get('enable_vah_rejection', True),
            enable_poc_breakdown=params.get('enable_poc_breakdown', True),
            enable_breakdown=params.get('enable_breakdown', True),
            enable_sustained_breakdown=params.get('enable_sustained_breakdown', True),
            enable_prior_vah_rejection=params.get('enable_vah_rejection', True),
            enable_prior_poc_breakdown=True
        )
        
        # Risk management parameters
        enable_stop_loss = params.get('enable_stop_loss', False)
        stop_loss_percent = params.get('stop_loss_percent', 50)
        enable_take_profit = params.get('enable_take_profit', False)
        take_profit_percent = params.get('take_profit_percent', 100)
        enable_trailing_stop = params.get('enable_trailing_stop', False)
        trailing_stop_percent = params.get('trailing_stop_percent', 25)
        trailing_stop_activation = params.get('trailing_stop_activation', 50)
        min_hold_bars = params.get('min_hold_bars', 0)
        
        # Kelly position sizing parameters
        kelly_fraction = params.get('kelly_fraction', 0.0)  # 0 = fixed size, 0.5 = half Kelly, etc.
        max_contracts = params.get('max_contracts', 5)
        base_contracts = 1  # Minimum contracts
        
        # Track trades
        trades = []
        current_trade = None
        trade_counter = 0
        current_date = None
        daily_trade_count = 0
        equity = starting_capital
        equity_curve = [starting_capital]
        daily_pnl = {}
        
        # Rolling stats for Kelly calculation (use last N trades)
        rolling_window = 20
        recent_wins = []
        recent_losses = []
        
        for bar_data in bars:
            bar = Bar(
                timestamp=bar_data['datetime'],
                open=bar_data['open'],
                high=bar_data['high'],
                low=bar_data['low'],
                close=bar_data['close'],
                volume=bar_data['volume']
            )
            
            bar_date = bar.timestamp.date()
            
            # New day reset
            if current_date != bar_date:
                current_date = bar_date
                daily_trade_count = 0
                if bar_date not in daily_pnl:
                    daily_pnl[bar_date] = 0.0
            
            # Time filter - morning only if enabled
            if params.get('use_time_filter', False):
                if bar.timestamp.hour >= 12:
                    # Still need to check risk management for open positions
                    if current_trade:
                        pass  # Continue to risk management check below
                    else:
                        continue
            
            # Update bars held and check risk management for open trade
            if current_trade:
                current_trade['bars_held'] += 1
                
                # Calculate current P&L
                entry_price = current_trade['entry_price']
                current_price = bar.close
                delta = current_trade.get('delta', target_delta)
                
                if current_trade['direction'] == 'LONG':
                    underlying_move = current_price - entry_price
                else:
                    underlying_move = entry_price - current_price
                
                option_entry = current_trade['option_entry']
                option_move = underlying_move * delta
                current_option_price = max(0.01, option_entry + option_move)
                current_pnl_pct = ((current_option_price / option_entry) - 1) * 100
                
                # Track high water mark for trailing stop
                if 'high_water_mark' not in current_trade:
                    current_trade['high_water_mark'] = current_pnl_pct
                else:
                    current_trade['high_water_mark'] = max(current_trade['high_water_mark'], current_pnl_pct)
                
                # Check risk management exits (only after min hold period)
                exit_reason = None
                if current_trade['bars_held'] >= min_hold_bars:
                    
                    # Stop Loss
                    if enable_stop_loss and current_pnl_pct <= -stop_loss_percent:
                        exit_reason = f"Stop Loss ({stop_loss_percent}%)"
                    
                    # Take Profit
                    elif enable_take_profit and current_pnl_pct >= take_profit_percent:
                        exit_reason = f"Take Profit ({take_profit_percent}%)"
                    
                    # Trailing Stop
                    elif enable_trailing_stop:
                        hwm = current_trade['high_water_mark']
                        if hwm >= trailing_stop_activation:
                            trail_level = hwm - trailing_stop_percent
                            if current_pnl_pct <= trail_level:
                                exit_reason = f"Trailing Stop (from {hwm:.1f}%)"
                
                # Execute risk management exit
                if exit_reason:
                    contracts = current_trade.get('contracts', base_contracts)
                    pnl = (current_option_price - option_entry) * 100 * contracts - commission * 2 * contracts
                    current_trade['exit_price'] = current_price
                    current_trade['exit_reason'] = exit_reason
                    current_trade['pnl'] = pnl
                    trades.append(current_trade)
                    
                    # Update rolling stats for Kelly
                    if pnl > 0:
                        recent_wins.append(pnl / contracts)  # Per-contract win
                        if len(recent_wins) > rolling_window:
                            recent_wins.pop(0)
                    else:
                        recent_losses.append(abs(pnl / contracts))  # Per-contract loss
                        if len(recent_losses) > rolling_window:
                            recent_losses.pop(0)
                    
                    equity += pnl
                    equity_curve.append(equity)
                    daily_pnl[bar_date] += pnl
                    
                    current_trade = None
                    continue  # Don't check for new signals this bar
            
            # Check for signal
            signal = detector.add_bar(bar)
            
            if signal and daily_trade_count < max_daily_trades:
                # Close existing trade on opposite signal
                if current_trade and current_trade['direction'] != signal.direction.value:
                    # Only close if past min hold period
                    if current_trade['bars_held'] >= min_hold_bars:
                        entry_price = current_trade['entry_price']
                        exit_price = bar.close
                        delta = current_trade.get('delta', target_delta)
                        option_entry = current_trade['option_entry']
                        contracts = current_trade.get('contracts', base_contracts)
                        
                        if current_trade['direction'] == 'LONG':
                            underlying_move = exit_price - entry_price
                        else:
                            underlying_move = entry_price - exit_price
                        
                        option_move = underlying_move * delta
                        option_exit = max(0.01, option_entry + option_move)
                        pnl = (option_exit - option_entry) * 100 * contracts - commission * 2 * contracts
                        
                        current_trade['exit_price'] = exit_price
                        current_trade['exit_reason'] = f"Opposite signal: {signal.signal_type.value}"
                        current_trade['pnl'] = pnl
                        trades.append(current_trade)
                        
                        # Update rolling stats for Kelly
                        if pnl > 0:
                            recent_wins.append(pnl / contracts)
                            if len(recent_wins) > rolling_window:
                                recent_wins.pop(0)
                        else:
                            recent_losses.append(abs(pnl / contracts))
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
                    
                    # Determine delta based on time
                    delta = afternoon_delta if bar.timestamp.hour >= 12 else target_delta
                    option_entry = max(0.50, bar.close * (0.003 + (delta - 0.30) * 0.01))
                    
                    # Calculate Kelly-based position size
                    if kelly_fraction > 0 and len(recent_wins) >= 5 and len(recent_losses) >= 5:
                        # Kelly % = W - [(1-W) / R]
                        # W = win rate, R = avg win / avg loss
                        total_trades_so_far = len(recent_wins) + len(recent_losses)
                        win_rate = len(recent_wins) / total_trades_so_far
                        avg_win = sum(recent_wins) / len(recent_wins) if recent_wins else 1
                        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 1
                        
                        if avg_loss > 0:
                            payoff_ratio = avg_win / avg_loss
                            kelly_pct = win_rate - ((1 - win_rate) / payoff_ratio)
                            
                            # Apply Kelly fraction (e.g., half Kelly)
                            kelly_pct = max(0, kelly_pct * kelly_fraction)
                            
                            # Convert to contracts: kelly% * equity / option_cost
                            option_cost = option_entry * 100  # Cost per contract
                            kelly_contracts = int((kelly_pct * equity) / option_cost)
                            contracts = max(base_contracts, min(kelly_contracts, max_contracts))
                        else:
                            contracts = base_contracts
                    else:
                        # Not enough history, use base size
                        contracts = base_contracts
                    
                    current_trade = {
                        'id': trade_counter,
                        'signal': signal.signal_type.value,
                        'direction': signal.direction.value,
                        'entry_time': bar.timestamp,
                        'entry_price': bar.close,
                        'option_entry': option_entry,
                        'delta': delta,
                        'contracts': contracts,
                        'bars_held': 0,
                        'exit_price': None,
                        'exit_reason': None,
                        'pnl': 0,
                        'high_water_mark': 0
                    }
        
        # Close any remaining trade
        if current_trade and bars:
            last_bar = bars[-1]
            entry_price = current_trade['entry_price']
            exit_price = last_bar['close']
            delta = current_trade.get('delta', target_delta)
            option_entry = current_trade['option_entry']
            contracts = current_trade.get('contracts', base_contracts)
            
            if current_trade['direction'] == 'LONG':
                underlying_move = exit_price - entry_price
            else:
                underlying_move = entry_price - exit_price
            
            option_move = underlying_move * delta
            option_exit = max(0.01, option_entry + option_move)
            pnl = (option_exit - option_entry) * 100 * contracts - commission * 2 * contracts
            
            current_trade['pnl'] = pnl
            current_trade['exit_reason'] = "End of backtest"
            trades.append(current_trade)
            equity += pnl
            equity_curve.append(equity)
        
        # Calculate metrics
        result = OptimizationResult(params=params)
        
        if not trades:
            return result
        
        result.total_trades = len(trades)
        result.winning_trades = len([t for t in trades if t['pnl'] > 0])
        result.losing_trades = len([t for t in trades if t['pnl'] <= 0])
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades * 100
            result.total_pnl = sum(t['pnl'] for t in trades)
            result.avg_trade = result.total_pnl / result.total_trades
            result.avg_bars_held = sum(t['bars_held'] for t in trades) / result.total_trades
        
        # Profit factor
        gross_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # Max drawdown
        peak = starting_capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd
        
        # Sharpe ratio
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 1:
            import statistics
            avg_daily = statistics.mean(daily_returns)
            std_daily = statistics.stdev(daily_returns)
            if std_daily > 0:
                result.sharpe_ratio = (avg_daily / std_daily) * (252 ** 0.5)
        
        # Composite score - prioritize win rate > 60% AND profitability
        # Score formula: (win_rate_score * profit_score * trade_count_score)
        win_rate_score = max(0, (result.win_rate - 40) / 30)  # 0 at 40%, 1 at 70%
        profit_score = max(0, 1 + result.total_pnl / 1000)  # Scales with profit
        trade_score = min(1, result.total_trades / 50)  # Want at least 50 trades
        pf_score = min(2, result.profit_factor) / 2  # Cap at 2
        
        # Bonus for >60% win rate
        win_rate_bonus = 2.0 if result.win_rate >= 60 else 1.0
        
        # Bonus for low drawdown
        dd_score = max(0.5, 1 - (result.max_drawdown / starting_capital))
        
        result.score = win_rate_score * profit_score * trade_score * pf_score * win_rate_bonus * dd_score
        
        return result
        
    except Exception as e:
        logger.debug(f"Error in backtest: {e}")
        return OptimizationResult(params=params)


def worker_backtest(params: Dict, bars: List[Dict]) -> OptimizationResult:
    """Worker function for parallel processing"""
    return run_single_backtest(bars, params)


def fetch_historical_data(days: int = 90) -> List[Dict]:
    """Fetch historical bar data in small chunks to avoid timeouts"""
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    from config import config
    
    print("Connecting to Schwab API...")
    
    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri,
        token_file=config.schwab.token_file
    )
    
    if not auth.is_authenticated:
        print("No valid tokens - starting auth flow...")
        if not auth.authorize_interactive():
            raise RuntimeError("Failed to authenticate with Schwab")
    else:
        auth.refresh_access_token()
    
    client = SchwabClient(auth)
    
    # Calculate trading days needed (roughly 5 trading days per week)
    trading_days_needed = days
    calendar_days_needed = int(days * 1.5)  # Account for weekends
    
    end = datetime.now()
    start = end - timedelta(days=calendar_days_needed)
    
    print(f"Fetching {trading_days_needed} trading days of SPY data...")
    
    all_bars = []
    
    # Schwab API limitation: max 10 days of minute data per request
    # Use 7-day chunks to be safe and avoid edge cases
    chunk_size_days = 7
    
    # Calculate number of chunks needed
    num_chunks = (calendar_days_needed // chunk_size_days) + 2
    
    chunk_end = end
    successful_chunks = 0
    failed_chunks = 0
    start_time = time_module.time()
    last_refresh_time = start_time
    
    for chunk_num in range(num_chunks):
        # Calculate chunk boundaries
        chunk_start = chunk_end - timedelta(days=chunk_size_days)
        
        # Don't go before our start date
        if chunk_start < start:
            chunk_start = start
        
        # Skip if we've gone past our start date
        if chunk_end <= start:
            break
        
        # Proactively refresh token every 5 minutes to avoid expiration mid-fetch
        current_time = time_module.time()
        if current_time - last_refresh_time > 300:  # 5 minutes
            try:
                auth.refresh_access_token()
                last_refresh_time = current_time
            except Exception as e:
                print(f"\n  ⚠ Token refresh failed: {e}")
        
        # Progress bar
        progress = (chunk_num + 1) / num_chunks
        elapsed = time_module.time() - start_time
        if progress > 0:
            eta = (elapsed / progress) - elapsed
        else:
            eta = 0
        
        bar_width = 40
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"\r  [{bar}] {progress*100:5.1f}% | ETA: {eta:5.1f}s", end='', flush=True)
        
        try:
            chunk_bars = client.get_price_history(
                symbol="SPY",
                period_type="day",
                period=10,
                frequency_type="minute",
                frequency=5,
                extended_hours=False,
                start_date=chunk_start,
                end_date=chunk_end
            )
            
            if chunk_bars:
                all_bars = chunk_bars + all_bars
                successful_chunks += 1
                
        except Exception as e:
            failed_chunks += 1
            # If auth error, try to re-authenticate
            if "401" in str(e) or "Unauthorized" in str(e):
                try:
                    print(f"\n  Refreshing token...")
                    auth.refresh_access_token()
                    last_refresh_time = time_module.time()
                except:
                    pass
        
        # Move window back (with 1 day overlap to catch edge cases)
        chunk_end = chunk_start
        
        # Small delay between API calls to avoid rate limiting
        if chunk_num < num_chunks - 1:
            time_module.sleep(0.5)
    
    # Complete progress bar
    print(f"\r  [{'█' * bar_width}] 100.0% | Done!     ")
    
    # Remove duplicates and sort
    seen = set()
    unique_bars = []
    for bar in sorted(all_bars, key=lambda x: x['datetime']):
        bar_key = bar['datetime'].isoformat()
        if bar_key not in seen:
            seen.add(bar_key)
            unique_bars.append(bar)
    
    # Calculate actual trading days
    trading_dates = set(bar['datetime'].date() for bar in unique_bars)
    actual_trading_days = len(trading_dates)
    
    print(f"  ✓ {len(unique_bars):,} bars across {actual_trading_days} trading days")
    
    if failed_chunks > 0:
        print(f"  ⚠ {failed_chunks} chunks failed")
    
    return unique_bars


def run_optimization(
    bars: List[Dict],
    param_grid: Dict = None,
    num_workers: int = None,
    min_trades: int = 30
) -> List[OptimizationResult]:
    """Run parallel optimization"""
    if param_grid is None:
        param_grid = PARAM_GRID_FAST
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    combinations = generate_param_combinations(param_grid)
    total_combos = len(combinations)
    
    print(f"\nOptimizing {total_combos:,} parameter combinations using {num_workers} workers...")
    
    # Create worker function with bars pre-loaded
    worker_fn = partial(worker_backtest, bars=bars)
    
    results = []
    start_time = time_module.time()
    bar_width = 40
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, combinations)):
            results.append(result)
            
            # Progress bar
            progress = (i + 1) / total_combos
            elapsed = time_module.time() - start_time
            if progress > 0:
                eta = (elapsed / progress) - elapsed
            else:
                eta = 0
            
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # Calculate current best
            valid = [r for r in results if r.total_trades >= min_trades and r.win_rate > 0]
            if valid:
                best_wr = max(r.win_rate for r in valid)
                best_pnl = max(r.total_pnl for r in valid)
                status = f"Best: {best_wr:.0f}% WR, ${best_pnl:,.0f}"
            else:
                status = "Searching..."
            
            print(f"\r  [{bar}] {progress*100:5.1f}% | ETA: {eta:5.0f}s | {status}      ", end='', flush=True)
    
    # Complete
    elapsed_total = time_module.time() - start_time
    print(f"\r  [{'█' * bar_width}] 100.0% | Done in {elapsed_total:.1f}s                    ")
    
    # Filter results with minimum trades and sort by score
    valid_results = [r for r in results if r.total_trades >= min_trades]
    valid_results.sort(key=lambda x: -x.score)
    
    print(f"  ✓ {len(valid_results):,} valid configurations (>={min_trades} trades)")
    
    high_wr = len([r for r in valid_results if r.win_rate >= 60])
    if high_wr > 0:
        print(f"  🎯 {high_wr} configurations with 60%+ win rate!")
    
    return valid_results


def run_optimization_with_combos(
    bars: List[Dict],
    combinations: List[Dict],
    num_workers: int = None,
    min_trades: int = 30
) -> List[OptimizationResult]:
    """Run parallel optimization with pre-generated combinations"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    total_combos = len(combinations)
    
    print(f"\nOptimizing {total_combos:,} parameter combinations using {num_workers} workers...")
    
    # Create worker function with bars pre-loaded
    worker_fn = partial(worker_backtest, bars=bars)
    
    results = []
    start_time = time_module.time()
    bar_width = 40
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, combinations)):
            results.append(result)
            
            # Progress bar
            progress = (i + 1) / total_combos
            elapsed = time_module.time() - start_time
            if progress > 0:
                eta = (elapsed / progress) - elapsed
            else:
                eta = 0
            
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # Calculate current best
            valid = [r for r in results if r.total_trades >= min_trades and r.win_rate > 0]
            if valid:
                best_wr = max(r.win_rate for r in valid)
                best_pnl = max(r.total_pnl for r in valid)
                status = f"Best: {best_wr:.0f}% WR, ${best_pnl:,.0f}"
            else:
                status = "Searching..."
            
            print(f"\r  [{bar}] {progress*100:5.1f}% | ETA: {eta:5.0f}s | {status}      ", end='', flush=True)
    
    # Complete
    elapsed_total = time_module.time() - start_time
    print(f"\r  [{'█' * bar_width}] 100.0% | Done in {elapsed_total:.1f}s                    ")
    
    # Filter results with minimum trades and sort by score
    valid_results = [r for r in results if r.total_trades >= min_trades]
    valid_results.sort(key=lambda x: -x.score)
    
    print(f"  ✓ {len(valid_results):,} valid configurations (>={min_trades} trades)")
    
    high_wr = len([r for r in valid_results if r.win_rate >= 60])
    if high_wr > 0:
        print(f"  🎯 {high_wr} configurations with 60%+ win rate!")
    
    return valid_results


def print_results(results: List[OptimizationResult], top_n: int = 10):
    """Print top results"""
    print("\n" + "=" * 90)
    print("                              OPTIMIZATION RESULTS")
    print("=" * 90)
    
    # Filter for 60%+ win rate first
    high_wr = [r for r in results if r.win_rate >= 60]
    
    if high_wr:
        print(f"\n🎯 Found {len(high_wr)} configurations with 60%+ win rate!\n")
        display_results = high_wr[:top_n]
    else:
        print(f"\n⚠️  No configurations achieved 60% win rate. Showing best available:\n")
        display_results = results[:top_n]
    
    for i, r in enumerate(display_results, 1):
        print(f"\n{'─' * 90}")
        print(f"RANK #{i} | Score: {r.score:.2f}")
        print(f"{'─' * 90}")
        print(f"  Win Rate:      {r.win_rate:.1f}%  ({r.winning_trades}W / {r.losing_trades}L)")
        print(f"  Total P&L:     ${r.total_pnl:,.2f}")
        print(f"  Avg Trade:     ${r.avg_trade:,.2f}")
        print(f"  Profit Factor: {r.profit_factor:.2f}")
        print(f"  Sharpe Ratio:  {r.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:  ${r.max_drawdown:,.2f}")
        print(f"  Avg Hold:      {r.avg_bars_held:.1f} bars")
        print(f"  Total Trades:  {r.total_trades}")
        print(f"\n  Signal Parameters:")
        for k, v in r.params.items():
            if k.startswith('enable_') or k in ['signal_cooldown_bars', 'min_confirmation_bars', 
                                                   'sustained_bars_required', 'volume_threshold',
                                                   'use_time_filter', 'use_or_bias_filter']:
                print(f"    {k}: {v}")
        
        # Position sizing params
        pos_params = {k: v for k, v in r.params.items() 
                      if k in ['kelly_fraction', 'max_contracts']}
        if pos_params:
            print(f"\n  Position Sizing (Kelly):")
            for k, v in pos_params.items():
                if k == 'kelly_fraction':
                    if v == 0:
                        print(f"    {k}: {v} (fixed size)")
                    elif v == 0.25:
                        print(f"    {k}: {v} (quarter Kelly)")
                    elif v == 0.5:
                        print(f"    {k}: {v} (half Kelly)")
                    elif v == 0.75:
                        print(f"    {k}: {v} (3/4 Kelly)")
                    elif v == 1.0:
                        print(f"    {k}: {v} (full Kelly)")
                    else:
                        print(f"    {k}: {v}")
                else:
                    print(f"    {k}: {v}")
        
        # Risk management params
        risk_params = {k: v for k, v in r.params.items() 
                      if 'stop' in k or 'profit' in k or 'trailing' in k or 'hold' in k}
        if risk_params:
            print(f"\n  Risk Management:")
            for k, v in risk_params.items():
                print(f"    {k}: {v}")
    
    print("\n" + "=" * 90)
    
    # Generate config snippet for best result
    if display_results:
        best = display_results[0]
        print("\n📋 RECOMMENDED CONFIG (copy to config.py):\n")
        print("@dataclass")
        print("class SignalConfig:")
        print(f"    signal_cooldown_bars: int = {best.params.get('signal_cooldown_bars', 8)}")
        print(f"    min_confirmation_bars: int = {best.params.get('min_confirmation_bars', 2)}")
        print(f"    sustained_bars_required: int = {best.params.get('sustained_bars_required', 3)}")
        print(f"    volume_threshold: float = {best.params.get('volume_threshold', 1.3)}")
        print(f"    use_or_bias_filter: bool = {best.params.get('use_or_bias_filter', True)}")
        print(f"    ")
        print(f"    # Signal enables")
        print(f"    enable_val_bounce: bool = {best.params.get('enable_val_bounce', True)}")
        print(f"    enable_vah_rejection: bool = {best.params.get('enable_vah_rejection', True)}")
        print(f"    enable_poc_reclaim: bool = {best.params.get('enable_poc_reclaim', True)}")
        print(f"    enable_poc_breakdown: bool = {best.params.get('enable_poc_breakdown', True)}")
        print(f"    enable_breakout: bool = {best.params.get('enable_breakout', True)}")
        print(f"    enable_breakdown: bool = {best.params.get('enable_breakdown', True)}")
        print(f"    enable_sustained_breakout: bool = {best.params.get('enable_sustained_breakout', True)}")
        print(f"    enable_sustained_breakdown: bool = {best.params.get('enable_sustained_breakdown', True)}")
        print()
        print("@dataclass")
        print("class TradingConfig:")
        print(f"    # Position Sizing (Kelly Criterion)")
        kelly = best.params.get('kelly_fraction', 0.0)
        print(f"    kelly_fraction: float = {kelly}  # 0=fixed, 0.5=half Kelly, 1.0=full Kelly")
        print(f"    max_contracts: int = {best.params.get('max_contracts', 5)}")
        print(f"    ")
        print(f"    # Risk Management")
        print(f"    enable_stop_loss: bool = {best.params.get('enable_stop_loss', False)}")
        print(f"    stop_loss_percent: float = {best.params.get('stop_loss_percent', 50.0)}")
        print(f"    enable_take_profit: bool = {best.params.get('enable_take_profit', False)}")
        print(f"    take_profit_percent: float = {best.params.get('take_profit_percent', 100.0)}")
        print(f"    enable_trailing_stop: bool = {best.params.get('enable_trailing_stop', False)}")
        print(f"    trailing_stop_percent: float = {best.params.get('trailing_stop_percent', 25.0)}")
        print(f"    trailing_stop_activation: float = {best.params.get('trailing_stop_activation', 50.0)}")
        print(f"    min_hold_bars: int = {best.params.get('min_hold_bars', 0)}")
        print()


def export_results(results: List[OptimizationResult], filename: str):
    """Export results to CSV"""
    with open(filename, 'w', newline='') as f:
        # Build header from first result
        if not results:
            return
        
        param_keys = list(results[0].params.keys())
        headers = ['rank', 'score', 'win_rate', 'total_pnl', 'total_trades', 
                   'profit_factor', 'sharpe_ratio', 'max_drawdown', 'avg_trade', 
                   'avg_bars_held'] + param_keys
        
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, r in enumerate(results, 1):
            row = [
                i, f"{r.score:.4f}", f"{r.win_rate:.1f}", f"{r.total_pnl:.2f}",
                r.total_trades, f"{r.profit_factor:.2f}", f"{r.sharpe_ratio:.2f}",
                f"{r.max_drawdown:.2f}", f"{r.avg_trade:.2f}", f"{r.avg_bars_held:.1f}"
            ]
            row += [r.params.get(k, '') for k in param_keys]
            writer.writerow(row)
    
    logger.info(f"Exported {len(results)} results to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--days', type=int, default=90, help='Days of historical data to use')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--top', type=int, default=10, help='Number of top results to show')
    parser.add_argument('--output', type=str, help='Export all results to CSV')
    parser.add_argument('--full', action='store_true', help='Use full parameter grid (slower)')
    parser.add_argument('--risk', action='store_true', help='Focus on risk management optimization')
    parser.add_argument('--min-trades', type=int, default=30, help='Minimum trades for valid result')
    
    args = parser.parse_args()
    
    # Fetch data
    try:
        bars = fetch_historical_data(days=args.days)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        sys.exit(1)
    
    if not bars:
        print("No data returned")
        sys.exit(1)
    
    # Select parameter grid or generate combinations
    if args.risk:
        # Use smart generator for risk optimization (no redundant combos)
        combinations = generate_risk_combinations()
        print(f"Using RISK-focused optimization ({len(combinations)} combinations)")
        
        # Run optimization with pre-generated combinations
        results = run_optimization_with_combos(
            bars=bars,
            combinations=combinations,
            num_workers=args.workers,
            min_trades=args.min_trades
        )
    else:
        if args.full:
            param_grid = PARAM_GRID
            print("Using FULL parameter grid")
        else:
            param_grid = PARAM_GRID_FAST
            print("Using FAST parameter grid")
        
        # Run optimization with grid
        results = run_optimization(
            bars=bars,
            param_grid=param_grid,
            num_workers=args.workers,
            min_trades=args.min_trades
        )
    
    # Print results
    print_results(results, top_n=args.top)
    
    # Export if requested
    if args.output:
        export_results(results, args.output)


if __name__ == "__main__":
    main()