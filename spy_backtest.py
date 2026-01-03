#!/usr/bin/env python3
"""
SPY Butterfly Backtest

Uses the same signal detection as the original backtest that got 71% win rate.
Tests the butterfly credit stacking strategy on SPY with 1-wide strikes.

For backtesting purposes, we pretend SPY is cash-settled like SPX.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import csv
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SCHWAB API
# =============================================================================

def connect_to_schwab():
    """Connect to Schwab API"""
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    
    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri,
        token_file=config.schwab.token_file
    )
    
    if not auth.is_authenticated:
        logger.error("Not authenticated. Run schwab_auth.py first.")
        return None
    
    auth.refresh_access_token()
    return SchwabClient(auth)


def fetch_spy_bars(client, days: int = 365) -> List[Bar]:
    """Fetch SPY 5-min bars in chunks"""
    
    logger.info(f"Fetching {days} days of SPY 5-min bars...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    chunk_size_days = 7
    num_chunks = (days // chunk_size_days) + 2
    
    all_bars = []
    chunk_end = end_date
    
    for chunk_num in range(num_chunks):
        chunk_start = chunk_end - timedelta(days=chunk_size_days)
        
        if chunk_start < start_date:
            chunk_start = start_date
        
        if chunk_end <= start_date:
            break
        
        try:
            bars_data = client.get_price_history(
                symbol='SPY',
                period_type='day',
                period=10,
                frequency_type='minute',
                frequency=5,
                extended_hours=False,
                start_date=chunk_start,
                end_date=chunk_end
            )
            
            for bar_data in bars_data:
                ts = bar_data.get('datetime')
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts / 1000)
                all_bars.append(Bar(
                    timestamp=ts,
                    open=bar_data['open'],
                    high=bar_data['high'],
                    low=bar_data['low'],
                    close=bar_data['close'],
                    volume=bar_data.get('volume', 0)
                ))
            
        except Exception as e:
            logger.error(f"Error fetching chunk: {e}")
        
        chunk_end = chunk_start
    
    # Sort and dedupe
    all_bars.sort(key=lambda b: b.timestamp)
    seen = set()
    unique_bars = []
    for bar in all_bars:
        key = bar.timestamp
        if key not in seen:
            seen.add(key)
            unique_bars.append(bar)
    
    logger.info(f"Fetched {len(unique_bars)} unique SPY bars")
    return unique_bars


# =============================================================================
# OPTION PRICER
# =============================================================================

class SPYOptionPricer:
    """
    Price SPY 0DTE options - calibrated to real market prices.
    
    SPY 0DTE (5 hrs left, ~$600 underlying):
    - ATM: ~$1.20, delta ~0.50
    - 1 OTM: ~$0.55-0.60, delta ~0.35
    - 2 OTM: ~$0.25-0.30, delta ~0.20
    
    Extrinsic decays ~50% per $1 OTM (steeper than BS model)
    """
    
    def get_option_price(
        self,
        underlying: float,
        strike: float,
        option_type: str,
        minutes_to_expiry: int
    ) -> Tuple[float, float]:
        """Get option bid/ask prices"""
        
        t = max(0.001, minutes_to_expiry / 390)
        sqrt_t = math.sqrt(t)
        
        # Moneyness
        if option_type == 'C':
            moneyness = underlying - strike
        else:
            moneyness = strike - underlying
        
        intrinsic = max(0, moneyness)
        distance = abs(moneyness) if moneyness < 0 else 0
        
        # ATM extrinsic: ~$1.20 at open, ~$0.30 at 1hr left
        atm_extrinsic = 1.20 * sqrt_t
        
        # OTM extrinsic decay: ~50% per $1 OTM
        if distance <= 0.05:
            extrinsic = atm_extrinsic
        else:
            # 1 OTM = 50% of ATM
            # 2 OTM = 25% of ATM
            decay = 0.7  # ~50% per $1
            extrinsic = atm_extrinsic * math.exp(-decay * distance)
        
        extrinsic = max(0.02, extrinsic)
        mid = intrinsic + extrinsic
        
        # SPY has penny-wide spreads
        spread = 0.01
        
        bid = max(0.01, round(mid - spread / 2, 2))
        ask = round(mid + spread / 2, 2)
        
        return bid, ask


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class FillStatus(Enum):
    PENDING = "PENDING"
    FILLED_TRAILING_STOP = "TRAILING_STOP"  # Hit trailing stop
    FILLED_TIMEOUT = "TIMEOUT"  # Max time reached


class TrailingStopState:
    """Track trailing stop for a pending butterfly"""
    def __init__(self, wing_cost: float, activation_pct: float, stop_pct: float):
        self.wing_cost = wing_cost
        self.activation_pct = activation_pct
        self.stop_pct = stop_pct
        
        self.peak_credit = 0.0
        self.is_activated = False
        self.stop_level = 0.0
    
    def update(self, current_credit: float) -> Tuple[bool, float]:
        """
        Update trailing stop state with current credit.
        Returns (should_fill, exit_credit)
        """
        # Track peak
        if current_credit > self.peak_credit:
            self.peak_credit = current_credit
            peak_pct = (self.peak_credit / self.wing_cost - 1) if self.wing_cost > 0 else 0
            
            # Check activation
            if not self.is_activated and peak_pct >= self.activation_pct:
                self.is_activated = True
                self.stop_level = self.peak_credit * (1 - self.stop_pct)
            elif self.is_activated:
                # Update stop level (trails up)
                self.stop_level = self.peak_credit * (1 - self.stop_pct)
        
        # Check if stop triggered - exit at STOP LEVEL (that's where limit order fills)
        if self.is_activated and current_credit <= self.stop_level:
            return True, self.stop_level
        
        return False, current_credit


@dataclass
class ButterflyTrade:
    """Single butterfly trade"""
    id: str
    direction: Direction
    signal_type: SignalType
    
    signal_time: datetime
    wing_fill_time: datetime
    middle_fill_time: Optional[datetime] = None
    
    entry_price: float = 0
    lower_strike: float = 0
    middle_strike: float = 0
    upper_strike: float = 0
    
    wing_cost: float = 0
    middle_credit: float = 0
    net_credit: float = 0
    
    # Trailing stop tracking
    peak_credit: float = 0
    trailing_stop: Optional[TrailingStopState] = None
    
    contracts: int = 1
    
    fill_status: FillStatus = FillStatus.PENDING
    fill_minutes: int = 0
    fill_price: float = 0
    move_points: float = 0
    
    settlement_price: float = 0
    pnl: float = 0
    
    equity_at_entry: float = 0
    
    @property
    def option_type(self) -> str:
        return 'C' if self.direction == Direction.LONG else 'P'
    
    @property
    def credit_pct(self) -> float:
        if self.wing_cost <= 0:
            return 0
        return (self.net_credit / self.wing_cost) * 100
    
    def calculate_settlement(self, settle_price: float):
        """Calculate P&L at settlement (pretend cash-settled like SPX)"""
        self.settlement_price = settle_price
        
        # 1-wide butterfly intrinsic
        if self.option_type == 'C':
            if settle_price <= self.lower_strike:
                intrinsic = 0
            elif settle_price >= self.upper_strike:
                intrinsic = 0
            elif settle_price <= self.middle_strike:
                intrinsic = settle_price - self.lower_strike
            else:
                intrinsic = self.upper_strike - settle_price
        else:
            if settle_price >= self.upper_strike:
                intrinsic = 0
            elif settle_price <= self.lower_strike:
                intrinsic = 0
            elif settle_price >= self.middle_strike:
                intrinsic = self.upper_strike - settle_price
            else:
                intrinsic = settle_price - self.lower_strike
        
        # P&L = (credit + intrinsic) * 100 * contracts
        self.pnl = (self.net_credit + intrinsic) * 100 * self.contracts


@dataclass 
class BacktestConfig:
    # Trailing stop settings (from optimized config)
    trailing_stop_activation: float = 0.32  # Activate after 32% gain
    trailing_stop_percent: float = 0.21     # Sell after 21% pullback from peak
    
    max_fill_minutes: int = 30  # Max time to wait before completing
    starting_capital: float = 500.0
    
    # Kelly sizing
    base_kelly_fraction: float = 0.10
    max_kelly_fraction: float = 0.25
    kelly_scale_threshold: float = 2.0
    
    max_contracts: int = 10
    min_contracts: int = 1


def calculate_position_size(
    equity: float,
    wing_cost: float,
    config: BacktestConfig,
    win_rate: float = 0.50
) -> int:
    """Calculate contracts based on Kelly"""
    
    if equity <= 0:
        return 0
    
    # Simple Kelly: scale with equity
    equity_multiple = equity / config.starting_capital
    
    if equity_multiple <= 1:
        kelly_frac = config.base_kelly_fraction
    else:
        scale = min(1.0, (equity_multiple - 1) / (config.kelly_scale_threshold - 1))
        kelly_frac = config.base_kelly_fraction + (config.max_kelly_fraction - config.base_kelly_fraction) * scale
    
    # Max loss per contract = wing_cost * 100
    max_loss = wing_cost * 100
    
    if max_loss <= 0:
        return config.min_contracts
    
    position_dollars = equity * kelly_frac
    contracts = int(position_dollars / max_loss)
    
    return max(config.min_contracts, min(config.max_contracts, contracts))


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_spy_backtest(bars: List[Bar], bt_config: BacktestConfig) -> Dict:
    """Run SPY butterfly backtest"""
    
    print("*** BACKTEST VERSION 2 - EXIT AT STOP LEVEL ***")
    
    logger.info(f"Running SPY butterfly backtest on {len(bars)} bars...")
    logger.info(f"Starting capital: ${bt_config.starting_capital:,.2f}")
    
    pricer = SPYOptionPricer()
    
    # Use signal config from config.py
    sig = config.signal
    logger.info(f"Signal config: cooldown={sig.signal_cooldown_bars}, vol_thresh={sig.volume_threshold}")
    
    detector = SignalDetector(
        length_period=sig.length_period,
        value_area_percent=sig.value_area_percent,
        volume_threshold=sig.volume_threshold,
        use_relaxed_volume=sig.use_relaxed_volume,
        min_confirmation_bars=sig.min_confirmation_bars,
        sustained_bars_required=sig.sustained_bars_required,
        signal_cooldown_bars=sig.signal_cooldown_bars,
        use_or_bias_filter=sig.use_or_bias_filter,
        or_buffer_points=sig.or_buffer_points,
        rth_only=True,
        enable_val_bounce=sig.enable_val_bounce,
        enable_poc_reclaim=sig.enable_poc_reclaim,
        enable_breakout=sig.enable_breakout,
        enable_sustained_breakout=sig.enable_sustained_breakout,
        enable_prior_val_bounce=True,
        enable_prior_poc_reclaim=False,
        enable_vah_rejection=sig.enable_vah_rejection,
        enable_poc_breakdown=sig.enable_poc_breakdown,
        enable_breakdown=sig.enable_breakdown,
        enable_sustained_breakdown=sig.enable_sustained_breakdown,
        enable_prior_vah_rejection=True,
        enable_prior_poc_breakdown=False
    )
    
    # Group by date
    bars_by_date = defaultdict(list)
    for bar in bars:
        bars_by_date[bar.timestamp.date()].append(bar)
    
    dates = sorted(bars_by_date.keys())
    logger.info(f"Processing {len(dates)} trading days")
    
    all_trades: List[ButterflyTrade] = []
    daily_results = {}
    
    equity = bt_config.starting_capital
    equity_curve = [equity]
    wins = []
    losses = []
    
    for day in dates:
        day_bars = sorted(bars_by_date[day], key=lambda b: b.timestamp)
        if len(day_bars) < 10:
            continue
        
        settlement_price = day_bars[-1].close
        detector.reset_session()
        
        pending: List[ButterflyTrade] = []
        completed: List[ButterflyTrade] = []
        
        market_close = datetime.combine(day, dt_time(16, 0))
        
        for bar in day_bars:
            bar_time = bar.timestamp
            if bar_time.tzinfo:
                bar_time = bar_time.replace(tzinfo=None)
            
            minutes_to_close = max(1, (market_close - bar_time).total_seconds() / 60)
            price = bar.close
            
            signal = detector.add_bar(bar)
            
            # Check pending positions
            still_pending = []
            for trade in pending:
                minutes_waiting = (bar_time - trade.wing_fill_time).total_seconds() / 60
                
                # Get current middle price
                m_bid, m_ask = pricer.get_option_price(
                    price, 
                    trade.middle_strike, 
                    trade.option_type, 
                    int(minutes_to_close)
                )
                
                current_credit = m_bid * 2
                
                # Update trailing stop - returns (triggered, exit_credit)
                should_fill, exit_credit = trade.trailing_stop.update(current_credit)
                
                # Track move
                if trade.direction == Direction.LONG:
                    move = price - trade.entry_price
                else:
                    move = trade.entry_price - price
                
                if should_fill:
                    # Trailing stop triggered - use stop level as exit credit
                    trade.middle_fill_time = bar_time
                    trade.middle_credit = exit_credit  # Exit at stop level, not current
                    trade.net_credit = exit_credit - trade.wing_cost
                    trade.peak_credit = trade.trailing_stop.peak_credit
                    trade.fill_status = FillStatus.FILLED_TRAILING_STOP
                    trade.fill_minutes = int(minutes_waiting)
                    trade.fill_price = price
                    trade.move_points = move
                    completed.append(trade)
                    
                    peak_pct = (trade.peak_credit / trade.wing_cost - 1) * 100
                    exit_pct = (exit_credit / trade.wing_cost - 1) * 100
                    logger.info(f"  ✓ TRAILING STOP after {int(minutes_waiting)} min!")
                    logger.info(f"    SPY moved {move:+.2f} to {price:.2f}")
                    logger.info(f"    Peak credit: ${trade.peak_credit:.2f} ({peak_pct:+.0f}%)")
                    logger.info(f"    Stop level exit: ${exit_credit:.2f} ({exit_pct:+.0f}%)")
                    logger.info(f"    Net credit: ${trade.net_credit:.2f}")
                    
                elif minutes_waiting >= bt_config.max_fill_minutes or minutes_to_close < 15:
                    # Timeout - complete at current price
                    trade.middle_fill_time = bar_time
                    trade.middle_credit = current_credit
                    trade.net_credit = current_credit - trade.wing_cost
                    trade.peak_credit = trade.trailing_stop.peak_credit
                    trade.fill_status = FillStatus.FILLED_TIMEOUT
                    trade.fill_minutes = int(minutes_waiting)
                    trade.fill_price = price
                    trade.move_points = move
                    completed.append(trade)
                    
                    credit_pct = (current_credit / trade.wing_cost - 1) * 100 if trade.wing_cost > 0 else 0
                    peak_pct = (trade.peak_credit / trade.wing_cost - 1) * 100 if trade.peak_credit > 0 else 0
                    logger.info(f"  ✗ TIMEOUT after {int(minutes_waiting)} min")
                    logger.info(f"    SPY moved {move:+.2f} to {price:.2f}")
                    logger.info(f"    Peak credit: ${trade.peak_credit:.2f} ({peak_pct:+.0f}%)")
                    logger.info(f"    Exit credit: ${current_credit:.2f} ({credit_pct:+.0f}%)")
                    logger.info(f"    Net: ${trade.net_credit:.2f} ({'PROFIT' if trade.net_credit > 0 else 'LOSS'})")
                    
                else:
                    still_pending.append(trade)
            
            pending = still_pending
            
            # New signal
            if signal and signal.direction in [Direction.LONG, Direction.SHORT]:
                if minutes_to_close < 60:
                    continue
                if equity <= 0:
                    continue
                
                # Calculate strikes (1-wide butterfly)
                # SPY strikes are $1 apart
                atm = round(price)
                
                if signal.direction == Direction.LONG:
                    lower = atm
                    middle = atm + 1
                    upper = atm + 2
                    opt_type = 'C'
                else:
                    upper = atm
                    middle = atm - 1
                    lower = atm - 2
                    opt_type = 'P'
                
                # Get wing prices
                l_bid, l_ask = pricer.get_option_price(price, lower, opt_type, int(minutes_to_close))
                m_bid, m_ask = pricer.get_option_price(price, middle, opt_type, int(minutes_to_close))
                u_bid, u_ask = pricer.get_option_price(price, upper, opt_type, int(minutes_to_close))
                
                wing_cost = l_ask + u_ask
                current_middle_credit = m_bid * 2
                activation_credit = wing_cost * (1 + bt_config.trailing_stop_activation)
                
                logger.info(f"  SIGNAL: {signal.direction.value} {signal.signal_type.value} @ SPY={price:.2f}")
                logger.info(f"    Strikes: {lower}/{middle}/{upper} {opt_type}")
                logger.info(f"    Lower {opt_type} bid/ask: ${l_bid:.2f}/${l_ask:.2f}")
                logger.info(f"    Middle {opt_type} bid/ask: ${m_bid:.2f}/${m_ask:.2f}")
                logger.info(f"    Upper {opt_type} bid/ask: ${u_bid:.2f}/${u_ask:.2f}")
                logger.info(f"    Wing cost: ${wing_cost:.2f}")
                logger.info(f"    Current 2x middle: ${current_middle_credit:.2f}")
                logger.info(f"    TS activation ({bt_config.trailing_stop_activation*100:.0f}%): ${activation_credit:.2f}")
                logger.info(f"    Minutes to close: {int(minutes_to_close)}")
                
                # Position size
                win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.50
                contracts = calculate_position_size(equity, wing_cost, bt_config, win_rate)
                
                if contracts <= 0:
                    continue
                
                trade = ButterflyTrade(
                    id=f"{day.strftime('%Y%m%d')}-{len(all_trades)+len(pending)+1:03d}",
                    direction=signal.direction,
                    signal_type=signal.signal_type,
                    signal_time=bar_time,
                    wing_fill_time=bar_time,
                    entry_price=price,
                    lower_strike=lower,
                    middle_strike=middle,
                    upper_strike=upper,
                    wing_cost=wing_cost,
                    contracts=contracts,
                    equity_at_entry=equity,
                    trailing_stop=TrailingStopState(
                        wing_cost=wing_cost,
                        activation_pct=bt_config.trailing_stop_activation,
                        stop_pct=bt_config.trailing_stop_percent
                    )
                )
                
                # Initialize peak with current 2x middle
                trade.trailing_stop.update(current_middle_credit)
                logger.info(f"    Initial 2x middle: ${current_middle_credit:.2f}, peak set to: ${trade.trailing_stop.peak_credit:.2f}")
                
                pending.append(trade)
        
        # Complete remaining pending at EOD
        for trade in pending:
            m_bid, _ = pricer.get_option_price(settlement_price, trade.middle_strike, trade.option_type, 1)
            trade.middle_credit = m_bid * 2
            trade.net_credit = trade.middle_credit - trade.wing_cost
            trade.fill_status = FillStatus.FILLED_TIMEOUT
            trade.fill_price = settlement_price
            completed.append(trade)
        
        # Calculate P&L
        day_pnl = 0
        for trade in completed:
            trade.calculate_settlement(settlement_price)
            day_pnl += trade.pnl
            
            credit_pct = (trade.net_credit / trade.wing_cost) * 100 if trade.wing_cost > 0 else 0
            peak_pct = (trade.peak_credit / trade.wing_cost - 1) * 100 if trade.peak_credit > 0 and trade.wing_cost > 0 else 0
            
            logger.info(f"  TRADE COMPLETE: {trade.id}")
            logger.info(f"    Direction: {trade.direction.value} | Signal: {trade.signal_type.value}")
            logger.info(f"    Entry: SPY @ ${trade.entry_price:.2f} | Exit: SPY @ ${trade.fill_price:.2f}")
            logger.info(f"    Move: {trade.move_points:+.2f} pts in {trade.fill_minutes} min")
            logger.info(f"    Strikes: {trade.lower_strike}/{trade.middle_strike}/{trade.upper_strike}")
            logger.info(f"    Wing cost: ${trade.wing_cost:.2f}")
            logger.info(f"    Peak 2x mid: ${trade.peak_credit:.2f} ({peak_pct:+.0f}%)")
            logger.info(f"    Exit 2x mid: ${trade.middle_credit:.2f}")
            logger.info(f"    Net credit: ${trade.net_credit:.2f} ({credit_pct:+.0f}%)")
            logger.info(f"    Fill status: {trade.fill_status.value}")
            logger.info(f"    Settlement: ${settlement_price:.2f}")
            logger.info(f"    FINAL P&L: ${trade.pnl:.2f}")
            logger.info(f"    ---")
            
            if trade.pnl > 0:
                wins.append(trade.pnl)
            else:
                losses.append(trade.pnl)
        
        equity += day_pnl
        equity_curve.append(equity)
        
        all_trades.extend(completed)
        
        trailing_stop_fills = sum(1 for t in completed if t.fill_status == FillStatus.FILLED_TRAILING_STOP)
        total_contracts = sum(t.contracts for t in completed)
        
        daily_results[day] = {
            'trades': len(completed),
            'contracts': total_contracts,
            'trailing_stop_fills': trailing_stop_fills,
            'pnl': day_pnl,
            'equity': equity
        }
        
        if completed:
            logger.info(f"{day}: {len(completed)} trades ({total_contracts} cts), "
                       f"TS fills: {trailing_stop_fills}, P&L: ${day_pnl:,.2f}, Equity: ${equity:,.2f}")
    
    # Summary
    total_trades = len(all_trades)
    if total_trades == 0:
        return {'error': 'No trades'}
    
    trailing_stop_fills = [t for t in all_trades if t.fill_status == FillStatus.FILLED_TRAILING_STOP]
    timeout_fills = [t for t in all_trades if t.fill_status == FillStatus.FILLED_TIMEOUT]
    total_pnl = sum(t.pnl for t in all_trades)
    winners = [t for t in all_trades if t.pnl > 0]
    total_contracts = sum(t.contracts for t in all_trades)
    
    # Credit stats
    avg_credit_pct = sum(t.credit_pct for t in all_trades) / total_trades
    avg_peak_pct = sum((t.peak_credit / t.wing_cost - 1) * 100 for t in all_trades if t.wing_cost > 0) / total_trades
    
    # Max drawdown
    max_dd = 0
    peak = equity_curve[0]
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'start_date': dates[0],
        'end_date': dates[-1],
        'trading_days': len(dates),
        'total_trades': total_trades,
        'total_contracts': total_contracts,
        'trailing_stop_fills': len(trailing_stop_fills),
        'trailing_stop_rate': len(trailing_stop_fills) / total_trades * 100,
        'timeout_fills': len(timeout_fills),
        'avg_credit_pct': avg_credit_pct,
        'avg_peak_credit_pct': avg_peak_pct,
        'starting_capital': bt_config.starting_capital,
        'ending_capital': equity,
        'total_return': (equity / bt_config.starting_capital - 1) * 100,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / total_trades,
        'win_rate': len(winners) / total_trades * 100,
        'best_trade': max(t.pnl for t in all_trades),
        'worst_trade': min(t.pnl for t in all_trades),
        'max_drawdown': max_dd * 100,
        'equity_curve': equity_curve,
        'daily_results': daily_results,
        'trades': all_trades
    }


def print_results(results: Dict):
    """Print results"""
    
    print("\n" + "=" * 70)
    print("🦋 SPY BUTTERFLY BACKTEST (Trailing Stop)")
    print("=" * 70)
    
    print(f"\nPeriod: {results['start_date']} to {results['end_date']}")
    print(f"Trading Days: {results['trading_days']}")
    print(f"Total Trades: {results['total_trades']} ({results['total_contracts']} contracts)")
    
    print(f"\n{'─' * 40}")
    print("CAPITAL GROWTH")
    print(f"{'─' * 40}")
    print(f"  Starting Capital: ${results['starting_capital']:,.2f}")
    print(f"  Ending Capital:   ${results['ending_capital']:,.2f}")
    print(f"  Total Return:     {results['total_return']:+.1f}%")
    print(f"  Max Drawdown:     {results['max_drawdown']:.1f}%")
    
    print(f"\n{'─' * 40}")
    print("FILL STATISTICS")
    print(f"{'─' * 40}")
    print(f"  Trailing Stop Fills: {results['trailing_stop_fills']} ({results['trailing_stop_rate']:.1f}%)")
    print(f"  Timeout Fills:       {results['timeout_fills']}")
    print(f"  Avg Credit %:        {results['avg_credit_pct']:+.1f}%")
    print(f"  Avg Peak Credit %:   {results['avg_peak_credit_pct']:+.1f}%")
    
    print(f"\n{'─' * 40}")
    print("P&L STATISTICS")
    print(f"{'─' * 40}")
    print(f"  Total P&L:    ${results['total_pnl']:,.2f}")
    print(f"  Avg P&L:      ${results['avg_pnl']:,.2f}")
    print(f"  Win Rate:     {results['win_rate']:.1f}%")
    print(f"  Best Trade:   ${results['best_trade']:,.2f}")
    print(f"  Worst Trade:  ${results['worst_trade']:,.2f}")
    
    print(f"\n{'─' * 40}")
    print("DAILY BREAKDOWN (showing first/last 10)")
    print(f"{'─' * 40}")
    
    sorted_days = sorted(results['daily_results'].items())
    days_to_show = sorted_days[:10] + ([('...', {})] if len(sorted_days) > 20 else []) + sorted_days[-10:]
    
    for day, stats in days_to_show:
        if day == '...':
            print("  ...")
            continue
        contracts = stats.get('contracts', stats['trades'])
        ts_fills = stats.get('trailing_stop_fills', 0)
        print(f"  {day}: {stats['trades']} trades ({contracts} cts) "
              f"[{ts_fills} TS] "
              f"P&L: ${stats['pnl']:>8,.2f} | Equity: ${stats['equity']:>10,.2f}")


def save_bars(bars: List[Bar], filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for bar in bars:
            writer.writerow([bar.timestamp.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume])
    logger.info(f"Saved {len(bars)} bars to {filename}")


def load_bars(filename: str) -> List[Bar]:
    bars = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(Bar(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))
    logger.info(f"Loaded {len(bars)} bars from {filename}")
    return bars


def main():
    parser = argparse.ArgumentParser(description='SPY Butterfly Backtest')
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--file', type=str, help='Load from CSV')
    parser.add_argument('--save', type=str, default='spy_bars.csv', help='Save to CSV')
    parser.add_argument('--capital', type=float, default=500, help='Starting capital')
    parser.add_argument('--activation', type=float, default=0.32, help='Trailing stop activation (0.32 = 32%)')
    parser.add_argument('--stop', type=float, default=0.21, help='Trailing stop pullback (0.21 = 21%)')
    parser.add_argument('--max-time', type=int, default=30, help='Max fill minutes')
    
    args = parser.parse_args()
    
    bt_config = BacktestConfig(
        trailing_stop_activation=args.activation,
        trailing_stop_percent=args.stop,
        max_fill_minutes=args.max_time,
        starting_capital=args.capital
    )
    
    if args.file:
        bars = load_bars(args.file)
    else:
        client = connect_to_schwab()
        if not client:
            print("Cannot connect to Schwab.")
            return
        
        bars = fetch_spy_bars(client, args.days)
        save_bars(bars, args.save)
    
    results = run_spy_backtest(bars, bt_config)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print_results(results)


if __name__ == "__main__":
    main()
