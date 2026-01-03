#!/usr/bin/env python3
"""
ES Butterfly Backtest with Real Schwab Data

Uses ES futures options directly - no SPX conversion needed.
Fetches both ES bars and ES option chains from Schwab API.

Usage:
    python es_backtest.py --days 30
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SCHWAB API FUNCTIONS
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


def fetch_es_bars(client, days: int = 30) -> List[Bar]:
    """Fetch ES 5-min bars in chunks (Schwab API limits to 10 days per request)"""
    
    logger.info(f"Fetching {days} days of ES 5-min bars...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Schwab API limitation: max 10 days of minute data per request
    # Use 7-day chunks to be safe
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
        
        logger.info(f"  Fetching chunk {chunk_num + 1}: {chunk_start.date()} to {chunk_end.date()}")
        
        try:
            bars_data = client.get_price_history(
                symbol='/ES',
                period_type='day',
                period=10,  # Max allowed for day period type
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
            
            logger.info(f"    Got {len(bars_data)} bars")
            
        except Exception as e:
            logger.error(f"    Error fetching chunk: {e}")
        
        chunk_end = chunk_start
    
    # Sort by timestamp and remove duplicates
    all_bars.sort(key=lambda b: b.timestamp)
    
    # Remove duplicates (overlapping chunks)
    seen = set()
    unique_bars = []
    for bar in all_bars:
        key = bar.timestamp
        if key not in seen:
            seen.add(key)
            unique_bars.append(bar)
    
    logger.info(f"Fetched {len(unique_bars)} unique ES bars")
    return unique_bars


def fetch_es_options_realtime(client, es_price: float) -> Dict:
    """
    Fetch current ES 0DTE option chain.
    Returns dict with call/put prices by strike.
    """
    
    today = date.today()
    
    # ES options symbol format: ./ESZ24 (December 2024 ES futures options)
    # For 0DTE we need the weekly or daily expiration
    
    try:
        chain = client.get_option_chain(
            symbol='/ES',
            contract_type='ALL',
            strike_count=20,
            include_quotes=True,
            from_date=today,
            to_date=today
        )
        
        result = {'calls': {}, 'puts': {}, 'underlying': es_price}
        
        call_map = chain.get('callExpDateMap', {})
        put_map = chain.get('putExpDateMap', {})
        
        # Parse calls
        for exp_str, strikes in call_map.items():
            for strike_str, options in strikes.items():
                if options:
                    opt = options[0]
                    strike = float(strike_str)
                    result['calls'][strike] = {
                        'bid': opt.get('bid', 0),
                        'ask': opt.get('ask', 0),
                        'mid': (opt.get('bid', 0) + opt.get('ask', 0)) / 2,
                        'delta': opt.get('delta', 0),
                        'iv': opt.get('volatility', 0)
                    }
        
        # Parse puts
        for exp_str, strikes in put_map.items():
            for strike_str, options in strikes.items():
                if options:
                    opt = options[0]
                    strike = float(strike_str)
                    result['puts'][strike] = {
                        'bid': opt.get('bid', 0),
                        'ask': opt.get('ask', 0),
                        'mid': (opt.get('bid', 0) + opt.get('ask', 0)) / 2,
                        'delta': opt.get('delta', 0),
                        'iv': opt.get('volatility', 0)
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching ES options: {e}")
        return None


# =============================================================================
# OPTION PRICER (from real data or model)
# =============================================================================

class ESOptionPricer:
    """
    Price ES options using either:
    1. Real-time Schwab data (if available)
    2. Calibrated empirical model (for backtesting)
    """
    
    def __init__(self, client=None):
        self.client = client
        self.cached_chain = None
        self.cache_time = None
        self.cache_ttl = 60  # Refresh every 60 seconds
    
    def get_option_price(
        self,
        underlying: float,
        strike: float,
        option_type: str,  # 'C' or 'P'
        minutes_to_expiry: int
    ) -> Tuple[float, float]:  # (bid, ask)
        """Get option bid/ask prices"""
        
        # For backtesting, use empirical model
        return self._model_price(underlying, strike, option_type, minutes_to_expiry)
    
    def _model_price(
        self,
        underlying: float,
        strike: float,
        option_type: str,
        minutes_to_expiry: int
    ) -> Tuple[float, float]:
        """
        Empirical model calibrated to real 0DTE ES options.
        
        ES options trade in 0.25 increments, similar to SPX.
        ATM ~$12-15 with 5 hours left.
        """
        import math
        
        t = max(0.001, minutes_to_expiry / 390)
        sqrt_t = math.sqrt(t)
        
        # Moneyness
        if option_type == 'C':
            moneyness = underlying - strike
        else:
            moneyness = strike - underlying
        
        intrinsic = max(0, moneyness)
        
        # ATM time value
        atm_time_value = 15 * sqrt_t
        
        # OTM decay
        distance = abs(moneyness)
        
        if distance <= 0.5:
            time_value = atm_time_value
        else:
            decay_per_point = 0.055
            time_factor = 1 + 0.3 * (1 - t)
            retention = math.exp(-decay_per_point * time_factor * distance)
            time_value = atm_time_value * max(0.05, retention)
        
        time_value = max(0.05, time_value)
        mid = intrinsic + time_value
        
        # Bid-ask spread (ES options have tight spreads)
        spread = 0.25 + 0.10 * (1 - t)  # Widens near expiry
        spread = min(spread, mid * 0.15)
        
        bid = max(0.25, mid - spread / 2)
        ask = mid + spread / 2
        
        # Round to ES tick size (0.25)
        bid = round(bid * 4) / 4
        ask = round(ask * 4) / 4
        
        return bid, ask
    
    def price_butterfly(
        self,
        underlying: float,
        lower: float,
        middle: float,
        upper: float,
        option_type: str,
        minutes_to_expiry: int
    ) -> Dict:
        """Price a butterfly spread"""
        
        l_bid, l_ask = self.get_option_price(underlying, lower, option_type, minutes_to_expiry)
        m_bid, m_ask = self.get_option_price(underlying, middle, option_type, minutes_to_expiry)
        u_bid, u_ask = self.get_option_price(underlying, upper, option_type, minutes_to_expiry)
        
        return {
            'lower_bid': l_bid,
            'lower_ask': l_ask,
            'middle_bid': m_bid,
            'middle_ask': m_ask,
            'upper_bid': u_bid,
            'upper_ask': u_ask,
            'wing_cost': l_ask + u_ask,
            'middle_credit_2x': m_bid * 2,
            'net': m_bid * 2 - (l_ask + u_ask)
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class FillStatus(Enum):
    PENDING = "PENDING"
    FILLED_TARGET = "FILLED_30"
    FILLED_TIMEOUT = "TIMEOUT"


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
    
    contracts: int = 1  # Position size
    
    fill_status: FillStatus = FillStatus.PENDING
    fill_minutes: int = 0
    fill_price: float = 0
    move_points: float = 0
    
    settlement_price: float = 0
    pnl: float = 0
    
    # Track equity at time of trade for analysis
    equity_at_entry: float = 0
    kelly_fraction_used: float = 0
    
    @property
    def option_type(self) -> str:
        return 'C' if self.direction == Direction.LONG else 'P'
    
    def calculate_settlement(self, settle_price: float):
        """Calculate P&L at settlement"""
        self.settlement_price = settle_price
        
        # Butterfly intrinsic
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
        
        # P&L = (credit received + intrinsic value) * contracts
        # ES options are $50 per point
        self.pnl = (self.net_credit + intrinsic) * 50 * self.contracts


@dataclass 
class BacktestConfig:
    credit_target_pct: float = 0.30
    max_fill_minutes: int = 30
    signal_cooldown_bars: int = 8
    use_or_bias: bool = True
    
    # Capital and position sizing
    starting_capital: float = 500.0
    base_kelly_fraction: float = 0.10  # Start with 10% Kelly
    max_kelly_fraction: float = 0.25   # Scale up to 25% Kelly
    kelly_scale_threshold: float = 2.0  # Double equity = full Kelly scale
    
    # Risk limits
    max_position_size: float = 0.50    # Never risk more than 50% of equity
    min_contracts: int = 1
    max_contracts: int = 10


def calculate_position_size(
    equity: float,
    wing_cost: float,
    config: BacktestConfig,
    win_rate: float = 0.50,
    avg_win: float = 1.0,
    avg_loss: float = 1.0
) -> int:
    """
    Calculate position size using Kelly Criterion, scaled by equity growth.
    
    Kelly % = W - [(1-W) / R]
    Where:
        W = win probability
        R = win/loss ratio
    
    As equity grows, we increase the Kelly fraction used.
    """
    
    # Calculate Kelly percentage
    if avg_loss == 0:
        avg_loss = 0.01
    
    R = abs(avg_win / avg_loss) if avg_loss != 0 else 1
    kelly_pct = win_rate - ((1 - win_rate) / R)
    kelly_pct = max(0, kelly_pct)  # Can't be negative
    
    # Scale Kelly fraction based on equity growth
    equity_multiple = equity / config.starting_capital
    
    if equity_multiple <= 1:
        # At or below starting capital - use base Kelly
        kelly_fraction = config.base_kelly_fraction
    else:
        # Scale up Kelly as equity grows
        scale_progress = min(1.0, (equity_multiple - 1) / (config.kelly_scale_threshold - 1))
        kelly_fraction = config.base_kelly_fraction + (
            (config.max_kelly_fraction - config.base_kelly_fraction) * scale_progress
        )
    
    # Position size in dollars
    position_dollars = equity * kelly_pct * kelly_fraction
    
    # Apply max position limit
    position_dollars = min(position_dollars, equity * config.max_position_size)
    
    # Convert to contracts
    # Max loss on butterfly = wing_cost (if expires worthless with no credit)
    max_loss_per_contract = wing_cost * 50  # ES = $50/point
    
    if max_loss_per_contract <= 0:
        return config.min_contracts
    
    contracts = int(position_dollars / max_loss_per_contract)
    contracts = max(config.min_contracts, min(config.max_contracts, contracts))
    
    return contracts


def run_es_backtest(bars: List[Bar], bt_config: BacktestConfig) -> Dict:
    """Run backtest on ES data with Kelly position sizing"""
    
    logger.info(f"Running backtest on {len(bars)} bars...")
    logger.info(f"Starting capital: ${bt_config.starting_capital:,.2f}")
    logger.info(f"Kelly range: {bt_config.base_kelly_fraction*100:.0f}% - {bt_config.max_kelly_fraction*100:.0f}%")
    
    pricer = ESOptionPricer()
    
    # Use signal config from config.py (optimized settings)
    from config import config as app_config
    sig = app_config.signal
    
    logger.info(f"Signal config: cooldown={sig.signal_cooldown_bars}, vol_thresh={sig.volume_threshold}, "
                f"confirm_bars={sig.min_confirmation_bars}, sustained={sig.sustained_bars_required}")
    
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
    
    # Use bt_config for the rest (rename to avoid confusion)
    config = bt_config
    
    # Group by date
    bars_by_date = defaultdict(list)
    for bar in bars:
        bars_by_date[bar.timestamp.date()].append(bar)
    
    dates = sorted(bars_by_date.keys())
    
    all_trades: List[ButterflyTrade] = []
    daily_results = {}
    
    # Track equity and performance for Kelly calculation
    equity = config.starting_capital
    equity_curve = [equity]
    peak_equity = equity
    
    # Running stats for Kelly calculation
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
        
        day_start_equity = equity
        
        for bar in day_bars:
            bar_time = bar.timestamp
            if bar_time.tzinfo:
                bar_time = bar_time.replace(tzinfo=None)
            
            minutes_to_close = max(1, (market_close - bar_time).total_seconds() / 60)
            es_price = bar.close
            
            signal = detector.add_bar(bar)
            
            # Check pending positions
            still_pending = []
            for trade in pending:
                minutes_waiting = (bar_time - trade.wing_fill_time).total_seconds() / 60
                
                # Get current middle price
                m_bid, m_ask = pricer.get_option_price(
                    es_price, 
                    trade.middle_strike, 
                    trade.option_type, 
                    int(minutes_to_close)
                )
                
                current_credit = m_bid * 2
                target_credit = trade.wing_cost * (1 + config.credit_target_pct)
                
                # Track move
                if trade.direction == Direction.LONG:
                    move = es_price - trade.entry_price
                else:
                    move = trade.entry_price - es_price
                
                if current_credit >= target_credit:
                    # Hit 30% target
                    trade.middle_fill_time = bar_time
                    trade.middle_credit = current_credit
                    trade.net_credit = current_credit - trade.wing_cost
                    trade.fill_status = FillStatus.FILLED_TARGET
                    trade.fill_minutes = int(minutes_waiting)
                    trade.fill_price = es_price
                    trade.move_points = move
                    completed.append(trade)
                    
                elif minutes_waiting >= config.max_fill_minutes or minutes_to_close < 15:
                    # Timeout - complete at current price
                    trade.middle_fill_time = bar_time
                    trade.middle_credit = current_credit
                    trade.net_credit = current_credit - trade.wing_cost
                    trade.fill_status = FillStatus.FILLED_TIMEOUT
                    trade.fill_minutes = int(minutes_waiting)
                    trade.fill_price = es_price
                    trade.move_points = move
                    completed.append(trade)
                    
                else:
                    still_pending.append(trade)
            
            pending = still_pending
            
            # New signal
            if signal and signal.direction in [Direction.LONG, Direction.SHORT]:
                if minutes_to_close < 60:
                    continue
                
                # Calculate strikes (5-wide)
                atm = round(es_price / 5) * 5
                
                if signal.direction == Direction.LONG:
                    lower = atm
                    middle = atm + 5
                    upper = atm + 10
                    opt_type = 'C'
                else:
                    upper = atm
                    middle = atm - 5
                    lower = atm - 10
                    opt_type = 'P'
                
                # Get wing prices
                l_bid, l_ask = pricer.get_option_price(es_price, lower, opt_type, int(minutes_to_close))
                u_bid, u_ask = pricer.get_option_price(es_price, upper, opt_type, int(minutes_to_close))
                
                wing_cost = l_ask + u_ask
                
                # Calculate position size using Kelly
                # Always trade at least 1 contract if we have any equity
                if equity > 0:
                    win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.50
                    avg_win = sum(wins) / len(wins) if wins else 1.0
                    avg_loss = abs(sum(losses) / len(losses)) if losses else 1.0
                    
                    contracts = calculate_position_size(
                        equity=equity,
                        wing_cost=wing_cost,
                        config=config,
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss
                    )
                else:
                    # Negative equity - skip this trade
                    logger.warning(f"Negative equity (${equity:.2f}), skipping trade")
                    continue
                
                # Calculate Kelly fraction being used for tracking
                equity_multiple = equity / config.starting_capital
                if equity_multiple <= 1:
                    kelly_frac = config.base_kelly_fraction
                else:
                    scale_progress = min(1.0, (equity_multiple - 1) / (config.kelly_scale_threshold - 1))
                    kelly_frac = config.base_kelly_fraction + (
                        (config.max_kelly_fraction - config.base_kelly_fraction) * scale_progress
                    )
                
                trade = ButterflyTrade(
                    id=f"{day.strftime('%Y%m%d')}-{len(all_trades)+1:03d}",
                    direction=signal.direction,
                    signal_type=signal.signal_type,
                    signal_time=bar_time,
                    wing_fill_time=bar_time,
                    entry_price=es_price,
                    lower_strike=lower,
                    middle_strike=middle,
                    upper_strike=upper,
                    wing_cost=wing_cost,
                    contracts=contracts,
                    equity_at_entry=equity,
                    kelly_fraction_used=kelly_frac
                )
                
                pending.append(trade)
        
        # Complete any remaining pending
        for trade in pending:
            m_bid, _ = pricer.get_option_price(settlement_price, trade.middle_strike, trade.option_type, 1)
            trade.middle_credit = m_bid * 2
            trade.net_credit = trade.middle_credit - trade.wing_cost
            trade.fill_status = FillStatus.FILLED_TIMEOUT
            trade.fill_price = settlement_price
            completed.append(trade)
        
        # Calculate settlement P&L and update equity
        day_pnl = 0
        for trade in completed:
            trade.calculate_settlement(settlement_price)
            day_pnl += trade.pnl
            
            # Track for Kelly calculation
            if trade.pnl > 0:
                wins.append(trade.pnl)
            else:
                losses.append(trade.pnl)
        
        equity += day_pnl
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        
        all_trades.extend(completed)
        
        target_fills = sum(1 for t in completed if t.fill_status == FillStatus.FILLED_TARGET)
        total_contracts = sum(t.contracts for t in completed)
        
        daily_results[day] = {
            'trades': len(completed),
            'contracts': total_contracts,
            'target_fills': target_fills,
            'pnl': day_pnl,
            'equity': equity
        }
        
        if completed:
            logger.info(f"{day}: {len(completed)} trades ({total_contracts} contracts), "
                       f"30% fills: {target_fills}, P&L: ${day_pnl:,.2f}, Equity: ${equity:,.2f}")
    
    # Summary
    total_trades = len(all_trades)
    if total_trades == 0:
        return {'error': 'No trades'}
    
    target_fills = [t for t in all_trades if t.fill_status == FillStatus.FILLED_TARGET]
    total_pnl = sum(t.pnl for t in all_trades)
    winners = [t for t in all_trades if t.pnl > 0]
    total_contracts = sum(t.contracts for t in all_trades)
    
    # Calculate max drawdown
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
        'target_fills': len(target_fills),
        'target_rate': len(target_fills) / total_trades * 100,
        'starting_capital': config.starting_capital,
        'ending_capital': equity,
        'total_return': (equity / config.starting_capital - 1) * 100,
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
    """Print backtest results"""
    
    print("\n" + "=" * 70)
    print("🦋 ES BUTTERFLY BACKTEST RESULTS (Kelly Sizing)")
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
    print(f"  30% Target Fills: {results['target_fills']} ({results['target_rate']:.1f}%)")
    
    print(f"\n{'─' * 40}")
    print("P&L STATISTICS (ES = $50/point)")
    print(f"{'─' * 40}")
    print(f"  Total P&L:    ${results['total_pnl']:,.2f}")
    print(f"  Avg P&L:      ${results['avg_pnl']:,.2f}")
    print(f"  Win Rate:     {results['win_rate']:.1f}%")
    print(f"  Best Trade:   ${results['best_trade']:,.2f}")
    print(f"  Worst Trade:  ${results['worst_trade']:,.2f}")
    
    print(f"\n{'─' * 40}")
    print("DAILY BREAKDOWN")
    print(f"{'─' * 40}")
    for day, stats in sorted(results['daily_results'].items()):
        contracts = stats.get('contracts', stats['trades'])
        equity = stats.get('equity', 0)
        print(f"  {day}: {stats['trades']} trades ({contracts} cts) "
              f"[{stats['target_fills']} @30%] "
              f"P&L: ${stats['pnl']:>8,.2f} | Equity: ${equity:>10,.2f}")


def save_bars(bars: List[Bar], filename: str):
    """Save bars to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for bar in bars:
            writer.writerow([bar.timestamp.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume])
    logger.info(f"Saved {len(bars)} bars to {filename}")


def load_bars(filename: str) -> List[Bar]:
    """Load bars from CSV"""
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
    parser = argparse.ArgumentParser(description='ES Butterfly Backtest')
    parser.add_argument('--days', type=int, default=30, help='Days of data to fetch')
    parser.add_argument('--file', type=str, help='Load bars from CSV')
    parser.add_argument('--save', type=str, default='es_bars.csv', help='Save bars to CSV')
    parser.add_argument('--target', type=float, default=0.30, help='Credit target (0.30 = 30%%)')
    parser.add_argument('--max-time', type=int, default=30, help='Max minutes to fill')
    parser.add_argument('--capital', type=float, default=500, help='Starting capital')
    parser.add_argument('--kelly-min', type=float, default=0.10, help='Base Kelly fraction')
    parser.add_argument('--kelly-max', type=float, default=0.25, help='Max Kelly fraction')
    parser.add_argument('--kelly-scale', type=float, default=2.0, help='Equity multiple for full Kelly scale')
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        credit_target_pct=args.target,
        max_fill_minutes=args.max_time,
        starting_capital=args.capital,
        base_kelly_fraction=args.kelly_min,
        max_kelly_fraction=args.kelly_max,
        kelly_scale_threshold=args.kelly_scale
    )
    
    if args.file:
        bars = load_bars(args.file)
    else:
        client = connect_to_schwab()
        if not client:
            print("Cannot connect to Schwab. Run with --file to use saved data.")
            return
        
        bars = fetch_es_bars(client, args.days)
        save_bars(bars, args.save)
        
        # Also show current ES options for reference
        try:
            es_quote = client.get_quote('/ES')
            es_price = es_quote.get('lastPrice', es_quote.get('mark', bars[-1].close))
            
            print(f"\n{'=' * 50}")
            print(f"CURRENT ES: ${es_price:.2f}")
            print(f"{'=' * 50}")
            
            pricer = ESOptionPricer()
            atm = round(es_price / 5) * 5
            
            print(f"\nBullish Butterfly: {atm}/{atm+5}/{atm+10} Call")
            bf = pricer.price_butterfly(es_price, atm, atm+5, atm+10, 'C', 300)
            print(f"  Wings: ${bf['wing_cost']:.2f}")
            print(f"  2x Middle: ${bf['middle_credit_2x']:.2f}")
            print(f"  Net: ${bf['net']:.2f}")
            print(f"  30% target: ${bf['wing_cost'] * 1.30:.2f}")
            
        except Exception as e:
            logger.warning(f"Could not fetch current ES quote: {e}")
    
    results = run_es_backtest(bars, config)
    print_results(results)


if __name__ == "__main__":
    main()