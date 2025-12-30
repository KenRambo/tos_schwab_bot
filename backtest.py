"""
Backtesting Module - Replay historical data through signal detector

Usage:
    python backtest.py                    # Last 30 trading days
    python backtest.py --days 60          # Last 60 trading days
    python backtest.py --start 2024-12-01 # From specific date
    python backtest.py --output results.csv  # Export trades to CSV
"""
import os
import sys
import logging
import argparse
from datetime import datetime, date, timedelta, time as dt_time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, SignalConfig
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Eastern timezone
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone
    ET = timezone(timedelta(hours=-5))


@dataclass
class BacktestTrade:
    """Represents a simulated trade"""
    id: int
    signal_type: SignalType
    direction: Direction
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_percent: float = 0.0
    bars_held: int = 0
    
    # Option simulation (rough estimate)
    option_delta: float = 0.30
    option_entry_price: float = 0.0  # Estimated option price
    option_exit_price: float = 0.0
    contracts: int = 1


@dataclass
class BacktestResults:
    """Aggregated backtest results"""
    start_date: date
    end_date: date
    trading_days: int = 0
    total_bars: int = 0
    
    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    gross_pnl: float = 0.0
    total_pnl: float = 0.0  # After estimated commissions
    
    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_bars_held: float = 0.0
    
    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # By signal type
    signals_by_type: Dict[str, int] = field(default_factory=dict)
    pnl_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Daily stats
    daily_pnl: List[float] = field(default_factory=list)
    best_day: float = 0.0
    worst_day: float = 0.0


class Backtester:
    """
    Backtests the signal detection strategy on historical data.
    
    Simulates:
    - Signal detection using the same logic as live trading
    - Entry/exit based on opposite signals
    - Option P&L estimation based on underlying move and delta
    """
    
    def __init__(
        self,
        symbol: str = "SPY",
        signal_config: SignalConfig = None,
        starting_capital: float = 10000.0,
        contracts_per_trade: int = 1,
        target_delta: float = 0.30,
        afternoon_delta: float = 0.40,
        commission_per_contract: float = 0.65,
        max_daily_trades: int = 3
    ):
        self.symbol = symbol
        self.signal_config = signal_config or config.signal
        self.starting_capital = starting_capital
        self.contracts = contracts_per_trade
        self.target_delta = target_delta
        self.afternoon_delta = afternoon_delta
        self.commission = commission_per_contract
        self.max_daily_trades = max_daily_trades
        
        # State
        self.detector: Optional[SignalDetector] = None
        self.trades: List[BacktestTrade] = []
        self.current_trade: Optional[BacktestTrade] = None
        self.trade_counter = 0
        
        # Daily tracking
        self.current_date: Optional[date] = None
        self.daily_trade_count = 0
        
        # Results
        self.equity_curve: List[float] = [starting_capital]
        self.daily_pnl: List[Tuple[date, float]] = []
        
    def _init_detector(self) -> None:
        """Initialize a fresh signal detector"""
        self.detector = SignalDetector(
            length_period=self.signal_config.length_period,
            value_area_percent=self.signal_config.value_area_percent,
            volume_threshold=self.signal_config.volume_threshold,
            use_relaxed_volume=self.signal_config.use_relaxed_volume,
            min_confirmation_bars=self.signal_config.min_confirmation_bars,
            sustained_bars_required=self.signal_config.sustained_bars_required,
            signal_cooldown_bars=self.signal_config.signal_cooldown_bars,
            use_or_bias_filter=self.signal_config.use_or_bias_filter,
            or_buffer_points=self.signal_config.or_buffer_points,
            rth_only=True
        )
    
    def _get_delta_for_time(self, bar_time: datetime) -> float:
        """Get appropriate delta based on time of day"""
        if bar_time.hour >= 12:
            return self.afternoon_delta
        return self.target_delta
    
    def _estimate_option_price(self, underlying_price: float, delta: float) -> float:
        """
        Rough estimate of ATM-ish option price based on underlying and delta.
        This is a simplification - real options depend on IV, time to expiry, etc.
        
        For 0DTE SPY options:
        - 30 delta option might be ~$1.50-3.00
        - 40 delta option might be ~$2.50-4.00
        """
        # Base price scales with underlying (roughly 0.3-0.5% for 30 delta)
        base_pct = 0.003 + (delta - 0.30) * 0.01
        estimated = underlying_price * base_pct
        
        # Minimum option price
        return max(0.50, estimated)
    
    def _estimate_option_pnl(
        self,
        entry_price: float,
        exit_price: float,
        delta: float,
        direction: Direction
    ) -> Tuple[float, float]:
        """
        Estimate option P&L based on underlying move and delta.
        
        Returns: (dollar_pnl, percent_pnl) per contract
        """
        underlying_move = exit_price - entry_price
        
        # For calls: positive move = positive P&L
        # For puts: negative move = positive P&L
        if direction == Direction.LONG:
            # Long = bought calls
            option_move = underlying_move * delta
        else:
            # Short = bought puts (delta is negative effectively)
            option_move = -underlying_move * delta
        
        # Estimate entry option price
        option_entry = self._estimate_option_price(entry_price, delta)
        option_exit = option_entry + option_move
        
        # Can't go below 0
        option_exit = max(0.01, option_exit)
        
        dollar_pnl = (option_exit - option_entry) * 100  # Per contract
        pct_pnl = ((option_exit / option_entry) - 1) * 100 if option_entry > 0 else 0
        
        return dollar_pnl, pct_pnl, option_entry, option_exit
    
    def _close_trade(self, bar: Bar, reason: str) -> None:
        """Close the current trade"""
        if not self.current_trade:
            return
        
        delta = self._get_delta_for_time(self.current_trade.entry_time)
        
        dollar_pnl, pct_pnl, opt_entry, opt_exit = self._estimate_option_pnl(
            self.current_trade.entry_price,
            bar.close,
            delta,
            self.current_trade.direction
        )
        
        self.current_trade.exit_time = bar.timestamp
        self.current_trade.exit_price = bar.close
        self.current_trade.exit_reason = reason
        self.current_trade.pnl = dollar_pnl * self.contracts
        self.current_trade.pnl_percent = pct_pnl
        self.current_trade.option_entry_price = opt_entry
        self.current_trade.option_exit_price = opt_exit
        self.current_trade.option_delta = delta
        self.current_trade.contracts = self.contracts
        
        # Subtract commission
        self.current_trade.pnl -= self.commission * self.contracts * 2  # Entry + exit
        
        self.trades.append(self.current_trade)
        
        # Update equity
        new_equity = self.equity_curve[-1] + self.current_trade.pnl
        self.equity_curve.append(new_equity)
        
        logger.debug(f"Closed trade: {self.current_trade.direction.value} "
                    f"${self.current_trade.entry_price:.2f} -> ${bar.close:.2f} "
                    f"P&L: ${self.current_trade.pnl:.2f}")
        
        self.current_trade = None
    
    def _open_trade(self, signal: Signal, bar: Bar) -> None:
        """Open a new trade"""
        self.trade_counter += 1
        self.daily_trade_count += 1
        
        delta = self._get_delta_for_time(bar.timestamp)
        
        self.current_trade = BacktestTrade(
            id=self.trade_counter,
            signal_type=signal.signal_type,
            direction=signal.direction,
            entry_time=bar.timestamp,
            entry_price=bar.close,
            option_delta=delta,
            contracts=self.contracts
        )
        
        logger.debug(f"Opened trade #{self.trade_counter}: {signal.direction.value} "
                    f"@ ${bar.close:.2f} ({signal.signal_type.value})")
    
    def run(self, bars: List[Dict]) -> BacktestResults:
        """
        Run backtest on historical bars.
        
        Args:
            bars: List of OHLCV bar dicts with 'datetime', 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            BacktestResults with performance metrics
        """
        if not bars:
            raise ValueError("No bars provided for backtesting")
        
        self._init_detector()
        
        logger.info(f"Starting backtest with {len(bars)} bars")
        logger.info(f"Date range: {bars[0]['datetime'].date()} to {bars[-1]['datetime'].date()}")
        
        # Track unique trading days
        trading_days = set()
        daily_pnl_tracker: Dict[date, float] = {}
        
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
            trading_days.add(bar_date)
            
            # Check for new day
            if self.current_date != bar_date:
                # Save previous day's P&L
                if self.current_date and self.current_date in daily_pnl_tracker:
                    pass  # Already tracked
                
                self.current_date = bar_date
                self.daily_trade_count = 0
                
                if bar_date not in daily_pnl_tracker:
                    daily_pnl_tracker[bar_date] = 0.0
            
            # Update bars held if in a trade
            if self.current_trade:
                self.current_trade.bars_held += 1
            
            # Add bar to detector and check for signal
            signal = self.detector.add_bar(bar)
            
            if signal:
                # Check daily limit
                if self.daily_trade_count >= self.max_daily_trades:
                    logger.debug(f"Signal ignored - daily limit reached ({self.max_daily_trades})")
                    continue
                
                # If we have an open trade, close it on opposite signal
                if self.current_trade:
                    if self.current_trade.direction != signal.direction:
                        pnl_before = sum(t.pnl for t in self.trades)
                        self._close_trade(bar, f"Opposite signal: {signal.signal_type.value}")
                        pnl_after = sum(t.pnl for t in self.trades)
                        daily_pnl_tracker[bar_date] += (pnl_after - pnl_before)
                        
                        # Open new position
                        self._open_trade(signal, bar)
                else:
                    # No position, open new one
                    self._open_trade(signal, bar)
        
        # Close any remaining position at end
        if self.current_trade and bars:
            last_bar = Bar(
                timestamp=bars[-1]['datetime'],
                open=bars[-1]['open'],
                high=bars[-1]['high'],
                low=bars[-1]['low'],
                close=bars[-1]['close'],
                volume=bars[-1]['volume']
            )
            pnl_before = sum(t.pnl for t in self.trades)
            self._close_trade(last_bar, "End of backtest")
            pnl_after = sum(t.pnl for t in self.trades)
            if self.current_date:
                daily_pnl_tracker[self.current_date] += (pnl_after - pnl_before)
        
        # Calculate results
        results = self._calculate_results(
            bars[0]['datetime'].date(),
            bars[-1]['datetime'].date(),
            len(trading_days),
            len(bars),
            daily_pnl_tracker
        )
        
        return results
    
    def _calculate_results(
        self,
        start_date: date,
        end_date: date,
        trading_days: int,
        total_bars: int,
        daily_pnl: Dict[date, float]
    ) -> BacktestResults:
        """Calculate performance metrics from trades"""
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            total_bars=total_bars,
            total_trades=len(self.trades)
        )
        
        if not self.trades:
            return results
        
        # Win/loss counts
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        results.winning_trades = len(winning)
        results.losing_trades = len(losing)
        
        # P&L
        results.gross_pnl = sum(t.pnl for t in self.trades)
        results.total_pnl = results.gross_pnl  # Already includes commission
        
        # Averages
        if winning:
            results.avg_win = sum(t.pnl for t in winning) / len(winning)
        if losing:
            results.avg_loss = sum(t.pnl for t in losing) / len(losing)
        
        results.avg_trade = results.total_pnl / len(self.trades)
        results.avg_bars_held = sum(t.bars_held for t in self.trades) / len(self.trades)
        
        # Win rate
        results.win_rate = results.winning_trades / results.total_trades * 100
        
        # Profit factor
        gross_wins = sum(t.pnl for t in winning) if winning else 0
        gross_losses = abs(sum(t.pnl for t in losing)) if losing else 0.01
        results.profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0
        
        # Expectancy
        if results.total_trades > 0:
            results.expectancy = (
                (results.win_rate / 100 * results.avg_win) +
                ((100 - results.win_rate) / 100 * results.avg_loss)
            )
        
        # Max drawdown
        peak = self.starting_capital
        max_dd = 0
        max_dd_pct = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        results.max_drawdown = max_dd
        results.max_drawdown_pct = max_dd_pct
        
        # Daily P&L stats
        daily_pnls = list(daily_pnl.values())
        results.daily_pnl = daily_pnls
        if daily_pnls:
            results.best_day = max(daily_pnls)
            results.worst_day = min(daily_pnls)
            
            # Sharpe ratio (simplified - assuming 0 risk-free rate)
            import statistics
            if len(daily_pnls) > 1:
                avg_daily = statistics.mean(daily_pnls)
                std_daily = statistics.stdev(daily_pnls)
                if std_daily > 0:
                    results.sharpe_ratio = (avg_daily / std_daily) * (252 ** 0.5)
        
        # By signal type
        for trade in self.trades:
            sig_name = trade.signal_type.value
            results.signals_by_type[sig_name] = results.signals_by_type.get(sig_name, 0) + 1
            results.pnl_by_type[sig_name] = results.pnl_by_type.get(sig_name, 0) + trade.pnl
        
        return results
    
    def print_results(self, results: BacktestResults) -> None:
        """Print formatted backtest results"""
        print("")
        print("=" * 70)
        print("                        BACKTEST RESULTS")
        print("=" * 70)
        print(f"  Period: {results.start_date} to {results.end_date}")
        print(f"  Trading Days: {results.trading_days}")
        print(f"  Total Bars: {results.total_bars:,}")
        print("")
        print("-" * 70)
        print("  TRADE STATISTICS")
        print("-" * 70)
        print(f"  Total Trades:      {results.total_trades}")
        print(f"  Winning Trades:    {results.winning_trades}")
        print(f"  Losing Trades:     {results.losing_trades}")
        print(f"  Win Rate:          {results.win_rate:.1f}%")
        print("")
        print(f"  Avg Win:           ${results.avg_win:,.2f}")
        print(f"  Avg Loss:          ${results.avg_loss:,.2f}")
        print(f"  Avg Trade:         ${results.avg_trade:,.2f}")
        print(f"  Avg Bars Held:     {results.avg_bars_held:.1f}")
        print("")
        print("-" * 70)
        print("  PERFORMANCE")
        print("-" * 70)
        print(f"  Total P&L:         ${results.total_pnl:,.2f}")
        print(f"  Profit Factor:     {results.profit_factor:.2f}")
        print(f"  Expectancy:        ${results.expectancy:,.2f} per trade")
        print("")
        print(f"  Best Day:          ${results.best_day:,.2f}")
        print(f"  Worst Day:         ${results.worst_day:,.2f}")
        print(f"  Sharpe Ratio:      {results.sharpe_ratio:.2f}")
        print("")
        print("-" * 70)
        print("  RISK METRICS")
        print("-" * 70)
        print(f"  Starting Capital:  ${self.starting_capital:,.2f}")
        print(f"  Ending Capital:    ${self.equity_curve[-1]:,.2f}")
        print(f"  Return:            {((self.equity_curve[-1] / self.starting_capital) - 1) * 100:.1f}%")
        print(f"  Max Drawdown:      ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.1f}%)")
        print("")
        print("-" * 70)
        print("  SIGNALS BY TYPE")
        print("-" * 70)
        for sig_type, count in sorted(results.signals_by_type.items(), key=lambda x: -x[1]):
            pnl = results.pnl_by_type.get(sig_type, 0)
            print(f"  {sig_type:<25} {count:>4} trades    ${pnl:>10,.2f}")
        print("")
        print("=" * 70)
    
    def export_trades(self, filename: str) -> None:
        """Export trades to CSV"""
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Signal', 'Direction', 'Entry Time', 'Entry Price',
                'Exit Time', 'Exit Price', 'Exit Reason', 'Bars Held',
                'Delta', 'Option Entry', 'Option Exit', 'Contracts',
                'P&L', 'P&L %'
            ])
            
            for t in self.trades:
                writer.writerow([
                    t.id,
                    t.signal_type.value,
                    t.direction.value,
                    t.entry_time.strftime('%Y-%m-%d %H:%M'),
                    f"{t.entry_price:.2f}",
                    t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else '',
                    f"{t.exit_price:.2f}" if t.exit_price else '',
                    t.exit_reason,
                    t.bars_held,
                    f"{t.option_delta:.2f}",
                    f"{t.option_entry_price:.2f}",
                    f"{t.option_exit_price:.2f}",
                    t.contracts,
                    f"{t.pnl:.2f}",
                    f"{t.pnl_percent:.1f}%"
                ])
        
        logger.info(f"Exported {len(self.trades)} trades to {filename}")


def fetch_historical_data(
    symbol: str,
    days: int = 30,
    start_date: date = None
) -> List[Dict]:
    """
    Fetch historical bar data from Schwab API.
    
    Args:
        symbol: Stock symbol
        days: Number of trading days to fetch
        start_date: Optional specific start date
        
    Returns:
        List of bar dicts
    """
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    
    logger.info("Connecting to Schwab API...")
    
    # Use credentials from config
    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri
    )
    if not auth.authenticate():
        raise RuntimeError("Failed to authenticate with Schwab")
    
    client = SchwabClient(auth)
    
    # Calculate date range
    end = datetime.now()
    if start_date:
        start = datetime.combine(start_date, dt_time(0, 0))
    else:
        # Go back extra days to account for weekends/holidays
        start = end - timedelta(days=int(days * 1.5))
    
    logger.info(f"Fetching {symbol} data from {start.date()} to {end.date()}...")
    
    bars = client.get_price_history(
        symbol=symbol,
        period_type="day",
        period=days + 15,  # Extra buffer
        frequency_type="minute",
        frequency=5,
        extended_hours=False,  # RTH only for cleaner backtest
        start_date=start,
        end_date=end
    )
    
    logger.info(f"Fetched {len(bars)} bars")
    
    return bars


def main():
    parser = argparse.ArgumentParser(description='Backtest the trading strategy')
    parser.add_argument('--days', type=int, default=30, help='Number of trading days to backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to backtest')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    parser.add_argument('--contracts', type=int, default=1, help='Contracts per trade')
    parser.add_argument('--output', type=str, help='Export trades to CSV file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse start date if provided
    start_date = None
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    
    # Fetch historical data
    try:
        bars = fetch_historical_data(
            symbol=args.symbol,
            days=args.days,
            start_date=start_date
        )
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)
    
    if not bars:
        logger.error("No data returned")
        sys.exit(1)
    
    # Run backtest
    backtester = Backtester(
        symbol=args.symbol,
        starting_capital=args.capital,
        contracts_per_trade=args.contracts,
        target_delta=config.trading.target_delta,
        afternoon_delta=config.trading.afternoon_delta,
        max_daily_trades=config.trading.max_daily_trades
    )
    
    results = backtester.run(bars)
    
    # Print results
    backtester.print_results(results)
    
    # Export if requested
    if args.output:
        backtester.export_trades(args.output)


if __name__ == "__main__":
    main()