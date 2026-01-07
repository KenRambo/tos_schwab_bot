"""
Performance Analytics and Tracking Module

Tracks trading performance, generates reports, and alerts on drawdowns.
"""

import logging
import json
from datetime import datetime, date, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade"""
    timestamp: datetime
    pnl: float
    won: bool
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0
    exit_price: float = 0


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0
    peak_balance: float = 0
    drawdown: float = 0


class PerformanceTracker:
    """
    Tracks trading performance and generates reports.
    
    Features:
    - Daily P&L tracking
    - Win rate calculation
    - Drawdown monitoring with THROTTLED alerts
    - Scheduled daily/weekly reports
    """
    
    # Throttle drawdown alerts - only alert once per this interval
    DRAWDOWN_ALERT_INTERVAL = timedelta(minutes=30)
    
    def __init__(self):
        self.starting_balance: float = 0
        self.current_balance: float = 0
        self.peak_balance: float = 0
        
        # Trade history
        self.trades: List[TradeRecord] = []
        self.daily_stats: Dict[date, DailyStats] = {}
        
        # Settings
        self.enable_daily_summary = True
        self.enable_weekly_report = True
        self.enable_drawdown_alerts = True
        self.drawdown_alert_threshold = 10.0  # percent
        self.daily_summary_time = dt_time(16, 15)
        self.data_file = "performance_data.json"
        
        # Throttling state
        self._last_drawdown_alert: Optional[datetime] = None
        self._last_drawdown_pct: float = 0
        
        # Session tracking
        self._today: Optional[date] = None
        self._daily_pnl: float = 0
        self._daily_trades: int = 0
        self._daily_wins: int = 0
    
    def set_starting_balance(self, balance: float) -> None:
        """Set starting balance for the session"""
        self.starting_balance = balance
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        logger.info(f"Starting balance set: ${balance:,.2f}")
    
    def update_balance(self, balance: float) -> None:
        """Update current balance and check drawdown"""
        self.current_balance = balance
        
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Check drawdown
        if self.enable_drawdown_alerts and self.peak_balance > 0:
            drawdown_pct = ((self.peak_balance - balance) / self.peak_balance) * 100
            
            if drawdown_pct >= self.drawdown_alert_threshold:
                self._check_drawdown_alert(drawdown_pct)
    
    def _check_drawdown_alert(self, drawdown_pct: float) -> None:
        """
        Check if we should fire a drawdown alert (with throttling).
        
        Only alerts if:
        1. First alert of the session, OR
        2. Last alert was > 30 minutes ago, OR
        3. Drawdown has increased by > 5% since last alert
        """
        now = datetime.now()
        should_alert = False
        
        if self._last_drawdown_alert is None:
            # First alert
            should_alert = True
        elif now - self._last_drawdown_alert > self.DRAWDOWN_ALERT_INTERVAL:
            # Enough time has passed
            should_alert = True
        elif drawdown_pct > self._last_drawdown_pct + 5.0:
            # Drawdown significantly worse
            should_alert = True
        
        if should_alert:
            logger.warning(f"Drawdown alert: {drawdown_pct:.1f}% (threshold: {self.drawdown_alert_threshold}%)")
            self._last_drawdown_alert = now
            self._last_drawdown_pct = drawdown_pct
    
    def new_trading_day(self) -> None:
        """Reset daily stats for new trading day"""
        today = date.today()
        
        if self._today != today:
            # Save previous day if we have data
            if self._today and self._daily_trades > 0:
                self.daily_stats[self._today] = DailyStats(
                    date=self._today,
                    trades=self._daily_trades,
                    wins=self._daily_wins,
                    losses=self._daily_trades - self._daily_wins,
                    pnl=self._daily_pnl,
                    peak_balance=self.peak_balance,
                    drawdown=self._last_drawdown_pct
                )
            
            # Reset for new day
            self._today = today
            self._daily_pnl = 0
            self._daily_trades = 0
            self._daily_wins = 0
            self._last_drawdown_alert = None  # Reset throttle for new day
            self._last_drawdown_pct = 0
            
            logger.info(f"New trading day: {today}")
    
    def record_trade(self, pnl: float, won: bool, symbol: str = "", direction: str = "") -> None:
        """Record a completed trade"""
        trade = TradeRecord(
            timestamp=datetime.now(),
            pnl=pnl,
            won=won,
            symbol=symbol,
            direction=direction
        )
        self.trades.append(trade)
        
        # Update daily stats
        self._daily_trades += 1
        self._daily_pnl += pnl
        if won:
            self._daily_wins += 1
        
        logger.info(f"Trade recorded: {'WIN' if won else 'LOSS'} ${pnl:+.2f} | Daily P&L: ${self._daily_pnl:+.2f}")
    
    def check_scheduled_reports(self) -> None:
        """Check if any scheduled reports should run"""
        now = datetime.now()
        current_time = now.time()
        
        # Daily summary at configured time
        if self.enable_daily_summary:
            if (current_time.hour == self.daily_summary_time.hour and 
                current_time.minute == self.daily_summary_time.minute):
                self._generate_daily_summary()
        
        # Weekly report on Friday after close
        if self.enable_weekly_report:
            if now.weekday() == 4 and current_time.hour == 16 and current_time.minute == 30:
                self._generate_weekly_report()
    
    def _generate_daily_summary(self) -> None:
        """Generate and log daily summary"""
        if self._daily_trades == 0:
            logger.info("Daily Summary: No trades today")
            return
        
        win_rate = (self._daily_wins / self._daily_trades) * 100 if self._daily_trades > 0 else 0
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("DAILY SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Date: {self._today}")
        logger.info(f"  Trades: {self._daily_trades}")
        logger.info(f"  Wins: {self._daily_wins} ({win_rate:.1f}%)")
        logger.info(f"  P&L: ${self._daily_pnl:+,.2f}")
        logger.info(f"  Balance: ${self.current_balance:,.2f}")
        logger.info("=" * 50)
        logger.info("")
    
    def _generate_weekly_report(self) -> None:
        """Generate and log weekly report"""
        # Get this week's stats
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        
        week_trades = 0
        week_wins = 0
        week_pnl = 0
        
        for day, stats in self.daily_stats.items():
            if day >= week_start:
                week_trades += stats.trades
                week_wins += stats.wins
                week_pnl += stats.pnl
        
        # Include today
        week_trades += self._daily_trades
        week_wins += self._daily_wins
        week_pnl += self._daily_pnl
        
        win_rate = (week_wins / week_trades) * 100 if week_trades > 0 else 0
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("WEEKLY REPORT")
        logger.info("=" * 50)
        logger.info(f"  Week: {week_start} to {today}")
        logger.info(f"  Total Trades: {week_trades}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  P&L: ${week_pnl:+,.2f}")
        logger.info(f"  Balance: ${self.current_balance:,.2f}")
        logger.info("=" * 50)
        logger.info("")
    
    def get_stats(self) -> dict:
        """Get current performance statistics"""
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.won)
        total_pnl = sum(t.pnl for t in self.trades)
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown_pct": ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0,
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl
        }
    
    def save(self) -> None:
        """Save performance data to file"""
        try:
            data = {
                "starting_balance": self.starting_balance,
                "peak_balance": self.peak_balance,
                "trades": [
                    {
                        "timestamp": t.timestamp.isoformat(),
                        "pnl": t.pnl,
                        "won": t.won,
                        "symbol": t.symbol,
                        "direction": t.direction
                    }
                    for t in self.trades
                ]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Performance data saved to {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def load(self) -> None:
        """Load performance data from file"""
        try:
            if Path(self.data_file).exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                self.starting_balance = data.get("starting_balance", 0)
                self.peak_balance = data.get("peak_balance", 0)
                
                for t in data.get("trades", []):
                    self.trades.append(TradeRecord(
                        timestamp=datetime.fromisoformat(t["timestamp"]),
                        pnl=t["pnl"],
                        won=t["won"],
                        symbol=t.get("symbol", ""),
                        direction=t.get("direction", "")
                    ))
                
                logger.info(f"Loaded {len(self.trades)} trades from {self.data_file}")
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")


# Singleton instance
_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get the singleton performance tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker