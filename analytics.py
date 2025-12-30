"""
Analytics Module - Performance Tracking and Reporting

Features:
- Daily P&L summary notification at 4:15 PM ET
- Weekly performance report (win rate, avg win/loss, Sharpe)
- Drawdown alerts (notify if down X% from peak)
"""
import logging
import json
import os
from datetime import datetime, date, timedelta, time as dt_time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import statistics

from notifications import get_notifier

logger = logging.getLogger(__name__)

# Eastern timezone
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    from datetime import timezone
    ET = timezone(timedelta(hours=-5))


def get_et_now():
    """Get current time in Eastern"""
    return datetime.now(ET)


@dataclass
class DailyStats:
    """Statistics for a single trading day"""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    peak_balance: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades * 100


@dataclass 
class WeeklyStats:
    """Statistics for a trading week"""
    week_start: date
    week_end: date
    trading_days: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_daily_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    sharpe_ratio: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0


class PerformanceTracker:
    """
    Tracks trading performance and sends scheduled reports.
    
    Features:
    - Daily P&L tracking
    - Drawdown monitoring with alerts
    - Scheduled daily summary (4:15 PM ET)
    - Weekly performance reports (Friday close or Monday morning)
    """
    
    def __init__(
        self,
        data_file: str = "performance_data.json",
        drawdown_alert_threshold: float = 10.0,  # Alert at 10% drawdown
        daily_summary_time: dt_time = dt_time(16, 15),  # 4:15 PM ET
        enable_daily_summary: bool = True,
        enable_weekly_report: bool = True,
        enable_drawdown_alerts: bool = True
    ):
        self.data_file = data_file
        self.drawdown_alert_threshold = drawdown_alert_threshold
        self.daily_summary_time = daily_summary_time
        self.enable_daily_summary = enable_daily_summary
        self.enable_weekly_report = enable_weekly_report
        self.enable_drawdown_alerts = enable_drawdown_alerts
        
        # Current session tracking
        self.session_start_balance: float = 0.0
        self.peak_balance: float = 0.0
        self.current_balance: float = 0.0
        
        # Today's stats
        self.today_stats = DailyStats(date=date.today())
        
        # Trade-level data for detailed analytics
        self.trade_pnls: List[float] = []  # Individual trade P&Ls
        self.daily_pnls: List[float] = []  # Daily P&Ls for Sharpe calc
        
        # Tracking flags
        self._daily_summary_sent = False
        self._weekly_report_sent = False
        self._last_drawdown_alert: Optional[datetime] = None
        self._drawdown_alert_cooldown = timedelta(hours=1)  # Don't spam alerts
        
        # Historical data
        self.historical_days: List[DailyStats] = []
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load historical performance data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                self.peak_balance = data.get('peak_balance', 0)
                self.daily_pnls = data.get('daily_pnls', [])
                
                # Load historical days
                for day_data in data.get('historical_days', []):
                    day_data['date'] = datetime.strptime(day_data['date'], '%Y-%m-%d').date()
                    self.historical_days.append(DailyStats(**day_data))
                
                logger.info(f"Loaded performance data: {len(self.historical_days)} historical days")
                
            except Exception as e:
                logger.warning(f"Could not load performance data: {e}")
    
    def _save_data(self):
        """Save performance data to file"""
        try:
            data = {
                'peak_balance': self.peak_balance,
                'daily_pnls': self.daily_pnls[-252:],  # Keep last year of daily P&Ls
                'historical_days': [
                    {**asdict(day), 'date': day.date.isoformat()}
                    for day in self.historical_days[-252:]  # Keep last year
                ]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save performance data: {e}")
    
    def set_starting_balance(self, balance: float):
        """Set the starting balance for the session"""
        self.session_start_balance = balance
        self.current_balance = balance
        
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        self.today_stats.starting_balance = balance
        self.today_stats.peak_balance = balance
        
        logger.info(f"Starting balance set: ${balance:,.2f}, Peak: ${self.peak_balance:,.2f}")
    
    def update_balance(self, new_balance: float):
        """Update current balance and check for drawdown alerts"""
        self.current_balance = new_balance
        self.today_stats.ending_balance = new_balance
        
        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.today_stats.peak_balance = new_balance
            logger.debug(f"New peak balance: ${new_balance:,.2f}")
        
        # Check for drawdown alert
        self._check_drawdown_alert()
    
    def record_trade(self, pnl: float, is_win: bool):
        """Record a completed trade"""
        self.today_stats.trades += 1
        self.today_stats.gross_pnl += pnl
        self.today_stats.net_pnl += pnl  # TODO: Subtract fees
        
        if is_win:
            self.today_stats.wins += 1
        else:
            self.today_stats.losses += 1
        
        self.trade_pnls.append(pnl)
        
        # Update balance
        self.current_balance += pnl
        self.update_balance(self.current_balance)
        
        logger.info(f"Trade recorded: ${pnl:+.2f} | Today: {self.today_stats.trades} trades, ${self.today_stats.net_pnl:+.2f}")
    
    def _check_drawdown_alert(self):
        """Check if drawdown exceeds threshold and send alert"""
        if not self.enable_drawdown_alerts:
            return
        
        if self.peak_balance <= 0:
            return
        
        drawdown_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        
        if drawdown_pct >= self.drawdown_alert_threshold:
            # Check cooldown
            now = datetime.now()
            if self._last_drawdown_alert:
                if now - self._last_drawdown_alert < self._drawdown_alert_cooldown:
                    return  # Still in cooldown
            
            logger.warning(f"Drawdown alert: {drawdown_pct:.1f}% (threshold: {self.drawdown_alert_threshold}%)")
            
            get_notifier().drawdown_alert(
                current_drawdown=drawdown_pct,
                peak_balance=self.peak_balance,
                current_balance=self.current_balance
            )
            
            self._last_drawdown_alert = now
    
    def check_scheduled_reports(self):
        """Check if it's time to send scheduled reports"""
        now = get_et_now()
        current_time = now.time()
        today = now.date()
        
        # Daily summary at 4:15 PM ET
        if self.enable_daily_summary:
            if not self._daily_summary_sent:
                # Check if we're past the summary time
                if current_time >= self.daily_summary_time:
                    self._send_daily_summary()
                    self._daily_summary_sent = True
        
        # Weekly report - send on Friday after close or Monday morning
        if self.enable_weekly_report:
            is_friday_after_close = (
                now.weekday() == 4 and 
                current_time >= dt_time(16, 30)
            )
            is_monday_morning = (
                now.weekday() == 0 and 
                current_time >= dt_time(8, 0) and
                current_time <= dt_time(9, 30)
            )
            
            if (is_friday_after_close or is_monday_morning) and not self._weekly_report_sent:
                self._send_weekly_report()
                self._weekly_report_sent = True
        
        # Reset flags at midnight
        if current_time < dt_time(0, 5):
            self._daily_summary_sent = False
            if now.weekday() == 0:  # Monday
                self._weekly_report_sent = False
    
    def _send_daily_summary(self):
        """Send daily performance summary"""
        stats = self.today_stats
        
        logger.info("Sending daily summary notification")
        
        get_notifier().daily_summary(
            trades=stats.trades,
            wins=stats.wins,
            pnl=stats.net_pnl
        )
        
        # Archive today's stats
        self.historical_days.append(stats)
        self.daily_pnls.append(stats.net_pnl)
        
        # Save data
        self._save_data()
    
    def _send_weekly_report(self):
        """Send weekly performance report"""
        # Get stats for the last 5 trading days
        recent_days = self.historical_days[-5:] if self.historical_days else []
        
        if not recent_days:
            logger.info("No data for weekly report")
            return
        
        # Calculate weekly stats
        total_trades = sum(d.trades for d in recent_days)
        wins = sum(d.wins for d in recent_days)
        losses = sum(d.losses for d in recent_days)
        total_pnl = sum(d.net_pnl for d in recent_days)
        
        # Calculate avg win/loss from trade-level data
        winning_trades = [p for p in self.trade_pnls[-100:] if p > 0]  # Last 100 trades
        losing_trades = [p for p in self.trade_pnls[-100:] if p < 0]
        
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = statistics.mean(losing_trades) if losing_trades else 0
        
        # Sharpe ratio (simplified - daily returns)
        daily_returns = self.daily_pnls[-20:] if len(self.daily_pnls) >= 5 else []
        if len(daily_returns) >= 5:
            try:
                mean_return = statistics.mean(daily_returns)
                std_return = statistics.stdev(daily_returns)
                sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            except:
                sharpe = 0
        else:
            sharpe = 0
        
        # Best/worst days
        day_pnls = [d.net_pnl for d in recent_days]
        best_day = max(day_pnls) if day_pnls else 0
        worst_day = min(day_pnls) if day_pnls else 0
        
        logger.info("Sending weekly report notification")
        
        get_notifier().weekly_report(
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe=sharpe,
            best_day=best_day,
            worst_day=worst_day
        )
    
    def new_trading_day(self):
        """Called when a new trading day starts"""
        today = date.today()
        
        if self.today_stats.date != today:
            # Archive previous day if it had activity
            if self.today_stats.trades > 0:
                self.historical_days.append(self.today_stats)
                self.daily_pnls.append(self.today_stats.net_pnl)
                self._save_data()
            
            # Start fresh for new day
            self.today_stats = DailyStats(date=today)
            self._daily_summary_sent = False
            
            logger.info(f"New trading day: {today}")
    
    def force_daily_summary(self):
        """Force send daily summary (for testing or manual trigger)"""
        self._send_daily_summary()
    
    def force_weekly_report(self):
        """Force send weekly report (for testing or manual trigger)"""
        self._send_weekly_report()
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance * 100
    
    def get_session_pnl(self) -> float:
        """Get P&L since session start"""
        return self.current_balance - self.session_start_balance
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of current stats"""
        return {
            'today': {
                'trades': self.today_stats.trades,
                'wins': self.today_stats.wins,
                'losses': self.today_stats.losses,
                'pnl': self.today_stats.net_pnl,
                'win_rate': self.today_stats.win_rate
            },
            'session': {
                'starting_balance': self.session_start_balance,
                'current_balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'session_pnl': self.get_session_pnl(),
                'drawdown': self.get_current_drawdown()
            },
            'historical': {
                'total_days': len(self.historical_days),
                'total_trades': sum(d.trades for d in self.historical_days),
                'total_pnl': sum(d.net_pnl for d in self.historical_days)
            }
        }


# Global tracker instance
_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get or create the global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker