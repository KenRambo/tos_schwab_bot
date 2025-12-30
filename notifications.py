"""
Push Notification Support via Pushover

Setup:
1. Download Pushover app on iOS/Android ($5 one-time)
2. Create account at pushover.net
3. Get your User Key from the dashboard
4. Create an Application/API Token
5. Add to .env:
   PUSHOVER_USER_KEY=your-user-key
   PUSHOVER_API_TOKEN=your-api-token
"""
import os
import logging
import requests
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Pushover priority levels"""
    LOWEST = -2      # No notification
    LOW = -1         # Quiet notification
    NORMAL = 0       # Normal notification
    HIGH = 1         # High priority, bypasses quiet hours
    EMERGENCY = 2    # Requires acknowledgment


class PushoverNotifier:
    """Send push notifications via Pushover"""
    
    API_URL = "https://api.pushover.net/1/messages.json"
    
    def __init__(
        self,
        user_key: Optional[str] = None,
        api_token: Optional[str] = None,
        enabled: bool = True
    ):
        self.user_key = user_key or os.getenv('PUSHOVER_USER_KEY', '')
        self.api_token = api_token or os.getenv('PUSHOVER_API_TOKEN', '')
        self.enabled = enabled and bool(self.user_key and self.api_token)
        
        if enabled and not self.enabled:
            logger.warning("Pushover notifications disabled - missing PUSHOVER_USER_KEY or PUSHOVER_API_TOKEN")
    
    def send(
        self,
        message: str,
        title: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        sound: Optional[str] = None,
        url: Optional[str] = None,
        url_title: Optional[str] = None
    ) -> bool:
        """
        Send a push notification.
        
        Args:
            message: The notification message
            title: Optional title (defaults to "Trading Bot")
            priority: Notification priority level
            sound: Custom sound (see Pushover docs)
            url: Optional URL to include
            url_title: Title for the URL
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Notification skipped (disabled): {message}")
            return False
        
        try:
            data = {
                'token': self.api_token,
                'user': self.user_key,
                'message': message,
                'title': title or 'Trading Bot',
                'priority': priority.value,
            }
            
            if sound:
                data['sound'] = sound
            if url:
                data['url'] = url
            if url_title:
                data['url_title'] = url_title
            
            # Emergency priority requires retry/expire params
            if priority == NotificationPriority.EMERGENCY:
                data['retry'] = 60  # Retry every 60 seconds
                data['expire'] = 300  # Stop after 5 minutes
            
            response = requests.post(self.API_URL, data=data, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"Notification sent: {title or 'Trading Bot'} - {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    # Convenience methods for common notifications
    
    def signal_alert(self, signal_type: str, direction: str, price: float, symbol: str = "SPY"):
        """Notify on new trading signal"""
        emoji = "📈" if direction == "LONG" else "📉"
        self.send(
            message=f"{direction} @ ${price:.2f}\n{signal_type}",
            title=f"{emoji} {symbol} Signal",
            priority=NotificationPriority.HIGH,
            sound="cashregister"
        )
    
    def trade_executed(self, direction: str, symbol: str, option_symbol: str, price: float, quantity: int):
        """Notify on trade execution"""
        emoji = "🟢" if direction == "LONG" else "🔴"
        self.send(
            message=f"{direction} {quantity}x {option_symbol}\n@ ${price:.2f}",
            title=f"{emoji} Trade Executed",
            priority=NotificationPriority.HIGH,
            sound="bugle"
        )
    
    def trade_closed(self, direction: str, pnl: float, pnl_percent: float, reason: str):
        """Notify on position close"""
        emoji = "💰" if pnl >= 0 else "💸"
        sign = "+" if pnl >= 0 else ""
        self.send(
            message=f"P&L: {sign}${pnl:.2f} ({sign}{pnl_percent:.1f}%)\nReason: {reason}",
            title=f"{emoji} Position Closed",
            priority=NotificationPriority.NORMAL,
            sound="magic" if pnl >= 0 else "falling"
        )
    
    def trade_rejected(self, reason: str, details: str = ""):
        """Notify on trade rejection"""
        self.send(
            message=f"{reason}\n{details}" if details else reason,
            title="❌ Trade Rejected",
            priority=NotificationPriority.HIGH,
            sound="pushover"
        )
    
    def error_alert(self, error: str):
        """Notify on error"""
        self.send(
            message=error[:500],  # Truncate long errors
            title="⚠️ Bot Error",
            priority=NotificationPriority.HIGH,
            sound="siren"
        )
    
    def bot_started(self, mode: str, symbol: str):
        """Notify bot started"""
        self.send(
            message=f"Mode: {mode}\nSymbol: {symbol}",
            title="🤖 Bot Started",
            priority=NotificationPriority.LOW,
            sound="bike"
        )
    
    def bot_stopped(self, trades: int, pnl: float):
        """Notify bot stopped"""
        sign = "+" if pnl >= 0 else ""
        self.send(
            message=f"Trades: {trades}\nP&L: {sign}${pnl:.2f}",
            title="🛑 Bot Stopped",
            priority=NotificationPriority.LOW,
            sound="gamelan"
        )
    
    def daily_summary(self, trades: int, wins: int, pnl: float, symbol: str = "SPY"):
        """Send daily performance summary"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""
        self.send(
            message=f"Trades: {trades}\nWins: {wins} ({win_rate:.0f}%)\nP&L: {sign}${pnl:.2f}",
            title=f"{emoji} {symbol} Daily Summary",
            priority=NotificationPriority.NORMAL,
            sound="classical"
        )
    
    def weekly_report(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        avg_win: float,
        avg_loss: float,
        sharpe: float,
        best_day: float,
        worst_day: float
    ):
        """Send weekly performance report"""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        emoji = "📊"
        sign = "+" if total_pnl >= 0 else ""
        
        message = (
            f"Trades: {total_trades} ({wins}W / {losses}L)\n"
            f"Win Rate: {win_rate:.0f}%\n"
            f"P&L: {sign}${total_pnl:.2f}\n"
            f"Avg Win: +${avg_win:.2f}\n"
            f"Avg Loss: -${abs(avg_loss):.2f}\n"
            f"Sharpe: {sharpe:.2f}\n"
            f"Best Day: +${best_day:.2f}\n"
            f"Worst Day: -${abs(worst_day):.2f}"
        )
        
        self.send(
            message=message,
            title=f"{emoji} Weekly Report",
            priority=NotificationPriority.NORMAL,
            sound="classical"
        )
    
    def drawdown_alert(self, current_drawdown: float, peak_balance: float, current_balance: float):
        """Alert when drawdown exceeds threshold"""
        self.send(
            message=(
                f"Drawdown: {current_drawdown:.1f}%\n"
                f"Peak: ${peak_balance:,.2f}\n"
                f"Current: ${current_balance:,.2f}\n"
                f"Loss: ${peak_balance - current_balance:,.2f}"
            ),
            title="🔻 Drawdown Alert",
            priority=NotificationPriority.HIGH,
            sound="falling"
        )
    
    def buying_power_warning(self, available: float, required: float):
        """Notify on insufficient buying power"""
        self.send(
            message=f"Available: ${available:,.2f}\nRequired: ${required:,.2f}\nShortfall: ${required - available:,.2f}",
            title="💳 Insufficient Funds",
            priority=NotificationPriority.HIGH,
            sound="intermission"
        )


# Global notifier instance
_notifier: Optional[PushoverNotifier] = None


def get_notifier() -> PushoverNotifier:
    """Get or create the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = PushoverNotifier()
    return _notifier


def notify(message: str, title: str = None, priority: NotificationPriority = NotificationPriority.NORMAL):
    """Convenience function to send a notification"""
    return get_notifier().send(message, title, priority)