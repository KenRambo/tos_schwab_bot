"""
Notification System for Trading Bot

Supports:
- Pushover (iOS/Android push notifications)
- Discord (webhook)

All notifications include the trading symbol for multi-bot clarity.
"""

import os
import requests
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PushoverNotifier:
    """Send notifications via Pushover"""
    
    # Sound options
    SOUNDS = {
        'start': 'bike',
        'stop': 'gamelan', 
        'signal': 'cashregister',
        'trade': 'bugle',
        'profit': 'magic',
        'loss': 'falling',
        'error': 'siren',
        'warning': 'pushover',
        'summary': 'classical'
    }
    
    def __init__(self, user_key: str = None, api_token: str = None):
        self.user_key = user_key or os.getenv('PUSHOVER_USER_KEY')
        self.api_token = api_token or os.getenv('PUSHOVER_API_TOKEN')
        self.enabled = bool(self.user_key and self.api_token)
        
        if not self.enabled:
            logger.warning("Pushover not configured - notifications disabled")
    
    def send(self, title: str, message: str, priority: int = 0, sound: str = None) -> bool:
        """Send a notification via Pushover"""
        if not self.enabled:
            logger.debug(f"[NOTIFICATION] {title}: {message}")
            return False
        
        try:
            response = requests.post(
                'https://api.pushover.net/1/messages.json',
                data={
                    'token': self.api_token,
                    'user': self.user_key,
                    'title': title,
                    'message': message,
                    'priority': priority,
                    'sound': sound or 'pushover'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Notification sent: {title}")
                return True
            else:
                logger.warning(f"Pushover error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    # =========================================================================
    # Convenience Methods - All include symbol in title
    # =========================================================================
    
    def bot_started(self, mode: str, symbol: str) -> bool:
        """Notification when bot starts"""
        return self.send(
            title=f"🤖 {symbol} Bot Started",
            message=f"Mode: {mode}\nSymbol: {symbol}\nTime: {datetime.now().strftime('%H:%M:%S')}",
            priority=-1,
            sound=self.SOUNDS['start']
        )
    
    def bot_stopped(self, trades: int, pnl: float, symbol: str = "BOT") -> bool:
        """Notification when bot stops"""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        return self.send(
            title=f"🛑 {symbol} Bot Stopped",
            message=f"Trades: {trades}\nP&L: {pnl_emoji} ${pnl:,.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}",
            priority=0,
            sound=self.SOUNDS['stop']
        )
    
    def signal_alert(self, signal_type: str, direction: str, price: float, symbol: str) -> bool:
        """Notification when signal is detected"""
        emoji = "📈" if direction.upper() == "LONG" else "📉"
        return self.send(
            title=f"{emoji} {symbol} {direction} Signal",
            message=f"Type: {signal_type}\nPrice: ${price:.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}",
            priority=1,
            sound=self.SOUNDS['signal']
        )
    
    def trade_executed(self, direction: str, symbol: str, option_symbol: str, 
                       price: float, quantity: int) -> bool:
        """Notification when trade is executed"""
        emoji = "🟢" if direction.upper() == "LONG" else "🔴"
        return self.send(
            title=f"{emoji} {symbol} {direction} Trade Filled",
            message=f"Option: {option_symbol}\nPrice: ${price:.2f}\nQty: {quantity}",
            priority=1,
            sound=self.SOUNDS['trade']
        )
    
    def trade_closed(self, direction: str, pnl: float, pnl_percent: float, 
                     reason: str, symbol: str = "SPY") -> bool:
        """Notification when trade is closed"""
        if pnl >= 0:
            emoji = "💰"
            sound = self.SOUNDS['profit']
        else:
            emoji = "📉"
            sound = self.SOUNDS['loss']
        
        return self.send(
            title=f"{emoji} {symbol} Position Closed",
            message=f"Direction: {direction}\nP&L: ${pnl:,.2f} ({pnl_percent:+.1f}%)\nReason: {reason}",
            priority=0,
            sound=sound
        )
    
    def trade_rejected(self, reason: str, details: str = "", symbol: str = "SPY") -> bool:
        """Notification when trade is rejected"""
        return self.send(
            title=f"❌ {symbol} Trade Rejected",
            message=f"Reason: {reason}\n{details}" if details else f"Reason: {reason}",
            priority=0,
            sound=self.SOUNDS['warning']
        )
    
    def error_alert(self, error: str, symbol: str = "BOT") -> bool:
        """Notification for errors"""
        return self.send(
            title=f"⚠️ {symbol} Bot Error",
            message=str(error)[:200],  # Truncate long errors
            priority=1,
            sound=self.SOUNDS['error']
        )
    
    def daily_summary(self, trades: int, wins: int, pnl: float, symbol: str) -> bool:
        """End of day summary notification"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        trend_emoji = "📈" if pnl >= 0 else "📉"
        
        return self.send(
            title=f"{trend_emoji} {symbol} Daily Summary",
            message=f"Trades: {trades}\nWins: {wins} ({win_rate:.0f}%)\nP&L: {pnl_emoji} ${pnl:,.2f}",
            priority=0,
            sound=self.SOUNDS['summary']
        )
    
    def drawdown_alert(self, current_drawdown: float, threshold: float, symbol: str = "BOT") -> bool:
        """Alert when drawdown exceeds threshold"""
        return self.send(
            title=f"🚨 {symbol} Drawdown Alert",
            message=f"Current: {current_drawdown:.1f}%\nThreshold: {threshold:.1f}%",
            priority=2,  # High priority
            sound=self.SOUNDS['error']
        )


class DiscordNotifier:
    """Send notifications via Discord webhook"""
    
    # Color codes for embeds
    COLORS = {
        'green': 0x00FF00,
        'red': 0xFF0000,
        'blue': 0x0099FF,
        'yellow': 0xFFCC00,
        'purple': 0x9900FF
    }
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if self.enabled:
            logger.info(f"Discord notifications enabled (webhook configured)")
        else:
            logger.debug("Discord webhook not configured - set DISCORD_WEBHOOK_URL")
    
    def send(self, title: str, message: str, color: int = None, **kwargs) -> bool:
        """Send a Discord embed message"""
        if not self.enabled:
            return False
        
        # Auto-determine color from title
        if color is None:
            title_lower = title.lower()
            if any(w in title_lower for w in ['profit', 'win', 'started', '🟢', '💰']):
                color = self.COLORS['green']
            elif any(w in title_lower for w in ['loss', 'error', 'stopped', '🔴', 'alert']):
                color = self.COLORS['red']
            elif any(w in title_lower for w in ['signal', '📈', '📉']):
                color = self.COLORS['purple']
            else:
                color = self.COLORS['blue']
        
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            if response.status_code in [200, 204]:
                logger.debug(f"Discord notification sent: {title}")
                return True
            else:
                logger.warning(f"Discord error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False


class MultiNotifier:
    """Send to multiple notification services"""
    
    def __init__(self):
        self.pushover = PushoverNotifier()
        self.discord = DiscordNotifier()
    
    def send(self, title: str, message: str, **kwargs) -> bool:
        """Send to all configured services"""
        results = []
        
        if self.pushover.enabled:
            results.append(self.pushover.send(title, message, **kwargs))
        
        if self.discord.enabled:
            results.append(self.discord.send(title, message, **kwargs))
        
        return any(results)
    
    def _send_to_all(self, title: str, message: str, **kwargs) -> bool:
        """Helper to send formatted notification to all services"""
        results = []
        
        if self.pushover.enabled:
            results.append(self.pushover.send(title, message, **kwargs))
        
        if self.discord.enabled:
            results.append(self.discord.send(title, message))
        
        return any(results)
    
    # =========================================================================
    # Convenience Methods - Send to ALL configured services
    # =========================================================================
    
    def bot_started(self, mode: str, symbol: str) -> bool:
        """Notification when bot starts"""
        title = f"🤖 {symbol} Bot Started"
        message = f"Mode: {mode}\nSymbol: {symbol}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        return self._send_to_all(title, message, priority=-1, sound='bike')
    
    def bot_stopped(self, trades: int, pnl: float, symbol: str = "BOT") -> bool:
        """Notification when bot stops"""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        title = f"🛑 {symbol} Bot Stopped"
        message = f"Trades: {trades}\nP&L: {pnl_emoji} ${pnl:,.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        return self._send_to_all(title, message, priority=0, sound='gamelan')
    
    def signal_alert(self, signal_type: str, direction: str, price: float, symbol: str) -> bool:
        """Notification when signal is detected"""
        emoji = "📈" if direction.upper() == "LONG" else "📉"
        title = f"{emoji} {symbol} {direction} Signal"
        message = f"Type: {signal_type}\nPrice: ${price:.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        return self._send_to_all(title, message, priority=1, sound='cashregister')
    
    def trade_executed(self, direction: str, symbol: str, option_symbol: str, 
                       price: float, quantity: int) -> bool:
        """Notification when trade is executed"""
        emoji = "🟢" if direction.upper() == "LONG" else "🔴"
        title = f"{emoji} {symbol} {direction} Trade Filled"
        message = f"Option: {option_symbol}\nPrice: ${price:.2f}\nQty: {quantity}"
        return self._send_to_all(title, message, priority=1, sound='bugle')
    
    def trade_closed(self, direction: str, pnl: float, pnl_percent: float, 
                     reason: str, symbol: str = "SPY") -> bool:
        """Notification when trade is closed"""
        if pnl >= 0:
            emoji = "💰"
            sound = 'magic'
        else:
            emoji = "📉"
            sound = 'falling'
        
        title = f"{emoji} {symbol} Position Closed"
        message = f"Direction: {direction}\nP&L: ${pnl:,.2f} ({pnl_percent:+.1f}%)\nReason: {reason}"
        return self._send_to_all(title, message, priority=0, sound=sound)
    
    def trade_rejected(self, reason: str, details: str = "", symbol: str = "SPY") -> bool:
        """Notification when trade is rejected"""
        title = f"❌ {symbol} Trade Rejected"
        message = f"Reason: {reason}\n{details}" if details else f"Reason: {reason}"
        return self._send_to_all(title, message, priority=0, sound='pushover')
    
    def error_alert(self, error: str, symbol: str = "BOT") -> bool:
        """Notification for errors"""
        title = f"⚠️ {symbol} Bot Error"
        message = str(error)[:200]
        return self._send_to_all(title, message, priority=1, sound='siren')
    
    def daily_summary(self, trades: int, wins: int, pnl: float, symbol: str) -> bool:
        """End of day summary notification"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        trend_emoji = "📈" if pnl >= 0 else "📉"
        
        title = f"{trend_emoji} {symbol} Daily Summary"
        message = f"Trades: {trades}\nWins: {wins} ({win_rate:.0f}%)\nP&L: {pnl_emoji} ${pnl:,.2f}"
        return self._send_to_all(title, message, priority=0, sound='classical')
    
    def drawdown_alert(self, current_drawdown: float, threshold: float, symbol: str = "BOT") -> bool:
        """Alert when drawdown exceeds threshold"""
        title = f"🚨 {symbol} Drawdown Alert"
        message = f"Current: {current_drawdown:.1f}%\nThreshold: {threshold:.1f}%"
        return self._send_to_all(title, message, priority=2, sound='siren')


# Singleton instance
_notifier: Optional[MultiNotifier] = None

def get_notifier() -> MultiNotifier:
    """Get the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = MultiNotifier()
    return _notifier