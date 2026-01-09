"""
Economic Calendar Module
Provides high-impact economic event alerts using:
1. Static calendar (FOMC, CPI, NFP - known dates)
2. Optional API fallback (Trading Economics, FMP - paid)

Usage:
    # As standalone script (for testing)
    python economic_calendar.py
    
    # Integrated with bot - call at market open
    from economic_calendar import EconomicCalendar
    calendar = EconomicCalendar()
    calendar.send_daily_alert()
"""

import os
import logging
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class EconomicEvent:
    """Represents a single economic event."""
    event_name: str
    country: str
    date: datetime
    time: str
    impact: EventImpact
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    
    @property
    def is_high_impact(self) -> bool:
        return self.impact == EventImpact.HIGH
    
    @property
    def time_str(self) -> str:
        """Return formatted time string."""
        if self.time:
            return self.time
        return "TBD"
    
    def __str__(self) -> str:
        impact_emoji = {
            EventImpact.HIGH: "🔴",
            EventImpact.MEDIUM: "🟡",
            EventImpact.LOW: "🟢"
        }
        emoji = impact_emoji.get(self.impact, "⚪")
        return f"{emoji} {self.time_str} - {self.event_name}"


# =============================================================================
# STATIC CALENDAR DATA - Known high-impact events for 2026
# Source: Federal Reserve, BLS schedules (update annually)
# =============================================================================

# FOMC Meeting Dates 2026 (announcement days, typically 2:00 PM ET)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES_2026 = [
    "2026-01-28",  # Jan 27-28
    "2026-03-18",  # Mar 17-18
    "2026-05-06",  # May 5-6
    "2026-06-17",  # Jun 16-17
    "2026-07-29",  # Jul 28-29
    "2026-09-16",  # Sep 15-16
    "2026-11-04",  # Nov 3-4
    "2026-12-16",  # Dec 15-16
]

# CPI Release Dates 2026 (typically 8:30 AM ET)
# Source: https://www.bls.gov/schedule/news_release/cpi.htm
CPI_DATES_2026 = [
    "2026-01-14",  # Dec 2025 CPI
    "2026-02-11",  # Jan CPI
    "2026-03-11",  # Feb CPI
    "2026-04-10",  # Mar CPI
    "2026-05-12",  # Apr CPI
    "2026-06-10",  # May CPI
    "2026-07-14",  # Jun CPI
    "2026-08-12",  # Jul CPI
    "2026-09-11",  # Aug CPI
    "2026-10-13",  # Sep CPI
    "2026-11-12",  # Oct CPI
    "2026-12-10",  # Nov CPI
]

# Non-Farm Payrolls (NFP) Dates 2026 (first Friday, 8:30 AM ET)
# Source: https://www.bls.gov/schedule/news_release/empsit.htm
NFP_DATES_2026 = [
    "2026-01-09",  # Dec 2025 jobs
    "2026-02-06",  # Jan jobs
    "2026-03-06",  # Feb jobs
    "2026-04-03",  # Mar jobs
    "2026-05-08",  # Apr jobs
    "2026-06-05",  # Jun jobs
    "2026-07-02",  # Jun jobs
    "2026-08-07",  # Jul jobs
    "2026-09-04",  # Aug jobs
    "2026-10-02",  # Sep jobs
    "2026-11-06",  # Oct jobs
    "2026-12-04",  # Nov jobs
]

# GDP Release Dates 2026 (advance, second, third estimates - 8:30 AM ET)
GDP_DATES_2026 = [
    "2026-01-29",  # Q4 2025 Advance
    "2026-02-26",  # Q4 2025 Second
    "2026-03-26",  # Q4 2025 Third
    "2026-04-29",  # Q1 2026 Advance
    "2026-05-28",  # Q1 2026 Second
    "2026-06-25",  # Q1 2026 Third
    "2026-07-30",  # Q2 2026 Advance
    "2026-08-27",  # Q2 2026 Second
    "2026-09-24",  # Q2 2026 Third
    "2026-10-29",  # Q3 2026 Advance
    "2026-11-25",  # Q3 2026 Second
    "2026-12-22",  # Q3 2026 Third
]

# PCE (Fed's preferred inflation measure) - typically last Friday of month, 8:30 AM ET
PCE_DATES_2026 = [
    "2026-01-30",
    "2026-02-27",
    "2026-03-27",
    "2026-04-30",
    "2026-05-29",
    "2026-06-26",
    "2026-07-31",
    "2026-08-28",
    "2026-09-25",
    "2026-10-30",
    "2026-11-25",
    "2026-12-23",
]

# Build static calendar
STATIC_EVENTS: Dict[str, List[Dict]] = {}

def _add_events(dates: List[str], name: str, time: str):
    """Helper to add events to static calendar."""
    for date_str in dates:
        if date_str not in STATIC_EVENTS:
            STATIC_EVENTS[date_str] = []
        STATIC_EVENTS[date_str].append({
            "event": name,
            "time": time,
            "impact": "High"
        })

# Populate static events
_add_events(FOMC_DATES_2026, "FOMC Interest Rate Decision", "14:00 ET")
_add_events(CPI_DATES_2026, "CPI (Inflation)", "08:30 ET")
_add_events(NFP_DATES_2026, "Non-Farm Payrolls", "08:30 ET")
_add_events(GDP_DATES_2026, "GDP", "08:30 ET")
_add_events(PCE_DATES_2026, "Core PCE Price Index", "08:30 ET")


class EconomicCalendar:
    """
    Provides economic calendar events using static data (free) 
    with optional API fallback for additional events.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        notify_callback: Optional[callable] = None,
        use_api: bool = True  # Try API first if key is available
    ):
        """
        Initialize the economic calendar.
        
        Args:
            api_key: FMP API key (optional, uses FMP_API_KEY env var)
            notify_callback: Function to call for sending notifications.
                           Signature: notify_callback(message: str, title: str)
            use_api: Whether to try API (default True if key available)
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.notify_callback = notify_callback
        self.use_api = use_api and self.api_key is not None
        
        if self.use_api:
            logger.info("Economic calendar: API mode enabled (FMP)")
        else:
            logger.info("Economic calendar: Using static data (FOMC, CPI, NFP, GDP, PCE)")
    
    def get_static_events(self, target_date: date) -> List[EconomicEvent]:
        """Get events from static calendar."""
        date_str = target_date.strftime("%Y-%m-%d")
        events = []
        
        if date_str in STATIC_EVENTS:
            for item in STATIC_EVENTS[date_str]:
                event = EconomicEvent(
                    event_name=item["event"],
                    country="US",
                    date=datetime.combine(target_date, datetime.min.time()),
                    time=item["time"],
                    impact=EventImpact.HIGH
                )
                events.append(event)
        
        return events
    
    def fetch_events_api(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[EconomicEvent]:
        """
        Fetch economic events from FMP API (requires paid plan).
        Falls back to static data on error.
        """
        if not self.api_key or not self.use_api:
            return []
        
        from_date = from_date or date.today()
        to_date = to_date or date.today()
        
        # Correct endpoint: /stable/economic-calendar
        url = "https://financialmodelingprep.com/stable/economic-calendar"
        params = {
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key
        }
        
        try:
            logger.debug(f"Fetching economic calendar API: {from_date} to {to_date}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"FMP API error: {data['Error Message']}")
                return []
            
            return self._parse_api_events(data)
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"API fetch failed (using static data): {e}")
            return []
    
    def _parse_api_events(self, data: List[Dict]) -> List[EconomicEvent]:
        """Parse raw API response into EconomicEvent objects."""
        events = []
        
        for item in data:
            try:
                impact_str = item.get("impact", "Low")
                try:
                    impact = EventImpact(impact_str)
                except ValueError:
                    impact = EventImpact.LOW
                
                date_str = item.get("date", "")
                try:
                    event_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    event_dt = datetime.now()
                
                time_str = event_dt.strftime("%H:%M ET") if event_dt else "TBD"
                
                event = EconomicEvent(
                    event_name=item.get("event", "Unknown Event"),
                    country=item.get("country", ""),
                    date=event_dt,
                    time=time_str,
                    impact=impact,
                    actual=item.get("actual"),
                    forecast=item.get("estimate"),
                    previous=item.get("previous")
                )
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
                continue
        
        return events
    
    def get_us_events(
        self,
        target_date: Optional[date] = None,
        high_impact_only: bool = True
    ) -> List[EconomicEvent]:
        """
        Get US economic events for a specific date.
        Uses API if available, falls back to static data.
        """
        target_date = target_date or date.today()
        
        events = []
        
        # Try API first if enabled
        if self.use_api:
            api_events = self.fetch_events_api(target_date, target_date)
            # Filter for US high-impact events
            for e in api_events:
                if e.country.upper() in ["US", "USA", "UNITED STATES"]:
                    if not high_impact_only or e.is_high_impact:
                        events.append(e)
        
        # If no API events, fall back to static data
        if not events:
            events = self.get_static_events(target_date)
        
        return events
    
    def get_week_events(self, high_impact_only: bool = True) -> Dict[date, List[EconomicEvent]]:
        """Get events for the current week."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        
        by_date: Dict[date, List[EconomicEvent]] = {}
        
        # Try to get all week's events from API in one call
        if self.use_api:
            friday = monday + timedelta(days=4)
            api_events = self.fetch_events_api(monday, friday)
            for e in api_events:
                if e.country.upper() in ["US", "USA", "UNITED STATES"]:
                    if not high_impact_only or e.is_high_impact:
                        event_date = e.date.date() if hasattr(e.date, 'date') else e.date
                        if event_date not in by_date:
                            by_date[event_date] = []
                        by_date[event_date].append(e)
        
        # Fall back to static if no API results
        if not by_date:
            for i in range(5):
                day = monday + timedelta(days=i)
                events = self.get_static_events(day)
                if events:
                    by_date[day] = events
        
        return by_date
    
    def format_daily_alert(self, events: List[EconomicEvent]) -> str:
        """Format events into a notification message."""
        today = date.today()
        day_name = today.strftime("%A")
        date_str = today.strftime("%b %d, %Y")
        
        if not events:
            return f"📅 {day_name}, {date_str}\n\n✅ No high-impact economic events today."
        
        events_sorted = sorted(events, key=lambda e: e.time)
        
        lines = [
            f"📅 {day_name}, {date_str}",
            f"⚠️ {len(events)} High-Impact Event{'s' if len(events) > 1 else ''} Today",
            "─" * 30,
        ]
        
        for event in events_sorted:
            lines.append(str(event))
            if event.forecast:
                lines.append(f"   Forecast: {event.forecast} | Previous: {event.previous or 'N/A'}")
        
        lines.extend([
            "─" * 30,
            "💡 Consider adjusting position size near event times."
        ])
        
        return "\n".join(lines)
    
    def format_weekly_summary(self, events_by_date: Dict[date, List[EconomicEvent]]) -> str:
        """Format weekly events into a summary message."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        
        lines = [
            f"📆 Week of {monday.strftime('%b %d, %Y')}",
            "═" * 35,
        ]
        
        if not events_by_date:
            lines.append("✅ No high-impact events this week.")
            return "\n".join(lines)
        
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        
        for i in range(5):
            day_date = monday + timedelta(days=i)
            day_events = events_by_date.get(day_date, [])
            
            day_str = day_names[i]
            date_str = day_date.strftime("%m/%d")
            
            if day_events:
                lines.append(f"\n{day_str} {date_str}:")
                for event in sorted(day_events, key=lambda e: e.time):
                    lines.append(f"  {event}")
            else:
                lines.append(f"{day_str} {date_str}: —")
        
        return "\n".join(lines)
    
    def send_daily_alert(self) -> bool:
        """Send daily economic calendar alert."""
        events = self.get_us_events(high_impact_only=True)
        message = self.format_daily_alert(events)
        
        logger.info(f"Daily alert: {len(events)} high-impact events")
        logger.info(f"\n{message}")
        
        if self.notify_callback:
            try:
                title = "📊 Economic Calendar"
                self.notify_callback(message, title)
                return True
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                return False
        else:
            print(message)
            return True
    
    def send_weekly_summary(self) -> bool:
        """Send weekly economic calendar summary."""
        events_by_date = self.get_week_events(high_impact_only=True)
        message = self.format_weekly_summary(events_by_date)
        
        total_events = sum(len(e) for e in events_by_date.values())
        logger.info(f"Weekly summary: {total_events} high-impact events")
        logger.info(f"\n{message}")
        
        if self.notify_callback:
            try:
                title = "📆 Weekly Economic Calendar"
                self.notify_callback(message, title)
                return True
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                return False
        else:
            print(message)
            return True
    
    def has_high_impact_today(self) -> bool:
        """Check if there are any high-impact events today."""
        events = self.get_us_events(high_impact_only=True)
        return len(events) > 0
    
    def get_next_event_time(self) -> Optional[str]:
        """Get the time of the next high-impact event today."""
        events = self.get_us_events(high_impact_only=True)
        if not events:
            return None
        return min(events, key=lambda e: e.time).time


# =============================================================================
# Standalone execution for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Economic Calendar Alerts")
    parser.add_argument("--daily", action="store_true", help="Send daily alert")
    parser.add_argument("--weekly", action="store_true", help="Send weekly summary")
    parser.add_argument("--check", action="store_true", help="Check for high-impact events today")
    parser.add_argument("--date", help="Check specific date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    calendar = EconomicCalendar()
    
    if args.date:
        check_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        events = calendar.get_us_events(check_date, high_impact_only=True)
        if events:
            print(f"📅 Events on {args.date}:")
            for e in events:
                print(f"  {e}")
        else:
            print(f"✅ No high-impact events on {args.date}")
    elif args.weekly:
        calendar.send_weekly_summary()
    elif args.check:
        if calendar.has_high_impact_today():
            print("⚠️ High-impact events today!")
            events = calendar.get_us_events(high_impact_only=True)
            for e in events:
                print(f"  {e}")
        else:
            print("✅ No high-impact events today")
    else:
        # Default to daily alert
        calendar.send_daily_alert()