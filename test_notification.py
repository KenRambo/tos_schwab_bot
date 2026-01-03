#!/usr/bin/env python3
"""
Test Pushover notification - run this to verify notifications work
"""
from dotenv import load_dotenv
load_dotenv()

import os
print(f"PUSHOVER_USER_KEY set: {bool(os.getenv('PUSHOVER_USER_KEY'))}")
print(f"PUSHOVER_API_TOKEN set: {bool(os.getenv('PUSHOVER_API_TOKEN'))}")

from notifications import get_notifier

notifier = get_notifier()
print(f"Notifier enabled: {notifier.enabled}")
print(f"User key: {notifier.user_key[:8]}..." if notifier.user_key else "User key: NOT SET")
print(f"API token: {notifier.api_token[:8]}..." if notifier.api_token else "API token: NOT SET")

print("\nSending test notification...")
result = notifier.send(
    message="Trades: 0\nCredits: +$0.00\nButterflies: 0",
    title="🦋 Butterfly Bot Stopped",
    sound="gamelan"
)
print(f"Result: {result}")

# Also try the bot_stopped method to compare
print("\nTrying bot_stopped method...")
result2 = notifier.bot_stopped(trades=0, pnl=0.0)
print(f"Result: {result2}")