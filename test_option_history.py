#!/usr/bin/env python3
"""
Test fetching historical SPX option prices from Schwab API.

The Schwab API might support price history for individual option symbols.
Option symbols follow OCC format: SPX   250103C06000000
                                  ROOT  YYMMDD C/P STRIKE*1000

Run this on your local machine with Schwab API credentials.
"""

import sys
import os
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schwab_auth import SchwabAuth
from schwab_client import SchwabClient
from config import config


def build_option_symbol(underlying: str, expiry: date, strike: float, option_type: str) -> str:
    """
    Build OCC option symbol.
    
    Format: ROOT  YYMMDD C/P STRIKE*1000 (padded to 8 digits)
    Example: SPX   250103C06000000
    """
    root = underlying.ljust(6)  # Pad to 6 chars
    exp_str = expiry.strftime('%y%m%d')
    opt_char = 'C' if option_type.upper() in ['C', 'CALL'] else 'P'
    strike_str = f"{int(strike * 1000):08d}"
    
    return f"{root}{exp_str}{opt_char}{strike_str}"


def test_option_price_history():
    """Test if we can fetch historical prices for options"""
    
    print("Connecting to Schwab API...")
    
    auth = SchwabAuth(
        app_key=config.schwab.app_key,
        app_secret=config.schwab.app_secret,
        redirect_uri=config.schwab.redirect_uri,
        token_file=config.schwab.token_file
    )
    
    if not auth.is_authenticated:
        print("Not authenticated. Run schwab_auth.py first.")
        return
    
    auth.refresh_access_token()
    client = SchwabClient(auth)
    
    # Build a recent SPX option symbol
    # Use today's expiration (0DTE)
    today = date.today()
    
    # Try a few option symbols
    test_symbols = [
        build_option_symbol('$SPX', today, 6000, 'C'),
        build_option_symbol('SPX', today, 6000, 'C'),
        build_option_symbol('SPXW', today, 6000, 'C'),  # Weekly
    ]
    
    print(f"\nTesting option symbols:")
    for sym in test_symbols:
        print(f"  {sym}")
    
    print("\n--- Testing Price History for Options ---")
    
    for symbol in test_symbols:
        print(f"\nTrying: {symbol}")
        try:
            bars = client.get_price_history(
                symbol=symbol,
                period_type="day",
                period=1,
                frequency_type="minute",
                frequency=5
            )
            
            if bars:
                print(f"  SUCCESS! Got {len(bars)} bars")
                print(f"  First bar: {bars[0]}")
                print(f"  Last bar: {bars[-1]}")
            else:
                print(f"  No data returned")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n--- Testing Quote for Options ---")
    
    for symbol in test_symbols:
        print(f"\nTrying quote: {symbol}")
        try:
            quote = client.get_quote(symbol)
            print(f"  Bid: ${quote.bid:.2f}, Ask: ${quote.ask:.2f}, Last: ${quote.last_price:.2f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n--- Testing Option Chain ---")
    
    print(f"\nFetching SPX option chain for {today}...")
    try:
        chain = client.get_option_chain(
            symbol='$SPX',
            contract_type='ALL',
            strike_count=5,
            include_quotes=True,
            from_date=today,
            to_date=today
        )
        
        # Parse the chain
        call_map = chain.get('callExpDateMap', {})
        put_map = chain.get('putExpDateMap', {})
        
        print(f"  Call expirations: {list(call_map.keys())[:3]}")
        print(f"  Put expirations: {list(put_map.keys())[:3]}")
        
        # Get a sample option
        if call_map:
            first_exp = list(call_map.keys())[0]
            strikes = call_map[first_exp]
            print(f"\n  Sample calls for {first_exp}:")
            for strike_str in list(strikes.keys())[:3]:
                opt = strikes[strike_str][0]
                print(f"    Strike {strike_str}: bid=${opt.get('bid', 0):.2f}, ask=${opt.get('ask', 0):.2f}, delta={opt.get('delta', 0):.2f}")
    
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    test_option_price_history()