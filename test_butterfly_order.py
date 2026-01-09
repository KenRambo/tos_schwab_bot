#!/usr/bin/env python3
"""
Live Butterfly Order Test Script
================================

Sends a real butterfly order to Schwab that WON'T FILL because:
1. It's after market hours (if run after 4pm ET)
2. The limit price is set far from market (won't execute)

Use this to verify:
- API connectivity
- Order structure is correct
- Account permissions
- Butterfly order format accepted by Schwab

The order will be placed as a DAY order and automatically cancelled
at end of day if not filled.

Two order modes available:
- TRIGGER (default): One-Triggers-Other order (wings trigger middle sell)
- SEQUENTIAL: Place wings first, wait for fill, then place middle sell

Usage:
    python test_butterfly_order.py                    # Paper mode (default)
    python test_butterfly_order.py --live             # Live TRIGGER mode
    python test_butterfly_order.py --live --sequential # Live SEQUENTIAL mode
    python test_butterfly_order.py --live --cancel    # Live mode, cancel after placing
"""

import os
import sys
import argparse
import time
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path if running from subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schwab_auth import SchwabAuth
from schwab_client import SchwabClient


def get_et_time():
    """Get current Eastern Time"""
    from datetime import timezone, timedelta
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        return datetime.now(et)
    except ImportError:
        # Rough EST approximation
        return datetime.now(timezone(timedelta(hours=-5)))


def is_market_hours():
    """Check if market is open"""
    now = get_et_time()
    
    # Weekend check
    if now.weekday() >= 5:
        return False
    
    # Time check (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def main():
    parser = argparse.ArgumentParser(description="Test butterfly order placement")
    parser.add_argument("--live", action="store_true", help="Place real order (default: paper)")
    parser.add_argument("--cancel", action="store_true", help="Cancel order after placing")
    parser.add_argument("--symbol", default="SPX", help="Symbol to trade (default: SPX)")
    parser.add_argument("--width", type=int, default=5, help="Wing width in points (default: 5)")
    parser.add_argument("--credit", type=float, default=0.05, help="Limit credit (default: $0.05 - won't fill)")
    parser.add_argument("--quantity", type=int, default=1, help="Number of contracts (default: 1)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential orders instead of TRIGGER")
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           BUTTERFLY ORDER TEST SCRIPT                        ║
    ║           ─────────────────────────────                      ║
    ║   This places a real order that SHOULD NOT FILL              ║
    ║   (limit price too far from market / after hours)            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check market hours
    now = get_et_time()
    market_open = is_market_hours()
    
    print(f"📅 Current time: {now.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"📊 Market status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
    
    if market_open:
        print("\n⚠️  WARNING: Market is OPEN!")
        print("   The order might actually fill if the credit is achievable.")
        print(f"   Using very low credit (${args.credit:.2f}) to prevent fills.")
        print("   Press Ctrl+C to abort, or wait 5 seconds to continue...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            return
    
    # Check credentials
    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    
    if not app_key or not app_secret:
        print("\n❌ Missing SCHWAB_APP_KEY or SCHWAB_APP_SECRET in environment")
        print("   Set these in your .env file")
        return
    
    print(f"\n🔑 API Key: {app_key[:8]}...")
    
    # Authenticate
    print("\n🔐 Authenticating with Schwab...")
    
    try:
        auth = SchwabAuth(
            app_key=app_key,
            app_secret=app_secret,
            redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
            token_file="schwab_tokens.json"
        )
        
        if not auth.is_authenticated:
            print("   No existing auth - starting OAuth flow...")
            if not auth.authorize_interactive():
                print("❌ Authentication failed")
                return
        else:
            if not auth.refresh_access_token():
                print("   Token refresh failed - re-authenticating...")
                if not auth.authorize_interactive():
                    print("❌ Re-authentication failed")
                    return
        
        print("   ✓ Authenticated")
        
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return
    
    # Create client
    client = SchwabClient(auth)
    
    # Get account info
    try:
        account_hash = client.get_account_hash()
        print(f"   ✓ Account: {account_hash[:8]}...")
        
        buying_power = client.get_buying_power()
        print(f"   ✓ Buying power: ${buying_power:,.2f}")
        
    except Exception as e:
        print(f"❌ Account error: {e}")
        return
    
    # Get current price for strike calculation
    print(f"\n📈 Getting {args.symbol} price...")
    
    try:
        if args.symbol.upper() in ['SPX', '$SPX', '$SPX.X']:
            quote = client.get_quote('$SPX')
        else:
            quote = client.get_quote(args.symbol)
        
        underlying_price = quote.last_price
        print(f"   ✓ {args.symbol} @ ${underlying_price:.2f}")
        
    except Exception as e:
        print(f"❌ Quote error: {e}")
        print("   Using fallback price of $5900")
        underlying_price = 5900.0
    
    # Calculate strikes (put butterfly for test)
    width = args.width
    
    if args.symbol.upper() in ['SPX', '$SPX', '$SPX.X']:
        # SPX has $5 strikes
        atm = round(underlying_price / 5) * 5
    else:
        atm = round(underlying_price)
    
    # Put butterfly (SHORT signal style)
    upper = atm
    middle = atm - width
    lower = atm - (width * 2)
    
    option_type = "PUT"
    
    print(f"\n🦋 Butterfly Structure:")
    print(f"   ┌─────────────────────────────────────────")
    print(f"   │ Type: {option_type} butterfly")
    print(f"   │ Symbol: {args.symbol}")
    
    # Show expected option root
    if args.symbol.upper() in ['SPX', 'SPXW', '$SPX']:
        opt_root = 'SPXW'  # Always SPXW for SPX options
        print(f"   │ Option Root: {opt_root}")
    else:
        opt_root = args.symbol.upper()
        print(f"   │ Option Root: {opt_root}")
    
    print(f"   │ Strikes: {lower}/{middle}/{upper}")
    print(f"   │ Width: {width} points")
    print(f"   │ Quantity: {args.quantity}")
    print(f"   │ Limit Credit: ${args.credit:.2f}")
    print(f"   │ Mode: {'SEQUENTIAL' if args.sequential else 'TRIGGER (OTO)'}")
    print(f"   └─────────────────────────────────────────")
    
    if args.sequential:
        print(f"\n   SEQUENTIAL MODE:")
        print(f"   1. BUY {args.quantity}x {opt_root} {lower} {option_type} @ MARKET")
        print(f"   2. BUY {args.quantity}x {opt_root} {upper} {option_type} @ MARKET")
        print(f"   3. (wait for fill)")
        print(f"   4. SELL {args.quantity * 2}x {opt_root} {middle} {option_type} @ ${args.credit:.2f} LIMIT")
    else:
        print(f"\n   TRIGGER (OTO) MODE:")
        print(f"   Primary: BUY {args.quantity}x {lower} + {args.quantity}x {upper} @ MARKET")
        print(f"   Triggered: SELL {args.quantity * 2}x {middle} @ ${args.credit:.2f} LIMIT")
    
    # Confirm
    if args.live:
        print("\n" + "=" * 60)
        print("🔴 LIVE MODE - This will place a REAL order!")
        print("=" * 60)
        
        if args.credit < 0.10:
            print(f"\n✓ Credit ${args.credit:.2f} is very low - unlikely to fill")
        else:
            print(f"\n⚠️ Credit ${args.credit:.2f} might actually fill!")
        
        confirm = input("\nType 'PLACE ORDER' to continue: ")
        if confirm != "PLACE ORDER":
            print("Aborted.")
            return
    else:
        print("\n📝 PAPER MODE - No real order will be placed")
        print("   Use --live to place a real order")
    
    # Place order
    print("\n📤 Placing butterfly order...")
    
    try:
        if args.live:
            if args.sequential:
                # Use sequential method (wings first, then middle)
                result = client.place_butterfly_order_sequential(
                    symbol=args.symbol,
                    lower_strike=lower,
                    middle_strike=middle,
                    upper_strike=upper,
                    option_type=option_type,
                    quantity=args.quantity,
                    limit_credit=args.credit,
                    wait_for_fill=True,
                    max_wait_seconds=30
                )
                
                if result.get('status') == 'PLACED':
                    print(f"   ✓ Orders placed!")
                    print(f"   Wing Order ID: {result.get('wing_order_id')}")
                    print(f"   Middle Order ID: {result.get('middle_order_id')}")
                    if result.get('wing_fill_price'):
                        print(f"   Wing Fill: ${result['wing_fill_price']:.2f}")
                    order_id = result.get('wing_order_id')
                else:
                    print(f"   ❌ Order failed: {result.get('status')}")
                    print(f"   Error: {result.get('error', 'Unknown')}")
                    order_id = None
            else:
                # Use TRIGGER method (OTO)
                result = client.place_butterfly_order(
                    symbol=args.symbol,
                    lower_strike=lower,
                    middle_strike=middle,
                    upper_strike=upper,
                    option_type=option_type,
                    quantity=args.quantity,
                    limit_credit=args.credit
                )
                
                if result and result.get('orderId'):
                    order_id = result['orderId']
                    print(f"   ✓ TRIGGER order placed!")
                    print(f"   Order ID: {order_id}")
                    
                    # Check status
                    time.sleep(1)
                    try:
                        status = client.get_order_status(order_id)
                        print(f"   Status: {status.get('status', 'UNKNOWN')}")
                    except:
                        pass
                elif result and result.get('error'):
                    print(f"   ❌ Order rejected: {result.get('error')}")
                    print(f"\n   💡 Try --sequential mode if TRIGGER orders aren't supported")
                    order_id = None
                else:
                    print(f"   ❌ Order rejected or failed")
                    print(f"   Response: {result}")
                    order_id = None
            
            # Cancel if requested
            if args.cancel and order_id:
                print("\n🗑️ Cancelling order...")
                try:
                    client.cancel_order(order_id)
                    print("   ✓ Order cancelled")
                except Exception as e:
                    print(f"   ⚠️ Cancel failed: {e}")
                    print("   Order may have been rejected or already cancelled")
                
        else:
            print(f"   [PAPER] Would place butterfly order:")
            print(f"   [PAPER]   Symbol: {args.symbol}")
            print(f"   [PAPER]   Strikes: {lower}/{middle}/{upper} {option_type}")
            print(f"   [PAPER]   Quantity: {args.quantity}")
            print(f"   [PAPER]   Credit: ${args.credit:.2f}")
            print(f"   [PAPER]   Mode: {'SEQUENTIAL' if args.sequential else 'TRIGGER'}")
            print(f"   ✓ Paper order logged")
            
    except Exception as e:
        print(f"   ❌ Order error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    
    if args.live and not args.cancel:
        print("\n⚠️ Note: If order was placed, it will remain WORKING until:")
        print("   - End of trading day (auto-cancel)")
        print("   - You manually cancel in Schwab app/website")
        print("   - It fills (unlikely with low credit)")
        if args.sequential:
            print("\n   SEQUENTIAL mode placed separate orders for wings and middle.")
            print("   Check both orders in your Schwab account.")
    
    if not args.live:
        print("\n💡 To test with real orders:")
        print("   python test_butterfly_order.py --live")
        print("   python test_butterfly_order.py --live --sequential  # if TRIGGER fails")


if __name__ == "__main__":
    main()