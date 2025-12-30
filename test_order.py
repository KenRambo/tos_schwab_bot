"""
Test Live Order Placement

Places a small limit order far from market price, verifies it works, then cancels.
This confirms your API can actually place orders.
"""
import os
import sys
import time
from datetime import date, timedelta

from schwab_auth import SchwabAuth
from schwab_client import SchwabClient, OrderType, OrderInstruction

def test_order():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ORDER PLACEMENT TEST                                 ║
    ║         Will place a limit order and immediately cancel      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Connect to Schwab
    auth = SchwabAuth(
        app_key=os.getenv("SCHWAB_APP_KEY", ""),
        app_secret=os.getenv("SCHWAB_APP_SECRET", ""),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
        token_file="schwab_tokens.json"
    )
    
    if not auth.is_authenticated:
        print("❌ Not authenticated. Run trading_bot.py first.")
        return False
    
    auth.refresh_access_token()
    client = SchwabClient(auth)
    
    # Get account info
    print("1. Checking account...")
    account_hash = client.get_account_hash()
    print(f"   ✓ Account hash: {account_hash[:16]}...")
    
    # Get SPY quote
    print("\n2. Getting SPY quote...")
    quote = client.get_quote("SPY")
    print(f"   ✓ SPY last price: ${quote.last_price:.2f}")
    
    # Find a cheap OTM option
    print("\n3. Finding a cheap OTM option...")
    
    # Look for a PUT way OTM (will be cheap)
    option = client.get_nearest_otm_option(
        symbol="SPY",
        option_type="PUT",
        min_dte=1,
        max_dte=7
    )
    
    if not option:
        print("   ❌ Could not find an option. Market may be closed.")
        return False
    
    print(f"   ✓ Found: {option.symbol}")
    print(f"     Strike: ${option.strike}")
    print(f"     Expiry: {option.expiration}")
    print(f"     Bid: ${option.bid:.2f} / Ask: ${option.ask:.2f}")
    
    # Place a limit order WAY below market (won't fill)
    limit_price = 0.01  # $1 total - won't fill, just testing
    
    print(f"\n4. Placing test order...")
    print(f"   Symbol: {option.symbol}")
    print(f"   Action: BUY TO OPEN")
    print(f"   Quantity: 1")
    print(f"   Type: LIMIT @ ${limit_price:.2f} (won't fill - too low)")
    
    confirm = input("\n   Type 'YES' to place test order: ")
    if confirm.upper() != 'YES':
        print("   Cancelled.")
        return False
    
    try:
        result = client.place_option_order(
            option_symbol=option.symbol,
            instruction=OrderInstruction.BUY_TO_OPEN,
            quantity=1,
            order_type=OrderType.LIMIT,
            price=limit_price
        )
        print(f"   ✓ Order placed!")
        print(f"   Response: {result}")
        
    except Exception as e:
        print(f"   ❌ Order failed: {e}")
        return False
    
    # Wait a moment
    print("\n5. Waiting 2 seconds...")
    time.sleep(2)
    
    # Check orders
    print("\n6. Checking open orders...")
    try:
        orders = client.get_orders()
        print(f"   Found {len(orders)} orders")
        
        for order in orders:
            print(f"   - {order.order_id}: {order.symbol} {order.status}")
            
            # Cancel test order
            if order.status in ['QUEUED', 'PENDING_ACTIVATION', 'WORKING', 'AWAITING_PARENT_ORDER']:
                print(f"\n7. Cancelling test order {order.order_id}...")
                cancelled = client.cancel_order(order.order_id)
                if cancelled:
                    print("   ✓ Order cancelled successfully!")
                else:
                    print("   ⚠️ Could not cancel - may need to cancel manually in Schwab")
                    
    except Exception as e:
        print(f"   ❌ Error checking orders: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\n✓ If you saw 'Order placed!' - your bot can place real orders!")
    print("✓ Check your Schwab account to confirm no orders are hanging.")
    
    return True


if __name__ == "__main__":
    test_order()
