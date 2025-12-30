"""
Debug script to test Schwab API connection
"""
import os
import json
import requests
from schwab_auth import SchwabAuth

def test_api():
    # Load auth
    auth = SchwabAuth(
        app_key=os.getenv("SCHWAB_APP_KEY", ""),
        app_secret=os.getenv("SCHWAB_APP_SECRET", ""),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
        token_file="schwab_tokens.json"
    )
    
    if not auth.is_authenticated:
        print("❌ Not authenticated. Run trading_bot.py first to authenticate.")
        return
    
    # Refresh token
    print("Refreshing token...")
    auth.refresh_access_token()
    
    headers = auth.get_headers()
    print(f"\n✓ Got access token: {auth.access_token[:30]}...")
    
    # Test 1: Try accounts endpoint (new format)
    print("\n" + "=" * 50)
    print("TEST 1: Account Numbers")
    print("=" * 50)
    
    url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
    print(f"URL: {url}")
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    # Test 2: Try accounts without accountNumbers
    print("\n" + "=" * 50)
    print("TEST 2: Accounts (direct)")
    print("=" * 50)
    
    url = "https://api.schwabapi.com/trader/v1/accounts"
    print(f"URL: {url}")
    
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    # Test 3: Try market data (different API)
    print("\n" + "=" * 50)
    print("TEST 3: Market Data - SPY Quote")
    print("=" * 50)
    
    url = "https://api.schwabapi.com/marketdata/v1/quotes"
    params = {"symbols": "SPY", "fields": "quote"}
    print(f"URL: {url}")
    
    response = requests.get(url, headers=headers, params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    # Test 4: Check token info
    print("\n" + "=" * 50)
    print("TOKEN INFO")
    print("=" * 50)
    
    with open("schwab_tokens.json", "r") as f:
        tokens = json.load(f)
    
    print(f"Access token (first 50 chars): {tokens.get('access_token', '')[:50]}...")
    print(f"Refresh token exists: {bool(tokens.get('refresh_token'))}")
    print(f"Expires at: {tokens.get('expires_at')}")
    
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

if __name__ == "__main__":
    test_api()
