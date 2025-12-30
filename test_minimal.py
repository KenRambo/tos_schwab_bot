"""
Minimal API test - debug exact request
"""
import requests

# Your token (paste from schwab_tokens.json)
access_token = "I0.b2F1dGgyLmJkYy5zY2h3YWIuY29t.AjtMVF0lCug2I2LvJ9yW-Vllv1M_sQ_AlXsntJMIvTM@"

# Test 1: Accounts
print("=" * 50)
print("TEST 1: Account Numbers")
print("=" * 50)

url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
}

print(f"URL: {url}")
print(f"Auth header: Bearer {access_token[:30]}...")

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 2: Market Data (might have different auth requirements)
print("\n" + "=" * 50)
print("TEST 2: SPY Quote")
print("=" * 50)

url = "https://api.schwabapi.com/marketdata/v1/quotes?symbols=SPY"
response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:500] if response.text else 'empty'}")

# Test 3: Try with Content-Type header too
print("\n" + "=" * 50)
print("TEST 3: With Content-Type header")
print("=" * 50)

url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 4: Accounts list (not accountNumbers)
print("\n" + "=" * 50)
print("TEST 4: Accounts list")
print("=" * 50)

url = "https://api.schwabapi.com/trader/v1/accounts"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
}

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
