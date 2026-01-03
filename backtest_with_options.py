#!/usr/bin/env python3
"""
Butterfly Backtest with Real Option Pricing

This backtest:
1. Fetches historical ES bars for signal detection
2. For EACH signal, fetches the CURRENT SPX option chain
3. Simulates what would have happened if you traded at those prices

IMPORTANT: This gives you TODAY's option prices, not historical.
For true historical backtesting, you'd need a historical options database.

However, this approach lets you:
- Validate the signal logic works
- See realistic option pricing
- Test the execution flow

Usage:
    python backtest_with_options.py --paper   # Paper trade mode
    python backtest_with_options.py --live    # Live signals, real prices
"""

import sys
import os
import time
import logging
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from signal_detector import SignalDetector, Signal, Bar, Direction, SignalType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """Quote for a single option"""
    symbol: str
    strike: float
    option_type: str
    bid: float
    ask: float
    mid: float
    delta: float
    
    
@dataclass
class ButterflyQuote:
    """Quotes for a butterfly spread"""
    lower: OptionQuote
    middle: OptionQuote
    upper: OptionQuote
    
    @property
    def wing_cost(self) -> float:
        """Cost to buy wings at ask"""
        return self.lower.ask + self.upper.ask
    
    @property
    def middle_credit(self) -> float:
        """Credit from selling 2x middle at bid"""
        return self.middle.bid * 2
    
    @property
    def net(self) -> float:
        """Net debit (negative) or credit (positive)"""
        return self.middle_credit - self.wing_cost
    
    @property
    def credit_pct(self) -> float:
        """Credit as percentage of wing cost"""
        if self.wing_cost == 0:
            return 0
        return (self.middle_credit / self.wing_cost - 1) * 100


class SchwabOptionPricer:
    """
    Fetches real option prices from Schwab API.
    """
    
    def __init__(self, client):
        self.client = client
        self._chain_cache = {}
        self._cache_time = None
        self._cache_ttl = 60  # Cache for 60 seconds
    
    def get_option_chain(self, symbol: str = '$SPX', expiry: date = None) -> Dict:
        """Fetch option chain, with caching"""
        
        if expiry is None:
            expiry = date.today()
        
        cache_key = f"{symbol}_{expiry}"
        now = datetime.now()
        
        # Check cache
        if (cache_key in self._chain_cache and 
            self._cache_time and 
            (now - self._cache_time).seconds < self._cache_ttl):
            return self._chain_cache[cache_key]
        
        # Fetch fresh chain
        logger.debug(f"Fetching option chain for {symbol}, expiry {expiry}")
        
        chain = self.client.get_option_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=20,  # Get 20 strikes above/below ATM
            include_quotes=True,
            from_date=expiry,
            to_date=expiry
        )
        
        self._chain_cache[cache_key] = chain
        self._cache_time = now
        
        return chain
    
    def get_option_quote(
        self,
        underlying_price: float,
        strike: float,
        option_type: str,  # 'C' or 'P'
        expiry: date = None
    ) -> Optional[OptionQuote]:
        """Get quote for a specific option"""
        
        chain = self.get_option_chain(expiry=expiry)
        
        if option_type.upper() in ['C', 'CALL']:
            exp_map = chain.get('callExpDateMap', {})
        else:
            exp_map = chain.get('putExpDateMap', {})
        
        # Find the expiration
        for exp_str, strikes in exp_map.items():
            # Look for our strike
            strike_str = f"{strike:.1f}"
            
            if strike_str in strikes:
                opt_data = strikes[strike_str][0]  # First contract at this strike
                
                return OptionQuote(
                    symbol=opt_data.get('symbol', ''),
                    strike=strike,
                    option_type=option_type,
                    bid=opt_data.get('bid', 0),
                    ask=opt_data.get('ask', 0),
                    mid=(opt_data.get('bid', 0) + opt_data.get('ask', 0)) / 2,
                    delta=opt_data.get('delta', 0)
                )
        
        return None
    
    def get_butterfly_quote(
        self,
        underlying_price: float,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        option_type: str,
        expiry: date = None
    ) -> Optional[ButterflyQuote]:
        """Get quotes for a butterfly spread"""
        
        lower = self.get_option_quote(underlying_price, lower_strike, option_type, expiry)
        middle = self.get_option_quote(underlying_price, middle_strike, option_type, expiry)
        upper = self.get_option_quote(underlying_price, upper_strike, option_type, expiry)
        
        if not all([lower, middle, upper]):
            return None
        
        return ButterflyQuote(lower=lower, middle=middle, upper=upper)


def run_live_test():
    """
    Run a live test - watch for signals and get real option prices.
    """
    from schwab_auth import SchwabAuth
    from schwab_client import SchwabClient
    
    print("\n" + "=" * 60)
    print("🦋 BUTTERFLY LIVE SIGNAL TEST")
    print("=" * 60)
    
    # Connect to Schwab
    print("\nConnecting to Schwab API...")
    
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
    pricer = SchwabOptionPricer(client)
    
    # Get current SPX price
    spx_quote = client.get_quote('$SPX')
    spx_price = spx_quote.last_price
    print(f"SPX: ${spx_price:.2f}")
    
    # Calculate ATM strikes
    atm = round(spx_price / 5) * 5
    width = 5
    
    # Test call butterfly
    lower, middle, upper = atm, atm + width, atm + (width * 2)
    print(f"\nCall Butterfly: {lower}/{middle}/{upper}")
    
    bf_quote = pricer.get_butterfly_quote(spx_price, lower, middle, upper, 'C')
    
    if bf_quote:
        print(f"  Lower {lower}C: bid=${bf_quote.lower.bid:.2f}, ask=${bf_quote.lower.ask:.2f}, delta={bf_quote.lower.delta:.2f}")
        print(f"  Middle {middle}C: bid=${bf_quote.middle.bid:.2f}, ask=${bf_quote.middle.ask:.2f}, delta={bf_quote.middle.delta:.2f}")
        print(f"  Upper {upper}C: bid=${bf_quote.upper.bid:.2f}, ask=${bf_quote.upper.ask:.2f}, delta={bf_quote.upper.delta:.2f}")
        print(f"\n  Wing Cost: ${bf_quote.wing_cost:.2f}")
        print(f"  Middle Credit: ${bf_quote.middle_credit:.2f}")
        print(f"  Net: ${bf_quote.net:.2f} ({bf_quote.credit_pct:+.1f}%)")
    else:
        print("  Could not get butterfly quote")
    
    # Test put butterfly
    lower, middle, upper = atm - (width * 2), atm - width, atm
    print(f"\nPut Butterfly: {lower}/{middle}/{upper}")
    
    bf_quote = pricer.get_butterfly_quote(spx_price, lower, middle, upper, 'P')
    
    if bf_quote:
        print(f"  Lower {lower}P: bid=${bf_quote.lower.bid:.2f}, ask=${bf_quote.lower.ask:.2f}, delta={bf_quote.lower.delta:.2f}")
        print(f"  Middle {middle}P: bid=${bf_quote.middle.bid:.2f}, ask=${bf_quote.middle.ask:.2f}, delta={bf_quote.middle.delta:.2f}")
        print(f"  Upper {upper}P: bid=${bf_quote.upper.bid:.2f}, ask=${bf_quote.upper.ask:.2f}, delta={bf_quote.upper.delta:.2f}")
        print(f"\n  Wing Cost: ${bf_quote.wing_cost:.2f}")
        print(f"  Middle Credit: ${bf_quote.middle_credit:.2f}")
        print(f"  Net: ${bf_quote.net:.2f} ({bf_quote.credit_pct:+.1f}%)")
    else:
        print("  Could not get butterfly quote")
    
    print("\n" + "=" * 60)
    print("This shows CURRENT option prices.")
    print("For backtesting, you'd need historical option data.")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Butterfly Backtest with Real Options')
    parser.add_argument('--live', action='store_true', help='Run live signal test')
    
    args = parser.parse_args()
    
    if args.live:
        run_live_test()
    else:
        print("Usage:")
        print("  python backtest_with_options.py --live   # Test with live option prices")


if __name__ == "__main__":
    main()