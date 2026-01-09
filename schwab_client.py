"""
Schwab API Client

Handles all trading operations including:
- Account information
- Market data (quotes, options chains)
- Order placement and management
- Complex multi-leg orders (butterflies, spreads)

UPDATED 2026-01-08:
- Added place_butterfly_order() for butterfly credit spreads
- Added get_order_status() for fill verification
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import requests

from schwab_auth import SchwabAuth

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    NET_CREDIT = "NET_CREDIT"
    NET_DEBIT = "NET_DEBIT"


class OrderInstruction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class Duration(Enum):
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    FOK = "FILL_OR_KILL"


class AssetType(Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    INDEX = "INDEX"


@dataclass
class Quote:
    """Stock/ETF quote data"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    high: float
    low: float
    open: float
    close: float
    volume: int
    timestamp: datetime


@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str  # Full OCC symbol
    underlying: str
    strike: float
    expiration: date
    option_type: str  # "CALL" or "PUT"
    bid: float
    ask: float
    last: float
    delta: float
    gamma: float
    theta: float
    vega: float
    volume: int
    open_interest: int
    in_the_money: bool


@dataclass
class Position:
    """Account position"""
    symbol: str
    asset_type: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


@dataclass
class Order:
    """Order details"""
    order_id: str
    symbol: str
    instruction: str
    quantity: int
    order_type: str
    status: str
    price: Optional[float]
    filled_quantity: int
    remaining_quantity: int
    entered_time: datetime


class SchwabClient:
    """Schwab API Client for trading operations"""
    
    BASE_URL = "https://api.schwabapi.com"
    TRADER_URL = f"{BASE_URL}/trader/v1"
    MARKETDATA_URL = f"{BASE_URL}/marketdata/v1"
    
    def __init__(self, auth: SchwabAuth):
        self.auth = auth
        self._account_hash: Optional[str] = None
    
    def _request(self, method: str, url: str, retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request with retry logic"""
        # Only include Content-Type for POST/PUT/PATCH
        include_content_type = method.upper() in ('POST', 'PUT', 'PATCH')
        headers = self.auth.get_headers(include_content_type=include_content_type)
        
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.request(method, url, headers=headers, **kwargs)
                
                # If 500 error, retry
                if response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, attempt {attempt + 1}/{retries}")
                    if attempt < retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                response.raise_for_status()
                
                if response.content:
                    return response.json()
                return {}
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response: {e.response.text}")
                
                if attempt < retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
        
        raise last_error
    
    # ==================== ACCOUNT METHODS ====================
    
    def get_account_numbers(self) -> List[Dict[str, str]]:
        """Get all linked account numbers and hashes"""
        url = f"{self.TRADER_URL}/accounts/accountNumbers"
        response = self._request("GET", url)
        return response if isinstance(response, list) else []
    
    def get_account_hash(self) -> str:
        """Get the primary account hash (cached)"""
        if self._account_hash:
            return self._account_hash
        
        accounts = self.get_account_numbers()
        if not accounts:
            raise ValueError("No accounts found")
        
        # Use first account
        self._account_hash = accounts[0]['hashValue']
        logger.info(f"Using account: {accounts[0].get('accountNumber', 'Unknown')}")
        return self._account_hash
    
    def get_account(self, include_positions: bool = True) -> Dict[str, Any]:
        """Get account details including positions"""
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}"
        
        params = {}
        if include_positions:
            params['fields'] = 'positions'
        
        return self._request("GET", url, params=params)
    
    def get_buying_power(self) -> float:
        """Get available buying power for the account"""
        account = self.get_account(include_positions=False)
        securities = account.get('securitiesAccount', {})
        balances = securities.get('currentBalances', {})
        
        # Try different balance fields
        buying_power = balances.get('buyingPower', 0)
        if buying_power == 0:
            buying_power = balances.get('availableFunds', 0)
        if buying_power == 0:
            buying_power = balances.get('cashBalance', 0)
        
        return float(buying_power)
    
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        account = self.get_account(include_positions=True)
        positions = []
        
        securities = account.get('securitiesAccount', {})
        for pos in securities.get('positions', []):
            instrument = pos.get('instrument', {})
            
            positions.append(Position(
                symbol=instrument.get('symbol', ''),
                asset_type=instrument.get('assetType', ''),
                quantity=pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
                average_price=pos.get('averagePrice', 0),
                current_price=pos.get('currentDayProfitLossPercentage', 0),  # Placeholder
                market_value=pos.get('marketValue', 0),
                unrealized_pnl=pos.get('currentDayProfitLoss', 0),
                unrealized_pnl_percent=pos.get('currentDayProfitLossPercentage', 0)
            ))
        
        return positions
    
    def get_option_positions(self, underlying: str = None) -> List[Position]:
        """Get option positions, optionally filtered by underlying"""
        positions = self.get_positions()
        options = [p for p in positions if p.asset_type == 'OPTION']
        
        if underlying:
            options = [p for p in options if underlying.upper() in p.symbol.upper()]
        
        return options
    
    # ==================== MARKET DATA METHODS ====================
    
    def get_quote(self, symbol: str) -> Quote:
        """Get quote for a symbol"""
        url = f"{self.MARKETDATA_URL}/quotes"
        params = {'symbols': symbol, 'fields': 'quote'}
        
        response = self._request("GET", url, params=params)
        
        quote_data = response.get(symbol, {}).get('quote', {})
        
        return Quote(
            symbol=symbol,
            last_price=quote_data.get('lastPrice', 0),
            bid=quote_data.get('bidPrice', 0),
            ask=quote_data.get('askPrice', 0),
            high=quote_data.get('highPrice', 0),
            low=quote_data.get('lowPrice', 0),
            open=quote_data.get('openPrice', 0),
            close=quote_data.get('closePrice', 0),
            volume=quote_data.get('totalVolume', 0),
            timestamp=datetime.now()
        )
    
    def get_price_history(
        self,
        symbol: str,
        period_type: str = "day",  # day, month, year, ytd
        period: int = 1,
        frequency_type: str = "minute",  # minute, daily, weekly, monthly
        frequency: int = 5,  # 1, 5, 10, 15, 30 for minute
        extended_hours: bool = True,  # Include pre/post market
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict]:
        """
        Get historical price candles/bars.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            period_type: day, month, year, ytd
            period: Number of periods
            frequency_type: minute, daily, weekly, monthly
            frequency: Bar interval (1, 5, 10, 15, 30 for minute)
            extended_hours: Include extended hours data
            start_date: Start datetime (optional)
            end_date: End datetime (optional)
            
        Returns:
            List of candle dicts with: datetime, open, high, low, close, volume
        """
        url = f"{self.MARKETDATA_URL}/pricehistory"
        
        params = {
            'symbol': symbol,
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency,
            'needExtendedHoursData': str(extended_hours).lower()
        }
        
        # Use specific date range if provided
        if start_date:
            params['startDate'] = int(start_date.timestamp() * 1000)
        if end_date:
            params['endDate'] = int(end_date.timestamp() * 1000)
        
        response = self._request("GET", url, params=params)
        
        candles = response.get('candles', [])
        
        # Convert to list of dicts with proper datetime
        bars = []
        for candle in candles:
            bars.append({
                'datetime': datetime.fromtimestamp(candle['datetime'] / 1000),
                'open': candle.get('open', 0),
                'high': candle.get('high', 0),
                'low': candle.get('low', 0),
                'close': candle.get('close', 0),
                'volume': candle.get('volume', 0)
            })
        
        return bars
    
    def get_recent_bars(
        self,
        symbol: str,
        num_bars: int = 20,
        bar_minutes: int = 5,
        extended_hours: bool = True
    ) -> List[Dict]:
        """
        Get the most recent N bars for initialization.
        
        Args:
            symbol: Stock symbol
            num_bars: Number of bars to fetch
            bar_minutes: Bar interval in minutes (5, 10, 15, 30)
            extended_hours: Include extended hours
            
        Returns:
            List of the most recent bars
        """
        # Use explicit date range to ensure we get today's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Go back 5 days to handle weekends/holidays
        
        bars = self.get_price_history(
            symbol=symbol,
            period_type="day",
            period=5,  # 5 days to handle weekends
            frequency_type="minute",
            frequency=bar_minutes,
            extended_hours=extended_hours,
            start_date=start_date,
            end_date=end_date
        )
        
        # Log what we got for debugging
        if bars:
            logger.debug(f"Got {len(bars)} bars from {bars[0]['datetime']} to {bars[-1]['datetime']}")
        else:
            logger.warning("No bars returned from price history")
        
        # Return the last N bars
        if len(bars) > num_bars:
            return bars[-num_bars:]
        return bars
    
    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "ALL",  # "CALL", "PUT", or "ALL"
        strike_count: int = 10,
        include_quotes: bool = True,
        from_date: date = None,
        to_date: date = None,
        expiration_month: str = None
    ) -> Dict[str, Any]:
        """
        Get option chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            contract_type: CALL, PUT, or ALL
            strike_count: Number of strikes above/below ATM
            include_quotes: Include bid/ask quotes
            from_date: Start date for expirations
            to_date: End date for expirations
        """
        url = f"{self.MARKETDATA_URL}/chains"
        
        params = {
            'symbol': symbol,
            'contractType': contract_type,
            'strikeCount': strike_count,
            'includeQuotes': str(include_quotes).upper(),
            'strategy': 'SINGLE'
        }
        
        if from_date:
            params['fromDate'] = from_date.isoformat()
        if to_date:
            params['toDate'] = to_date.isoformat()
        if expiration_month:
            params['expirationMonth'] = expiration_month
        
        return self._request("GET", url, params=params)
    
    def get_nearest_otm_option(
        self,
        symbol: str,
        option_type: str,  # "CALL" or "PUT"
        min_dte: int = 0,
        max_dte: int = 0,  # 0 = today only (0DTE)
        target_delta: float = None  # If set, find option closest to this delta
    ) -> Optional[OptionContract]:
        """
        Find an option contract by delta or nearest OTM.
        
        Args:
            symbol: Underlying symbol (e.g., "SPY")
            option_type: "CALL" or "PUT"
            min_dte: Minimum days to expiration (0 = today)
            max_dte: Maximum days to expiration (0 = today only)
            target_delta: Target delta (e.g., 0.30 for 30 delta). If None, uses nearest OTM.
        
        For delta targeting:
            - CALL: Finds option closest to +target_delta (e.g., +0.30)
            - PUT: Finds option closest to -target_delta (e.g., -0.30)
        
        For nearest OTM (when target_delta is None):
            - CALL: First strike above current price
            - PUT: First strike below current price
        """
        # Get current price
        quote = self.get_quote(symbol)
        current_price = quote.last_price
        
        # Check if market is closed - if so, use next trading day
        now = datetime.now()
        market_closed = now.hour >= 16 or now.hour < 9 or (now.hour == 9 and now.minute < 30)
        is_weekend = now.weekday() >= 5
        
        if market_closed or is_weekend:
            # Adjust to next trading day
            days_ahead = 1
            if now.weekday() == 4 and now.hour >= 16:  # Friday after close
                days_ahead = 3  # Monday
            elif now.weekday() == 5:  # Saturday
                days_ahead = 2  # Monday
            elif now.weekday() == 6:  # Sunday
                days_ahead = 1  # Monday
            
            base_date = date.today() + timedelta(days=days_ahead)
            logger.info(f"Market closed - using next trading day: {base_date}")
        else:
            base_date = date.today()
        
        if target_delta:
            logger.info(f"Looking for {option_type} option, {symbol} @ {current_price:.2f}, target delta: {target_delta:.0%}, DTE: {min_dte}-{max_dte}")
        else:
            logger.info(f"Looking for {option_type} option, {symbol} @ {current_price:.2f}, DTE: {min_dte}-{max_dte}")
        
        # Get option chain
        from_date = base_date + timedelta(days=min_dte)
        to_date = base_date + timedelta(days=max_dte if max_dte > 0 else 1)
        
        chain = self.get_option_chain(
            symbol=symbol,
            contract_type=option_type,
            strike_count=30,  # Get more strikes for delta targeting
            from_date=from_date,
            to_date=to_date
        )
        
        # Parse options
        if option_type == "CALL":
            exp_map = chain.get('callExpDateMap', {})
        else:
            exp_map = chain.get('putExpDateMap', {})
        
        best_option = None
        best_delta_diff = float('inf')
        best_distance = float('inf')
        best_dte = float('inf')
        
        for exp_date_str, strikes in exp_map.items():
            # Parse expiration date (format: "2025-12-30:0")
            try:
                exp_date = datetime.strptime(exp_date_str.split(':')[0], '%Y-%m-%d').date()
            except:
                continue
            
            # Check DTE from base date
            dte = (exp_date - base_date).days
            
            # For 0DTE, only accept the base date's expiration
            if max_dte == 0 and dte != 0:
                continue
            
            if dte < min_dte or dte > max(max_dte, 1):
                continue
            
            for strike_str, options in strikes.items():
                strike = float(strike_str)
                opt_data = options[0] if options else {}
                
                # Get delta from option data
                option_delta = opt_data.get('delta', 0)
                
                if target_delta:
                    # Delta targeting mode
                    # For PUTs, delta is negative, so we compare absolute values
                    if option_type == "PUT":
                        delta_diff = abs(abs(option_delta) - target_delta)
                    else:
                        delta_diff = abs(option_delta - target_delta)
                    
                    # Skip if delta is 0 (no data) or too far ITM/OTM
                    if option_delta == 0:
                        continue
                    
                    # Prefer: 1) Closer to target delta, 2) Sooner expiration
                    if delta_diff < best_delta_diff or (delta_diff == best_delta_diff and dte < best_dte):
                        best_delta_diff = delta_diff
                        best_dte = dte
                        best_option = OptionContract(
                            symbol=opt_data.get('symbol', ''),
                            underlying=symbol,
                            strike=strike,
                            expiration=exp_date,
                            option_type=option_type,
                            bid=opt_data.get('bid', 0),
                            ask=opt_data.get('ask', 0),
                            last=opt_data.get('last', 0),
                            delta=option_delta,
                            gamma=opt_data.get('gamma', 0),
                            theta=opt_data.get('theta', 0),
                            vega=opt_data.get('vega', 0),
                            volume=opt_data.get('totalVolume', 0),
                            open_interest=opt_data.get('openInterest', 0),
                            in_the_money=opt_data.get('inTheMoney', False)
                        )
                else:
                    # Nearest OTM mode (original behavior)
                    # Check if OTM
                    if option_type == "CALL" and strike <= current_price:
                        continue
                    if option_type == "PUT" and strike >= current_price:
                        continue
                    
                    # Calculate distance from ATM
                    distance = abs(strike - current_price)
                    
                    # Prefer: 1) Closer to ATM, 2) Sooner expiration
                    if dte < best_dte or (dte == best_dte and distance < best_distance):
                        best_distance = distance
                        best_dte = dte
                        
                        best_option = OptionContract(
                            symbol=opt_data.get('symbol', ''),
                            underlying=symbol,
                            strike=strike,
                            expiration=exp_date,
                            option_type=option_type,
                            bid=opt_data.get('bid', 0),
                            ask=opt_data.get('ask', 0),
                            last=opt_data.get('last', 0),
                            delta=option_delta,
                            gamma=opt_data.get('gamma', 0),
                            theta=opt_data.get('theta', 0),
                            vega=opt_data.get('vega', 0),
                            volume=opt_data.get('totalVolume', 0),
                            open_interest=opt_data.get('openInterest', 0),
                            in_the_money=opt_data.get('inTheMoney', False)
                        )
        
        if best_option:
            delta_str = f" | Delta: {best_option.delta:.2f}" if best_option.delta else ""
            logger.info(f"Selected: {best_option.symbol} | Strike: {best_option.strike} | Exp: {best_option.expiration} | DTE: {best_dte}{delta_str}")
        else:
            logger.warning(f"No {option_type} option found for {symbol} with DTE {min_dte}-{max_dte}")
        
        return best_option
    
    # ==================== ORDER METHODS ====================
    
    def place_option_order(
        self,
        option_symbol: str,
        instruction: OrderInstruction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        duration: Duration = Duration.DAY
    ) -> Dict[str, Any]:
        """
        Place an option order.
        
        Args:
            option_symbol: Full OCC option symbol
            instruction: BUY_TO_OPEN, SELL_TO_CLOSE, etc.
            quantity: Number of contracts
            order_type: MARKET, LIMIT, etc.
            price: Limit price (required for LIMIT orders)
            duration: DAY, GTC, etc.
        """
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders"
        
        order = {
            "orderType": order_type.value,
            "session": "NORMAL",
            "duration": duration.value,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction.value,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": option_symbol,
                        "assetType": "OPTION"
                    }
                }
            ]
        }
        
        if order_type == OrderType.LIMIT and price is not None:
            order["price"] = str(price)
        
        response = self._request("POST", url, json=order)
        
        # Get order ID from response headers if available
        logger.info(f"Order placed: {instruction.value} {quantity} {option_symbol}")
        return response
    
    def buy_to_open(
        self,
        option_symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = None
    ) -> Dict[str, Any]:
        """Buy to open an option position"""
        return self.place_option_order(
            option_symbol=option_symbol,
            instruction=OrderInstruction.BUY_TO_OPEN,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
    
    def sell_to_close(
        self,
        option_symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = None
    ) -> Dict[str, Any]:
        """Sell to close an option position"""
        return self.place_option_order(
            option_symbol=option_symbol,
            instruction=OrderInstruction.SELL_TO_CLOSE,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
    
    # ==================== COMPLEX ORDER METHODS ====================
    
    def _build_option_symbol(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        option_type: str
    ) -> str:
        """
        Build OCC option symbol for Schwab API.
        
        Format (21 characters total):
        - Underlying Symbol: 6 characters (padded with trailing spaces)
        - Expiration: 6 characters (YYMMDD)
        - Call/Put: 1 character (C or P)
        - Strike Price: 8 characters (strike * 1000, zero-padded)
        
        Example: 'SPXW  260109P05910000' (SPXW + 2 spaces, Jan 9 2026, 5910 Put)
        
        From official Schwab docs:
        "XYZ 210115C00050000" = XYZ (padded to 6), Jan 15 2021, $50 Call
        """
        # Clean underlying (remove $ prefix if present)
        underlying_clean = underlying.replace('$', '').replace('.X', '').upper()
        
        # Determine option root symbol
        if underlying_clean in ['SPX', 'SPXW']:
            root = 'SPXW'
        elif underlying_clean == 'XSP':
            root = 'XSP'
        elif underlying_clean in ['NDX', 'NDXP']:
            root = 'NDXP'
        else:
            root = underlying_clean
        
        # Pad root to 6 characters with trailing spaces
        root_padded = f"{root:<6}"
        
        # Format expiration
        exp_str = expiration.strftime('%y%m%d')
        
        # Option type
        opt_char = 'C' if option_type.upper() in ['C', 'CALL'] else 'P'
        
        # Strike: 8 digits (strike * 1000, zero-padded)
        strike_val = int(strike * 1000)
        strike_str = f"{strike_val:08d}"
        
        # Format: 6-char root + YYMMDD + C/P + 8-digit strike = 21 chars total
        symbol = f"{root_padded}{exp_str}{opt_char}{strike_str}"
        logger.debug(f"Built option symbol: {underlying} {strike} {option_type} -> '{symbol}' ({len(symbol)} chars)")
        
        return symbol
        
        return symbol
    
    def place_butterfly_order(
        self,
        symbol: str,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        option_type: str,  # "CALL" or "PUT"
        quantity: int = 1,
        limit_credit: float = None,
        duration: Duration = Duration.DAY
    ) -> Dict[str, Any]:
        """
        Place a butterfly credit spread as a triggered order (OTO).
        
        Structure:
        1. PRIMARY ORDER: Buy wings (lower + upper) at market
        2. TRIGGERED ORDER: Sell 2x middle at limit (for net credit)
        
        Args:
            symbol: Underlying symbol (e.g., "SPX", "$SPX")
            lower_strike: Lower wing strike price
            middle_strike: Middle (short) strike price
            upper_strike: Upper wing strike price
            option_type: "CALL" or "PUT"
            quantity: Number of butterfly spreads
            limit_credit: Net credit target (used to calculate middle limit price)
            duration: Order duration (DAY, GTC)
        
        Returns:
            Order response with orderId if successful
        """
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders"
        
        # Determine expiration (0DTE)
        now = datetime.now()
        if now.hour >= 16 or now.weekday() >= 5:
            # After hours or weekend - use next trading day
            days_ahead = 1
            if now.weekday() == 4 and now.hour >= 16:
                days_ahead = 3
            elif now.weekday() == 5:
                days_ahead = 2
            elif now.weekday() == 6:
                days_ahead = 1
            expiration = date.today() + timedelta(days=days_ahead)
        else:
            expiration = date.today()
        
        # Build option symbols
        lower_symbol = self._build_option_symbol(symbol, expiration, lower_strike, option_type)
        middle_symbol = self._build_option_symbol(symbol, expiration, middle_strike, option_type)
        upper_symbol = self._build_option_symbol(symbol, expiration, upper_strike, option_type)
        
        logger.info(f"Building butterfly TRIGGER order:")
        logger.info(f"  Underlying: {symbol}")
        logger.info(f"  Lower:  '{lower_symbol}'")
        logger.info(f"  Middle: '{middle_symbol}'")
        logger.info(f"  Upper:  '{upper_symbol}'")
        
        # For the triggered sell, we need to calculate the limit price for the middle
        # If user wants net credit X, and wings cost W, middle must sell for (W + X) / 2 each
        # But since we're buying wings at market, we'll use limit_credit as the middle sell price
        # In practice: middle_limit = (estimated_wing_cost + limit_credit) / 2
        # For simplicity, we'll set the middle to sell at a high limit that ensures credit
        middle_limit_each = limit_credit if limit_credit else 0.01
        
        # Build TRIGGER order (OTO - One Triggers Other)
        # Primary: Buy both wings at MARKET
        # Child: Sell 2x middle at LIMIT (triggered when wings fill)
        order = {
            "orderStrategyType": "TRIGGER",
            "session": "NORMAL",
            "duration": duration.value,
            "orderType": "MARKET",
            "orderLegCollection": [
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": lower_symbol,
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": upper_symbol,
                        "assetType": "OPTION"
                    }
                }
            ],
            "childOrderStrategies": [
                {
                    "orderStrategyType": "SINGLE",
                    "session": "NORMAL",
                    "duration": duration.value,
                    "orderType": "LIMIT",
                    "price": str(middle_limit_each),
                    "orderLegCollection": [
                        {
                            "instruction": "SELL_TO_OPEN",
                            "quantity": quantity * 2,
                            "instrument": {
                                "symbol": middle_symbol,
                                "assetType": "OPTION"
                            }
                        }
                    ]
                }
            ]
        }
        
        logger.info(f"Placing butterfly TRIGGER order: {quantity}x {lower_strike}/{middle_strike}/{upper_strike} {option_type}")
        logger.info(f"  Step 1: BUY {quantity}x {lower_strike} + {quantity}x {upper_strike} @ MARKET")
        logger.info(f"  Step 2: SELL {quantity * 2}x {middle_strike} @ ${middle_limit_each:.2f} LIMIT (triggered)")
        
        try:
            # Make the request and capture response headers for order ID
            include_content_type = True
            headers = self.auth.get_headers(include_content_type=include_content_type)
            
            response = requests.post(url, headers=headers, json=order)
            
            # Check for errors
            if response.status_code >= 400:
                logger.error(f"Order rejected: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            # Try to get order ID from Location header
            location = response.headers.get('Location', '')
            order_id = location.split('/')[-1] if location else None
            
            # Also try to parse response body
            result = {}
            if response.content:
                try:
                    result = response.json()
                except:
                    pass
            
            if order_id:
                result['orderId'] = order_id
                logger.info(f"Butterfly TRIGGER order placed: {order_id}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Butterfly order failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def place_butterfly_order_sequential(
        self,
        symbol: str,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        option_type: str,
        quantity: int = 1,
        limit_credit: float = None,
        wait_for_fill: bool = True,
        max_wait_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Place butterfly as sequential orders (fallback if TRIGGER not supported).
        
        1. Buy wings at market
        2. Wait for fill
        3. Sell middle at limit
        
        Args:
            symbol: Underlying symbol
            lower_strike, middle_strike, upper_strike: Strike prices
            option_type: "CALL" or "PUT"
            quantity: Number of butterflies
            limit_credit: Target credit (used for middle limit price)
            wait_for_fill: Whether to wait for wing fill before placing middle
            max_wait_seconds: Max time to wait for wing fill
        
        Returns:
            Dict with wing_order_id, middle_order_id, status
        """
        import time
        
        # Determine expiration
        now = datetime.now()
        if now.hour >= 16 or now.weekday() >= 5:
            days_ahead = 1
            if now.weekday() == 4 and now.hour >= 16:
                days_ahead = 3
            elif now.weekday() == 5:
                days_ahead = 2
            elif now.weekday() == 6:
                days_ahead = 1
            expiration = date.today() + timedelta(days=days_ahead)
        else:
            expiration = date.today()
        
        # Build symbols
        lower_symbol = self._build_option_symbol(symbol, expiration, lower_strike, option_type)
        middle_symbol = self._build_option_symbol(symbol, expiration, middle_strike, option_type)
        upper_symbol = self._build_option_symbol(symbol, expiration, upper_strike, option_type)
        
        result = {
            'wing_order_id': None,
            'middle_order_id': None,
            'status': 'PENDING',
            'wing_fill_price': None,
            'middle_fill_price': None
        }
        
        # Step 1: Place wing orders (can be combined or separate)
        logger.info(f"Step 1: Placing wing orders...")
        
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders"
        
        # Combined wing order
        wing_order = {
            "orderStrategyType": "SINGLE",
            "session": "NORMAL",
            "duration": "DAY",
            "orderType": "MARKET",
            "orderLegCollection": [
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": lower_symbol,
                        "assetType": "OPTION"
                    }
                },
                {
                    "instruction": "BUY_TO_OPEN",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": upper_symbol,
                        "assetType": "OPTION"
                    }
                }
            ]
        }
        
        try:
            headers = self.auth.get_headers(include_content_type=True)
            response = requests.post(url, headers=headers, json=wing_order)
            
            if response.status_code >= 400:
                logger.error(f"Wing order rejected: {response.text}")
                result['status'] = 'WING_REJECTED'
                result['error'] = response.text
                return result
            
            location = response.headers.get('Location', '')
            wing_order_id = location.split('/')[-1] if location else None
            result['wing_order_id'] = wing_order_id
            logger.info(f"Wing order placed: {wing_order_id}")
            
        except Exception as e:
            logger.error(f"Wing order failed: {e}")
            result['status'] = 'WING_FAILED'
            result['error'] = str(e)
            return result
        
        # Step 2: Wait for fill (if requested)
        if wait_for_fill and wing_order_id:
            logger.info(f"Waiting for wing fill (max {max_wait_seconds}s)...")
            start_time = time.time()
            
            while time.time() - start_time < max_wait_seconds:
                status = self.get_order_status(wing_order_id)
                if status:
                    order_status = status.get('status', '').upper()
                    if order_status == 'FILLED':
                        result['wing_fill_price'] = status.get('filledPrice') or status.get('price')
                        logger.info(f"Wings filled @ ${result['wing_fill_price']}")
                        break
                    elif order_status in ['REJECTED', 'CANCELLED', 'EXPIRED']:
                        logger.error(f"Wing order {order_status}")
                        result['status'] = f'WING_{order_status}'
                        return result
                time.sleep(0.5)
            else:
                logger.warning("Wing fill timeout - proceeding anyway")
        
        # Step 3: Place middle sell order
        logger.info(f"Step 2: Placing middle sell order...")
        
        middle_limit = limit_credit if limit_credit else 0.01
        
        middle_order = {
            "orderStrategyType": "SINGLE",
            "session": "NORMAL",
            "duration": "DAY",
            "orderType": "LIMIT",
            "price": str(middle_limit),
            "orderLegCollection": [
                {
                    "instruction": "SELL_TO_OPEN",
                    "quantity": quantity * 2,
                    "instrument": {
                        "symbol": middle_symbol,
                        "assetType": "OPTION"
                    }
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=middle_order)
            
            if response.status_code >= 400:
                logger.error(f"Middle order rejected: {response.text}")
                result['status'] = 'MIDDLE_REJECTED'
                result['error'] = response.text
                return result
            
            location = response.headers.get('Location', '')
            middle_order_id = location.split('/')[-1] if location else None
            result['middle_order_id'] = middle_order_id
            result['status'] = 'PLACED'
            result['orderId'] = wing_order_id  # Return primary order ID for tracking
            
            logger.info(f"Middle order placed: {middle_order_id}")
            logger.info(f"Butterfly order complete: wings={wing_order_id}, middle={middle_order_id}")
            
        except Exception as e:
            logger.error(f"Middle order failed: {e}")
            result['status'] = 'MIDDLE_FAILED'
            result['error'] = str(e)
        
        return result
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order details including status, filledQuantity, price, etc.
        """
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}"
        
        try:
            response = self._request("GET", url)
            return response
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    def get_orders(
        self,
        from_date: datetime = None,
        to_date: datetime = None,
        status: str = None
    ) -> List[Order]:
        """Get orders for the account"""
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders"
        
        # Schwab requires fromEnteredTime and toEnteredTime
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1)
        if to_date is None:
            to_date = datetime.now() + timedelta(days=1)
        
        params = {
            'fromEnteredTime': from_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'toEnteredTime': to_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        
        if status:
            params['status'] = status
        
        response = self._request("GET", url, params=params)
        orders = []
        
        for order_data in response if isinstance(response, list) else []:
            legs = order_data.get('orderLegCollection', [])
            symbol = legs[0].get('instrument', {}).get('symbol', '') if legs else ''
            instruction = legs[0].get('instruction', '') if legs else ''
            
            orders.append(Order(
                order_id=str(order_data.get('orderId', '')),
                symbol=symbol,
                instruction=instruction,
                quantity=order_data.get('quantity', 0),
                order_type=order_data.get('orderType', ''),
                status=order_data.get('status', ''),
                price=order_data.get('price'),
                filled_quantity=order_data.get('filledQuantity', 0),
                remaining_quantity=order_data.get('remainingQuantity', 0),
                entered_time=datetime.now()  # Parse actual time
            ))
        
        return orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        account_hash = self.get_account_hash()
        url = f"{self.TRADER_URL}/accounts/{account_hash}/orders/{order_id}"
        
        try:
            self._request("DELETE", url)
            logger.info(f"Order {order_id} cancelled")
            return True
        except:
            return False
    
    def close_all_option_positions(self, underlying: str) -> List[Dict[str, Any]]:
        """Close all option positions for an underlying"""
        positions = self.get_option_positions(underlying)
        results = []
        
        for pos in positions:
            if pos.quantity > 0:
                # Long position - sell to close
                result = self.sell_to_close(pos.symbol, int(abs(pos.quantity)))
                results.append(result)
            elif pos.quantity < 0:
                # Short position - buy to close
                result = self.buy_to_open(pos.symbol, int(abs(pos.quantity)))
                results.append(result)
        
        return results