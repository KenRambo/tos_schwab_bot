"""
Test Mode - Simulate trading signals after hours

This script simulates the bot's behavior using fake price data
so you can test the full flow without live market data.
"""
import os
import sys
import time
import random
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import bot components
from config import BotConfig
from schwab_auth import SchwabAuth
from schwab_client import SchwabClient
from signal_detector import SignalDetector, Bar, Direction
from position_manager import PositionManager


class MarketSimulator:
    """Simulates SPY price movement for testing"""
    
    def __init__(self, start_price: float = 590.0):
        self.price = start_price
        self.high = start_price
        self.low = start_price
        self.open = start_price
        self.volume_base = 100000
        self.trend = 0  # -1 down, 0 neutral, 1 up
        self.trend_strength = 0
        self.bar_count = 0
        
    def next_bar(self) -> Bar:
        """Generate next simulated bar"""
        self.bar_count += 1
        
        # Randomly change trend occasionally
        if random.random() < 0.1:
            self.trend = random.choice([-1, 0, 1])
            self.trend_strength = random.uniform(0.5, 2.0)
        
        # Calculate price movement
        base_move = random.gauss(0, 0.3)  # Random noise
        trend_move = self.trend * self.trend_strength * random.uniform(0, 0.5)
        total_move = base_move + trend_move
        
        # Generate OHLC
        self.open = self.price
        
        if total_move > 0:
            self.high = self.price + abs(total_move) + random.uniform(0, 0.2)
            self.low = self.price - random.uniform(0, 0.3)
            self.price = self.price + total_move
        else:
            self.high = self.price + random.uniform(0, 0.3)
            self.low = self.price - abs(total_move) - random.uniform(0, 0.2)
            self.price = self.price + total_move
        
        # Volume with some randomness (higher on trend moves)
        volume = int(self.volume_base * (1 + abs(total_move) * 0.5) * random.uniform(0.8, 1.5))
        
        # Create bar with simulated time (pretend it's market hours)
        sim_time = datetime.now().replace(hour=10, minute=0) + timedelta(minutes=5 * self.bar_count)
        
        return Bar(
            timestamp=sim_time,
            open=round(self.open, 2),
            high=round(self.high, 2),
            low=round(self.low, 2),
            close=round(self.price, 2),
            volume=volume
        )
    
    def force_signal(self, direction: str):
        """Force price movement to trigger a signal"""
        if direction == "long":
            # Big drop then bounce (VAL bounce setup)
            self.trend = -1
            self.trend_strength = 3.0
            logger.info("📉 Forcing downward move to setup VAL bounce...")
        elif direction == "short":
            # Big rally then rejection (VAH rejection setup)
            self.trend = 1
            self.trend_strength = 3.0
            logger.info("📈 Forcing upward move to setup VAH rejection...")


def run_simulation(num_bars: int = 50, speed: float = 0.5):
    """
    Run a trading simulation
    
    Args:
        num_bars: Number of bars to simulate
        speed: Seconds between bars (0.5 = fast, 2 = slow)
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ToS Signal Bot - SIMULATION MODE                     ║
    ║         Testing signal detection with fake data              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize components
    config = BotConfig()
    
    # Check if we can connect to Schwab (optional for simulation)
    schwab_connected = False
    client = None
    
    try:
        auth = SchwabAuth(
            app_key=os.getenv("SCHWAB_APP_KEY", ""),
            app_secret=os.getenv("SCHWAB_APP_SECRET", ""),
            redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182/callback"),
            token_file="schwab_tokens.json"
        )
        
        if auth.is_authenticated:
            auth.refresh_access_token()
            client = SchwabClient(auth)
            # Test connection
            client.get_account_hash()
            schwab_connected = True
            logger.info("✓ Schwab API connected - orders will be simulated (paper mode)")
    except Exception as e:
        logger.warning(f"Schwab not connected: {e}")
        logger.info("Running in pure simulation mode (no API)")
    
    # Initialize detector (with relaxed settings for testing)
    detector = SignalDetector(
        length_period=10,  # Shorter for faster signals
        volume_threshold=1.2,
        use_relaxed_volume=True,
        min_confirmation_bars=1,
        sustained_bars_required=2,
        signal_cooldown_bars=3,  # Shorter cooldown for testing
        use_or_bias_filter=False,  # Disable for testing
        use_time_filter=False,
        rth_only=False  # Allow any time
    )
    
    # Initialize position manager
    if client:
        position_manager = PositionManager(
            client=client,
            symbol="SPY",
            contracts=1,
            max_daily_trades=10,  # Higher for testing
            paper_trading=True  # Always paper in simulation
        )
    else:
        position_manager = None
    
    # Initialize simulator
    simulator = MarketSimulator(start_price=590.0)
    
    print(f"\n🎮 Starting simulation: {num_bars} bars, {speed}s between bars")
    print(f"📡 Schwab API: {'Connected (paper mode)' if schwab_connected else 'Not connected'}")
    print("\nCommands during simulation:")
    print("  - Simulation runs automatically")
    print("  - Watch for signal detection")
    print("  - Ctrl+C to stop\n")
    print("=" * 60)
    
    signals_detected = []
    
    try:
        for i in range(num_bars):
            # Generate bar
            bar = simulator.next_bar()
            
            # Process bar
            signal = detector.add_bar(bar)
            
            # Get current state
            state = detector.get_state_summary()
            
            # Print bar info
            direction_icon = "📈" if bar.close > bar.open else "📉" if bar.close < bar.open else "➡️"
            print(f"\nBar {i+1}/{num_bars} [{bar.timestamp.strftime('%H:%M')}] {direction_icon}")
            print(f"  Price: {bar.close:.2f} (O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f})")
            print(f"  Volume: {bar.volume:,}")
            print(f"  VAH: {state['vah']:.2f} | POC: {state['poc']:.2f} | VAL: {state['val']:.2f}")
            print(f"  Position: {state['position']} | Bars above VAH: {state['bars_above_vah']} | Below VAL: {state['bars_below_val']}")
            
            if signal:
                signals_detected.append(signal)
                print(f"\n  🚨 SIGNAL: {signal.signal_type.value}")
                print(f"     Direction: {signal.direction.value}")
                print(f"     Reason: {signal.reason}")
                
                # Try to execute if we have position manager
                if position_manager:
                    print(f"     Executing trade (paper)...")
                    trade = position_manager.process_signal(signal)
                    if trade:
                        print(f"     ✓ Trade executed: {trade.option_symbol}")
            
            # Pause between bars
            time.sleep(speed)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation stopped by user")
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Bars processed: {simulator.bar_count}")
    print(f"Signals detected: {len(signals_detected)}")
    
    if signals_detected:
        print("\nSignals:")
        for s in signals_detected:
            print(f"  - {s.signal_type.value} ({s.direction.value}) @ {s.price:.2f}")
    
    if position_manager:
        stats = position_manager.get_daily_stats()
        print(f"\nTrades executed: {stats['trades_taken']}")
        print(f"Current position: {stats['open_position']}")
    
    print("=" * 60)


def run_single_signal_test(direction: str = "long"):
    """
    Test a single signal type
    
    Args:
        direction: "long" or "short"
    """
    print(f"\n🧪 Testing {direction.upper()} signal generation...\n")
    
    detector = SignalDetector(
        length_period=10,
        signal_cooldown_bars=1,
        use_or_bias_filter=False,
        rth_only=False
    )
    
    simulator = MarketSimulator(start_price=590.0)
    
    # Build up some history
    print("Building price history...")
    for _ in range(15):
        bar = simulator.next_bar()
        detector.add_bar(bar)
    
    # Force movement to trigger signal
    simulator.force_signal(direction)
    
    # Generate bars until signal
    for i in range(20):
        bar = simulator.next_bar()
        signal = detector.add_bar(bar)
        
        state = detector.get_state_summary()
        print(f"Bar {i+1}: {bar.close:.2f} | VAH:{state['vah']:.2f} POC:{state['poc']:.2f} VAL:{state['val']:.2f}")
        
        if signal:
            print(f"\n✓ Signal detected: {signal.signal_type.value}")
            print(f"  Direction: {signal.direction.value}")
            print(f"  Reason: {signal.reason}")
            return True
    
    print("\n✗ No signal detected in test window")
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trading bot simulation')
    parser.add_argument('--bars', type=int, default=50, help='Number of bars to simulate')
    parser.add_argument('--speed', type=float, default=0.5, help='Seconds between bars')
    parser.add_argument('--test-long', action='store_true', help='Test long signal generation')
    parser.add_argument('--test-short', action='store_true', help='Test short signal generation')
    
    args = parser.parse_args()
    
    if args.test_long:
        run_single_signal_test("long")
    elif args.test_short:
        run_single_signal_test("short")
    else:
        run_simulation(num_bars=args.bars, speed=args.speed)
