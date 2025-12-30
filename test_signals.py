"""
Test Script - Verify bot components work correctly

Run this to test signal detection logic without API connection.
"""
import sys
from datetime import datetime, timedelta
from signal_detector import SignalDetector, Bar, Direction, SignalType


def create_test_bars():
    """Create sample bars that should trigger signals"""
    base_time = datetime(2024, 1, 15, 9, 30)  # Market open
    bars = []
    
    # Simulate opening range building (9:30-10:00)
    # Price starts at 475, builds range 474-476
    prices = [
        (475.0, 475.5, 474.5, 475.2, 100000),  # 9:30
        (475.2, 476.0, 475.0, 475.8, 120000),  # 9:35
        (475.8, 476.2, 475.5, 476.0, 110000),  # 9:40
        (476.0, 476.5, 475.8, 476.2, 130000),  # 9:45
        (476.2, 476.3, 474.0, 474.5, 150000),  # 9:50 - drop
        (474.5, 475.0, 474.2, 474.8, 140000),  # 9:55
    ]
    
    # OR complete, now simulate signals
    # Add more bars for OR to complete
    prices += [
        (474.8, 475.2, 474.5, 475.0, 100000),  # 10:00 - OR complete
        (475.0, 475.5, 474.8, 475.3, 110000),  # 10:05
        (475.3, 475.8, 475.0, 475.5, 105000),  # 10:10
    ]
    
    # Simulate VAL bounce setup
    # Price drops to VAL area, then bounces with volume
    prices += [
        (475.5, 475.6, 474.0, 474.2, 180000),  # 10:15 - drop toward VAL
        (474.2, 474.5, 473.8, 474.0, 200000),  # 10:20 - touch VAL
        (474.0, 475.0, 473.9, 474.8, 250000),  # 10:25 - BOUNCE! High volume
    ]
    
    # Continue upward
    prices += [
        (474.8, 475.5, 474.6, 475.3, 150000),  # 10:30
        (475.3, 476.0, 475.2, 475.8, 160000),  # 10:35
        (475.8, 476.5, 475.7, 476.3, 170000),  # 10:40
    ]
    
    # Build up for breakout above VAH
    prices += [
        (476.3, 477.0, 476.2, 476.8, 180000),  # 10:45
        (476.8, 477.5, 476.7, 477.2, 200000),  # 10:50 - breaking out
        (477.2, 478.0, 477.0, 477.8, 220000),  # 10:55 - confirmed above VAH
        (477.8, 478.2, 477.5, 478.0, 190000),  # 11:00 - sustained
    ]
    
    # Simulate rejection and reversal
    prices += [
        (478.0, 478.5, 477.0, 477.2, 180000),  # 11:05 - rejection candle
        (477.2, 477.5, 476.0, 476.2, 200000),  # 11:10 - breakdown
        (476.2, 476.5, 475.5, 475.8, 210000),  # 11:15 - continuing down
        (475.8, 476.0, 474.5, 474.8, 230000),  # 11:20 - below POC
        (474.8, 475.0, 474.0, 474.2, 250000),  # 11:25 - approaching VAL
    ]
    
    for i, (o, h, l, c, v) in enumerate(prices):
        bar = Bar(
            timestamp=base_time + timedelta(minutes=5*i),
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v
        )
        bars.append(bar)
    
    return bars


def test_signal_detection():
    """Test signal detection with sample data"""
    print("=" * 60)
    print("SIGNAL DETECTION TEST")
    print("=" * 60)
    
    detector = SignalDetector(
        length_period=10,  # Shorter for test
        volume_threshold=1.3,
        use_relaxed_volume=True,
        min_confirmation_bars=2,
        sustained_bars_required=3,
        signal_cooldown_bars=4,  # Shorter for test
        use_or_bias_filter=True,
        or_buffer_points=0.5,
        opening_range_minutes=30,
        use_time_filter=False,
        rth_only=True
    )
    
    bars = create_test_bars()
    signals_found = []
    
    print(f"\nProcessing {len(bars)} test bars...\n")
    
    for bar in bars:
        signal = detector.add_bar(bar)
        
        # Print bar info
        state = detector.get_state_summary()
        time_str = bar.timestamp.strftime("%H:%M")
        
        if signal:
            signals_found.append(signal)
            print(f"[{time_str}] SIGNAL: {signal.signal_type.value}")
            print(f"         Direction: {signal.direction.value}")
            print(f"         Price: {bar.close:.2f}")
            print(f"         Reason: {signal.reason}")
            print()
        else:
            # Show state changes
            if state['or_complete'] and not getattr(test_signal_detection, '_or_logged', False):
                print(f"[{time_str}] Opening Range Complete")
                print(f"         OR High: {state['or_high']:.2f}")
                print(f"         OR Low: {state['or_low']:.2f}")
                test_signal_detection._or_logged = True
    
    print("-" * 60)
    print(f"\nTotal signals detected: {len(signals_found)}")
    
    if signals_found:
        print("\nSignal Summary:")
        for i, sig in enumerate(signals_found, 1):
            print(f"  {i}. {sig.signal_type.value} ({sig.direction.value}) @ {sig.price:.2f}")
    
    # Verify detector state
    print("\n" + "-" * 60)
    print("Final Detector State:")
    state = detector.get_state_summary()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)
    return len(signals_found) > 0


def test_value_area_calculation():
    """Test Value Area calculation"""
    print("\n" + "=" * 60)
    print("VALUE AREA CALCULATION TEST")
    print("=" * 60)
    
    detector = SignalDetector(length_period=10)
    
    # Add bars with known price distribution
    base_time = datetime(2024, 1, 15, 9, 30)
    
    for i in range(15):
        bar = Bar(
            timestamp=base_time + timedelta(minutes=5*i),
            open=100.0 + i * 0.1,
            high=100.5 + i * 0.1,
            low=99.5 + i * 0.1,
            close=100.2 + i * 0.1,
            volume=100000 + i * 10000
        )
        detector.add_bar(bar)
    
    levels = detector.get_current_levels()
    
    if levels:
        print(f"\nValue Area Levels:")
        print(f"  VAH:  {levels.vah:.2f}")
        print(f"  POC:  {levels.poc:.2f}")
        print(f"  VAL:  {levels.val:.2f}")
        print(f"  VWAP: {levels.vwap:.2f}")
        print("\n✓ Value Area calculation working")
        return True
    else:
        print("\n✗ Value Area calculation failed")
        return False


def test_opening_range():
    """Test Opening Range detection"""
    print("\n" + "=" * 60)
    print("OPENING RANGE TEST")
    print("=" * 60)
    
    detector = SignalDetector(
        length_period=10,
        opening_range_minutes=30
    )
    
    base_time = datetime(2024, 1, 15, 9, 30)
    
    # Bars during OR period
    or_prices = [
        (100, 101, 99, 100.5),   # 9:30
        (100.5, 102, 100, 101),  # 9:35
        (101, 101.5, 98, 99),    # 9:40 - low of day
        (99, 100, 98.5, 99.5),   # 9:45
        (99.5, 103, 99, 102.5),  # 9:50 - high of day
        (102.5, 103, 102, 102),  # 9:55
    ]
    
    for i, (o, h, l, c) in enumerate(or_prices):
        bar = Bar(
            timestamp=base_time + timedelta(minutes=5*i),
            open=o, high=h, low=l, close=c,
            volume=100000
        )
        detector.add_bar(bar)
    
    state = detector.get_state_summary()
    print(f"\nAfter OR period (9:30-9:55):")
    print(f"  OR Complete: {state['or_complete']}")
    print(f"  OR High: {state['or_high']:.2f}")
    print(f"  OR Low: {state['or_low']:.2f}")
    
    # Bar after OR period
    bar = Bar(
        timestamp=base_time + timedelta(minutes=30),
        open=102, high=104, low=101.5, close=103.5,
        volume=150000
    )
    detector.add_bar(bar)
    
    state = detector.get_state_summary()
    print(f"\nAfter 10:00 bar (price broke above OR):")
    print(f"  OR Complete: {state['or_complete']}")
    print(f"  OR Bias: {state['or_bias']}")
    
    expected_high = 103  # From 9:50 bar
    expected_low = 98    # From 9:40 bar
    
    if (abs(state['or_high'] - expected_high) < 0.01 and 
        abs(state['or_low'] - expected_low) < 0.01 and
        state['or_complete']):
        print("\n✓ Opening Range detection working correctly")
        return True
    else:
        print("\n✗ Opening Range detection issue")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Value Area
    results.append(("Value Area Calculation", test_value_area_calculation()))
    
    # Test 2: Opening Range
    results.append(("Opening Range Detection", test_opening_range()))
    
    # Test 3: Signal Detection
    results.append(("Signal Detection", test_signal_detection()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Bot components working correctly.\n")
    else:
        print("\n✗ Some tests failed. Check implementation.\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
