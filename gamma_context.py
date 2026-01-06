"""
Gamma Context Module

Approximates 0DTE gamma exposure levels for SPX/ES trading.
Used to filter signals based on dealer gamma positioning.

GAMMA REGIME BEHAVIOR:
- Positive Gamma (above zero gamma): Dealers long gamma, mean reversion
  → Favor: VAL bounce, VAH rejection (fade moves)
  → Avoid: Breakouts (will get faded)
  
- Negative Gamma (below zero gamma): Dealers short gamma, momentum
  → Favor: Breakouts, breakdowns (ride momentum)
  → Avoid: Fading moves (will get run over)

Based on the SPX 0DTE Gamma Exposure Approximation study.
"""

import logging
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class GammaRegime(Enum):
    POSITIVE = "POSITIVE"  # Above zero gamma - mean reversion
    NEGATIVE = "NEGATIVE"  # Below zero gamma - momentum
    NEUTRAL = "NEUTRAL"    # Near zero gamma - no edge


@dataclass
class GammaLevels:
    """Gamma exposure levels for the current session"""
    zero_gamma: float
    
    # Call gamma levels (resistance)
    call_gamma_1: float
    call_gamma_2: float
    call_gamma_3: float
    
    # Put gamma levels (support)
    put_gamma_1: float
    put_gamma_2: float
    put_gamma_3: float
    
    # Big round number levels
    nearest_50: float
    wall_above: float  # Next $50 level up
    wall_below: float  # Next $50 level down
    
    # Nearest $100 levels (massive gamma)
    major_above: float
    major_below: float


class GammaContext:
    """
    Calculates and tracks gamma exposure context for trading decisions.
    
    Usage:
        gamma = GammaContext(symbol="SPX", today_open=6900.0)
        gamma.update(current_price=6920.0, current_time=datetime.now())
        
        if gamma.regime == GammaRegime.POSITIVE:
            # Mean reversion mode - fade moves
        else:
            # Momentum mode - ride moves
    """
    
    # Market hours (ET)
    MARKET_OPEN = dt_time(9, 30)
    MARKET_CLOSE = dt_time(16, 0)
    SESSION_SECONDS = 6.5 * 3600  # 6.5 hours
    
    def __init__(
        self,
        symbol: str = "SPX",
        today_open: Optional[float] = None,
        strike_width: int = 5,
        major_strike_width: int = 25,
        big_level_width: int = 50,
        neutral_zone_points: float = 5.0,  # Points from ZG to consider neutral
        enable_filtering: bool = True
    ):
        self.symbol = symbol
        self.today_open = today_open
        self.strike_width = strike_width
        self.major_strike_width = major_strike_width
        self.big_level_width = big_level_width
        self.neutral_zone_points = neutral_zone_points
        self.enable_filtering = enable_filtering
        
        # Current state
        self.current_price: Optional[float] = None
        self.current_time: Optional[datetime] = None
        self.levels: Optional[GammaLevels] = None
        self.regime: GammaRegime = GammaRegime.NEUTRAL
        
        # Time factors
        self.time_remaining_pct: float = 1.0
        self.gamma_acceleration: float = 1.0
        
        logger.info(f"GammaContext initialized for {symbol}")
    
    def set_today_open(self, open_price: float) -> None:
        """Set today's opening price (call at RTH open)"""
        self.today_open = open_price
        logger.info(f"Today's open set: ${open_price:.2f}")
    
    def update(self, current_price: float, current_time: Optional[datetime] = None) -> None:
        """Update gamma calculations with current price and time"""
        self.current_price = current_price
        self.current_time = current_time or datetime.now()
        
        # Calculate time remaining
        self._calculate_time_factors()
        
        # Calculate levels
        self.levels = self._calculate_levels()
        
        # Determine regime
        self.regime = self._determine_regime()
    
    def _calculate_time_factors(self) -> None:
        """Calculate time-based factors for gamma concentration"""
        if not self.current_time:
            self.time_remaining_pct = 1.0
            self.gamma_acceleration = 1.0
            return
        
        current = self.current_time.time()
        
        # Calculate seconds from open and to close
        current_seconds = current.hour * 3600 + current.minute * 60 + current.second
        open_seconds = self.MARKET_OPEN.hour * 3600 + self.MARKET_OPEN.minute * 60
        close_seconds = self.MARKET_CLOSE.hour * 3600 + self.MARKET_CLOSE.minute * 60
        
        # Time remaining as percentage (1.0 at open, 0.0 at close)
        if current_seconds < open_seconds:
            self.time_remaining_pct = 1.0
        elif current_seconds >= close_seconds:
            self.time_remaining_pct = 0.0
        else:
            elapsed = current_seconds - open_seconds
            self.time_remaining_pct = max(0, 1.0 - (elapsed / self.SESSION_SECONDS))
        
        # Gamma acceleration (increases as day progresses)
        # Early: gamma spread wide, Late: gamma concentrated
        self.gamma_acceleration = 1.0 / (self.time_remaining_pct + 0.01) ** 0.5
    
    def _calculate_levels(self) -> GammaLevels:
        """Calculate all gamma levels"""
        if not self.current_price:
            raise ValueError("Current price not set")
        
        # Zero gamma calculation
        # Anchors near open early, drifts toward current price as day progresses
        if self.today_open:
            anchor_weight = self.time_remaining_pct
            zero_gamma_raw = (self.today_open * anchor_weight) + \
                           (self.current_price * (1 - anchor_weight))
        else:
            zero_gamma_raw = self.current_price
        
        # Round to nearest strike
        zero_gamma = round(zero_gamma_raw / self.strike_width) * self.strike_width
        
        # Dynamic spacing based on time
        # Wider early (gamma spread out), tighter late (gamma concentrated)
        base_spacing = self.major_strike_width
        dynamic_spacing = max(
            self.strike_width,
            round((base_spacing / self.gamma_acceleration) / self.strike_width) * self.strike_width
        )
        spacing = max(self.strike_width, min(dynamic_spacing, self.major_strike_width))
        
        # Gamma levels
        call_1 = zero_gamma + spacing
        call_2 = zero_gamma + (spacing * 2)
        call_3 = zero_gamma + (spacing * 3)
        
        put_1 = zero_gamma - spacing
        put_2 = zero_gamma - (spacing * 2)
        put_3 = zero_gamma - (spacing * 3)
        
        # Big round number levels
        nearest_50 = round(self.current_price / self.big_level_width) * self.big_level_width
        wall_above = nearest_50 + self.big_level_width
        wall_below = nearest_50 - self.big_level_width
        
        # $100 levels
        major_above = round(self.current_price / 100) * 100 + 100
        major_below = round(self.current_price / 100) * 100 - 100
        
        return GammaLevels(
            zero_gamma=zero_gamma,
            call_gamma_1=call_1,
            call_gamma_2=call_2,
            call_gamma_3=call_3,
            put_gamma_1=put_1,
            put_gamma_2=put_2,
            put_gamma_3=put_3,
            nearest_50=nearest_50,
            wall_above=wall_above,
            wall_below=wall_below,
            major_above=major_above,
            major_below=major_below
        )
    
    def _determine_regime(self) -> GammaRegime:
        """Determine current gamma regime based on price vs zero gamma"""
        if not self.levels or not self.current_price:
            return GammaRegime.NEUTRAL
        
        distance = self.current_price - self.levels.zero_gamma
        
        if abs(distance) <= self.neutral_zone_points:
            return GammaRegime.NEUTRAL
        elif distance > 0:
            return GammaRegime.POSITIVE
        else:
            return GammaRegime.NEGATIVE
    
    def should_allow_signal(self, signal_type: str, direction: str) -> Tuple[bool, str]:
        """
        Check if a signal should be allowed based on gamma regime.
        
        Args:
            signal_type: VAL_BOUNCE, VAH_REJECTION, BREAKOUT, BREAKDOWN, etc.
            direction: LONG or SHORT
            
        Returns:
            (allowed: bool, reason: str)
        """
        if not self.enable_filtering:
            return True, "Gamma filter disabled"
        
        if self.regime == GammaRegime.NEUTRAL:
            return True, "Neutral gamma - all signals allowed"
        
        signal_type = signal_type.upper()
        direction = direction.upper()
        
        # Mean reversion signals (good in positive gamma)
        mean_reversion_signals = {'VAL_BOUNCE', 'VAH_REJECTION', 'POC_RECLAIM', 'POC_BREAKDOWN'}
        
        # Momentum signals (good in negative gamma)
        momentum_signals = {'BREAKOUT', 'BREAKDOWN', 'SUSTAINED_BREAKOUT', 'SUSTAINED_BREAKDOWN'}
        
        if self.regime == GammaRegime.POSITIVE:
            # Positive gamma = mean reversion environment
            if signal_type in mean_reversion_signals:
                return True, f"✓ {signal_type} allowed in +gamma (mean reversion)"
            elif signal_type in momentum_signals:
                return False, f"✗ {signal_type} blocked in +gamma (breakouts get faded)"
            else:
                return True, f"Unknown signal type {signal_type} - allowing"
        
        else:  # NEGATIVE gamma
            # Negative gamma = momentum environment
            if signal_type in momentum_signals:
                return True, f"✓ {signal_type} allowed in -gamma (momentum)"
            elif signal_type in mean_reversion_signals:
                # Still allow VAL/VAH signals but could be riskier
                # For now, allow but log warning
                logger.warning(f"⚠ {signal_type} in -gamma may be risky (momentum environment)")
                return True, f"⚠ {signal_type} allowed in -gamma (use caution)"
            else:
                return True, f"Unknown signal type {signal_type} - allowing"
    
    def get_bias(self) -> str:
        """Get current bias string for display"""
        if self.regime == GammaRegime.POSITIVE:
            return "FADE MOVES"
        elif self.regime == GammaRegime.NEGATIVE:
            return "RIDE MOMENTUM"
        else:
            return "NEUTRAL"
    
    def is_near_level(self, level_type: str = "any", threshold: float = 5.0) -> bool:
        """Check if price is near a gamma level"""
        if not self.levels or not self.current_price:
            return False
        
        checks = {
            "zero_gamma": self.levels.zero_gamma,
            "call_1": self.levels.call_gamma_1,
            "put_1": self.levels.put_gamma_1,
            "wall_above": self.levels.wall_above,
            "wall_below": self.levels.wall_below,
            "nearest_50": self.levels.nearest_50,
        }
        
        if level_type == "any":
            return any(abs(self.current_price - level) <= threshold for level in checks.values())
        elif level_type in checks:
            return abs(self.current_price - checks[level_type]) <= threshold
        
        return False
    
    def get_summary(self) -> dict:
        """Get summary dict for logging/display"""
        if not self.levels:
            return {"status": "Not initialized"}
        
        return {
            "regime": self.regime.value,
            "bias": self.get_bias(),
            "zero_gamma": self.levels.zero_gamma,
            "current_price": self.current_price,
            "distance_from_zg": round(self.current_price - self.levels.zero_gamma, 2) if self.current_price else 0,
            "call_gamma_1": self.levels.call_gamma_1,
            "put_gamma_1": self.levels.put_gamma_1,
            "nearest_50": self.levels.nearest_50,
            "time_remaining_pct": round(self.time_remaining_pct * 100, 1),
            "gamma_acceleration": round(self.gamma_acceleration, 2),
        }
    
    def log_status(self) -> None:
        """Log current gamma status"""
        if not self.levels:
            logger.info("Gamma context not initialized")
            return
        
        zg = self.levels.zero_gamma
        regime_emoji = "🟢" if self.regime == GammaRegime.POSITIVE else "🔴" if self.regime == GammaRegime.NEGATIVE else "⚪"
        
        logger.info(f"  Gamma: {regime_emoji} {self.regime.value} | ZG: ${zg:.0f} | Bias: {self.get_bias()}")
        logger.info(f"  Levels: PUT₁ ${self.levels.put_gamma_1:.0f} | ZG ${zg:.0f} | CALL₁ ${self.levels.call_gamma_1:.0f}")
        
        if self.current_price:
            dist = self.current_price - zg
            logger.info(f"  Price vs ZG: ${self.current_price:.2f} ({'+' if dist >= 0 else ''}{dist:.2f})")


# Convenience function
def create_gamma_context(
    symbol: str = "SPX",
    today_open: Optional[float] = None,
    enable_filtering: bool = True
) -> GammaContext:
    """Create a GammaContext instance"""
    return GammaContext(
        symbol=symbol,
        today_open=today_open,
        enable_filtering=enable_filtering
    )