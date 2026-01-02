# AMT Indicator - Simplified High Delta Version
# OPTIMIZED on 2026-01-02 12:13:21
# Focus: Core AMT signals, VIX regime, high delta ATM contracts

declare upper;

# ==================== INPUTS ====================

input lengthPeriod = 20;
input valueAreaPercent = 70;
input showValueArea = yes;
input showPOC = yes;
input showSignals = yes;
input enableAlerts = yes;

# Signal Parameters - OPTIMIZED
input volumeThreshold = 1.48; # OPTIMIZED
input minConfirmationBars = 2; # OPTIMIZED
input sustainedBarsRequired = 5; # OPTIMIZED
input signalCooldownBars = 17; # OPTIMIZED
input useRelaxedVolume = yes;

# Signal Enables - OPTIMIZED (POC and sustained disabled)
input enableVALBounce = yes; # OPTIMIZED
input enableVAHRejection = yes; # OPTIMIZED
input enableBreakout = no; # OPTIMIZED
input enableBreakdown = yes; # OPTIMIZED
input enablePOCReclaim = no;  # Disabled - noisy
input enablePOCBreakdown = no;  # Disabled - noisy
input enableSustainedBreakout = no;  # Disabled
input enableSustainedBreakdown = no;  # Disabled

# Opening Range Bias - OPTIMIZED
input useORBiasFilter = yes; # OPTIMIZED
input showOpeningRange = yes;
input orBufferPoints = 1.0;

# VIX Regime - OPTIMIZED
input useVixRegime = yes; # OPTIMIZED
input vixHighThreshold = 25;
input vixLowThreshold = 15;
input highVolCooldownMult = 1.43; # OPTIMIZED
input lowVolCooldownMult = 0.88; # OPTIMIZED

# Daily Trade Lockout
input maxDailyTrades = 6; # OPTIMIZED
input enableLockout = yes;
input showTradeCounter = yes;

# Display
input showExitSignals = yes;
input showPositionStatus = yes;
input rthOnlySignals = yes;
input useSessionVA = yes;

# ==================== VALUE AREA ====================

def isNewDay = GetDay() != GetDay()[1];
def sessionHigh = if isNewDay then high else if high > sessionHigh[1] then high else sessionHigh[1];
def sessionLow = if isNewDay then low else if low < sessionLow[1] then low else sessionLow[1];
def sessionVolume = if isNewDay then volume else sessionVolume[1] + volume;
def sessionVWAPSum = if isNewDay then (volume * (high + low + close) / 3) 
                     else sessionVWAPSum[1] + (volume * (high + low + close) / 3);
def sessionVolWeighted = if isNewDay then (volume * ((high + low) / 2))
                         else sessionVolWeighted[1] + (volume * ((high + low) / 2));

def lengthPeriodVal = lengthPeriod;
def rollingHigh = highest(high, lengthPeriodVal);
def rollingLow = lowest(low, lengthPeriodVal);
def rollingVolume = sum(volume, lengthPeriodVal);
def rollingVWAPSum = sum(volume * (high + low + close) / 3, lengthPeriodVal);
def rollingVolWeighted = sum(volume * ((high + low) / 2), lengthPeriodVal);

def priceHigh = if useSessionVA then sessionHigh else rollingHigh;
def priceLow = if useSessionVA then sessionLow else rollingLow;

def volumeSum = if useSessionVA then sessionVolume else rollingVolume;
def vwapSum = if useSessionVA then sessionVWAPSum else rollingVWAPSum;
def vwapValue = vwapSum / volumeSum;

def volWeightedSum = if useSessionVA then sessionVolWeighted else rollingVolWeighted;
def pocValue = volWeightedSum / volumeSum;

def stdDev = StDev(close, lengthPeriodVal);
def valueAreaHigh = vwapValue + (stdDev * 0.5);
def valueAreaLow = vwapValue - (stdDev * 0.5);

# ==================== OPENING RANGE ====================

def isInOpeningRange = SecondsFromTime(0930) >= 0 and SecondsFromTime(1000) < 0;
def orComplete = SecondsFromTime(1000) >= 0;

rec openingRangeHigh = if isNewDay then high 
                       else if isInOpeningRange and high > openingRangeHigh[1] then high
                       else openingRangeHigh[1];
             
rec openingRangeLow = if isNewDay then low
                      else if isInOpeningRange and low < openingRangeLow[1] then low
                      else openingRangeLow[1];

def orMid = (openingRangeHigh + openingRangeLow) / 2;
def aboveORHigh = close > (openingRangeHigh + orBufferPoints);
def belowORLow = close < (openingRangeLow - orBufferPoints);

def orBias = if !orComplete then 0
             else if aboveORHigh then 1
             else if belowORLow then -1
             else 0;

# ==================== PRICE POSITION ====================

def priceAboveVAH = close > valueAreaHigh;
def priceBelowVAL = close < valueAreaLow;
def priceAbovePOC = close > pocValue;
def priceBelowPOC = close < pocValue;

def prevAboveVAH = close[1] > valueAreaHigh[1];
def prevBelowVAL = close[1] < valueAreaLow[1];
def prevAbovePOC = close[1] > pocValue[1];

# ==================== VOLUME ====================

def avgVolume = Average(volume, lengthPeriodVal);
def volumeSpike = volume > (avgVolume * volumeThreshold);
def volumeElevated = volume > avgVolume;
def volCondition = if useRelaxedVolume then volumeElevated else volumeSpike;
def volIncreasing = volume > volume[1];

# ==================== VIX REGIME ====================

def vix = close("$VIX.X");
def vixRegime = if !useVixRegime then 0
                else if vix >= vixHighThreshold then 1
                else if vix <= vixLowThreshold then -1
                else 0;

def effectiveCooldown = if !useVixRegime then signalCooldownBars
                        else if vixRegime == 1 then Round(signalCooldownBars * highVolCooldownMult, 0)
                        else if vixRegime == -1 then Max(3, Round(signalCooldownBars * lowVolCooldownMult, 0))
                        else signalCooldownBars;

# ==================== FILTERS ====================

def timeNow = SecondsFromTime(0930);
def marketClose = SecondsTillTime(1600);
def isRTH = timeNow >= 0 and marketClose >= 0;
def rthFilter = if rthOnlySignals then isRTH else yes;

def orAllowLong = if !useORBiasFilter then yes
                  else if !orComplete then yes
                  else orBias >= 0;

def orAllowShort = if !useORBiasFilter then yes
                   else if !orComplete then yes
                   else orBias <= 0;

# ==================== SIGNALS ====================

# VAL Bounce (LONG)
def valTouch = low <= valueAreaLow and close > valueAreaLow;
def valBounce = valTouch and volCondition and close > open;
def valBounceRaw = enableVALBounce and valBounce and !valBounce[1] and rthFilter;

# Breakout (LONG)
def breakoutBar = !prevAboveVAH and priceAboveVAH and volCondition;
def barsAboveVAH = if priceAboveVAH then barsAboveVAH[1] + 1 else 0;
def breakoutAcceptance = breakoutBar[minConfirmationBars] and barsAboveVAH >= minConfirmationBars;
def breakoutRaw = enableBreakout and breakoutAcceptance and !breakoutAcceptance[1] and rthFilter;

# VAH Rejection (SHORT)
def vahTouch = high >= valueAreaHigh and close < valueAreaHigh;
def vahRejection = vahTouch and volCondition and close < open;
def vahRejectionRaw = enableVAHRejection and vahRejection and !vahRejection[1] and rthFilter;

# Breakdown (SHORT)
def breakdownBar = !prevBelowVAL and priceBelowVAL and volCondition;
def barsBelowVAL = if priceBelowVAL then barsBelowVAL[1] + 1 else 0;
def breakdownAcceptance = breakdownBar[minConfirmationBars] and barsBelowVAL >= minConfirmationBars;
def breakdownRaw = enableBreakdown and breakdownAcceptance and !breakdownAcceptance[1] and rthFilter;

# Cooldown
def anyRawSignal = valBounceRaw or breakoutRaw or vahRejectionRaw or breakdownRaw;
rec barsSinceLastSignal = if barsSinceLastSignal[1] >= effectiveCooldown and anyRawSignal then 0 
                          else barsSinceLastSignal[1] + 1;
def cooldownClear = barsSinceLastSignal[1] >= effectiveCooldown;

# Position filter
def allowLongSignals = close >= valueAreaLow;
def allowShortSignals = close <= valueAreaHigh;

# Final signals
def valBounceLong = valBounceRaw and cooldownClear and allowLongSignals and orAllowLong;
def breakoutLong = breakoutRaw and cooldownClear and allowLongSignals and orAllowLong;
def vahRejectionShort = vahRejectionRaw and cooldownClear and allowShortSignals and orAllowShort;
def breakdownShort = breakdownRaw and cooldownClear and allowShortSignals and orAllowShort;

def longSignal = valBounceLong or breakoutLong;
def shortSignal = vahRejectionShort or breakdownShort;

# ==================== POSITION TRACKING ====================

rec currentPosition = if longSignal then 1 
                      else if shortSignal then -1 
                      else currentPosition[1];

def isLong = currentPosition == 1;
def isShort = currentPosition == -1;
def hasPosition = currentPosition != 0;
def wasLong = currentPosition[1] == 1;
def wasShort = currentPosition[1] == -1;

def exitLongSignal = wasLong and shortSignal;
def exitShortSignal = wasShort and longSignal;

# ==================== DAILY TRADE COUNTER ====================

rec dailyTradeCount = if isNewDay then 0
                      else if (longSignal or shortSignal) and !isNewDay then dailyTradeCount[1] + 1
                      else dailyTradeCount[1];

def isLockedOut = enableLockout and dailyTradeCount >= maxDailyTrades;
def justLockedOut = isLockedOut and !isLockedOut[1];

# ==================== ENTRY TRACKING ====================

rec positionEntryPrice = if longSignal or shortSignal then close
                         else positionEntryPrice[1];

# ==================== PLOTS ====================

plot VAH = if showValueArea then valueAreaHigh else Double.NaN;
VAH.SetDefaultColor(Color.CYAN);
VAH.SetLineWeight(2);

plot VAL = if showValueArea then valueAreaLow else Double.NaN;
VAL.SetDefaultColor(Color.CYAN);
VAL.SetLineWeight(2);

plot POC = if showPOC then pocValue else Double.NaN;
POC.SetDefaultColor(Color.MAGENTA);
POC.SetLineWeight(2);
POC.SetStyle(Curve.SHORT_DASH);

plot ORHigh = if showOpeningRange and orComplete then openingRangeHigh else Double.NaN;
ORHigh.SetDefaultColor(Color.LIME);
ORHigh.SetStyle(Curve.LONG_DASH);

plot ORLow = if showOpeningRange and orComplete then openingRangeLow else Double.NaN;
ORLow.SetDefaultColor(Color.PINK);
ORLow.SetStyle(Curve.LONG_DASH);

# Signal Arrows
plot LongArrow = if showSignals and longSignal and !isLockedOut then low - (ATR(14) * 0.5) else Double.NaN;
LongArrow.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
LongArrow.SetLineWeight(3);
LongArrow.SetDefaultColor(Color.GREEN);

plot ShortArrow = if showSignals and shortSignal and !isLockedOut then high + (ATR(14) * 0.5) else Double.NaN;
ShortArrow.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
ShortArrow.SetLineWeight(3);
ShortArrow.SetDefaultColor(Color.RED);

# Exit Markers
plot ExitLong = if showExitSignals and exitLongSignal then high + (ATR(14) * 0.8) else Double.NaN;
ExitLong.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
ExitLong.SetLineWeight(4);
ExitLong.SetDefaultColor(Color.ORANGE);

plot ExitShort = if showExitSignals and exitShortSignal then low - (ATR(14) * 0.8) else Double.NaN;
ExitShort.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
ExitShort.SetLineWeight(4);
ExitShort.SetDefaultColor(Color.ORANGE);

# ==================== LABELS ====================

AddLabel(yes, "VAH: " + Round(valueAreaHigh, 2), Color.CYAN);
AddLabel(yes, "POC: " + Round(pocValue, 2), Color.MAGENTA);
AddLabel(yes, "VAL: " + Round(valueAreaLow, 2), Color.CYAN);

AddLabel(showOpeningRange and orComplete and orBias == 1, "Bias: BULL", Color.LIME);
AddLabel(showOpeningRange and orComplete and orBias == -1, "Bias: BEAR", Color.PINK);
AddLabel(showOpeningRange and orComplete and orBias == 0, "Bias: NEUTRAL", Color.GRAY);

# VIX Regime
AddLabel(useVixRegime, "VIX: " + Round(vix, 1), 
         if vixRegime == 1 then Color.RED else if vixRegime == -1 then Color.GREEN else Color.GRAY);
AddLabel(useVixRegime and vixRegime == 1, "HIGH VOL | CD:" + effectiveCooldown, Color.RED);
AddLabel(useVixRegime and vixRegime == -1, "LOW VOL | CD:" + effectiveCooldown, Color.GREEN);

AddLabel(showPositionStatus and isLong, "LONG @ " + Round(positionEntryPrice, 2), Color.GREEN);
AddLabel(showPositionStatus and isShort, "SHORT @ " + Round(positionEntryPrice, 2), Color.RED);
AddLabel(showPositionStatus and !hasPosition and !isLockedOut, "FLAT", Color.GRAY);

AddLabel(showTradeCounter, "Trades: " + dailyTradeCount + "/" + maxDailyTrades,
         if dailyTradeCount >= maxDailyTrades - 1 then Color.YELLOW else Color.WHITE);

AddLabel(isLockedOut, "🛑 LOCKED OUT", Color.RED);
AddLabel(rthOnlySignals and !isRTH, "OUTSIDE RTH", Color.GRAY);

AddLabel(longSignal and !isLockedOut, "🔔 LONG", Color.GREEN);
AddLabel(shortSignal and !isLockedOut, "🔔 SHORT", Color.RED);
AddLabel(showExitSignals and exitLongSignal, "🔄 EXIT LONG", Color.ORANGE);
AddLabel(showExitSignals and exitShortSignal, "🔄 EXIT SHORT", Color.ORANGE);

# ==================== ALERTS ====================

Alert(enableAlerts and valBounceLong and !isLockedOut, "LONG: VAL Bounce", Alert.BAR, Sound.Ding);
Alert(enableAlerts and breakoutLong and !isLockedOut, "LONG: Breakout", Alert.BAR, Sound.Bell);
Alert(enableAlerts and vahRejectionShort and !isLockedOut, "SHORT: VAH Rejection", Alert.BAR, Sound.Ding);
Alert(enableAlerts and breakdownShort and !isLockedOut, "SHORT: Breakdown", Alert.BAR, Sound.Bell);
Alert(enableAlerts and exitLongSignal, "EXIT LONG", Alert.BAR, Sound.Chimes);
Alert(enableAlerts and exitShortSignal, "EXIT SHORT", Alert.BAR, Sound.Chimes);
Alert(enableLockout and justLockedOut, "DAILY LIMIT!", Alert.BAR, Sound.Ring);
