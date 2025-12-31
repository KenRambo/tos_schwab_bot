# Auction Market Theory Indicator with Trade Signals - V3.5 OPTIMIZED
# Based on Volume Profile, Value Area, Point of Control, and Price Action
# OPTIMIZED: Parameters tuned by Bayesian optimizer on 2025-12-31 12:54:29
#
# To regenerate: python apply_config.py --from <results.json>

declare upper;

# ==================== INPUTS - OPTIMIZED VALUES ====================
input lengthPeriod = 20;
input valueAreaPercent = 70;
input showValueArea = yes;
input showPOC = yes;
input showSignals = yes;
input volumeThreshold = 1.30; # OPTIMIZED
input enableAlerts = yes;
input minConfirmationBars = 4; # OPTIMIZED
input useSessionVA = yes;

# Credit Optimization Settings
input minESPointMove = 6;
input wingEntryDelay = 1;
input bodyEntryDelay = 3;
input showCreditZones = yes;

# Take Profit Settings
input showTargets = no;
input target1Multiplier = 1.5; 
input target2Multiplier = 3.0; 
input useVATargets = no;

# Prior Day Settings
input showPriorDayVA = no;
input tradePriorDayLevels = yes; 

# Win Rate & EV Optimization Settings
input useTrendFilter = no;
input useTimeFilter = yes; # OPTIMIZED
input rthOnlySignals = yes;
input showStopLoss = no;
input riskRewardRatio = 2.0;
input tickSize = 0.25; 

# Display Simplification
input showOnlyActiveSignals = yes; 
input hideValueAreaCloud = yes;

# Signal Sensitivity Settings - OPTIMIZED
input useRelaxedVolume = yes;
input sustainedBarsRequired = 5; # OPTIMIZED
input showDebugLabels = no;
input signalCooldownBars = 21; # OPTIMIZED

# Exit Signal Settings
input showExitSignals = yes;
input showPositionStatus = yes;
input showHoldDuration = yes;
input exitAlertSound = yes;
input showSignalBubbles = no;

# Daily Trade Lockout System
input maxDailyTrades = 5.0; # OPTIMIZED
input enableLockout = yes;
input showTradeCounter = yes;
input lockoutWarningAt = 2;

# Opening Range Bias - OPTIMIZED
input openingRangeMinutes = 30;
input showOpeningRange = yes;
input useORBiasFilter = yes; # OPTIMIZED
input orBufferPoints = 1.0;

# Put/Call Zone Display
input showPutCallZones = yes;

# ==================== SIGNAL ENABLE TOGGLES - OPTIMIZED ====================
# These control which signals are active
input enableVALBounce = yes; # OPTIMIZED
input enableVAHRejection = yes; # OPTIMIZED
input enablePOCReclaim = no; # OPTIMIZED
input enablePOCBreakdown = no; # OPTIMIZED
input enableBreakout = yes; # OPTIMIZED
input enableBreakdown = no; # OPTIMIZED
input enableSustainedBreakout = no; # OPTIMIZED
input enableSustainedBreakdown = no; # OPTIMIZED

# ==================== VALUE AREA CALCULATIONS ====================

def isNewDay = GetDay() != GetDay()[1];
def sessionHigh = if isNewDay then high else if high > sessionHigh[1] then high else sessionHigh[1];
def sessionLow = if isNewDay then low else if low < sessionLow[1] then low else sessionLow[1];
def sessionVolume = if isNewDay then volume else sessionVolume[1] + volume;
def sessionVWAPSum = if isNewDay then (volume * (high + low + close) / 3) 
                     else sessionVWAPSum[1] + (volume * (high + low + close) / 3);
def sessionVolWeighted = if isNewDay then (volume * ((high + low) / 2))
                         else sessionVolWeighted[1] + (volume * ((high + low) / 2));

# ==================== OPENING RANGE CALCULATION ====================

def orStartTime = 0930;
def isInOpeningRange = SecondsFromTime(orStartTime) >= 0 and SecondsFromTime(1000) < 0;
def orComplete = SecondsFromTime(1000) >= 0;

rec openingRangeHigh = if isNewDay then high 
                       else if isInOpeningRange and high > openingRangeHigh[1] then high
                       else openingRangeHigh[1];
             
rec openingRangeLow = if isNewDay then low
                      else if isInOpeningRange and low < openingRangeLow[1] then low
                      else openingRangeLow[1];

def orMid = (openingRangeHigh + openingRangeLow) / 2;
def orSize = openingRangeHigh - openingRangeLow;

def aboveORHigh = close > (openingRangeHigh + orBufferPoints);
def belowORLow = close < (openingRangeLow - orBufferPoints);

def orBias = if !orComplete then 0
             else if aboveORHigh then 1
             else if belowORLow then -1
             else 0;

rec firstBreakoutDir = if isNewDay then 0
                       else if firstBreakoutDir[1] != 0 then firstBreakoutDir[1]
                       else if orComplete and aboveORHigh then 1
                       else if orComplete and belowORLow then -1
                       else 0;

# Rolling calculations
def lengthPeriodVal = lengthPeriod;
def rollingHigh = highest(high, lengthPeriodVal);
def rollingLow = lowest(low, lengthPeriodVal);
def rollingVolume = sum(volume, lengthPeriodVal);
def rollingVWAPSum = sum(volume * (high + low + close) / 3, lengthPeriodVal);
def rollingVolWeighted = sum(volume * ((high + low) / 2), lengthPeriodVal);

def priceHigh = if useSessionVA then sessionHigh else rollingHigh;
def priceLow = if useSessionVA then sessionLow else rollingLow;
def priceRange = priceHigh - priceLow;

def volumeSum = if useSessionVA then sessionVolume else rollingVolume;
def vwapSum = if useSessionVA then sessionVWAPSum else rollingVWAPSum;
def vwapValue = vwapSum / volumeSum;

def volWeightedSum = if useSessionVA then sessionVolWeighted else rollingVolWeighted;
def pocValue = volWeightedSum / volumeSum;

def stdDev = StDev(close, lengthPeriodVal);
def valueAreaHigh = vwapValue + (stdDev * 0.5);
def valueAreaLow = vwapValue - (stdDev * 0.5);

# ==================== PRIOR DAY VALUE AREA ====================

def priorDayVAH = if isNewDay then valueAreaHigh[1] else priorDayVAH[1];
def priorDayVAL = if isNewDay then valueAreaLow[1] else priorDayVAL[1];
def priorDayPOC = if isNewDay then pocValue[1] else priorDayPOC[1];
def hasPriorDay = !IsNaN(priorDayVAH) and priorDayVAH != 0;

# ==================== PRICE POSITION ====================

def priceAboveVAH = close > valueAreaHigh;
def priceBelowVAL = close < valueAreaLow;
def priceInsideVA = close >= valueAreaLow and close <= valueAreaHigh;
def priceAbovePOC = close > pocValue;
def priceBelowPOC = close < pocValue;

def prevAboveVAH = close[1] > valueAreaHigh[1];
def prevBelowVAL = close[1] < valueAreaLow[1];
def prevAbovePOC = close[1] > pocValue[1];
def prevBelowPOC = close[1] < pocValue[1];

def priceAbovePriorVAH = close > priorDayVAH;
def priceBelowPriorVAL = close < priorDayVAL;
def priceAbovePriorPOC = close > priorDayPOC;
def priceBelowPriorPOC = close < priorDayPOC;

def prevAbovePriorVAH = close[1] > priorDayVAH[1];
def prevBelowPriorVAL = close[1] > priorDayVAL[1];
def prevAbovePriorPOC = close[1] > priorDayPOC[1];
def prevBelowPriorPOC = close[1] < priorDayPOC[1];

# ==================== VOLUME ANALYSIS ====================

def avgVolume = Average(volume, lengthPeriodVal);
def volumeSpike = volume > (avgVolume * volumeThreshold);
def volumeElevated = volume > avgVolume;
def increasingVolume = volume > volume[1];
def increasingVolume2Bar = volume > volume[1] and volume[1] > volume[2];

def volCondition = if useRelaxedVolume then volumeElevated else volumeSpike;
def volIncreasing = if useRelaxedVolume then increasingVolume else increasingVolume2Bar;

# ==================== FILTERS ====================

def trendFilterLong = yes;
def trendFilterShort = yes;

def timeNow = SecondsFromTime(0930);
def marketClose = SecondsTillTime(1600);
def isRTH = timeNow >= 0 and marketClose >= 0;
def avoidLunch = timeNow >= 9000 and timeNow <= 12600;
def avoidClose = marketClose < 900;
def goodTradingTime = isRTH and !avoidLunch and !avoidClose;

def timeFilter = if useTimeFilter then goodTradingTime else yes;
def rthFilter = if rthOnlySignals then isRTH else yes;

# ==================== SIGNAL LOGIC - WITH ENABLE TOGGLES ====================

# --- RAW SIGNAL CONDITIONS ---

# LONG RAW 1: VAL Bounce
def valTouch = low <= valueAreaLow and close > valueAreaLow;
def valBounce = valTouch and volCondition and close > open;
def valBounceRaw = enableVALBounce and valBounce and !valBounce[1] and timeFilter and rthFilter;

# LONG RAW 2: POC Reclaim
def pocReclaim = priceBelowPOC[1] and priceAbovePOC and volIncreasing;
def pocReclaimRaw = enablePOCReclaim and pocReclaim and !pocReclaim[1] and timeFilter and rthFilter;

# LONG RAW 3: Breakout above VAH
def breakoutBar = !prevAboveVAH and priceAboveVAH and volCondition;
def barsAboveVAH = if priceAboveVAH then barsAboveVAH[1] + 1 else 0;
def breakoutAcceptance = breakoutBar[minConfirmationBars] and barsAboveVAH >= minConfirmationBars;
def breakoutRaw = enableBreakout and breakoutAcceptance and !breakoutAcceptance[1] and timeFilter and rthFilter;

# LONG RAW 4: Sustained breakout
def sustainedAboveVAH = barsAboveVAH >= sustainedBarsRequired;
def sustainedBreakoutRaw = enableSustainedBreakout and sustainedAboveVAH and !sustainedAboveVAH[1] and timeFilter and rthFilter;

# SHORT RAW 1: VAH Rejection
def vahTouch = high >= valueAreaHigh and close < valueAreaHigh;
def vahRejection = vahTouch and volCondition and close < open;
def vahRejectionRaw = enableVAHRejection and vahRejection and !vahRejection[1] and timeFilter and rthFilter;

# SHORT RAW 2: POC Breakdown
def pocBreakdown = prevAbovePOC and priceBelowPOC and volIncreasing;
def pocBreakdownRaw = enablePOCBreakdown and pocBreakdown and !pocBreakdown[1] and timeFilter and rthFilter;

# SHORT RAW 3: Breakdown below VAL
def breakdownBar = !prevBelowVAL and priceBelowVAL and volCondition;
def barsBelowVAL = if priceBelowVAL then barsBelowVAL[1] + 1 else 0;
def breakdownAcceptance = breakdownBar[minConfirmationBars] and barsBelowVAL >= minConfirmationBars;
def breakdownRaw = enableBreakdown and breakdownAcceptance and !breakdownAcceptance[1] and timeFilter and rthFilter;

# SHORT RAW 4: Sustained breakdown
def sustainedBelowVAL = barsBelowVAL >= sustainedBarsRequired;
def sustainedBreakdownRaw = enableSustainedBreakdown and sustainedBelowVAL and !sustainedBelowVAL[1] and timeFilter and rthFilter;

# --- COOLDOWN LOGIC ---
def anyRawLong = valBounceRaw or pocReclaimRaw or breakoutRaw or sustainedBreakoutRaw;
def anyRawShort = vahRejectionRaw or pocBreakdownRaw or breakdownRaw or sustainedBreakdownRaw;
def anyRawSignal = anyRawLong or anyRawShort;

rec barsSinceLastSignal = if barsSinceLastSignal[1] >= signalCooldownBars and anyRawSignal then 0 
                          else barsSinceLastSignal[1] + 1;

def cooldownClear = barsSinceLastSignal[1] >= signalCooldownBars;

# --- POSITION FILTER ---
def allowLongSignals = close >= valueAreaLow;
def allowShortSignals = close <= valueAreaHigh;

# --- OR BIAS FILTER ---
def orAllowLong = if !useORBiasFilter then yes
                  else if !orComplete then yes
                  else if orBias == 1 then yes
                  else if orBias == 0 then yes
                  else no;

def orAllowShort = if !useORBiasFilter then yes
                   else if !orComplete then yes
                   else if orBias == -1 then yes
                   else if orBias == 0 then yes
                   else no;

# --- FINAL SIGNALS ---
def valBounceLong = valBounceRaw and cooldownClear and allowLongSignals and orAllowLong;
def pocReclaimLong = pocReclaimRaw and cooldownClear and allowLongSignals and orAllowLong;
def breakoutLong = breakoutRaw and cooldownClear and allowLongSignals and orAllowLong;
def sustainedBreakoutNew = sustainedBreakoutRaw and cooldownClear and allowLongSignals and orAllowLong;

def vahRejectionShort = vahRejectionRaw and cooldownClear and allowShortSignals and orAllowShort;
def pocBreakdownShort = pocBreakdownRaw and cooldownClear and allowShortSignals and orAllowShort;
def breakdownShort = breakdownRaw and cooldownClear and allowShortSignals and orAllowShort;
def sustainedBreakdownNew = sustainedBreakdownRaw and cooldownClear and allowShortSignals and orAllowShort;

# ==================== PRIOR DAY VA SIGNALS ====================

def priorValTouch = hasPriorDay and low <= priorDayVAL and close > priorDayVAL;
def priorValBounce = priorValTouch and volCondition and close > open;
def priorValBounceRaw = tradePriorDayLevels and enableVALBounce and priorValBounce and !priorValBounce[1] and timeFilter and rthFilter;
def priorValBounceLong = priorValBounceRaw and cooldownClear and allowLongSignals and orAllowLong;

def priorPocReclaim = hasPriorDay and prevBelowPriorPOC and priceAbovePriorPOC and volIncreasing;
def priorPocReclaimRaw = tradePriorDayLevels and enablePOCReclaim and priorPocReclaim and !priorPocReclaim[1] and timeFilter and rthFilter;
def priorPocReclaimLong = priorPocReclaimRaw and cooldownClear and allowLongSignals and orAllowLong;

def priorVahTouch = hasPriorDay and high >= priorDayVAH and close < priorDayVAH;
def priorVahRejection = priorVahTouch and volCondition and close < open;
def priorVahRejectionRaw = tradePriorDayLevels and enableVAHRejection and priorVahRejection and !priorVahRejection[1] and timeFilter and rthFilter;
def priorVahRejectionShort = priorVahRejectionRaw and cooldownClear and allowShortSignals and orAllowShort;

def priorPocBreakdown = hasPriorDay and prevAbovePriorPOC and priceBelowPriorPOC and volIncreasing;
def priorPocBreakdownRaw = tradePriorDayLevels and enablePOCBreakdown and priorPocBreakdown and !priorPocBreakdown[1] and timeFilter and rthFilter;
def priorPocBreakdownShort = priorPocBreakdownRaw and cooldownClear and allowShortSignals and orAllowShort;

# Combined signals
def longSignal = valBounceLong or pocReclaimLong or breakoutLong or sustainedBreakoutNew or
                 priorValBounceLong or priorPocReclaimLong;
def shortSignal = vahRejectionShort or pocBreakdownShort or breakdownShort or sustainedBreakdownNew or
                  priorVahRejectionShort or priorPocBreakdownShort;

# ==================== POSITION TRACKING ====================

rec currentPosition = if longSignal then 1 
                      else if shortSignal then -1 
                      else currentPosition[1];

def previousPosition = currentPosition[1];
def hasPosition = currentPosition != 0;
def wasLong = previousPosition == 1;
def wasShort = previousPosition == -1;
def isLong = currentPosition == 1;
def isShort = currentPosition == -1;

# ==================== EXIT SIGNALS ====================

def exitLongSignal = wasLong and shortSignal;
def exitShortSignal = wasShort and longSignal;
def exitSignal = exitLongSignal or exitShortSignal;

# ==================== DAILY TRADE COUNTER ====================

rec dailyTradeCount = if isNewDay then 0
                      else if (longSignal or shortSignal) and !isNewDay then dailyTradeCount[1] + 1
                      else dailyTradeCount[1];

def isLockedOut = enableLockout and dailyTradeCount >= maxDailyTrades;
def isWarning = enableLockout and dailyTradeCount >= lockoutWarningAt and !isLockedOut;
def tradesRemaining = maxDailyTrades - dailyTradeCount;
def justLockedOut = isLockedOut and !isLockedOut[1];

# ==================== HOLD DURATION ====================

rec barsInPosition = if longSignal or shortSignal then 1
                     else if hasPosition then barsInPosition[1] + 1
                     else 0;

rec positionEntryPrice = if longSignal or shortSignal then close
                         else positionEntryPrice[1];

def unrealizedPL = if isLong then close - positionEntryPrice
                   else if isShort then positionEntryPrice - close
                   else 0;

def exitPrice = if exitSignal then close else Double.NaN;
def realizedPL = if exitLongSignal then close - positionEntryPrice[1]
                 else if exitShortSignal then positionEntryPrice[1] - close
                 else Double.NaN;

# ==================== CREDIT OPPORTUNITY ====================

def barsSinceSignal = if longSignal or shortSignal then 0 else barsSinceSignal[1] + 1;
def priceAtSignal = if barsSinceSignal == 0 then close else priceAtSignal[1];
def pointsMovedFromSignal = AbsValue(close - priceAtSignal);

def strongMomentum = pointsMovedFromSignal >= minESPointMove;
def volumeExpansion = volume > (avgVolume * 2.0);
def highCreditZone = strongMomentum and volumeExpansion and barsSinceSignal > 0 and barsSinceSignal <= 6;

# ==================== PLOTS ====================

plot VWAP = vwapValue;
VWAP.SetDefaultColor(Color.YELLOW);
VWAP.SetLineWeight(2);
VWAP.SetStyle(Curve.FIRM);

plot VAH = if showValueArea then valueAreaHigh else Double.NaN;
VAH.SetDefaultColor(Color.CYAN);
VAH.SetLineWeight(2);
VAH.SetStyle(Curve.FIRM);

plot VAL = if showValueArea then valueAreaLow else Double.NaN;
VAL.SetDefaultColor(Color.CYAN);
VAL.SetLineWeight(2);
VAL.SetStyle(Curve.FIRM);

plot POC = if showPOC then pocValue else Double.NaN;
POC.SetDefaultColor(Color.MAGENTA);
POC.SetLineWeight(2);
POC.SetStyle(Curve.SHORT_DASH);

AddCloud(if hideValueAreaCloud then Double.NaN else VAH, 
         if hideValueAreaCloud then Double.NaN else VAL, 
         Color.DARK_GRAY, Color.DARK_GRAY);

# Prior Day Lines
plot PriorVAH = if showPriorDayVA and hasPriorDay then priorDayVAH else Double.NaN;
PriorVAH.SetDefaultColor(Color.CYAN);
PriorVAH.SetLineWeight(1);
PriorVAH.SetStyle(Curve.SHORT_DASH);

plot PriorVAL = if showPriorDayVA and hasPriorDay then priorDayVAL else Double.NaN;
PriorVAL.SetDefaultColor(Color.CYAN);
PriorVAL.SetLineWeight(1);
PriorVAL.SetStyle(Curve.SHORT_DASH);

plot PriorPOC = if showPriorDayVA and hasPriorDay then priorDayPOC else Double.NaN;
PriorPOC.SetDefaultColor(Color.MAGENTA);
PriorPOC.SetLineWeight(1);
PriorPOC.SetStyle(Curve.SHORT_DASH);

# Opening Range Lines
plot ORHighLine = if showOpeningRange and orComplete then openingRangeHigh else Double.NaN;
ORHighLine.SetDefaultColor(Color.LIME);
ORHighLine.SetLineWeight(2);
ORHighLine.SetStyle(Curve.LONG_DASH);

plot ORLowLine = if showOpeningRange and orComplete then openingRangeLow else Double.NaN;
ORLowLine.SetDefaultColor(Color.PINK);
ORLowLine.SetLineWeight(2);
ORLowLine.SetStyle(Curve.LONG_DASH);

plot ORMidline = if showOpeningRange and orComplete then orMid else Double.NaN;
ORMidline.SetDefaultColor(Color.GRAY);
ORMidline.SetLineWeight(1);
ORMidline.SetStyle(Curve.SHORT_DASH);

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
plot ExitLongMarker = if showExitSignals and exitLongSignal then high + (ATR(14) * 0.8) else Double.NaN;
ExitLongMarker.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_DOWN);
ExitLongMarker.SetLineWeight(4);
ExitLongMarker.SetDefaultColor(Color.ORANGE);

plot ExitShortMarker = if showExitSignals and exitShortSignal then low - (ATR(14) * 0.8) else Double.NaN;
ExitShortMarker.SetPaintingStrategy(PaintingStrategy.BOOLEAN_ARROW_UP);
ExitShortMarker.SetLineWeight(4);
ExitShortMarker.SetDefaultColor(Color.ORANGE);

plot EntryLine = if hasPosition and showExitSignals then positionEntryPrice else Double.NaN;
EntryLine.SetDefaultColor(Color.WHITE);
EntryLine.SetLineWeight(1);
EntryLine.SetStyle(Curve.SHORT_DASH);

# ==================== LABELS ====================

AddLabel(yes, "VAH: " + Round(valueAreaHigh, 2), Color.CYAN);
AddLabel(yes, "POC: " + Round(pocValue, 2), Color.MAGENTA);
AddLabel(yes, "VAL: " + Round(valueAreaLow, 2), Color.CYAN);

AddLabel(showOpeningRange and !orComplete, "OR: Building...", Color.GRAY);
AddLabel(showOpeningRange and orComplete and orBias == 1, "Bias: BULL ▲", Color.LIME);
AddLabel(showOpeningRange and orComplete and orBias == -1, "Bias: BEAR ▼", Color.PINK);
AddLabel(showOpeningRange and orComplete and orBias == 0, "Bias: NEUTRAL", Color.GRAY);

AddLabel(showPositionStatus and isLong, "LONG @ " + Round(positionEntryPrice, 2), Color.GREEN);
AddLabel(showPositionStatus and isShort, "SHORT @ " + Round(positionEntryPrice, 2), Color.RED);
AddLabel(showPositionStatus and !hasPosition and !isLockedOut, "FLAT", Color.GRAY);

AddLabel(showTradeCounter and !isLockedOut, 
         "Trades: " + dailyTradeCount + "/" + maxDailyTrades,
         if isWarning then Color.YELLOW else Color.WHITE);

AddLabel(isWarning, "⚠️ LAST TRADE", Color.YELLOW);
AddLabel(isLockedOut, "🛑 LOCKED OUT - STOP TRADING", Color.RED);
AddLabel(rthOnlySignals and !isRTH and !isLockedOut, "OUTSIDE RTH", Color.GRAY);

AddLabel(longSignal and !wasShort and !isLockedOut, "🔔 LONG SIGNAL", Color.GREEN);
AddLabel(shortSignal and !wasLong and !isLockedOut, "🔔 SHORT SIGNAL", Color.RED);
AddLabel(showExitSignals and exitLongSignal, "🔄 EXIT LONG → SHORT", Color.ORANGE);
AddLabel(showExitSignals and exitShortSignal, "🔄 EXIT SHORT → LONG", Color.ORANGE);

# ==================== ALERTS ====================

def alertValBounceLong = if enableAlerts then valBounceLong else no;
def alertPocReclaimLong = if enableAlerts then pocReclaimLong else no;
def alertBreakoutLong = if enableAlerts then breakoutLong else no;
def alertSustainedBreakout = if enableAlerts then sustainedBreakoutNew else no;
def alertVahRejectionShort = if enableAlerts then vahRejectionShort else no;
def alertPocBreakdownShort = if enableAlerts then pocBreakdownShort else no;
def alertBreakdownShort = if enableAlerts then breakdownShort else no;
def alertSustainedBreakdown = if enableAlerts then sustainedBreakdownNew else no;
def alertExitLong = if enableAlerts and exitAlertSound then exitLongSignal else no;
def alertExitShort = if enableAlerts and exitAlertSound then exitShortSignal else no;

Alert(alertValBounceLong, "LONG SIGNAL: VAL Bounce", Alert.BAR, Sound.Ding);
Alert(alertPocReclaimLong, "LONG SIGNAL: POC Reclaim", Alert.BAR, Sound.Ding);
Alert(alertBreakoutLong, "LONG BREAKOUT!", Alert.BAR, Sound.Bell);
Alert(alertSustainedBreakout, "LONG: Sustained Breakout!", Alert.BAR, Sound.Bell);
Alert(alertVahRejectionShort, "SHORT SIGNAL: VAH Rejection", Alert.BAR, Sound.Ding);
Alert(alertPocBreakdownShort, "SHORT SIGNAL: POC Breakdown", Alert.BAR, Sound.Ding);
Alert(alertBreakdownShort, "SHORT BREAKDOWN!", Alert.BAR, Sound.Bell);
Alert(alertSustainedBreakdown, "SHORT: Sustained Breakdown!", Alert.BAR, Sound.Bell);
Alert(alertExitLong, "EXIT LONG - Flip to Short!", Alert.BAR, Sound.Chimes);
Alert(alertExitShort, "EXIT SHORT - Flip to Long!", Alert.BAR, Sound.Chimes);
Alert(enableLockout and justLockedOut, "DAILY LIMIT REACHED!", Alert.BAR, Sound.Ring);

# ==================== TIME LINES ====================

input showTimeLines = yes;

def isMarketOpen = SecondsFromTime(0930) >= 0 and SecondsFromTime(0930) < 300;
def isEuroClose = SecondsFromTime(1130) >= 0 and SecondsFromTime(1130) < 300;
def isLunchStart = SecondsFromTime(1200) >= 0 and SecondsFromTime(1200) < 300;
def isLunchEnd = SecondsFromTime(1330) >= 0 and SecondsFromTime(1330) < 300;
def isPowerHour = SecondsFromTime(1500) >= 0 and SecondsFromTime(1500) < 300;
def isMarketClose = SecondsFromTime(1600) >= 0 and SecondsFromTime(1600) < 300;

AddVerticalLine(showTimeLines and isMarketOpen, "OPEN 9:30", Color.WHITE, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isEuroClose, "EU Close 11:30", Color.ORANGE, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isLunchStart, "Lunch 12:00", Color.GRAY, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isLunchEnd, "Lunch End 1:30", Color.GRAY, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isPowerHour, "Power Hour 3:00", Color.CYAN, Curve.SHORT_DASH);
AddVerticalLine(showTimeLines and isMarketClose, "CLOSE 4:00", Color.WHITE, Curve.SHORT_DASH);

# ==================== ZONE CLOUDS ====================

def upperBound = Highest(high, 50);
def lowerBound = Lowest(low, 50);

def inCallZone = close > valueAreaHigh;
def inPutZone = close < valueAreaLow;

def callZoneUpper = if showPutCallZones and inCallZone then upperBound else Double.NaN;
def callZoneLower = if showPutCallZones and inCallZone then lowerBound else Double.NaN;
def putZoneUpper = if showPutCallZones and inPutZone then upperBound else Double.NaN;
def putZoneLower = if showPutCallZones and inPutZone then lowerBound else Double.NaN;
def creditUpper = if showCreditZones and highCreditZone then upperBound else Double.NaN;
def creditLower = if showCreditZones and highCreditZone then lowerBound else Double.NaN;

AddCloud(callZoneUpper, callZoneLower, Color.GREEN);
AddCloud(putZoneUpper, putZoneLower, Color.RED);
AddCloud(creditUpper, creditLower, Color.ORANGE);

# ==================== SIGNAL BUBBLES ====================

AddChartBubble(showSignals and showSignalBubbles and valBounceLong, low - (ATR(14) * 0.5), "VAL", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and pocReclaimLong, low - (ATR(14) * 0.5), "POC+", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and breakoutLong, low - (ATR(14) * 0.5), "BO", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and sustainedBreakoutNew, low - (ATR(14) * 0.5), "SUS-BO", Color.GREEN, no);
AddChartBubble(showSignals and showSignalBubbles and vahRejectionShort, high + (ATR(14) * 0.5), "VAH", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and pocBreakdownShort, high + (ATR(14) * 0.5), "POC-", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and breakdownShort, high + (ATR(14) * 0.5), "BD", Color.RED, yes);
AddChartBubble(showSignals and showSignalBubbles and sustainedBreakdownNew, high + (ATR(14) * 0.5), "SUS-BD", Color.RED, yes);

AddChartBubble(showExitSignals and showSignalBubbles and exitLongSignal, high + (ATR(14) * 1.2), "EXIT\nLONG", Color.ORANGE, yes);
AddChartBubble(showExitSignals and showSignalBubbles and exitShortSignal, low - (ATR(14) * 1.2), "EXIT\nSHORT", Color.ORANGE, no);
