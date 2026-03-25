//@version=6
strategy("TuTuTu - 測試訊號能否進場", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// === 先簡化一切 ===
rangePeriod = input.int(15, "Range Period")
atrPeriod   = input.int(14, "ATR Period")
multiplier  = input.float(1.0, "區間倍數")

// === 區間邏輯 ===
avgPrice   = ta.ema(close, rangePeriod)
volatility = ta.atr(atrPeriod) * multiplier

isInRange = true
for i = 0 to rangePeriod - 1
    if math.abs(close[i] - avgPrice) > volatility
        isInRange := false

var string boxState = "Idle"
var bool signalTriggered = false
var float upperBound = na
var float lowerBound = na

if isInRange and not isInRange[1]
    upperBound := avgPrice + volatility
    lowerBound := avgPrice - volatility
    boxState := "Active"
    signalTriggered := false

breakoutUp = close > upperBound and not na(upperBound)
breakoutDown = close < lowerBound and not na(lowerBound)

canEnter = strategy.opentrades == 0

if boxState == "Active" and not signalTriggered and canEnter
    if breakoutUp
        strategy.entry("Long", strategy.long)
        signalTriggered := true
        boxState := "Triggered"

    else if breakoutDown
        strategy.entry("Short", strategy.short)
        signalTriggered := true
        boxState := "Triggered"
