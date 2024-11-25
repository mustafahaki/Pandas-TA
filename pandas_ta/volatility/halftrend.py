import pandas as pd
from numpy import isnan, nan
from pandas import Series, DataFrame
from pandas_ta.volatility import atr
from pandas_ta.overlap import sma
from pandas_ta._typing import Int, DictLike
from pandas_ta.utils import v_pos_default, v_series

def nz(value, default):
    """Return default if value is NaN, otherwise return value."""
    return default if pd.isnull(value) else value

def na(value):
    """Check if a value is NaN."""
    return pd.isnull(value)

def halftrend(
    high: Series, low: Series, close: Series,
    atr_length: Int = None,
    amplitude: Int = None,
    channel_deviation: Int = None,
    **kwargs: DictLike
) -> DataFrame:
    """
    HalfTrend Indicator

    Sources:
        - https://docs.algotest.in/signals/pinescripts/halftrend_strategy/
        - https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/
        - https://github.com/ryu878/halftrend_python

    Args:
        high (pd.Series): Series of 'high's.
        low (pd.Series): Series of 'low's.
        close (pd.Series): Series of 'close's.
        atr_length (int): ATR length. Default is 14.
        amplitude (int): Amplitude. Default is 2.
        channel_deviation (int): Channel deviation. Default is 2.

    Returns:
        pd.DataFrame: DataFrame containing HalfTrend values.
    """
    # Validate inputs
    atr_length = v_pos_default(atr_length, 14)
    amplitude = v_pos_default(amplitude, 2)
    channel_deviation = v_pos_default(channel_deviation, 2)
    _length = max(atr_length, amplitude, channel_deviation) + 1

    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if high is None or low is None or close is None:
        return None

    # Initialize variables
    trend = nextTrend = 0
    up = down = atrHigh = atrLow = 0.0
    direction = None
    df_length = high.size

    arrTrend = [None] * df_length
    arrUp = [None] * df_length
    arrDown = [None] * df_length

    atrHighSeries = pd.Series([0.0] * df_length)
    atrLowSeries = pd.Series([0.0] * df_length)
    atrCloseSeries = pd.Series([0.0] * df_length)
    atrDirectionSeries = pd.Series([None] * df_length)

    maxLowPrice = low.iat[atr_length - 1]
    minHighPrice = high.iat[atr_length - 1]

    if close.iat[0] > low.iat[atr_length]:
        trend = nextTrend = 1

    # Main calculation loop
    atr_N = atr(high, low, close, window=atr_length)
    highma_N = sma(high, amplitude)
    lowma_N = sma(low, amplitude)
    highestbars = high.rolling(amplitude, min_periods=1).max()
    lowestbars = low.rolling(amplitude, min_periods=1).min()

    for i in range(1, df_length):
        atr2 = atr_N.iat[i] / 2.0
        dev = channel_deviation * atr2

        highPrice = highestbars.iat[i]
        lowPrice = lowestbars.iat[i]

        if nextTrend == 1:
            maxLowPrice = max(lowPrice, maxLowPrice)
            if highma_N.iat[i] < maxLowPrice and close.iat[i] < nz(low.iat[i - 1], low.iat[i]):
                trend = nextTrend = 0
                minHighPrice = highPrice
        else:
            minHighPrice = min(highPrice, minHighPrice)
            if lowma_N.iat[i] > minHighPrice and close.iat[i] > nz(high.iat[i - 1], high.iat[i]):
                trend = nextTrend = 1
                maxLowPrice = lowPrice

        arrTrend[i] = trend

        if trend == 0:
            up = max(maxLowPrice, nz(arrUp[i - 1], maxLowPrice))
            direction = "long"
            atrHigh, atrLow = up + dev, up - dev
            arrUp[i] = up
        else:
            down = min(minHighPrice, nz(arrDown[i - 1], minHighPrice))
            direction = "short"
            atrHigh, atrLow = down + dev, down - dev
            arrDown[i] = down

        atrHighSeries.iat[i] = atrHigh
        atrLowSeries.iat[i] = atrLow
        atrCloseSeries.iat[i] = up if trend == 0 else down
        atrDirectionSeries.iat[i] = direction

    # Output DataFrame
    _props = f"_{atr_length}_{amplitude}_{channel_deviation}"
    _name = "HALFTREND"

    data = {
        f"{_name}_atrHigh{_props}": atrHighSeries,
        f"{_name}_close{_props}": atrCloseSeries,
        f"{_name}_atrLow{_props}": atrLowSeries,
        f"{_name}_direction{_props}": atrDirectionSeries,
        f"{_name}_arrUp{_props}": arrUp,
        f"{_name}_arrDown{_props}": arrDown
    }

    df = DataFrame(data, index=close.index)
    df.name = f"{_name}{_props}"
    df.category = "volatility"

    return df
