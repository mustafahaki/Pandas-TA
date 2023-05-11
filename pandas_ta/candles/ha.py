# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas_ta.utils import v_offset, v_series


def ha(open_, high, low, close, offset=None, **kwargs):
    """Candle Type: Heikin Ashi"""
    # Validate Arguments
    open_ = v_series(open_,1)
    high = v_series(high,1)
    low = v_series(low,1)
    close = v_series(close,1)
    offset = v_offset(offset)

   # Calculate Result
    m = close.size
    # store of calcualtions
    open_zero = 0.5 * (open_.iloc[0] + close.iloc[0])
    close_zero = 0.25 * (open_ + high + low + close)
    df = DataFrame({
        "HA_open":  open_zero,
        "HA_high": high,
        "HA_low": low,
        "HA_close":close_zero,
    })
    # as lists
    ha_open = df['HA_open'].tolist()
    ha_close = close_zero.tolist()
    for i in range(1, m):
        ha_open[i] = (0.5 * (ha_open[i - 1] + ha_close[i - 1]))
    df["HA_open"] = ha_open
    df["HA_high"] = df[["HA_open", "HA_high", "HA_close"]].max(axis=1)
    df["HA_low"] = df[["HA_open", "HA_low", "HA_close"]].min(axis=1)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    df.name = "Heikin-Ashi"
    df.category = "candles"


ha.__doc__ = \
"""Heikin Ashi Candles (HA)

The Heikin-Ashi technique averages price data to create a Japanese
candlestick chart that filters out market noise. Heikin-Ashi charts,
developed by Munehisa Homma in the 1700s, share some characteristics
with standard candlestick charts but differ based on the values used
to create each candle. Instead of using the open, high, low, and close
like standard candlestick charts, the Heikin-Ashi technique uses a
modified formula based on two-period averages. This gives the chart a
smoother appearance, making it easier to spots trends and reversals,
but also obscures gaps and some price data.

Sources:
    https://www.investopedia.com/terms/h/heikinashi.asp

Calculation:
    HA_OPEN[0] = (open[0] + close[0]) / 2
    HA_CLOSE = (open[0] + high[0] + low[0] + close[0]) / 4

    for i > 1 in df.index:
        HA_OPEN = (HA_OPEN[i−1] + HA_CLOSE[i−1]) / 2

    HA_HIGH = MAX(HA_OPEN, HA_HIGH, HA_CLOSE)
    HA_LOW = MIN(HA_OPEN, HA_LOW, HA_CLOSE)

    How to Calculate Heikin-Ashi

    Use one period to create the first Heikin-Ashi (HA) candle, using
    the formulas. For example use the high, low, open, and close to
    create the first HA close price. Use the open and close to create
    the first HA open. The high of the period will be the first HA high,
    and the low will be the first HA low. With the first HA calculated,
    it is now possible to continue computing the HA candles per the formulas.
​​
Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ha_open, ha_high,ha_low, ha_close columns.
"""