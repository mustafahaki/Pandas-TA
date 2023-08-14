# -*- coding: utf-8 -*-
from numpy import empty_like, maximum, minimum
from numba import njit
from pandas import DataFrame, Series
from pandas_ta._typing import Array, DictLike, Int
from pandas_ta.utils import v_offset, v_series


@njit(cache=True)
def np_ha(np_open: Array, np_high: Array, np_low: Array, np_close: Array):
    """Heiken-Ashi - numpy/numba version"""
    ha_close = 0.25 * (np_open + np_high + np_low + np_close)
    ha_open = empty_like(ha_close)
    ha_open[0] = 0.5 * (np_open[0] + np_close[0])

    m = np_close.size
    for i in range(1, m):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])

    ha_high = maximum(maximum(ha_open, ha_close), np_high)
    ha_low = minimum(minimum(ha_open, ha_close), np_low)

    return ha_open, ha_high, ha_low, ha_close


def ha(
    open_: Series, high: Series, low: Series, close: Series,
    offset: Int = None, **kwargs: DictLike
) -> DataFrame:
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
    # Validate
    open_ = v_series(open_, 1)
    high = v_series(high, 1)
    low = v_series(low, 1)
    close = v_series(close, 1)
    offset = v_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return

    # Calculate
    np_open, np_high = open_.values, high.values
    np_low, np_close = low.values, close.values
    ha_open, ha_high, ha_low, ha_close = np_ha(np_open, np_high, np_low, np_close)
    ha_open = Series(ha_open, index=close.index)
    ha_high = Series(ha_high, index=close.index)
    ha_low = Series(ha_low, index=close.index)
    ha_close = Series(ha_close, index=close.index)
    ha_open.name = "HA_open"
    ha_high.name = "HA_high"
    ha_low.name = "HA_low"
    ha_close.name = "HA_close"
    ha_open.category = ha_high.category = ha_low.category = ha_close.category = "candles"
    ha_open.attrs["variable_type"] = ha_high.attrs["variable_type"] = ha_low.attrs["variable_type"] = ha_close.attrs["variable_type"] = "continuous"
    df = DataFrame({
        ha_open.name: ha_open,
        ha_high.name: ha_high,
        ha_low.name: ha_low,
        ha_close.name: ha_close,
    })

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Fill
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Category
    df.name = "Heikin-Ashi"
    df.category = "candles"

    return df
