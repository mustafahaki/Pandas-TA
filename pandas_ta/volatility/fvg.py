# -*- coding: utf-8 -*-
from numpy import floor, isnan, nan, zeros, zeros_like
from numba import njit
from pandas import Series, DataFrame
from pandas_ta._typing import DictLike, Int, IntFloat
from pandas_ta.utils import (
    v_bool,
    v_offset,
    v_pos_default,
    v_series,
)


def fvg(
    open: Series, high: Series, low: Series, close: Series,
    min_gap=None,
    offset: Int = None, **kwargs: DictLike
):
    """ Fair Value Gap (FVG)

    An FVG occurs when a strong momentum candle creates an imbalance between
    the high and low of surrounding candles, forming a price inefficiency that
    the market may revisit. So it can be representative of high volatility.

    Sources:
        https://www.tradingview.com/support/solutions/43000591664-zig-zag/#:~:text=Definition,trader%20visual%20the%20price%20action.
        https://school.stockcharts.com/doku.php?id=technical_indicators:zigzag

    Args:
        open (pd.Series): Series of 'open's
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's.
        min_gap (int): minimum percentage gap size. Default: 0
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.DataFrame: fvg_high and fvg_low, and fvg_type (bullish or bearish).
    """

    pass
