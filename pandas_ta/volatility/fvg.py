# -*- coding: utf-8 -*-
from numba import njit
from numpy import full, nan
from pandas import DataFrame, Series

from pandas_ta._typing import DictLike, Int, IntFloat
from pandas_ta.utils import (
    v_offset,
    v_pos_default,
    v_series,
)


@njit(cache=True)
def nb_fvg(
    np_open: Series,
    np_high: Series,
    np_low: Series,
    np_close: Series,
    min_gap: IntFloat,
):
    n = np_open.size

    fvg_high = full(n, nan)
    fvg_low = full(n, nan)
    fvg_type = full(n, nan)

    for i in range(1, n - 1):
        if np_close[i] > np_open[i]:
            if (np_low[i + 1] - np_high[i - 1]) > min_gap * np_close[i]:
                fvg_low[i] = np_high[i - 1]
                fvg_high[i] = np_low[i + 1]
                fvg_type[i] = 1  # bullish

        elif np_close[i] < np_open[i]:
            if (np_low[i - 1] - np_high[i + 1]) > min_gap * np_close[i]:
                fvg_low[i] = np_high[i + 1]
                fvg_high[i] = np_low[i - 1]
                fvg_type[i] = -1  # bearish

    return fvg_high, fvg_low, fvg_type


def fvg(
    open: Series,
    high: Series,
    low: Series,
    close: Series,
    min_gap: IntFloat = None,
    offset: Int = None,
    **kwargs: DictLike,
) -> DataFrame:
    """Fair Value Gap (FVG)

    An FVG occurs when a strong momentum candle creates an imbalance between
    the high and low of surrounding candles, forming a price inefficiency that
    the market may revisit. So it can be representative of high volatility.

    Sources:
        https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps

    Args:
        open (pd.Series): Series of 'open's
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's.
        min_gap (int|float|None): minimum percentage gap size. Default: 0
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.DataFrame: fvg_high and fvg_low, and fvg_type (1 for bullish or -1 for bearish).
    """

    # Validate
    _length = 3
    open = v_series(open, _length)
    high = v_series(high, _length)
    low = v_series(low, _length)
    close = v_series(close, _length)

    if open is None or high is None or low is None or close is None:
        return

    min_gap = v_pos_default(min_gap, 0)
    min_gap = min_gap / 100
    offset = v_offset(offset)

    # calculation
    np_open, np_high, np_low, np_close = (
        open.to_numpy(),
        high.to_numpy(),
        low.to_numpy(),
        close.to_numpy(),
    )
    fvg_high, fvg_low, fvg_type = nb_fvg(
        np_open=np_open,
        np_high=np_high,
        np_low=np_low,
        np_close=np_close,
        min_gap=min_gap,
    )

    # Offset
    if offset != 0:
        fvg_high = fvg_high.shift(offset)
        fvg_low = fvg_low.shift(offset)
        fvg_type = fvg_type.shift(offset)

    # Fill
    if "fillna" in kwargs:
        fvg_high.fillna(kwargs["fillna"], inplace=True)
        fvg_low.fillna(kwargs["fillna"], inplace=True)
        fvg_type.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fvg_high.fillna(method=kwargs["fill_method"], inplace=True)
        fvg_low.fillna(method=kwargs["fill_method"], inplace=True)
        fvg_type.fillna(method=kwargs["fill_method"], inplace=True)

    _props = f"_{min_gap}"
    data = {
        f"FVGh{_props}": fvg_high,
        f"FVGl{_props}": fvg_low,
        f"FVGt{_props}": fvg_type,
    }
    df = DataFrame(data, index=high.index)
    df.name = f"FVG{_props}"
    df.category = "volatility"

    return df
