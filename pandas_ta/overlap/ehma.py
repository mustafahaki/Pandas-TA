# -*- coding: utf-8 -*-
from numpy import sqrt
from pandas import Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.utils import v_offset, v_pos_default, v_series
from .ema import ema



def ehma(
    close: Series, length: Int = None,
    offset: Int = None, **kwargs: DictLike
) -> Series:
    """Hull Moving Average (EHMA)

    The Hull Exponential Moving Average attempts to reduce or remove lag
    in moving averages.

    Sources:
        https://alanhull.com/hull-moving-average

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 10)
    close = v_series(close, length + 2)

    if close is None:
        return

    offset = v_offset(offset)

    # Calculate
    half_length = int(length / 2)
    sqrt_length = int(sqrt(length))

    emaf = ema(close=close, length=half_length)
    emas = ema(close=close, length=length)
    ehma = ema(close=2 * emaf - emas, length=sqrt_length)

    # Offset
    if offset != 0:
        ehma = ehma.shift(offset)

    # Fill
    if "fillna" in kwargs:
        ehma.fillna(kwargs["fillna"], inplace=True)

    # Name and Category
    ehma.name = f"EHMA_{length}"
    ehma.category = "overlap"

    return ehma
