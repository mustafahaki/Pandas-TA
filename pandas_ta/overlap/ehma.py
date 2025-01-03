# -*- coding: utf-8 -*-
from numpy import sqrt as npSqrt
from .ema import ema
from pandas_ta.utils import get_offset, verify_series


def ehma(close, length=None, offset=None, **kwargs):
    """Indicator: Exponnetial Hull Moving Average (HMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    # Calculate Result
    half_length = int(length / 2)
    sqrt_length = int(npSqrt(length))

    emaf = ema(close=close, length=half_length)
    emas = ema(close=close, length=length)
    ehma = ema(close=(2 * emaf - emas), length=sqrt_length)

    # Offset
    if offset != 0:
        ehma = ehma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ehma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ehma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    ehma.name = f"EHMA_{length}"
    ehma.category = "overlap"

    return ehma


ehma.__doc__ = \
"""Exponential Hull Moving Average (HMA)

The Hull Exponential Moving Average attempts to reduce or remove lag in moving
averages.


Calculation:
    Default Inputs:
        length=10
    EMA = Weighted Moving Average
    half_length = int(0.5 * length)
    sqrt_length = int(sqrt(length))

    emaf = EMA(close, half_length)
    emas = EMA(close, length)
    EHMA = EMA(2 * emaf - emas, sqrt_length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
