# -*- coding: utf-8 -*-
from sys import modules as sys_modules
from numpy import sqrt
from pandas import Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.utils import v_offset, v_pos_default, v_series,v_mamode
from .wma import wma
from .ema import ema


def hma(
    close: Series, length: Int = None,mamode: str = None,
    offset: Int = None, **kwargs: DictLike
) -> Series:
    """Hull Moving Average (HMA)

    The Hull Exponential Moving Average attempts to reduce or remove lag
    in moving averages.

    Sources:
        https://alanhull.com/hull-moving-average

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        mamode (str): Options: 'ema', 'wma'. Default: 'wma'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    mamode = v_mamode(mamode, "wma")
    length = v_pos_default(length, 10)
    close = v_series(close, length + 2)

    if close is None:
        return

    offset = v_offset(offset)

    supported_mas = [
        "ema","wma"
    ]

    if mamode not in supported_mas:
        return

    # Calculate
    half_length = int(length / 2)
    sqrt_length = int(sqrt(length))

    fn=getattr(sys_modules[__name__],mamode)

    maf = fn(close=close, length=half_length)
    mas = fn(close=close, length=length)
    hma = fn(close=2 * maf - mas, length=sqrt_length)

    # Offset
    if offset != 0:
        hma = hma.shift(offset)

    # Fill
    if "fillna" in kwargs:
        hma.fillna(kwargs["fillna"], inplace=True)

    # Name and Category
    hma.name = f"HMA_{length}"
    hma.category = "overlap"

    return hma
