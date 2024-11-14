# -*- coding: utf-8 -*-
from pandas_ta import Imports
from pandas_ta.utils import get_offset, verify_series


def imi(open_, close, length=None, offset=None, **kwargs):
    """Indicator: Intraday Momentum Index (IMI)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    open_ = verify_series(open_, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    
    if open_ is None or close is None: return

    # Calculate Result
    
    up = (close > open_) * (close - open_)
    down = (close < open_) * (open_ - close)
    
    sum_up = up.rolling(length, min_periods = length).sum()
    sum_down = down.rolling(length, min_periods =length).sum()
    
    imi = 100*sum_up / (sum_up + sum_down)
   
    # Offset
    if offset != 0:
        imi = imi.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        imi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        imi.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    imi.name = f"IMI_{length}"
    imi.category = "momentum"

    return imi


imi.__doc__ = \
"""Intraday Momentum Index (IMI)

The Intraday Momentum Index (IMI), is a technical indicator that combines aspects of candlestick
analysis with the relative strength index (RSI) in order to generate overbought or oversold signals. 

Sources:
    https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp

Calculation:
    Default Inputs:
        length=14
    up = Gains
    down = Losses
    sum_up = sum(up, length)
    sum_down = sum(down, length)
    IMI = 100 * sum_up / (sum_up + sum_down)
    
Args:
    open_ (pd.Series): Series of 'open's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""