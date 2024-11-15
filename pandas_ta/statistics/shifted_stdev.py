# -*- coding: utf-8 -*-
from numpy import sqrt as npsqrt
from .variance import variance
from .shifted_variance import shifted_variance
from pandas_ta import Imports
from pandas_ta.utils import get_offset, verify_series


def shifted_stdev(close, length=None, new_mean=None, ddof=None, talib=None, offset=None, **kwargs):
    """Indicator: Standard Deviation Shifted to a new mean"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    ddof = int(ddof) if isinstance(ddof, int) and ddof >= 0 and ddof < length else 1
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    if new_mean is None: return stdev(close=close, length=length, ddof=ddof, talib=talib, offset=offset, **kwargs)
    
    # Calculate Result
 
    shifted_stdev = shifted_variance(close=close, length=length, new_mean=new_mean, ddof=ddof).apply(npsqrt)

    # Offset
    if offset != 0:
        shifted_stdev = shifted_stdev.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        shifted_stdev.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        shifted_stdev.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    shifted_stdev.name = f"SHIFTEDSTDEV_{length}_{new_mean}"
    shifted_stdev.category = "statistics"

    return shifted_stdev


shifted_stdev.__doc__ = \
"""Rolling Standard Deviation shifted to a new mean

Sources:

Calculation:
    Default Inputs:
        length=30
    VAR = shifted_variance
    SHIFTEDSTDEV_ = shifted_variance(close, length, new_mean).apply(np.sqrt)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    ddof (int): Delta Degrees of Freedom.
                The divisor used in calculations is N - ddof,
                where N represents the number of elements. Default: 1
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
