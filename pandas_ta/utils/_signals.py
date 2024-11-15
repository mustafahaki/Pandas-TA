# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat

from ._core import get_offset, verify_series
from ._math import zero


def _above_below(series_a: Series, series_b: Series, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)

    # Calculate Result
    if above:
        current = series_a >= series_b
    else:
        current = series_a <= series_b

    if asint:
        current = current.astype(int)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name & Category
    current.name = f"{series_a.name}_{'A' if above else 'B'}_{series_b.name}"
    current.category = "utility"

    return current


def above(series_a: Series, series_b: Series, asint: bool = True, offset: int = None, **kwargs):
    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)


def above_value(series_a: Series, value: float, asint: bool = True, offset: int = None, **kwargs):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)


def below(series_a: Series, series_b: Series, asint: bool = True, offset: int = None, **kwargs):
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)


def below_value(series_a: Series, value: float, asint: bool = True, offset: int = None, **kwargs):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)

def between(series_a: Series, series_b: Series, series_c: Series, asint: bool = True, offset: int = None, **kwargs):
    """Indicator: Between, checks if series_a is between series_b  and series_c.
        dynamically handling cases where series_b < series_c.
    """
    
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    series_c = verify_series(series_c)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)
    series_c.apply(zero)

    # Dynamically compute the lower and upper bounds
    lower_bound = concat([series_b, series_c], axis=1).min(axis=1)
    upper_bound = concat([series_b, series_c], axis=1).max(axis=1)

    # Check if series_a is within bounds
    current = (series_a >= lower_bound) & (series_a <= upper_bound)


    if asint:
        current = current.astype(int)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name & Category
    current.name = f"{series_c.name}_<=_{series_a.name}_<=_{series_b.name}"
    current.category = "utility"

    return current

def diff(series_a: Series, series_b: Series, asint: bool = False, offset: int = None, **kwargs):
    """Indicator: diff, calculates the difference between two series."""

    series_a = verify_series(series_a,0)
    series_b = verify_series(series_b,0)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)
   
    # Calculate Result
    current = series_a - series_b

    if asint:
        current = current.astype(int)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name & Category
    current.name = f"{series_a.name}_-_{series_b.name}"
    current.category = "utility"

    return current

import pandas as pd
from pandas import Series, concat

def price_deviation(series_a: Series, series_b: Series, length: int = 3, asint: bool = False, offset: int = None, **kwargs):
    """
    Indicator: price_deviation, calculates the absolute deviation between the price and the moving average (MA)
    over the past 'n' periods.
    
    Parameters:
    - series_a (Series): The series containing price data (e.g., 'close').
    - series_b (Series): The series containing moving average data (e.g., 'MA20').
    - periods (int): The number of periods over which to calculate the absolute deviation. Default is 3.
    - asint (bool): Whether to convert the result to integer type. Default is True.
    - offset (int): The number of periods to shift the result. Default is None.
    
    Returns:
    - Series: A Series containing the absolute deviations over the past 'n' periods.
    """
    # Ensure the series are valid
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)
    
    # Apply rolling window to compute absolute difference from MA for 'n' periods
    lower_bound = concat([series_a, series_b], axis=1).min(axis=1)
    upper_bound = concat([series_a, series_b], axis=1).max(axis=1)
    
    # Calculate the absolute deviation (difference between max and min in the rolling window)
    current = upper_bound - lower_bound
    current = current.rolling(window=length,min_periods=1).apply(lambda x: abs(x[-1] - x[0]), raw=False)
    
    # Convert to integer if required
    if asint:
        current = current.astype(int)
    
    # Offset (shift) the result if specified
    if offset != 0:
        current = current.shift(offset)
    
    
    # Handle fills
    if "fillna" in kwargs:
        current.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        current.fillna(method=kwargs["fill_method"], inplace=True)
    
    # Naming the output with the original series names
    current.name = f"PRICEDEVIATION_{series_a.name}_{series_b.name}_{length}"
    current.category = "utility"
    
    return current



def absolute_diff(series_a: Series, series_b: Series, asint: bool = False, offset: int = None, **kwargs):
    """Indicator: absolute_diff, calculates the absolute difference between two series."""

    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)
    
     # Dynamically compute the lower and upper bounds
    lower_bound = concat([series_a, series_b], axis=1).min(axis=1)
    upper_bound = concat([series_a, series_b], axis=1).max(axis=1)
    
    # Calculate Result
    current = upper_bound - lower_bound

    if asint:
        current = current.astype(int)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name & Category
    current.name = f"ABS_{series_a.name}_-_{series_b.name}"
    current.category = "utility"

    return current

def sum(series_a: Series, series_b: Series, asint: bool = False, offset: int = None, **kwargs):
    """Indicator: sum, calculates the sum of two series."""

    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)

    # Calculate Result
    current = series_a + series_b

    if asint:
        current = current.astype(int)

    # Offset
    if offset != 0:
        current = current.shift(offset)

    # Name & Category
    current.name = f"{series_a.name}_+_{series_b.name}"
    current.category = "utility"

    return current

def cross_value(series_a: Series, value: float, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return cross(series_a, series_b, above, asint, offset, **kwargs)


def cross(series_a: Series, series_b: Series, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)

    # Calculate Result
    current = series_a > series_b  # current is above
    previous = series_a.shift(1) < series_b.shift(1)  # previous is below
    # above if both are true, below if both are false
    cross = current & previous if above else ~current & ~previous

    if asint:
        cross = cross.astype(int)

    # Offset
    if offset != 0:
        cross = cross.shift(offset)

    # Name & Category
    cross.name = f"{series_a.name}_{'XA' if above else 'XB'}_{series_b.name}"
    cross.category = "utility"

    return cross


def signals(indicator, xa, xb, cross_values, xserie, xserie_a, xserie_b, cross_series, offset) -> DataFrame:
    df = DataFrame()
    if xa is not None and isinstance(xa, (int, float)):
        if cross_values:
            crossed_above_start = cross_value(indicator, xa, above=True, offset=offset)
            crossed_above_end = cross_value(indicator, xa, above=False, offset=offset)
            df[crossed_above_start.name] = crossed_above_start
            df[crossed_above_end.name] = crossed_above_end
        else:
            crossed_above = above_value(indicator, xa, offset=offset)
            df[crossed_above.name] = crossed_above

    if xb is not None and isinstance(xb, (int, float)):
        if cross_values:
            crossed_below_start = cross_value(indicator, xb, above=True, offset=offset)
            crossed_below_end = cross_value(indicator, xb, above=False, offset=offset)
            df[crossed_below_start.name] = crossed_below_start
            df[crossed_below_end.name] = crossed_below_end
        else:
            crossed_below = below_value(indicator, xb, offset=offset)
            df[crossed_below.name] = crossed_below

    # xseries is the default value for both xserie_a and xserie_b
    if xserie_a is None:
        xserie_a = xserie
    if xserie_b is None:
        xserie_b = xserie

    if xserie_a is not None and verify_series(xserie_a):
        if cross_series:
            cross_serie_above = cross(indicator, xserie_a, above=True, offset=offset)
        else:
            cross_serie_above = above(indicator, xserie_a, offset=offset)

        df[cross_serie_above.name] = cross_serie_above

    if xserie_b is not None and verify_series(xserie_b):
        if cross_series:
            cross_serie_below = cross(indicator, xserie_b, above=False, offset=offset)
        else:
            cross_serie_below = below(indicator, xserie_b, offset=offset)

        df[cross_serie_below.name] = cross_serie_below

    return df
