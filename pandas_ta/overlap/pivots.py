# -*- coding: utf-8 -*-
from numpy import NaN, greater, zeros_like
from numba import njit
from pandas import DataFrame, Series, Timedelta, infer_freq, to_datetime
from pandas_ta._typing import Array, DictLike
from pandas_ta.utils import np_non_zero_range, v_datetime_ordered, v_series, v_str


@njit(cache=True)
def pivot_camarilla(high: Array, low: Array, close: Array):
    """Camarilla Pivot Points - requires high, low, close"""
    tp = (high + low + close) / 3
    hl_range = np_non_zero_range(high, low)

    s1 = close - 11 / 120 * hl_range
    s2 = close - 11 / 60 * hl_range
    s3 = close - 0.275 * hl_range
    s4 = close - 0.55 * hl_range

    r1 = close + 11 / 120 * hl_range
    r2 = close + 11 / 60 * hl_range
    r3 = close + 0.275 * hl_range
    r4 = close + 0.55 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_classic(high: Array, low: Array, close: Array):
    """Classic Pivot Points - requires high, low, close"""
    tp = (high + low + close) / 3
    hl_range = np_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 3 * hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 3 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4



@njit(cache=True)
def pivot_demark(open_: Array, high: Array, low: Array, close: Array):
    """DeMark Pivot Points - requires open, high, low, close"""
    if (open_ == close).all():
        tp = 0.25 * (high + low + 2 * close)
    elif greater(close, open_).all():
        tp = 0.25 * (2 * high + low + close)
    else:
        tp = 0.25 * (high + 2 * low + close)

    s1 = 2 * tp - high
    r1 = 2 * tp - low

    return tp, s1, r1


@njit(cache=True)
def pivot_fibonacci(high: Array, low: Array, close: Array):
    """Fibonacci Pivot Points - requires high, low, close"""
    tp = (high + low + close) / 3
    hl_range = np_non_zero_range(high, low)

    s1 = tp - 0.382 * hl_range
    s2 = tp - 0.618 * hl_range
    s3 = tp - hl_range

    r1 = tp + 0.382 * hl_range
    r2 = tp + 0.618 * hl_range
    r3 = tp + hl_range

    return tp, s1, s2, s3, r1, r2, r3


@njit(cache=True)
def pivot_traditional(high: Array, low: Array, close: Array):
    """Traditional Pivot Points - requires high, low, close"""
    tp = (high + low + close) / 3
    hl_range = np_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 2 * hl_range

    r1 = 2 * tp - low
    r2 = tp +  hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 2 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_woodie(open_: Array, high: Array, low: Array):
    """Woodie Pivot Points - requires open, high, low"""
    tp = (2 * open_ + high + low) / 4
    hl_range = np_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = low - 2 * (high - tp)
    s4 = s3 - hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = high + 2 * (tp - low)
    r4 = r3 + hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


def pivots(
    open_: Series, high: Series,
    low: Series, close: Series,
    method: str = None, anchor: str = None,
    **kwargs: DictLike
) -> DataFrame:
    """Pivot Points

    Pivot Points are used to calculate possible support and resistance levels
    given previous price action. There are many different methods of
    calculating Pivot Points.

    The most common (and default) method is: Traditional. Other methods
    include: Camarilla, Classic, Demark, Fibonacci, and Woodie.

    Sources:
        https://www.sierrachart.com/index.php?page=doc/PivotPoints.html

    Args:
        open_ (pd.Series): Series of 'open's
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        method (str): The method to use. Default: 'traditional'
        anchor (str): The anchor frequency. Default: 'D'
            Pandas Offset Aliases include:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.DataFrame: New feature generated.
    """
    # Validate
    open_ = v_series(open_)
    high = v_series(high)
    low = v_series(low)
    close = v_series(close)

    if open_ is None or high is None or low is None or close is None:
        return None

    methods = [
        "traditional", "fibonacci", "woodie", "classic", "demark", "camarilla"
    ]
    method = v_str(method, methods[0])

    if close.index.size < 3:
        return  # Emergency Break

    if not v_datetime_ordered(close):
        print("[!] Pivots requires a datetime ordered index.")
        return

    dt_index = close.index
    freq = infer_freq(dt_index)

    if anchor and isinstance(anchor, str) and len(anchor) >= 1:
        anchor = anchor.upper()
    else:
        anchor = "D"

    # Resample if freq does not match the anchor
    if freq is not anchor:
        df = DataFrame(
            data={
                "open": open_.resample(anchor).first(),
                "high": high.resample(anchor).max(),
                "low": low.resample(anchor).min(),
                "close": close.resample(anchor).last()
            }
        )
        df.dropna(inplace=True)
    else:
        df = DataFrame(
            data={"open": open_, "high": high, "low": low, "close": close},
            index=dt_index
        )

    np_open = df.open.values
    np_high = df.high.values
    np_low = df.low.values
    np_close = df.close.values

    # Create nan arrays for "demark" and "fibonacci" pivots
    _nan_array = zeros_like(np_close)
    _nan_array[:] = NaN
    tp = s1 = s2 = s3 = s4 = r1 = r2 = r3 = r4 = _nan_array

    # Calculate
    if method == "camarilla":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = \
            pivot_camarilla(np_high, np_low, np_close)

    elif method == "classic":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = \
            pivot_classic(np_high, np_low, np_close)

    elif method == "demark":
        tp, s1, r1 = pivot_demark(np_open, np_high, np_low, np_close)

    elif method == "fibonacci":
        tp, s1, s2, s3, r1, r2, r3 = pivot_fibonacci(np_high, np_low, np_close)

    elif method == "woodie":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = \
            pivot_woodie(np_open, np_high, np_low)

    else: # Traditional
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = \
            pivot_traditional(np_high, np_low, np_close)

    # Name and Category
    _props = f"PIVOTS_{method[:4].upper()}_{anchor}"

    df[f"{_props}_P"] = tp
    df[f"{_props}_S1"], df[f"{_props}_S2"] = s1, s2
    df[f"{_props}_S3"], df[f"{_props}_S4"] = s3, s4
    df[f"{_props}_R1"], df[f"{_props}_R2"] = r1, r2
    df[f"{_props}_R3"], df[f"{_props}_R4"] = r3, r4
    
    df[f"{_props}_P"].category = df[f"{_props}_S1"].category = df[f"{_props}_S2"].category = df[f"{_props}_S3"].category = df[f"{_props}_S4"].category = df[f"{_props}_R1"].category = df[f"{_props}_R2"].category = df[f"{_props}_R3"].category = df[f"{_props}_R4"].category = "overlap"
    df[f"{_props}_P"].attrs["variable_type"] = df[f"{_props}_S1"].attrs["variable_type"] = df[f"{_props}_S2"].attrs["variable_type"] = df[f"{_props}_S3"].attrs["variable_type"] = df[f"{_props}_S4"].attrs["variable_type"] = df[f"{_props}_R1"].attrs["variable_type"] = df[f"{_props}_R2"].attrs["variable_type"] = df[f"{_props}_R3"].attrs["variable_type"] = df[f"{_props}_R4"].attrs["variable_type"] = "continuous"

    df.index = df.index + Timedelta(1, anchor.lower())
    if freq is not anchor:
        df = df.reindex(dt_index, method="ffill")
    df = df.iloc[:,4:]

    # hm = df.loc[lambda x: (x.index.hour == 23) & (x.index.minute == 59)].iloc[0].name
    # df.loc[:hm] = NaN

    if method in ["demark", "fibonacci"]:
        df.drop(columns=[x for x in df.columns if all(df[x].isna())], inplace=True)

    df.name = _props
    df.category = "overlap"

    return df
