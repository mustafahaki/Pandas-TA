# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from pandas import DataFrame, Series
from pandas_ta._typing import DictLike, Int, IntFloat
from pandas_ta.utils import v_offset, v_pos_default, v_series, zero


def psar(
    high: Series, low: Series, close: Series = None,
    af0: IntFloat = None, af: IntFloat = None, max_af: IntFloat = None, tv=False,
    offset: Int = None, **kwargs: DictLike
) -> DataFrame:
    """Parabolic Stop and Reverse (psar)
    ...

    (Docstring remains the same)
    """

    # Validate
    _length = 1
    high = v_series(high, _length)
    low = v_series(low, _length)

    if high is None or low is None:
        return

    orig_high = high.copy()
    orig_low = low.copy()
    # Numpy arrays offer some performance improvements
    high_values, low_values = high.values, low.values

    # Corrected parameter initialization
    af_increment = v_pos_default(af, 0.02)  # Acceleration Factor Increment
    af0 = v_pos_default(af0, 0.02)          # Initial Acceleration Factor
    max_af = v_pos_default(max_af, 0.2)
    offset = v_offset(offset)

    # Initialize variables
    m = high_values.size
    sar = np.zeros(m, dtype=np.float64)
    af_array = np.zeros(m, dtype=np.float64)
    reversal = np.zeros(m, dtype=np.int64)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)

    # Determine initial trend direction
    falling = _falling(orig_high.iloc[:2], orig_low.iloc[:2])
    if falling:
        ep = low_values[0]
        sar[0] = high_values[0] if close is None else close.iloc[0]
        short[0] = sar[0]
    else:
        ep = high_values[0]
        sar[0] = low_values[0] if close is None else close.iloc[0]
        long[0] = sar[0]

    af = af0
    af_array[0] = af

    # Calculate using Numba-accelerated function
    sar, long, short, af_array, reversal = _psar_numba(
        high_values, low_values, sar, long, short,
        af0, af_increment, max_af, falling, ep, af_array, reversal
    )

    # Convert arrays back to Series
    _af = Series(af_array, index=orig_high.index)
    long = Series(long, index=orig_high.index)
    short = Series(short, index=orig_high.index)
    reversal = Series(reversal, index=orig_high.index)

    # Apply offset if provided
    if offset != 0:
        _af = _af.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)
        reversal = reversal.shift(offset)

    # Fill NaN values if 'fillna' is provided in kwargs
    if "fillna" in kwargs:
        _af.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
        reversal.fillna(kwargs["fillna"], inplace=True)

    # Prepare DataFrame to return
    _props = f"_{af0}_{max_af}"
    data = {
        f"PSARl{_props}": long,
        f"PSARs{_props}": short,
        f"PSARaf{_props}": _af,
        f"PSARr{_props}": reversal
    }
    df = DataFrame(data, index=orig_high.index)
    df.name = f"PSAR{_props}"
    df.category = long.category = short.category = "trend"

    return df


@njit
def _psar_numba(high, low, sar, long, short,
                af0, af_increment, max_af, falling, ep, af_array, reversal):
    m = len(high)
    af = af0

    for i in range(1, m):
        prior_sar = sar[i - 1]
        prior_ep = ep
        prior_af = af
        prior_falling = falling

        if not prior_falling:  # Rising trend
            sar_i = prior_sar + prior_af * (prior_ep - prior_sar)
            # Adjust SAR to not exceed the last two lows
            if i >= 2:
                sar_i = min(sar_i, low[i - 1], low[i - 2])
            else:
                sar_i = min(sar_i, low[i - 1])

            reverse = low[i] < sar_i
            if high[i] > prior_ep:
                ep = high[i]
                af = min(prior_af + af_increment, max_af)
            else:
                ep = prior_ep
                af = prior_af

            if reverse:
                sar_i = prior_ep
                af = af0
                falling = True
                ep = low[i]
                reversal[i] = 1
                short[i] = sar_i  # Assign short signal
            else:
                falling = False
                long[i] = sar_i  # Assign long signal

            sar[i] = sar_i
            af_array[i] = af

        else:  # Falling trend
            sar_i = prior_sar + prior_af * (prior_ep - prior_sar)
            # Adjust SAR to not fall below the last two highs
            if i >= 2:
                sar_i = max(sar_i, high[i - 1], high[i - 2])
            else:
                sar_i = max(sar_i, high[i - 1])

            reverse = high[i] > sar_i
            if low[i] < prior_ep:
                ep = low[i]
                af = min(prior_af + af_increment, max_af)
            else:
                ep = prior_ep
                af = prior_af

            if reverse:
                sar_i = prior_ep
                af = af0
                falling = False
                ep = high[i]
                reversal[i] = 1
                long[i] = sar_i  # Assign long signal
            else:
                falling = True
                short[i] = sar_i  # Assign short signal

            sar[i] = sar_i
            af_array[i] = af

    return sar, long, short, af_array, reversal


def _falling(high, low, drift: int = 1):
    """Determines if the initial trend is falling based on -DM."""
    up = high - high.shift(drift)
    dn = low.shift(drift) - low
    _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
    return _dmn > 0
