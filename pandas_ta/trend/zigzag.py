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



@njit(cache=True)
def nb_rolling_hl(np_high, np_low, window_size):
    """
    Find high and low pivots.
    Using a centered rolling window.
    
    Args:
        np_high (1d np array) : High values
        np_low (1d np array) : Low values
        window_size (int) : Window size (will be made uneven to have equal size sides)

    Returns:
        1d np array : Detected pivot indices
        1d np array : Detected pivot directions
        1d np array : Detected pivot values
    """

    m = np_high.size
    idx = zeros(m)
    swing = zeros(m)  # where a high = 1 and low = -1
    value = zeros(m)

    extremes = 0
    left = int(floor(window_size / 2))
    right = left + 1
    # sample_array = [*[left-window], *[center], *[right-window]]
    for i in range(left, m - right):
        low_center = np_low[i]
        high_center = np_high[i]
        low_window = np_low[i - left: i + right]
        high_window = np_high[i - left: i + right]

        if (low_center <= low_window).all():
            idx[extremes] = i
            swing[extremes] = -1
            value[extremes] = low_center
            extremes += 1

        if (high_center >= high_window).all():
            idx[extremes] = i
            swing[extremes] = 1
            value[extremes] = high_center
            extremes += 1

    return idx[:extremes], swing[:extremes], value[:extremes]

@njit(cache=True)
def nb_find_zigzags_backtest(idx, swing, value, deviation):
    """
    Calculate zigzag points using pre-calculated unfiltered pivots.
    
    Args:
        idx (1d np array) : Pivot indices
        swing (1d np array) : Pivot swing direction -1 or 1
        value (1d np array) : Pivot values
        deviation (float) : Deviation percentage for reversal detection
        backtest_mode (bool) : Use backtest mode

    Returns:
        1d np array : Zigzag point indices on original data
        1d np array : Zigzag swing directions
        1d np array : Zigzag point values
        1d np array : Zigzag point deviation
    """

    zz_idx = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev = zeros_like(idx)

    zigzags = 0
    changes = 0
    zz_idx[zigzags] = idx[0]
    zz_swing[zigzags] = swing[0]
    zz_value[zigzags] = value[0]
    zz_dev[zigzags] = 0

    m = idx.size
    for i in range(m+1):
        last_zz_value = zz_value[zigzags - (1+changes)]
        current_dev = (value[i] - last_zz_value) / last_zz_value
        # Last point in zigzag is bottom
        if zz_swing[zigzags] == -1:
            if swing[i] == -1:
                # If the current pivot is lower than the last ZZ bottom:
                # create a new point and log it as a change
                if value[i] < zz_value[zigzags] and zigzags > 1:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    changes += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
            else:
                # If the deviation between pivot and the last ZZ bottom is great enough create new ZZ point.
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
                    changes = 0

        # last point in zigzag is top
        else:
            if swing[i] == 1:
                # If the current pivot is higher than the last ZZ top:
                # create a new point and log it as a change
                if value[i] > zz_value[zigzags] and zigzags > 1:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    changes += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev
            else:
                # If the deviation between pivot and the last ZZ top is great enough create new ZZ point.
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
                    changes = 0

    _n = zigzags + 1
    return zz_idx[:_n], zz_swing[:_n], zz_value[:_n], zz_dev[:_n]


@njit(cache=True)
def nb_find_zigzags(idx, swing, value, deviation):
    """
    Calculate zigzag points using pre-calculated unfiltered pivots.
    
    Args:
        idx (1d np array) : Pivot indices
        swing (1d np array) : Pivot swing direction -1 or 1
        value (1d np array) : Pivot values
        deviation (float) : Deviation percentage for reversal detection

    Returns:
        1d np array : Zigzag point indices on original data
        1d np array : Zigzag swing directions
        1d np array : Zigzag point values
        1d np array : Zigzag point deviation
    """

    zz_idx = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev = zeros_like(idx)

    zigzags = 0
    zz_idx[zigzags] = idx[-1]
    zz_swing[zigzags] = swing[-1]
    zz_value[zigzags] = value[-1]
    zz_dev[zigzags] = 0 

    m = idx.size
    for i in range(m - 2, -1, -1):
        # Next point in zigzag is bottom
        if zz_swing[zigzags] == -1:
            if swing[i] == -1:
                # If the current pivot is lower than the next ZZ bottom in time, move it to the pivot.
                # As this lower value invalidates the other one
                if value[i] < zz_value[zigzags] and zigzags > 1:
                    current_dev = (zz_value[zigzags - 1] - value[i]) / value[i]
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev
            else:
                # If the deviation between pivot and the next ZZ bottom is great enough create new ZZ point.
                current_dev = (value[i] - zz_value[zigzags]) / value[i]
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev

        # Next point in zigzag is top
        else:
            if swing[i] == 1:
                # If the current pivot is greater than the next ZZ top in time, move it to the pivot.
                # As this higher value invalidates the other one
                if value[i] > zz_value[zigzags] and zigzags > 1:
                    zigzags += 1
                    current_dev = (value[i] - zz_value[zigzags - 1]) / value[i]
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev
            else:
                # If the deviation between pivot and the next ZZ top is great enough create new ZZ point.
                current_dev = (zz_value[zigzags] - value[i]) / value[i]
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev

    _n = zigzags + 1
    return zz_idx[:_n], zz_swing[:_n], zz_value[:_n], zz_dev[:_n]



@njit(cache=True)
def nb_map_zigzag(idx, swing, value, deviation, n):
    """
    Maps nb_find_zigzag results back onto the original data indices.

    Args:
        idx (1d np array): indices from nb_find_zigzag
        swing (1d np array): swing directions from nb_find_zigzag
        value (1d np array): values from nb_find_zigzag
        deviation (1d np array): deviations from nb_find_zigzag
        n (int): Length of original high low data

    Returns:
        1d np array : swing map
        1d np array : value map
        1d np array : deviation map
    """    

    swing_map = zeros(n)
    value_map = zeros(n)
    dev_map = zeros(n)

    for j, i in enumerate(idx):
        i = int(i)
        swing_map[i] = swing[j]
        value_map[i] = value[j]
        dev_map[i] = deviation[j]

    for i in range(n):
        if swing_map[i] == 0:
            swing_map[i] = nan
            value_map[i] = nan
            dev_map[i] = nan

    return swing_map, value_map, dev_map



def zigzag(
    high: Series, low: Series, close: Series = None,
    legs: int = None, deviation: IntFloat = None,
    retrace: bool = None, last_extreme: bool = None,
    offset: Int = None, backtest_mode: bool = False,
    **kwargs: DictLike
):
    """ Zigzag (ZIGZAG)

    Zigzag attempts to filter out smaller price movments while highlighting
    trend direction. It does not predict future trends, but it does identify
    swing highs and lows. When 'deviation' is set to 10, it will ignore
    all price movements less than 10%; only price movements greater than 10%
    would be shown.

    Note: Zigzag lines are not permanent and a price reversal will create a
        new line.

    Sources:
        https://www.tradingview.com/support/solutions/43000591664-zig-zag/#:~:text=Definition,trader%20visual%20the%20price%20action.
        https://school.stockcharts.com/doku.php?id=technical_indicators:zigzag

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's. Default: None
        legs (int): Pivot detection window size.
            Pivots will be detected at the peak HL values in this window.
            These pivots will still be filtered by the deviation criteria.
            Minimum: 2. Default: 10
        deviation (float): Price deviation percentage for a reversal.
            Default: 5
        retrace (bool): Default: False **NOT IMPLEMENTED**
        last_extreme (bool): Default: True **NOT IMPLEMENTED**
        offset (int): How many periods to offset the result. Default: 0
        backtest_mode (bool) : Ensures the returned DF is safe for backtesting purposes.
            By default swing points are returned on the index of the pivot.
            Along with that intermediate swing values aren't returned at all.
            Backtest mode ensures swing detection are placed on the candle that they would have been detected.
            And changes in swing levels are included as well instead of only final values.
            To get the true index of a pivot use the following formula:
                p_i = i-int(floor(legs/2))

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method

    Returns:
        pd.DataFrame with columns:
            ZIGZAGs : Swing type (bottom: -1, top: 1)
            ZIGZAGv : Price levels of the swing points
            ZIGZAGd : Deviation from the last confirmed swing point
    """
    # Validate
    legs = v_pos_default(legs, 10)
    _length = legs + 1
    high = v_series(high, _length)
    low = v_series(low, _length)

    if high is None or low is None:
        return

    if close is not None:
        close = v_series(close,_length)
        np_close = close.values
        if close is None:
            return

    deviation = v_pos_default(deviation, 5.0)
    retrace = v_bool(retrace, False)
    last_extreme = v_bool(last_extreme, True)
    offset = v_offset(offset)
    backtest_mode = v_bool(backtest_mode, False)

    if backtest_mode:
        # Ensure signals are offset to their confirmation bar
        offset+=int(floor(legs/2))

    # Calculation
    np_high, np_low = high.to_numpy(), low.to_numpy()
    hli, hls, hlv = nb_rolling_hl(np_high, np_low, legs)
    zzi, zzs, zzv, zzd = nb_find_zigzags_backtest(hli, hls, hlv, deviation) if backtest_mode else nb_find_zigzags(hli, hls, hlv, deviation)
    zz_swing, zz_value, zz_dev = nb_map_zigzag(zzi, zzs, zzv, zzd, np_high.size)

    # Offset
    if offset != 0:
        zz_swing = zz_swing.shift(offset)
        zz_value = zz_value.shift(offset)
        zz_dev = zz_dev.shift(offset)

    # Fill
    if "fillna" in kwargs:
        zz_swing.fillna(kwargs["fillna"], inplace=True)
        zz_value.fillna(kwargs["fillna"], inplace=True)
        zz_dev.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        zz_swing.fillna(method=kwargs["fill_method"], inplace=True)
        zz_value.fillna(method=kwargs["fill_method"], inplace=True)
        zz_dev.fillna(method=kwargs["fill_method"], inplace=True)

    _props = f"_{deviation}%_{legs}"
    data = {
        f"ZIGZAGs{_props}": zz_swing,
        f"ZIGZAGv{_props}": zz_value,
        f"ZIGZAGd{_props}": zz_dev,
    }
    df = DataFrame(data, index=high.index)
    df.name = f"ZIGZAG{_props}"
    df.category = "trend"

    return df
