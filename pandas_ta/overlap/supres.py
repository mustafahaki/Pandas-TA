# -*- coding: utf-8 -*-
from numpy import nan
from pandas import DataFrame, Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.overlap import hma
from pandas_ta.momentum import rsi
from pandas_ta.utils import v_offset, v_series



def supres(open_: Series, high: Series, low: Series,
    close: Series, offset: Int = None,
    suffix: str = None, **kwargs: DictLike
) -> DataFrame:
    # Validate and prepare data
    length = len(close)
    open_, high, low, close = map(lambda s: v_series(s, length),
                                  [open_, high, low, close])

    # Compute basic indicators
    src1 = hma(open_, length=5, offset=1)
    src2 = hma(close, length=12)
    momm1, momm2 = src1.diff(), src2.diff()
    rsi_new = rsi(close, length=9)
    sh = high.rolling(2).max().rolling(2).mean()
    sl = low.rolling(2).min().rolling(2).mean()

    # Calculate momentum and CMO
    m1 = momm1 >= momm2
    m2 = -1 * momm1.where(momm1 < momm2, 0)
    sm1, sm2 = m1.rolling(1).sum(), m2.rolling(1).sum()
    CMO = 100 * (sm1 - sm2) / (sm1 + sm2).replace(0, nan)

    # Calculate support and resistance
    h1 = (high - sh).abs().expanding().mean()
    l1 = (low - sl).abs().expanding().mean()
    hpivot, lpivot = h1 != 0, l1 != 0
    hpivot_new, lpivot_new = high.where(hpivot), low.where(lpivot)

    RESISTANCE = hpivot_new.where((rsi_new > 75) & (CMO < -50))
    SUPPORT = lpivot_new.where((rsi_new < 25) & (CMO > 50))

    # Prepare DataFrame to return
    props = f"_{length}" if suffix == "length" else ""
    df = DataFrame({f"SUPPORT{props}": SUPPORT, f"RESISTANCE{props}": RESISTANCE},
                   index=close.index)
    df.name = f"SUPPORT{props}"
    df.category = "overlap"
    df.ffill(inplace=True)

    # Apply offset if needed
    return df.shift(v_offset(offset))
