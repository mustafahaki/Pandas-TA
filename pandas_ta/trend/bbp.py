
from pandas_ta.overlap import ema
from pandas import DataFrame
from pandas_ta.utils import verify_series

def bbp(high , low , close , length=None , **kwargs):
     """Indicator: BUll Bear Power (BBP)"""
     # Validate Arguments
     length = int(length) if length and length > 0 else 13
     high = verify_series(high , length)
     low = verify_series(low , length)
     close = verify_series(close , length)
     if high is None or low is None or close is None: return
     ema = ema(close , length)
     Bull_Power = high - ema
     Bear_Power = low - ema
     BBP = Bull_Power + Bear_Power
      # Handle fills
     if "fillna" in kwargs:
        BBP.fillna(kwargs["fillna"], inplace=True)
     if "fill_method" in kwargs:
        BBP.fillna(method=kwargs["fill_method"], inplace=True)
     BBP.name = f"BBP_{length}"
     BBP.category = "trend"
     return BBP
bbp.__doc__ = \
"""Bull Bear Power (BBP)

The Bull Bear Power (BBP) indicator, otherwise known as the Elder-Ray Index, 
estimates the relationship between the strength of bulls and bears on an instrument.
Sources:
    https://www.tradingview.com/wiki/Commodity_Channel_Index_(CCI)

Calculation:
    Default Inputs:
    length=13,
     ema = ema(close , length)
     Bull_Power = high - ema
     Bear_Power = low - ema
    BBP = Bull_Power + Bear_Power

 

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 13
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


    
