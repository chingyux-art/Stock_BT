import numpy as np
from ta.momentum import RSIIndicator,StochasticOscillator
from ta.volatility import BollingerBands

def add_indicators(df):

    close = df["Close"]
    high = df["High"]
    low = df["Low"]


    df["MA20"]=close.rolling(20).mean()
    df["MA60"]=close.rolling(60).mean()

    df["RSI"]=RSIIndicator(close,14).rsi()

    bb=BollingerBands(close,20)

    df["BB_H"]=bb.bollinger_hband()
    df["BB_L"]=bb.bollinger_lband()

    kd=StochasticOscillator(high,low,close)

    df["K"]=kd.stoch()
    df["D"]=kd.stoch_signal()

    return df

def compute_kd(df, n=9, k_period=3, d_period=3):

    df = df.copy()

    low_min = df["Low"].rolling(n).min()
    high_max = df["High"].rolling(n).max()

    # ⭐ 防止除以0
    denom = (high_max - low_min).replace(0, np.nan)

    rsv = (df["Close"] - low_min) / denom * 100

    # ⭐ RSV 補值（避免 NaN 傳染）
    rsv = rsv.fillna(50)

    df["K"] = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1/d_period, adjust=False).mean()

    return df
