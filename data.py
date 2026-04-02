import yfinance as yf
import pandas as pd

def get_data(stock, start=None, end=None):

    df = yf.download(stock, start=start, end=end)

    # ✅ 處理 MultiIndex（有些ETF會出現）
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    # ✅ 防止空資料
    if df.empty:
        return df

    # ✅ 去除缺值
    df.dropna(inplace=True)

    return df