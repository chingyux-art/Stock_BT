import yfinance as yf
import pandas as pd

def get_data(stock, start=None, end=None):
    """
    取得股票數據
    
    Parameters:
    - stock: 股票代碼（台股可用 2330 或 2330.TW，美股用 AAPL）
    - start: 開始日期 (YYYY-MM-DD)
    - end: 結束日期 (YYYY-MM-DD)
    """
    
    # ✅ 自動格式化台股代碼（補上 .TW）
    if stock.isdigit() or (stock[0].isdigit() and not '.' in stock):
        stock = f"{stock}.TW"
    
    # ✅ 嘗試下載數據
    try:
        df = yf.download(stock, start=start, end=end, progress=False)
    except Exception as e:
        print(f"❌ 無法下載 {stock}：{str(e)}")
        return pd.DataFrame()  # 回傳空 DataFrame
    
    # ✅ 處理 MultiIndex（有些ETF會出現）
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    
    # ✅ 防止空資料
    if df.empty:
        return df
    
    # ✅ 去除缺值
    df.dropna(inplace=True)
    
    return df
