import pandas as pd
import requests

def get_data(stock, start=None, end=None, data_source="yfinance"):
    """
    取得股票數據（支援多個數據源）
    
    Parameters:
    - stock: 股票代碼（6895 或 6895.TW）
    - start: 開始日期 (YYYY-MM-DD)
    - end: 結束日期 (YYYY-MM-DD)
    - data_source: "yfinance" 或 "finmind"
    """
    
    # 格式化股票代碼
    stock_code = stock.replace(".TW", "")
    
    if data_source == "finmind":
        return get_data_finmind(stock_code, start, end)
    else:
        return get_data_yfinance(stock, start, end)

def get_data_finmind(stock_code, start=None, end=None):
    """使用 FinMind API 獲取台股數據"""
    try:
        import finmind
        from finmind.data import DataLoader
        
        # 需要先註冊並取得 Token
        # 可在 https://finmind.github.io/ 申請
        loader = DataLoader()
        
        df = loader.taiwan_stock(
            stock_id=stock_code,
            start_date=start,
            end_date=end
        )
        
        if df.empty:
            print(f"❌ FinMind 無法下載 {stock_code}")
            return pd.DataFrame()
        
        # 標準化列名
        df.columns = df.columns.str.upper()
        return df
        
    except Exception as e:
        print(f"❌ FinMind 錯誤 {stock_code}：{str(e)}")
        return pd.DataFrame()

def get_data_yfinance(stock, start=None, end=None):
    """使用 yfinance 獲取數據"""
    import yfinance as yf
    
    # 自動格式化台股代碼
    if stock.isdigit() or (stock[0].isdigit() and not '.' in stock):
        stock = f"{stock}.TW"
    
    try:
        df = yf.download(stock, start=start, end=end, progress=False)
    except Exception as e:
        print(f"❌ yfinance 無法下載 {stock}：{str(e)}")
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    
    if df.empty:
        return df
    
    df.dropna(inplace=True)
    return df
