import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf

def get_data(stock, start=None, end=None):
    """
    取得股票數據（台股用台證所 API，美股用 yfinance）
    
    Parameters:
    - stock: 股票代碼（台股可用 2330 或 2330.TW，美股用 AAPL）
    - start: 開始日期 (YYYY-MM-DD)
    - end: 結束日期 (YYYY-MM-DD)
    """
    
    # ✅ 自動判斷是台股還是美股
    if stock.isdigit() or (stock[0].isdigit() and '.' not in stock):
        # 台股：純數字或數字開頭
        stock_code = stock.replace(".TW", "")
        return get_data_twse(stock_code, start, end)
    else:
        # 美股：字母開頭
        return get_data_yfinance(stock, start, end)

def get_data_twse(stock_code, start=None, end=None):
    """
    使用台灣證交所官方 API 獲取台股歷史數據
    API 說明：https://www.twse.com.tw/en/page/information/about/web-api.html
    """
    try:
        import time
        
        # ✅ 預設日期
        if start is None:
            start = "2020-01-01"
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        # ✅ 轉換為日期物件
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        
        all_data = []
        current_date = start_date
        
        print(f"📥 從台證所下載 {stock_code}（{start} ~ {end}）...")
        
        # ✅ 逐日請求（台證所 API 一次只回傳一天）
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            
            try:
                url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
                params = {
                    "response": "json",
                    "date": date_str,
                    "stockNo": stock_code
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # ✅ 檢查是否有資料
                    if data.get("data"):
                        # 解析每一筆資料
                        for row in data["data"]:
                            try:
                                all_data.append({
                                    "Date": row[0],  # 日期 (例: "20240107")
                                    "Open": float(row[3]),   # 開盤
                                    "High": float(row[4]),   # 最高
                                    "Low": float(row[5]),    # 最低
                                    "Close": float(row[6]),  # 收盤
                                    "Volume": int(row[1].replace(",", ""))  # 成交量
                                })
                            except (ValueError, IndexError):
                                continue
                
            except Exception as e:
                # 跳過這一天，繼續下一天
                pass
            
            # ✅ 移到下一天
            current_date += timedelta(days=1)
            
            # ⏱️ 避免被 API 限流
            time.sleep(0.1)
        
        if not all_data:
            print(f"❌ 台證所無 {stock_code} 的數據")
            return pd.DataFrame()
        
        # ✅ 轉換為 DataFrame
        df = pd.DataFrame(all_data)
        
        # ✅ 轉換日期格式
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        
        # ✅ 設置日期為索引
        df.set_index("Date", inplace=True)
        
        # ✅ 排序
        df = df.sort_index()
        
        # ✅ 去除重複
        df = df[~df.index.duplicated(keep='first')]
        
        # ✅ 去除缺值
        df.dropna(inplace=True)
        
        print(f"✅ 成功下載 {len(df)} 筆 {stock_code} 數據")
        
        return df
        
    except Exception as e:
        print(f"❌ 台證所錯誤 {stock_code}：{str(e)}")
        return pd.DataFrame()

def get_data_yfinance(stock, start=None, end=None):
    """
    使用 yfinance 獲取美股數據（備用方案）
    """
    try:
        # ✅ 自動補上 .TW（以防萬一）
        if stock.isdigit() or (stock[0].isdigit() and '.' not in stock):
            stock = f"{stock}.TW"
        
        print(f"📥 從 Yahoo Finance 下載 {stock}...")
        
        df = yf.download(stock, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"❌ yfinance 無 {stock} 的數據")
            return pd.DataFrame()
        
        # ✅ 處理 MultiIndex（有些 ETF 會出現）
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        
        # ✅ 標準化列名
        df.columns = df.columns.str.upper()
        
        # ✅ 去除缺值
        df.dropna(inplace=True)
        
        print(f"✅ 成功下載 {len(df)} 筆 {stock} 數據")
        
        return df
        
    except Exception as e:
        print(f"❌ yfinance 錯誤 {stock}：{str(e)}")
        return pd.DataFrame()
