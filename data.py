import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf

def get_data(stock, start=None, end=None):
    """
    自動判斷是上市股票(TWSE)、上櫃股票(TPEX)，還是美股(yfinance)
    """
    if stock.isdigit() or (stock[0].isdigit() and '.' not in stock):
        stock_code = stock.replace(".TW", "")
        
        # ✅ 先試試上市股票
        df = get_data_twse(stock_code, start, end)
        if not df.empty:
            return df
        
        # ✅ 再試試上櫃股票（使用新方法）
        df = get_data_tpex_via_twse_api(stock_code, start, end)
        if not df.empty:
            return df
        
        # ❌ 都找不到
        print(f"❌ {stock_code} 在上市、上櫃都找不到")
        return pd.DataFrame()
    else:
        return get_data_yfinance(stock, start, end)

def get_data_twse(stock_code, start=None, end=None):
    """
    台灣證交所（上市股票）
    """
    try:
        import time
        
        if start is None:
            start = "2020-01-01"
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        
        all_data = []
        current_date = start_date
        
        print(f"📥 從台證所(上市)下載 {stock_code}（{start} ~ {end}）...")
        
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
                    
                    if data.get("data"):
                        for row in data["data"]:
                            try:
                                all_data.append({
                                    "Date": row[0],
                                    "Open": float(row[3]),
                                    "High": float(row[4]),
                                    "Low": float(row[5]),
                                    "Close": float(row[6]),
                                    "Volume": int(row[1].replace(",", ""))
                                })
                            except (ValueError, IndexError):
                                continue
                
            except Exception as e:
                pass
            
            current_date += timedelta(days=1)
            time.sleep(0.1)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(inplace=True)
        
        print(f"✅ 成功下載 {len(df)} 筆 {stock_code} 數據")
        return df
        
    except Exception as e:
        print(f"❌ 台證所錯誤 {stock_code}：{str(e)}")
        return pd.DataFrame()

def get_data_tpex_via_twse_api(stock_code, start=None, end=None):
    """
    ✅ 使用 TWSE 即時行情 API 取得上櫃股票歷史資料
    API: https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=otc_XXXX.tw
    
    這個 API 同時支援上市和上櫃股票，更穩定！
    """
    try:
        import time
        
        if start is None:
            start = "2020-01-01"
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        
        all_data = []
        current_date = start_date
        
        print(f"📥 從TWSE API(上櫃)下載 {stock_code}（{start} ~ {end}）...")
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            
            try:
                # ✅ 使用 TWSE 即時行情 API（支援上櫃股票，前綴為 otc_）
                url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
                params = {
                    "ex_ch": f"otc_{stock_code}.tw",
                    "json": "1"
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.encoding = 'utf-8'
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # ✅ API 返回結構：msgArray
                        if data.get("msgArray"):
                            for row in data["msgArray"]:
                                try:
                                    # 檢查是否是我們要找的股票
                                    if row.get("c", "").strip() == stock_code:
                                        all_data.append({
                                            "Date": date_str,
                                            "Open": float(row.get("o", 0)),
                                            "High": float(row.get("h", 0)),
                                            "Low": float(row.get("l", 0)),
                                            "Close": float(row.get("z", 0)),
                                            "Volume": int(float(row.get("v", 0))) * 1000  # 轉為股數
                                        })
                                        break
                                except (ValueError, KeyError, TypeError):
                                    continue
                    except ValueError:
                        # JSON 解析失敗，跳過該日期
                        pass
                
            except Exception as e:
                pass
            
            current_date += timedelta(days=1)
            time.sleep(0.1)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(inplace=True)
        
        print(f"✅ 成功下載 {len(df)} 筆 {stock_code} 數據")
        return df
        
    except Exception as e:
        print(f"❌ TWSE API 錯誤 {stock_code}：{str(e)}")
        return pd.DataFrame()

def get_data_yfinance(stock, start=None, end=None):
    """使用 yfinance 獲取美股數據"""
    try:
        if stock.isdigit() or (stock[0].isdigit() and '.' not in stock):
            stock = f"{stock}.TW"
        
        print(f"📥 從 Yahoo Finance 下載 {stock}...")
        
        df = yf.download(stock, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"❌ yfinance 無 {stock} 的數據")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        
        df.columns = df.columns.str.upper()
        df.dropna(inplace=True)
        
        print(f"✅ 成功下載 {len(df)} 筆 {stock} 數據")
        
        return df
        
    except Exception as e:
        print(f"❌ yfinance 錯誤 {stock}：{str(e)}")
        return pd.DataFrame()
