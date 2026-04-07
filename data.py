import pandas as pd
import requests
from datetime import datetime
import yfinance as yf
import time

# ==============================
# ⭐ 主入口
# ==============================
def get_data(stock, start="2020-01-01", end=None):

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"\n🔍 取得資料：{stock}")

    # ==============================
    # 台股（純數字）
    # ==============================
    if stock.isdigit():

        # 1️⃣ 上市（TWSE）
        df = get_data_twse_monthly(stock, start, end)
        if not df.empty:
            print("✅ 使用 TWSE（上市）")
            return df

        # 2️⃣ 上櫃（TPEX）
        df = get_data_tpex_monthly(stock, start, end)
        if not df.empty:
            print("✅ 使用 TPEX（上櫃）")
            return df

        # 3️⃣ fallback → Yahoo
        print("⚠️ 改用 Yahoo Finance")
        df = get_data_yfinance_tw(stock, start, end)
        return df

    # ==============================
    # 非台股（美股等）
    # ==============================
    else:
        return get_data_yfinance(stock, start, end)


# ==============================
# ⭐ TWSE（月資料，快速）
# ==============================
def get_data_twse_monthly(stock_code, start, end):

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end, "%Y-%m-%d")

        all_data = []

        print(f"📥 TWSE（月）下載 {stock_code}...")

        current = start_dt.replace(day=1)

        while current <= end_dt:

            date_str = current.strftime("%Y%m01")

            url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
            params = {
                "response": "json",
                "date": date_str,
                "stockNo": stock_code
            }

            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                data = r.json()

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
                        except:
                            continue

            current = (current.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
            time.sleep(0.1)

        return clean_df(all_data)

    except Exception as e:
        print(f"❌ TWSE error: {e}")
        return pd.DataFrame()


# ==============================
# ⭐ TPEX（月資料，正確來源）
# ==============================
def get_data_tpex_monthly(stock_code, start, end):

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end, "%Y-%m-%d")

        all_data = []

        print(f"📥 TPEX（月）下載 {stock_code}...")

        current = start_dt.replace(day=1)

        while current <= end_dt:

            # 民國年
            year = current.year - 1911
            month = current.month

            date_str = f"{year}/{month:02d}"

            url = "https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php"
            params = {
                "l": "zh-tw",
                "d": date_str,
                "stkno": stock_code
            }

            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                data = r.json()

                if data.get("aaData"):
                    for row in data["aaData"]:
                        try:
                            all_data.append({
                                "Date": row[0],
                                "Open": float(row[4]),
                                "High": float(row[5]),
                                "Low": float(row[6]),
                                "Close": float(row[2]),
                                "Volume": int(row[1].replace(",", ""))
                            })
                        except:
                            continue

            current = (current.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
            time.sleep(0.1)

        return clean_df(all_data)

    except Exception as e:
        print(f"❌ TPEX error: {e}")
        return pd.DataFrame()


# ==============================
# ⭐ Yahoo（台股）
# ==============================
def get_data_yfinance_tw(stock, start, end):

    # 先試上市
    df = yf.download(f"{stock}.TW", start=start, end=end, progress=False)

    if df.empty:
        # 再試上櫃
        df = yf.download(f"{stock}.TWO", start=start, end=end, progress=False)

    return format_yf(df)


# ==============================
# ⭐ Yahoo（全球）
# ==============================
def get_data_yfinance(stock, start, end):

    print(f"📥 Yahoo 下載 {stock}...")

    df = yf.download(stock, start=start, end=end, progress=False)

    return format_yf(df)


# ==============================
# ⭐ 格式整理
# ==============================
def clean_df(data):

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # 日期轉換（兼容民國/西元）
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
    except:
        df["Date"] = pd.to_datetime(df["Date"])

    df.set_index("Date", inplace=True)
    df = df.sort_index()
    df = df[~df.index.duplicated()]

    df.dropna(inplace=True)

    print(f"✅ 共 {len(df)} 筆")

    return df


def format_yf(df):

    if df.empty:
        print("❌ Yahoo 無資料")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df.columns = df.columns.str.capitalize()

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    print(f"✅ Yahoo {len(df)} 筆")

    return df
