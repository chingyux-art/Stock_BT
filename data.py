def get_data_twse(stock_code, start=None, end=None):
    """
    從台灣證交所下載歷史數據（最可靠）
    """
    import requests
    import pandas as pd
    from datetime import datetime
    
    try:
        # 台灣證交所公開資訊觀測站
        # 格式：YYYYMMDD
        if start:
            start_date = datetime.strptime(start, '%Y-%m-%d').strftime('%Y%m%d')
        else:
            start_date = '20200101'
        
        if end:
            end_date = datetime.strptime(end, '%Y-%m-%d').strftime('%Y%m%d')
        else:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 台證所 API
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        params = {
            'response': 'json',
            'date': start_date,
            'stockNo': stock_code
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"❌ 台證所無 {stock_code} 數據")
            return pd.DataFrame()
        
        data = response.json()
        
        if not data.get('data'):
            return pd.DataFrame()
        
        # 解析為 DataFrame
        df = pd.DataFrame(data['data'])
        # 自行處理列名與日期格式
        
        return df
        
    except Exception as e:
        print(f"❌ 台證所錯誤 {stock_code}：{str(e)}")
        return pd.DataFrame()
