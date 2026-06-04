import numpy as np
import pandas as pd


# ==============================
# 條件類型定義 - 簡化版
# ==============================
CONDITION_CATEGORIES = {
    "KD": {
        "display_name": "KD 隨機指標",
        "signals": {
            "KD黃金交叉": "K 上穿 D（黃金交叉）",
            "KD死亡交叉": "K 下穿 D（死亡交叉）",
            "KD_K值大於": "K 值 > X",
            "KD_K值小於": "K 值 < X",
        }
    },
    "MACD": {
        "display_name": "MACD 移動平均收斂散度",
        "signals": {
            "MACD由零翻正": "MACD 由零翻正",
            "MACD黃金交叉": "MACD 上穿 Signal（黃金交叉）",
            "MACD死亡交叉": "MACD 下穿 Signal（死亡交叉）",
        }
    },
    "收盤價": {
        "display_name": "收盤價 vs MA",
        "signals": {
            "收盤大於MA": "收盤價 > MA",
            "收盤小於MA": "收盤價 < MA",
        }
    },
    "RSI": {
        "display_name": "RSI 相對強弱指數",
        "signals": {
            "RSI超賣反彈": "RSI < X（超賣反彈）",
            "RSI超買回吐": "RSI > X（超買回吐）",
        }
    },
    "Volume": {
        "display_name": "成交量",
        "signals": {
            "交易量大於": "Volume > X（張數）",
            "交易量小於": "Volume < X（張數）",
            "交易量倍數>前一日": "Volume > 前一日 × X 倍",
            "交易量倍數<前一日": "Volume < 前一日 × X 倍",
            "交易量倍數>前三天": "Volume > 前三天平均 × X 倍",
            "交易量倍數<前三天": "Volume < 前三天平均 × X 倍",
        }
    },
    "布林通道": {
        "display_name": "布林通道",
        "signals": {
            "布林_大於下軌": "收盤價 > 下軌",
            "布林_小於下軌": "收盤價 < 下軌",
            "布林_大於中軌": "收盤價 > 中軌",
            "布林_小於中軌": "收盤價 < 中軌",
            "布林_大於上軌": "收盤價 > 上軌",
            "布林_小於上軌": "收盤價 < 上軌",
        }
    },
}

# 平坦化的條件類型（向後相容）
CONDITION_TYPES = {}
for category, info in CONDITION_CATEGORIES.items():
    for signal_key, signal_name in info["signals"].items():
        CONDITION_TYPES[signal_key] = signal_name

# 每種條件需要的參數及預設值
CONDITION_PARAMS = {
    "KD黃金交叉":   {},
    "KD死亡交叉":   {},
    "KD_K值大於":   {"threshold": 50},
    "KD_K值小於":   {"threshold": 50},
    "MACD由零翻正": {"fast": 12, "slow": 26, "sig": 9},
    "MACD黃金交叉": {"fast": 12, "slow": 26, "sig": 9},
    "MACD死亡交叉": {"fast": 12, "slow": 26, "sig": 9},
    "收盤大於MA":   {"period": 20},
    "收盤小於MA":   {"period": 20},
    "RSI超賣反彈":  {"period": 14, "buy": 30},
    "RSI超買回吐":  {"period": 14, "sell": 70},
    "交易量大於":        {"threshold": 10},
    "交易量小於":        {"threshold": 10},
    "交易量倍數>前一日":  {"multiple": 2.0},
    "交易量倍數<前一日":  {"multiple": 2.0},
    "交易量倍數>前三天": {"multiple": 2.0},
    "交易量倍數<前三天": {"multiple": 2.0},
    "布林_大於下軌": {"n": 20, "k": 2.0},
    "布林_小於下軌": {"n": 20, "k": 2.0},
    "布林_大於中軌": {"n": 20, "k": 2.0},
    "布林_小於中軌": {"n": 20, "k": 2.0},
    "布林_大於上軌": {"n": 20, "k": 2.0},
    "布林_小於上軌": {"n": 20, "k": 2.0},
    "停損_虧損10%": {},
}


def check_condition(df: pd.DataFrame, condition_type: str, **params) -> pd.Series:
    """
    根據條件類型和參數，計算每個時間點是否觸發該條件。

    Returns:
        pd.Series (bool)，True 表示該時間點觸發條件。
    """
    df = df.copy()
    false_series = pd.Series(False, index=df.index)

    if condition_type == "KD黃金交叉":
        if "K" not in df.columns or "D" not in df.columns:
            return false_series
        return (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))

    elif condition_type == "KD死亡交叉":
        if "K" not in df.columns or "D" not in df.columns:
            return false_series
        return (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))

    elif condition_type == "KD_K值大於":
        if "K" not in df.columns:
            return false_series
        threshold = float(params.get("threshold", 50))
        return df["K"] > threshold

    elif condition_type == "KD_K值小於":
        if "K" not in df.columns:
            return false_series
        threshold = float(params.get("threshold", 50))
        return df["K"] < threshold

    elif condition_type in ("MACD由零翻正", "MACD黃金交叉", "MACD死亡交叉"):
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        sig  = int(params.get("sig",  9))

        close = df["Close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=sig, adjust=False).mean()

        if condition_type == "MACD由零翻正":
            return (macd > 0) & (macd.shift(1) <= 0)
        elif condition_type == "MACD黃金交叉":
            return (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        else:  # MACD死亡交叉
            return (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

    elif condition_type in ("收盤大於MA", "收盤小於MA"):
        period = int(params.get("period", 20))
        ma = df["Close"].rolling(period).mean()
        if condition_type == "收盤大於MA":
            return df["Close"] > ma
        else:
            return df["Close"] < ma

    elif condition_type == "RSI超賣反彈":
        period = int(params.get("period", 14))
        buy    = float(params.get("buy", 30))
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        return (rsi < buy) & (rsi.shift(1) >= buy)

    elif condition_type == "RSI超買回吐":
        period = int(params.get("period", 14))
        sell   = float(params.get("sell", 70))
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        return (rsi > sell) & (rsi.shift(1) <= sell)

    elif condition_type == "交易量大於":
        if "Volume" not in df.columns:
            return false_series
        threshold = float(params.get("threshold", 10)) * 1000
        return df["Volume"] > threshold

    elif condition_type == "交易量小於":
        if "Volume" not in df.columns:
            return false_series
        threshold = float(params.get("threshold", 10)) * 1000
        return df["Volume"] < threshold

    elif condition_type in ("交易量倍數>前一日", "交易量倍數<前一日"):
        if "Volume" not in df.columns:
            return false_series
        multiple = float(params.get("multiple", 2.0))
        prev_vol = df["Volume"].shift(1)
        if condition_type == "交易量倍數>前一日":
            return df["Volume"] > prev_vol * multiple
        else:
            return df["Volume"] < prev_vol * multiple

    elif condition_type in ("交易量倍數>前三天", "交易量倍數<前三天"):
        if "Volume" not in df.columns:
            return false_series
        multiple = float(params.get("multiple", 2.0))
        # 計算前三天的平均成交量
        three_day_avg = df["Volume"].rolling(3).mean()
        if condition_type == "交易量倍數>前三天":
            return df["Volume"] > three_day_avg * multiple
        else:
            return df["Volume"] < three_day_avg * multiple

    elif condition_type.startswith("布林_"):
        if "Close" not in df.columns:
            return false_series
        n = int(params.get("n", 20))
        k = float(params.get("k", 2.0))
        
        bb_ma   = df["Close"].rolling(n).mean()
        bb_std  = df["Close"].rolling(n).std()
        bb_up   = bb_ma + k * bb_std
        bb_dn   = bb_ma - k * bb_std
        
        if condition_type == "布林_大於下軌":
            return df["Close"] > bb_dn
        elif condition_type == "布林_小於下軌":
            return df["Close"] < bb_dn
        elif condition_type == "布林_大於中軌":
            return df["Close"] > bb_ma
        elif condition_type == "布林_小於中軌":
            return df["Close"] < bb_ma
        elif condition_type == "布林_大於上軌":
            return df["Close"] > bb_up
        elif condition_type == "布林_小於上軌":
            return df["Close"] < bb_up

    elif condition_type == "停損_虧損10%":
        # 這個條件在 backtest 中特殊處理
        return false_series

    return false_series


def combine_signals(signals: list, logic_mode: str = "AND", min_count: int = 1) -> pd.Series:
    """
    將多個布林訊號 Series 組合成最終進/出場訊號（0 或 1）。

    Args:
        signals:    list of pd.Series (bool)，不可為空
        logic_mode: "AND" 全部符合 | "OR" 至少 N 項符合
        min_count:  logic_mode="OR" 時最少需幾項符合（min_count >= 1）
                    在 AND 模式下此參數無效。

    Returns:
        pd.Series，1 表示觸發，0 表示不觸發。

    Raises:
        ValueError: 若 signals 為空列表。
    """
    if not signals:
        raise ValueError("signals 不可為空列表")

    combined = pd.concat(signals, axis=1).fillna(False)

    if logic_mode == "AND":
        result = combined.all(axis=1)
    else:  # OR / 至少 N 項
        result = combined.sum(axis=1) >= min_count

    return result.astype(int)
