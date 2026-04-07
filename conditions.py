import numpy as np
import pandas as pd


# ==============================
# 條件類型定義
# ==============================
CONDITION_TYPES = {
    "KD黃金交叉":    "KD 黃金交叉 (K 上穿 D)",
    "KD死亡交叉":    "KD 死亡交叉 (K 下穿 D)",
    "MACD由零翻正":  "MACD 由零翻正",
    "MACD黃金交叉":  "MACD 黃金交叉 (MACD 上穿 Signal)",
    "MACD死亡交叉":  "MACD 死亡交叉 (MACD 下穿 Signal)",
    "收盤大於MA":    "收盤 > MA (指定週期)",
    "收盤小於MA":    "收盤 < MA (指定週期)",
    "RSI超賣反彈":   "RSI 超賣反彈 (RSI 跌破買進門檻)",
    "RSI超買回吐":   "RSI 超買回吐 (RSI 突破賣出門檻)",
}

# 每種條件需要的參數及預設值
CONDITION_PARAMS = {
    "KD黃金交叉":   {},
    "KD死亡交叉":   {},
    "MACD由零翻正": {"fast": 12, "slow": 26, "sig": 9},
    "MACD黃金交叉": {"fast": 12, "slow": 26, "sig": 9},
    "MACD死亡交叉": {"fast": 12, "slow": 26, "sig": 9},
    "收盤大於MA":   {"period": 20},
    "收盤小於MA":   {"period": 20},
    "RSI超賣反彈":  {"period": 14, "buy": 30},
    "RSI超買回吐":  {"period": 14, "sell": 70},
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
