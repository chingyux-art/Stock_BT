import numpy as np
import pandas as pd

def ma_strategy(df, short=20, long=60):

    # ========= 防呆 =========
    if short >= long:
        raise ValueError("short MA 必須小於 long MA")

    df = df.copy()

    # ========= MA =========
    df["MA_short"] = df["Close"].rolling(int(short)).mean()
    df["MA_long"]  = df["Close"].rolling(int(long)).mean()

    # ========= 訊號 =========
    df["signal"] = 0

    # 黃金交叉
    cross_up = (
        (df["MA_short"] > df["MA_long"]) &
        (df["MA_short"].shift(1) <= df["MA_long"].shift(1))
    )

    #死亡交叉
    cross_down = (
        (df["MA_short"] < df["MA_long"]) &
        (df["MA_short"].shift(1) >= df["MA_long"].shift(1))
    )

    df.loc[cross_up, "signal"] = 1
    df.loc[cross_down, "signal"] = -1

    return df["signal"]

def rsi_strategy(df, period=14, buy=30, sell=70):

    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["signal"] = 0

    # ⭐ 進場（從上往下跌破 buy）
    buy_signal = (
        (df["RSI"] < buy) &
        (df["RSI"].shift(1) >= buy)
    )

    # ⭐ 出場（從下往上突破 sell）
    sell_signal = (
        (df["RSI"] > sell) &
        (df["RSI"].shift(1) <= sell)
    )

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1

    return df["signal"]

# 執行RSI 最佳化策略
def optimize_rsi(df, backtest_func, performance_func):

    results = []

    for period in range(10, 30, 2):
        for buy in range(20, 40, 5):
            for sell in range(60, 90, 5):

                if buy >= sell:
                    continue

                signal = rsi_strategy(df, period, buy, sell)

                trades, equity = backtest_func(df, signal)
                perf = performance_func(trades)

                if perf:
                    total, winrate, sharpe = perf

                    results.append({
                        "period": period,
                        "buy": buy,
                        "sell": sell,
                        "return": total,
                        "winrate": winrate,
                        "sharpe": sharpe
                    })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="return", ascending=False)

    return result_df

#=====Booling=====
def bb_strategy(df, n=20, k=2):

    df = df.copy()

    # ===== 布林通道 =====
    # n:週期（MA長度）, k: 標準差倍數
    df["MA"] = df["Close"].rolling(n).mean()
    df["STD"] = df["Close"].rolling(n).std()

    df["BB_H"] = df["MA"] + k * df["STD"]
    df["BB_L"] = df["MA"] - k * df["STD"]

    # ===== 訊號 =====
    signal = np.zeros(len(df))

    signal[df["Close"] < df["BB_L"]] = 1   # 買
    signal[df["Close"] > df["BB_H"]] = -1  # 賣

    return pd.Series(signal, index=df.index)

#====KD======

def kd_strategy(df, low=20, high=80):

    signal = np.zeros(len(df))

    # 進場：超賣 + K 上穿 D
    signal[(df["K"] < low) & (df["K"] > df["D"])] = 1

    # 出場：超買 + K 下穿 D
    signal[(df["K"] > high) & (df["K"] < df["D"])] = -1

    return pd.Series(signal, index=df.index)


#======= MA + RSI 組合策略 （進場：MA 多頭 + RSI 超賣反彈｜出場：MA 空頭 或 RSI 超買）
def ma_rsi_strategy(df, ma_short=20, ma_long=60, rsi_period=14, buy=30, sell=70):

    if ma_short >= ma_long:
        raise ValueError("ma_short 必須小於 ma_long")

    df = df.copy()

    # ========= MA =========
    df["MA_short"] = df["Close"].rolling(ma_short).mean()
    df["MA_long"]  = df["Close"].rolling(ma_long).mean()

    # ========= RSI =========
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ========= 訊號 =========
    df["signal"] = 0

    buy_signal = (
        (df["MA_short"] > df["MA_long"]) &
        (df["RSI"] < buy) &
        (df["RSI"].shift(1) >= buy)
    )

    sell_signal = (
        ((df["MA_short"] < df["MA_long"]) &
        (df["MA_short"].shift(1) >= df["MA_long"].shift(1))) |
        ((df["RSI"] > sell) &
        (df["RSI"].shift(1) <= sell))
    )

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1

    return df["signal"]

#====== MA + RSI「雙參數最佳化」
def optimize_ma_rsi(df, backtest_func, performance_func):

    results = []

    for ma_short in range(5, 30, 5):
        for ma_long in range(30, 100, 10):

            if ma_short >= ma_long:
                continue

            for rsi_period in range(10, 30, 5):
                for buy in range(20, 40, 5):
                    for sell in range(60, 90, 5):

                        if buy >= sell:
                            continue

                        signal = ma_rsi_strategy(
                            df,
                            ma_short,
                            ma_long,
                            rsi_period,
                            buy,
                            sell
                        )

                        trades, equity = backtest_func(df, signal)
                        perf = performance_func(trades)

                        # ========= 防呆 =========
                        if not perf or len(trades) < 5:
                            continue

                        total, winrate, sharpe = perf

                        # ========= 過濾爛策略 =========
                        if sharpe < 0:
                            continue

                        # ========= 評分系統（核心🔥）=========
                        score = (
                            total * 0.5 +
                            sharpe * 0.3 +
                            winrate * 0.2
                        )

                        results.append({
                            "ma_short": ma_short,
                            "ma_long": ma_long,
                            "rsi_period": rsi_period,
                            "buy": buy,
                            "sell": sell,
                            "return": total,
                            "winrate": winrate,
                            "sharpe": sharpe,
                            "trades": len(trades),
                            "score": score
                        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # ⭐ 用 score 排序（不是 return）
    result_df = result_df.sort_values(by="score", ascending=False)

    return result_df