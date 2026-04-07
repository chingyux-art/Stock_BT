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
    result_df = result_df.sort

