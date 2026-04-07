def run_backtest(df, signal):

    import numpy as np
    import pandas as pd

    # ✅ 自動轉型（關鍵修正）
    if isinstance(signal, np.ndarray):
        signal = pd.Series(signal, index=df.index)

    position = 0
    buy_price = 0

    equity = [1]
    trades = []

    for i in range(1, len(df)):

        price = df["Close"].iloc[i]

        # 進場
        if signal.iloc[i] == 1 and position == 0:
            position = 1
            buy_price = price

        # 出場
        elif signal.iloc[i] == -1 and position == 1:
            position = 0

            ret = (price - buy_price) / buy_price
            trades.append({
                "return": ret,
                "buy_price": buy_price,
                "sell_price": price
            })

        # 每日 equity
        if position == 1:
            ret = (price - df["Close"].iloc[i-1]) / df["Close"].iloc[i-1]
            equity.append(equity[-1] * (1 + ret))
        else:
            equity.append(equity[-1])

    # 強制平倉
    if position == 1:
        price = df["Close"].iloc[-1]
        ret = (price - buy_price) / buy_price
        trades.append({
            "return": ret,
            "buy_price": buy_price,
            "sell_price": price
        })

    return trades, equity

    



import numpy as np
import pandas as pd

def performance(trades):

    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)

    # ===== 報酬 =====
    returns = df["return"]

    total_return = (1 + returns).prod() - 1

    # ===== 勝率 =====
    winrate = (returns > 0).mean()

    # ===== Sharpe =====
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    return total_return, winrate, sharpe
