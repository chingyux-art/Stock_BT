import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import date, datetime

from data import get_data
from indicators import add_indicators
from indicators import compute_kd, add_macd
from strategies import *
from backtest import *
from conditions import (
    CONDITION_TYPES,
    CONDITION_PARAMS,
    check_condition,
    combine_signals,
)

st.title("Stock Backtest Application")

# ==============================
# 股票
# ==============================
stock = st.text_input("Stock", "2330")
st.caption("台股: 2330 | 美股: AAPL")

# ==============================
# 日期
# ==============================
start_date = st.date_input("開始日期", date(2020, 1, 1))
end_date   = st.date_input("結束日期", date(2025, 1, 1))
st.caption("時間格式(yyyy, mm, dd)")

# ==============================
# 策略選擇
# ==============================
selected = st.multiselect(
    "策略組合",
    ["MA", "RSI", "KD", "Bollinger", "MACD"],
    default=["MA"]
)

# ==============================
# 參數 UI（⭐關鍵）
# ==============================
params_dict = {}

if "MA" in selected:
    params_dict["MA"] = {
        "short": st.slider("MA short", 5, 50, 20),
        "long": st.slider("MA long", 20, 200, 60)
    }

if "RSI" in selected:
    params_dict["RSI"] = {
        "period": st.slider("RSI period", 5, 30, 14),
        "buy": st.slider("RSI buy", 10, 40, 30),
        "sell": st.slider("RSI sell", 60, 90, 70)
    }

if "Bollinger" in selected:
    params_dict["Bollinger"] = {
        "n": st.slider("BB n", 10, 60, 20),
        "k": st.slider("BB k", 1.0, 3.5, 2.0)
    }

if "KD" in selected:
    params_dict["KD"] = {
        "n": st.slider("KD n", 5, 20, 9),
        "k_period": st.slider("KD K period", 1, 10, 3),
        "d_period": st.slider("KD D period", 1, 10, 3),
        "low": st.slider("KD low", 0, 50, 20),
        "high": st.slider("KD high", 50, 100, 80)
    }

if "MACD" in selected:
    params_dict["MACD"] = {
        "fast_period": st.slider("MACD 快速期", 5, 20, 12),
        "slow_period": st.slider("MACD 慢速期", 20, 50, 26),
        "signal_period": st.slider("MACD Signal 期", 3, 15, 9)
    }

# ==============================
# 進出場條件設定
# ==============================
def render_condition_ui(prefix: str, label: str):
    """
    渲染一組「最多 5 個條件 + 邏輯模式」的 UI，
    返回 (conditions_cfg, logic_mode, min_count)。
    conditions_cfg = list of (condition_type, params_dict)
    """
    st.subheader(label)

    condition_names = list(CONDITION_TYPES.keys())
    display_names   = ["(不設定)"] + [CONDITION_TYPES[k] for k in condition_names]

    logic_mode_label = st.radio(
        f"{label} 邏輯模式",
        ["全部符合 (AND)", "至少 N 項符合 (OR)"],
        key=f"{prefix}_logic_mode",
        horizontal=True,
    )
    logic_mode = "AND" if "AND" in logic_mode_label else "OR"

    min_count = 1
    if logic_mode == "OR":
        min_count = st.number_input(
            "最少符合幾項",
            min_value=1,
            max_value=5,
            value=1,
            key=f"{prefix}_min_count",
        )

    conditions_cfg = []
    for i in range(1, 6):
        col_sel, col_params = st.columns([2, 3])
        with col_sel:
            chosen_display = st.selectbox(
                f"條件 {i}",
                display_names,
                key=f"{prefix}_cond_{i}",
            )

        if chosen_display == "(不設定)":
            continue

        # 反查 key
        cond_key = condition_names[display_names.index(chosen_display) - 1]
        param_defaults = CONDITION_PARAMS.get(cond_key, {})
        param_values = {}

        with col_params:
            if "fast" in param_defaults:
                param_values["fast"] = st.slider(
                    f"快速期 (條件{i})", 5, 20,
                    int(param_defaults["fast"]),
                    key=f"{prefix}_cond_{i}_fast",
                )
            if "slow" in param_defaults:
                param_values["slow"] = st.slider(
                    f"慢速期 (條件{i})", 10, 60,
                    int(param_defaults["slow"]),
                    key=f"{prefix}_cond_{i}_slow",
                )
            if "sig" in param_defaults:
                param_values["sig"] = st.slider(
                    f"Signal 期 (條件{i})", 3, 15,
                    int(param_defaults["sig"]),
                    key=f"{prefix}_cond_{i}_sig",
                )
            if "period" in param_defaults:
                param_values["period"] = st.slider(
                    f"週期 (條件{i})", 5, 120,
                    int(param_defaults["period"]),
                    key=f"{prefix}_cond_{i}_period",
                )
            if "buy" in param_defaults:
                param_values["buy"] = st.slider(
                    f"買進門檻 (條件{i})", 10, 50,
                    int(param_defaults["buy"]),
                    key=f"{prefix}_cond_{i}_buy",
                )
            if "sell" in param_defaults:
                param_values["sell"] = st.slider(
                    f"賣出門檻 (條件{i})", 50, 90,
                    int(param_defaults["sell"]),
                    key=f"{prefix}_cond_{i}_sell",
                )
            if "threshold" in param_defaults:
                param_values["threshold"] = st.slider(
                    f"交易量門檻 (張數) (條件{i})", 1, 10000,
                    int(param_defaults["threshold"]),
                    key=f"{prefix}_cond_{i}_threshold",
                )
            if "multiple" in param_defaults:
                param_values["multiple"] = st.slider(
                    f"倍數 (條件{i})", 1.0, 5.0,
                    float(param_defaults["multiple"]),
                    step=0.1,
                    key=f"{prefix}_cond_{i}_multiple",
                )

        conditions_cfg.append((cond_key, param_values))

    return conditions_cfg, logic_mode, int(min_count)


with st.expander("⚙️ 自訂進場條件設定（最多 5 個）", expanded=False):
    entry_conditions, entry_logic, entry_min = render_condition_ui("entry", "進場條件")

with st.expander("⚙️ 自訂出場條件設定（最多 5 個）", expanded=False):
    exit_conditions, exit_logic, exit_min = render_condition_ui("exit", "出場條件")

# ==============================
# 最佳化開關（要提前）
# ==============================
optimize = st.checkbox("啟用最佳化")

# ==============================
# 最佳化參數設定
# ==============================
def get_param_grid(strategy):

    if strategy == "MA":
        return [
            {"short": s, "long": l}
            for s in range(5, 50, 5)
            for l in range(20, 200, 10)
            if s < l
        ]

    elif strategy == "RSI":
        return [
            {"period": p, "buy": b, "sell": s}
            for p in range(10, 30, 2)
            for b in range(20, 40, 5)
            for s in range(60, 90, 5)
            if b < s
        ]

    elif strategy == "Bollinger":
        return [
            {"n": n, "k": k}
            for n in range(10, 50, 5)
            for k in np.arange(1.5, 3.5, 0.5)
        ]

    elif strategy == "KD":
        return [
            {
                "n": n,
                "k_period": k,
                "d_period": d,
                "low": low,
                "high": high
            }
            for n in range(5, 20, 2)
            for k in range(2, 6)
            for d in range(2, 6)
            for low in range(10, 40, 5)
            for high in range(60, 90, 5)
        ]

    elif strategy == "MACD":
        return [
            {"fast_period": f, "slow_period": s, "signal_period": sig}
            for f in range(5, 20, 2)
            for s in range(20, 50, 5)
            for sig in range(3, 15, 2)
            if f < s
        ]

# ==============================
# 統一策略執行器
# ==============================
def run_strategy(df, strategy, p):

    if strategy == "MA":
        return ma_strategy(df, p["short"], p["long"])

    elif strategy == "RSI":
        return rsi_strategy(df, p["period"], p["buy"], p["sell"])

    elif strategy == "Bollinger":
        return bb_strategy(df, p["n"], p["k"])

    elif strategy == "KD":
        df_kd = compute_kd(
            df.copy(),
            n=p["n"],
            k_period=p["k_period"],
            d_period=p["d_period"]
        )
        return kd_strategy(df_kd, p["low"], p["high"])

    elif strategy == "MACD":
        return macd_strategy(df, p["fast_period"], p["slow_period"], p["signal_period"])

# ==============================
# 統一策略執行器
# ==============================
def optimize_strategy(df, strategy):

    grid = get_param_grid(strategy)

    best_score = -999
    best_params = None
    results = []

    for i, p in enumerate(grid):

        signal = run_strategy(df, strategy, p)

        trades, equity = run_backtest(df, signal)
        perf = performance(trades)

        if not perf:
            continue

        if len(trades) < 2:
            continue

        total, winrate, sharpe = perf

        score = total * 0.5 + sharpe * 0.3 + winrate * 0.2

        results.append({**p, "score": score})

        if score > best_score:
            best_score = score
            best_params = p

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return None, result_df

    return best_params, result_df.sort_values("score", ascending=False)

# ==============================
# 執行
# ==============================
if st.button("Run Backtest"):

    # ---------- 防呆 ----------
    if start_date >= end_date:
        st.error("開始日期必須早於結束日期")
        st.stop()

    if not stock:
        st.error("請輸入股票")
        st.stop()

    if end_date > datetime.today().date():
        st.error("結束日期不能超過今天")
        st.stop()

    # ---------- 抓資料 ----------
    df = get_data(stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df.empty:
        st.error("抓不到資料")
        st.stop()

    df = add_indicators(df)

    # ==============================
    # ⭐ 訊號產生（正確位置🔥）
    # ==============================
    signals = []

    if optimize:

        best_params_dict = {}

        opt_results = {}

        for s in selected:
            best_params, result_df = optimize_strategy(df, s)

            if best_params is None:
                st.warning(f"{s} 無最佳參數")
                continue

            best_params_dict[s] = best_params
            opt_results[s] = result_df   # ⭐ 存下來

            if best_params is None:
                st.warning(f"{s} 無最佳參數")
                continue

            best_params_dict[s] = best_params

        for s in best_params_dict:
            signals.append(run_strategy(df, s, best_params_dict[s]))

    else:
        for s in selected:
            signals.append(run_strategy(df, s, params_dict[s]))

    # ===== 合併訊號 =====
    # ---- 自訂條件訊號 ----
    if entry_conditions or exit_conditions:
        # 為 KD 條件補齊 K/D 欄位
        df_cond = df.copy()
        if any(c in ("KD黃金交叉", "KD死亡交叉") for c, _ in (entry_conditions + exit_conditions)):
            df_cond = compute_kd(df_cond)

        custom_signal = pd.Series(0, index=df_cond.index)

        if entry_conditions:
            entry_bool_list = [
                check_condition(df_cond, ctype, **cparams)
                for ctype, cparams in entry_conditions
            ]
            entry_trigger = combine_signals(entry_bool_list, entry_logic, entry_min)
            custom_signal[entry_trigger == 1] = 1

        if exit_conditions:
            exit_bool_list = [
                check_condition(df_cond, ctype, **cparams)
                for ctype, cparams in exit_conditions
            ]
            exit_trigger = combine_signals(exit_bool_list, exit_logic, exit_min)
            # 出場訊號優先：若同一時間點同時觸發進/出場，以出場（-1）為準
            custom_signal[exit_trigger == 1] = -1

        signals.append(custom_signal)

    if not signals:
        st.error("沒有產生任何策略訊號")
        st.stop()

    signal = sum(signals)
    signal = signal.clip(-1, 1)

    # ==============================
    # 回測
    # ==============================
    trades, equity = run_backtest(df, signal)
    perf = performance(trades)

    # ==============================
    # 顯示績效
    # ==============================
    if perf:
        total, winrate, sharpe = perf

        st.subheader("📊 回測績效")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("總報酬", f"{total*100:.2f}%")
        col2.metric("勝率", f"{winrate*100:.2f}%")
        col3.metric("Sharpe", f"{sharpe:.2f}")
        col4.metric("交易次數", len(trades))

        st.subheader("⚙️ 最佳參數")
        
        if optimize: 
            for strat, params in best_params_dict.items():
    
                st.markdown(f"### 📌 {strat}")
    
                if strat == "MA":
                    st.write(f"短均線：{params['short']}")
                    st.write(f"長均線：{params['long']}")
    
                elif strat == "RSI":
                    st.write(f"週期：{params['period']}")
                    st.write(f"買進門檻：{params['buy']}")
                    st.write(f"賣出門檻：{params['sell']}")
    
                elif strat == "Bollinger":
                    st.write(f"期間 n：{params['n']}")
                    st.write(f"標準差 k：{params['k']}")
    
                elif strat == "KD":
                    st.write(f"週期 n：{params['n']}")
                    st.write(f"K 期：{params['k_period']}")
                    st.write(f"D 期：{params['d_period']}")
                    st.write(f"低檔：{params['low']}")
                    st.write(f"高檔：{params['high']}")

                elif strat == "MACD":
                    st.write(f"快速期：{params['fast_period']}")
                    st.write(f"慢速期：{params['slow_period']}")
                    st.write(f"Signal 期：{params['signal_period']}")

            st.subheader("🏆 最佳化結果 Top 5")
            for s in opt_results:
                st.markdown(f"### 📌 {s}")
                df_show = opt_results[s].head(5).copy()
                # ⭐ 排序 + 格式化
                df_show = df_show.round(4)
    
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    height=220
                )
     
    else:
        st.warning("沒有產生交易績效")



    # ==============================
    # 📊 計算績效（只算一次🔥）
    # ==============================
    if len(equity) > 1:

        final_return = equity[-1] - 1

        peak = equity[0]
        max_drawdown = 0

        for v in equity:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_drawdown:
                max_drawdown = dd

        # ==============================
        # 📈 圖 + KPI（左右排版🔥）
        # ==============================
        col1, col2 = st.columns([3, 1])

        with col1:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=df.index,
                y=equity,
                mode="lines",
                name="資產淨值",
                line=dict(color="royalblue", width=2),
            ))
            fig_eq.update_layout(
                title="資產淨值曲線（Equity Curve）",
                xaxis_title="日期",
                yaxis_title="資產倍數",
                template="plotly_white",
                height=350,
                hovermode="x unified",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col2:
            st.markdown("### 📊 重點指標")

            st.metric("總報酬", f"{final_return*100:.2f}%")
            st.metric("最大回撤", f"{max_drawdown*100:.2f}%")

            if final_return > 0:
                st.success("策略為獲利")
            else:
                st.error("策略為虧損")

        # ==============================
        # 📈 技術分析圖表
        # ==============================
        st.subheader("📈 技術分析圖表")

        # 決定顯示哪些指標
        show_ma_chart        = "MA" in selected
        show_bollinger_chart = "Bollinger" in selected
        show_kd_chart        = "KD" in selected
        show_macd_chart      = "MACD" in selected
        show_rsi_chart       = "RSI" in selected

        # 自訂條件也可觸發副圖
        all_cond_types = [c for c, _ in (entry_conditions + exit_conditions)]
        if any(c in ("KD黃金交叉", "KD死亡交叉") for c in all_cond_types):
            show_kd_chart = True
        if any(c in ("MACD由零翻正", "MACD黃金交叉", "MACD死亡交叉") for c in all_cond_types):
            show_macd_chart = True

        # 副圖指標清單
        sub_indicators = []
        if show_kd_chart:
            sub_indicators.append("KD")
        if show_macd_chart:
            sub_indicators.append("MACD")
        if show_rsi_chart:
            sub_indicators.append("RSI")

        # 建立多子圖佈局
        n_rows = 1 + len(sub_indicators)
        if sub_indicators:
            main_h = 0.55
            sub_h  = round(0.45 / len(sub_indicators), 4)
            row_heights    = [main_h] + [sub_h] * len(sub_indicators)
            subplot_titles = ["價格走勢"] + sub_indicators
        else:
            row_heights    = [1.0]
            subplot_titles = ["價格走勢"]

        fig_tech = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
        )

        # 主圖：收盤價
        fig_tech.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="收盤價",
                       line=dict(color="royalblue", width=1.5)),
            row=1, col=1,
        )

        # 均線
        if show_ma_chart:
            ma_p    = params_dict.get("MA", {"short": 20, "long": 60})
            short_p = int(ma_p["short"])
            long_p  = int(ma_p["long"])
            fig_tech.add_trace(
                go.Scatter(x=df.index, y=df["Close"].rolling(short_p).mean(),
                           name=f"MA{short_p}", line=dict(color="orange", width=1.2)),
                row=1, col=1,
            )
            fig_tech.add_trace(
                go.Scatter(x=df.index, y=df["Close"].rolling(long_p).mean(),
                           name=f"MA{long_p}", line=dict(color="green", width=1.2)),
                row=1, col=1,
            )

        # 布林通道
        if show_bollinger_chart:
            bb_p    = params_dict.get("Bollinger", {"n": 20, "k": 2.0})
            bb_n    = int(bb_p["n"])
            bb_k    = float(bb_p["k"])
            bb_ma   = df["Close"].rolling(bb_n).mean()
            bb_std  = df["Close"].rolling(bb_n).std()
            bb_up   = bb_ma + bb_k * bb_std
            bb_dn   = bb_ma - bb_k * bb_std
            fig_tech.add_trace(
                go.Scatter(x=df.index, y=bb_up, name="BB上軌",
                           line=dict(color="rgba(128,0,128,0.6)", dash="dash", width=1)),
                row=1, col=1,
            )
            fig_tech.add_trace(
                go.Scatter(x=df.index, y=bb_dn, name="BB下軌",
                           line=dict(color="rgba(128,0,128,0.6)", dash="dash", width=1),
                           fill="tonexty", fillcolor="rgba(128,0,128,0.05)"),
                row=1, col=1,
            )
            fig_tech.add_trace(
                go.Scatter(x=df.index, y=bb_ma, name="BB中線",
                           line=dict(color="purple", width=1, dash="dot")),
                row=1, col=1,
            )

        # 買賣點標記
        if trades:
            buy_dates_list  = [t["buy_date"]   for t in trades]
            sell_dates_list = [t["sell_date"]  for t in trades]
            buy_px_list     = [t["buy_price"]  for t in trades]
            sell_px_list    = [t["sell_price"] for t in trades]

            fig_tech.add_trace(
                go.Scatter(
                    x=buy_dates_list, y=buy_px_list,
                    mode="markers", name="買入",
                    marker=dict(symbol="triangle-up", size=14, color="green"),
                    hovertemplate="買入<br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
            sell_hover = [
                f"賣出<br>日期: {t['sell_date']}<br>價格: {t['sell_price']:.2f}"
                f"<br>報酬: {t['return']*100:.2f}%"
                for t in trades
            ]
            fig_tech.add_trace(
                go.Scatter(
                    x=sell_dates_list, y=sell_px_list,
                    mode="markers", name="賣出",
                    marker=dict(symbol="triangle-down", size=14, color="red"),
                    text=sell_hover, hoverinfo="text",
                ),
                row=1, col=1,
            )

        # 副圖
        for sub_idx, indicator in enumerate(sub_indicators, start=2):
            if indicator == "KD":
                if "K" in df.columns and "D" in df.columns:
                    fig_tech.add_trace(
                        go.Scatter(x=df.index, y=df["K"], name="K",
                                   line=dict(color="blue", width=1.2)),
                        row=sub_idx, col=1,
                    )
                    fig_tech.add_trace(
                        go.Scatter(x=df.index, y=df["D"], name="D",
                                   line=dict(color="orange", width=1.2)),
                        row=sub_idx, col=1,
                    )
                    fig_tech.update_yaxes(range=[0, 100], row=sub_idx, col=1)

            elif indicator == "MACD":
                macd_p   = params_dict.get("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9})
                df_macd  = add_macd(df, macd_p.get("fast_period", 12),
                                    macd_p.get("slow_period", 26),
                                    macd_p.get("signal_period", 9))
                hist_col = ["green" if v >= 0 else "red" for v in df_macd["MACD_hist"]]
                fig_tech.add_trace(
                    go.Bar(x=df_macd.index, y=df_macd["MACD_hist"],
                           name="MACD Hist", marker_color=hist_col),
                    row=sub_idx, col=1,
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_macd.index, y=df_macd["MACD"], name="MACD",
                               line=dict(color="blue", width=1.2)),
                    row=sub_idx, col=1,
                )
                fig_tech.add_trace(
                    go.Scatter(x=df_macd.index, y=df_macd["MACD_signal"], name="Signal",
                               line=dict(color="orange", width=1.2)),
                    row=sub_idx, col=1,
                )

            elif indicator == "RSI":
                rsi_p      = params_dict.get("RSI", {"period": 14, "buy": 30, "sell": 70})
                rsi_period = int(rsi_p.get("period", 14))
                delta      = df["Close"].diff()
                gain       = delta.clip(lower=0).rolling(rsi_period).mean()
                loss       = (-delta.clip(upper=0)).rolling(rsi_period).mean()
                rs         = gain / loss.where(loss != 0, np.nan)
                rsi_vals   = 100 - (100 / (1 + rs))
                buy_lv     = float(rsi_p.get("buy", 30))
                sell_lv    = float(rsi_p.get("sell", 70))
                fig_tech.add_trace(
                    go.Scatter(x=df.index, y=rsi_vals, name="RSI",
                               line=dict(color="purple", width=1.5)),
                    row=sub_idx, col=1,
                )
                fig_tech.add_hline(y=buy_lv,  line_dash="dash", line_color="green",
                                   row=sub_idx, col=1)
                fig_tech.add_hline(y=sell_lv, line_dash="dash", line_color="red",
                                   row=sub_idx, col=1)
                fig_tech.update_yaxes(range=[0, 100], row=sub_idx, col=1)

        fig_tech.update_layout(
            title="技術分析圖表",
            template="plotly_white",
            height=400 + 200 * len(sub_indicators),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig_tech, use_container_width=True)

        # ==============================
        # 📖 圖表解讀（放下面🔥）
        # ==============================
        st.subheader("📖 圖表解讀")

        st.write(f"👉 最終報酬：約 {final_return*100:.2f}%")
        st.write(f"👉 最大回撤：約 {max_drawdown*100:.2f}%")

        # ⭐ 智能解讀
        if final_return > 0:
            st.success("此策略在回測期間呈現穩定獲利趨勢")
        else:
            st.error("此策略整體為虧損，建議調整參數或更換策略")

        if max_drawdown < -0.3:
            st.warning("⚠️ 回撤過大，風險偏高（可能不適合實盤）")
        elif max_drawdown < -0.15:
            st.info("回撤屬於中等範圍，可視風險承受度使用")
        else:
            st.success("回撤控制良好 👍（策略穩定性佳）")
