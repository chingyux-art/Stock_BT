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
from conditions import ＊


st.set_page_config(layout="wide")
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
# 參數 UI
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
# 回測模式選擇（互斥）
# ==============================
st.markdown("---")
st.subheader("⚙️ 回測模式選擇")

mode = st.radio(
    "選擇回測方式",
    ["🎯 自訂進出場條件", "🔄 策略最佳化"],
    horizontal=True
)

optimize = (mode == "🔄 策略最佳化")

# 最佳化說明
if optimize:
    with st.expander("📘 策略最佳化使用說明", expanded=True):
        st.markdown("""
### 🎯 最佳化邏輯說明

**最佳化目的**: 自動搜尋各策略的最優參數，而不是手動設定

**評分方式** (加權組合):
- **總報酬 (50%)**: 策略獲利能力
- **Sharpe 夏普比率 (30%)**: 風險調整後報酬（越高越穩定）
- **勝率 (20%)**: 獲利次數比例

**運作流程**:
1. 自動遍歷參數空間（短均線、週期、門檻值等）
2. 對每組參數進行回測
3. 計算評分（Return×0.5 + Sharpe×0.3 + Winrate×0.2）
4. 返回最高評分的參數組合

**注意事項**:
- ⏱️ 最佳化會耗費較多時間（取決於參數空間大小）
- 📊 結果會顯示 Top 5 最優參數組合
- ⚠️ 過去績效不代表未來走勢，建議在實盤前進行充分驗證
        """)
    
    # 最佳化模式下不顯示自訂條件
    entry_conditions = []
    exit_conditions = []
    use_stop_loss = False

else:
    # ==============================
    # 技術指標說明（摺疊展開）
    # ==============================
    with st.expander("📚 技術指標說明與建議", expanded=False):
        
        tabs = st.tabs(["KD 隨機指標", "MACD", "RSI", "布林通道"])
        
        with tabs[0]:
            st.markdown("""
### KD 指標（隨機指標，Stochastic Oscillator）

KD 指標是由 George Lane 提出的技術指標，用來衡量股價在最近一段時間內的相對強弱位置，數值範圍在 0~100 之間。

**計算方式**：
- %K線（快速線）= [(當日收盤價 - 過去N天最低價) ÷ (過去N天最高價 - 過去N天最低價)] × 100
- %D線（慢速線）= %K線的 M 期簡單移動平均線（通常 M=3）

**標準參數**：(14,3,3) 或 (9,3,3)

**交易信號**：
- K線向上穿越D線 = **金叉（買進訊號）**
- K線向下穿越D線 = **死叉（賣出訊號）**
- K > 80 = **超買（可能回檔）**
- K < 20 = **超賣（可能反彈）**

#### 📊 參數建議表

| 交易類型 | 建議參數 | 特色 | 適合時間框架 | 注意事項 |
|---------|---------|------|------------|---------|
| 當日沖/極短線 | (5,3,3) 或 (7,3,3) | 極敏感，訊號多 | 1~15分鐘線 | 假訊號非常多 |
| 短線/波段（最推薦） | (9,3,3) | 平衡，適合台股個股 | 日線 | 台股主流設定 |
| 中線波段/趨勢 | (14,3,3) | 較平滑，減少噪音 | 日線~周線 | 假訊號較少 |
| 長線/存股 | (21,3,3) 或 (36,5,5) | 更穩定，訊號少 | 周線 | 搭配均線使用 |
            """)
        
        with tabs[1]:
            st.markdown("""
### MACD（移動平均收斂散度）

MACD 是由 Gerald Appel 提出的技術分析中最受歡迎的趨勢指標之一。

**組成成分**：
- **MACD線**（快線）= 短期 EMA(12) - 長期 EMA(26)
- **訊號線（Signal Line）** = MACD線的 9日 EMA
- **柱狀圖（Histogram）** = MACD線 - 訊號線

**交易信號**：
- **由零翻正**：MACD 從負值轉為正值（趨勢向上）
- **黃金交叉**：MACD 上穿 Signal 線（買進訊號）
- **死亡交叉**：MACD 下穿 Signal 線（賣出訊號）

#### 📊 參數建議表

| 交易類型 | 建議參數 | 特色 | 適合時間框架 |
|---------|---------|------|------------|
| 當日沖/極短線 | (5,13,9) 或 (8,17,9) | 極敏感，訊號多 | 5分~60分線 |
| 短線/波段（最推薦） | (12,26,9) | 平衡，假訊號適中 | 日線 |
| 中線波段 | (12,26,9) 或 (19,39,9) | 較平滑，可靠性較高 | 日線 |
| 長線/趨勢 | (19,39,9) 或 (12,26,12) | 訊號少，適合抓大趨勢 | 日線~周線 |
            """)
        
        with tabs[2]:
            st.markdown("""
### RSI（Relative Strength Index，相對強弱指數）

RSI 用來衡量股價漲跌的強弱程度，範圍在 0~100

**計算方式**：
1. 取過去 N 天（預設 N=14）的平均漲幅（Average Gain）與平均跌幅（Average Loss）
2. 相對強弱值 RS = 平均漲幅 ÷ 平均跌幅
3. RSI = 100 - 100 / (1 + RS)

**交易信號**：
- RSI < 30（或自訂值） = **超賣（可能反彈）** → **買進信號**
- RSI > 70（或自訂值） = **超買（可能回檔）** → **賣出信號**

#### 📊 參數建議表

| 交易類型 | 建議週期 | 超買/超賣線 | 特色 | 適合時間框架 |
|---------|---------|----------|------|------------|
| 當日沖/極短線 | 5~9 | 75/25 或 80/20 | 訊號非常敏感 | 1~15分鐘線 |
| 短線/波段（最推薦） | 14 | 70/30 | 平衡，假訊號適中 | 日線 |
| 中線波段 | 14~21 | 70/30 或 75/25 | 較平滑，可靠性較高 | 日線 |
| 長線/趨勢 | 21~50 | 75/25 或 80/20 | 訊號少，適合大趨勢 | 日線~周線 |
            """)
        
        with tabs[3]:
            st.markdown("""
### 布林通道（Bollinger Bands）

布林通道是由 John Bollinger 提出的技術指標，用來衡量股價波動性與相對位置。

**組成成分**：
- **中軌** = 過去 20 日簡單移動平均線（SMA）
- **上軌** = 中軌 + 2 倍標準差
- **下軌** = 中軌 - 2 倍標準差

**通道特性**：
- **變寬** = 波動增大
- **變窄（Squeeze）** = 預示即將突破
- **靠近上軌** = 偏強（超買）
- **靠近下軌** = 偏弱（超賣）

#### 📊 參數建議表

| 交易類型 | 建議Period | 建議Std Dev | 特色 | 適合時間框架 |
|---------|----------|----------|------|------------|
| 短線/當日沖 | 10~15 | 1.5~2.0 | 更敏感，訊號多 | 5分~1小時 |
| 波段交易（最推薦） | 20 | 2.0 | 平衡，假訊號較少 | 日線 |
| 中長線/趨勢 | 50 | 2.2~2.5 | 更平滑，減少噪音 | 日線~周線 |
| 超長線 | 100~200 | 2.5~3.0 | 捕捉大趨勢，訊號很少 | 周線 |
            """)
    
    # ==============================
    # 簡化版進出場條件設定
    # ==============================
    st.markdown("---")
    st.subheader("🎯 自訂進出場條件")
    
    col_entry, col_exit = st.columns(2)
    
    def render_simple_condition(prefix: str):
        """簡化版條件設定"""
        conditions = []
        
        for i in range(1, 3):  # 最多 2 個條件
            with st.container():
                st.write(f"**條件 {i}**")
                
                col1, col2, col3 = st.columns(3)
                
                # 選擇條件類別
                with col1:
                    category = st.selectbox(
                        "類別",
                        ["(不設定)"] + list(CONDITION_CATEGORIES.keys()),
                        key=f"{prefix}_cat_{i}",
                        index=0
                    )
                
                if category == "(不設定)":
                    continue
                
                # 選擇具體信號
                with col2:
                    signals = CONDITION_CATEGORIES[category]["signals"]
                    signal = st.selectbox(
                        "信號",
                        list(signals.keys()),
                        key=f"{prefix}_sig_{i}",
                        format_func=lambda x: signals[x]
                    )
                
                # 設置參數
                with col3:
                    st.write("**參數**")
                    param_defaults = CONDITION_PARAMS.get(signal, {})
                    param_values = {}
                    
                    for param_name, default_val in param_defaults.items():
                        if isinstance(default_val, int):
                            param_values[param_name] = st.number_input(
                                f"{param_name}", 
                                min_value=0, 
                                value=int(default_val),
                                key=f"{prefix}_{signal}_{param_name}_{i}"
                            )
                        else:
                            param_values[param_name] = st.number_input(
                                f"{param_name}", 
                                value=float(default_val),
                                step=0.1,
                                key=f"{prefix}_{signal}_{param_name}_{i}"
                            )
                
                conditions.append((signal, param_values))
        
        return conditions
    
    with col_entry:
        st.write("### 📈 進場條件")
        entry_conditions = render_simple_condition("entry")
    
    with col_exit:
        st.write("### 📉 出場條件")
        exit_conditions = render_simple_condition("exit")
        
        # 停損選項
        use_stop_loss = st.checkbox("✋ 啟用 10% 停損", key="stop_loss_cb")
        if use_stop_loss:
            exit_conditions.append(("停損_虧損10%", {}))

# ==============================
# 📋 動態參數說明欄
# ==============================
st.markdown("---")
st.subheader("📋 回測參數設定說明")

with st.expander("詳細參數配置", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 回測基本設定")
        st.write(f"**股票代號:** {stock}")
        st.write(f"**開始日期:** {start_date}")
        st.write(f"**結束日期:** {end_date}")
        st.write(f"**策略組合:** {', '.join(selected) if selected else '(未選擇)'}")
        
    with col2:
        st.markdown("### ⚙️ 回測模式")
        if optimize:
            st.write("🔄 **策略最佳化模式**")
            st.write("自動搜尋各策略的最優參數")
        else:
            st.write("🎯 **自訂進出場條件模式**")
            st.write("使用自訂買賣信號進行回測")
    
    # 策略參數說明（動態）
    if optimize:
        st.markdown("### 📌 策略參數設定")
        
        param_info = []
        
        if "MA" in selected:
            params_used = params_dict.get("MA", {})
            param_info.append(f"""
**🔹 MA 均線策略**
- 短均線週期: **{params_used.get('short', 'N/A')}**
- 長均線週期: **{params_used.get('long', 'N/A')}**
            """)
        
        if "RSI" in selected:
            params_used = params_dict.get("RSI", {})
            param_info.append(f"""
**🔹 RSI 相對強弱指標**
- 週期: **{params_used.get('period', 'N/A')}** | 買進: **{params_used.get('buy', 'N/A')}** | 賣出: **{params_used.get('sell', 'N/A')}**
            """)
        
        if "KD" in selected:
            params_used = params_dict.get("KD", {})
            param_info.append(f"""
**🔹 KD 隨機指標**
- 週期(N): **{params_used.get('n', 'N/A')}** | K期: **{params_used.get('k_period', 'N/A')}** | D期: **{params_used.get('d_period', 'N/A')}**
            """)
        
        if "MACD" in selected:
            params_used = params_dict.get("MACD", {})
            param_info.append(f"""
**🔹 MACD 移動平均收斂散度**
- 快速期: **{params_used.get('fast_period', 'N/A')}** | 慢速期: **{params_used.get('slow_period', 'N/A')}** | Signal期: **{params_used.get('signal_period', 'N/A')}**
            """)
        
        if "Bollinger" in selected:
            params_used = params_dict.get("Bollinger", {})
            param_info.append(f"""
**🔹 布林通道**
- 週期(N): **{params_used.get('n', 'N/A')}** | 標準差(K): **{params_used.get('k', 'N/A')}**
            """)
        
        for info in param_info:
            st.write(info)


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
# 統一最佳化器
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
# 執行回測
# ==============================
if st.button("🚀 Run Backtest", use_container_width=True):

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

    if not selected:
        st.error("請選擇至少一個策略")
        st.stop()

    # 檢查互斥條件
    if not optimize and not entry_conditions:
        st.error("自訂進出場條件模式下，請至少設定一個進場條件")
        st.stop()

    # ---------- 抓資料 ----------
    with st.spinner("📊 正在獲取股票數據..."):
        df = get_data(stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df.empty:
        st.error("抓不到資料")
        st.stop()

    df = add_indicators(df)

    # ==============================
    # ⭐ 訊號產生
    # ==============================
    signals = []

    if optimize:

        best_params_dict = {}
        opt_results = {}

        with st.spinner("🔄 正在執行參數最佳化，請稍候..."):
            for s in selected:
                best_params, result_df = optimize_strategy(df, s)

                if best_params is None:
                    st.warning(f"{s} 無最佳參數")
                    continue

                best_params_dict[s] = best_params
                opt_results[s] = result_df

        for s in best_params_dict:
            signals.append(run_strategy(df, s, best_params_dict[s]))

    else:
        # 自訂條件模式
        df_cond = df.copy()
        if any(c in ("KD黃金交叉", "KD死亡交叉", "KD_K值大於", "KD_K值小於") for c, _ in (entry_conditions + exit_conditions)):
            df_cond = compute_kd(df_cond)

        custom_signal = pd.Series(0, index=df_cond.index)

        if entry_conditions:
            entry_bool_list = [
                check_condition(df_cond, ctype, **cparams)
                for ctype, cparams in entry_conditions
            ]
            if entry_bool_list:
                entry_trigger = combine_signals(entry_bool_list, "AND", 1)
                custom_signal[entry_trigger == 1] = 1

        if exit_conditions:
            exit_bool_list = [
                check_condition(df_cond, ctype, **cparams)
                for ctype, cparams in exit_conditions
            ]
            if exit_bool_list:
                exit_trigger = combine_signals(exit_bool_list, "OR", 1)
                custom_signal[exit_trigger == 1] = -1

        signals = [custom_signal]

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

        if optimize:
            st.subheader("⚙️ 最佳參數")
            
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
                df_show = df_show.round(4)
        
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    height=220
                )
     
    else:
        st.warning("沒有產生交易績效")


    # ==============================
    # 📊 計算績效
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
        # 📈 圖 + KPI
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
            subplot_titles = ["K線 & MA10/MA20"] + sub_indicators
        else:
            row_heights    = [1.0]
            subplot_titles = ["K線 & MA10/MA20"]

        fig_tech = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
        )

        # ⭐ 主圖：K棒圖表
        fig_tech.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K棒",
                increasing_line_color="red",
                decreasing_line_color="green"
            ),
            row=1, col=1,
        )

        # ⭐ MA10 和 MA20 均線
        ma_10 = df["Close"].rolling(10).mean()
        ma_20 = df["Close"].rolling(20).mean()
        
        fig_tech.add_trace(
            go.Scatter(x=df.index, y=ma_10, name="MA10",
                       line=dict(color="rgba(255, 100, 0, 0.8)", width=1.5)),
            row=1, col=1,
        )
        fig_tech.add_trace(
            go.Scatter(x=df.index, y=ma_20, name="MA20",
                       line=dict(color="rgba(0, 100, 255, 0.8)", width=1.5)),
            row=1, col=1,
        )

        # 均線（如果 MA 策略被選擇）
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
                           name=f"MA{long_p}", line=dict(color="purple", width=1.2)),
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
                    hovertemplate="買入<br>日期: %{x}<br>��格: %{y:.2f}<extra></extra>",
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
        # 📖 圖表解讀
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
