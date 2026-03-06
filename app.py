import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
from streamlit_lightweight_charts import renderLightweightCharts

# ================= 页面配置 =================
st.set_page_config(
    page_title="技术分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 专业级技术分析平台（多周期 / 多标的 / 策略回测）")
st.markdown("使用 Streamlit + Lightweight Charts + Yahoo Finance 构建的专业技术分析与简易回测终端")

# ================= 工具函数 =================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period_str, interval="1d"):
    """从 yfinance 获取数据并返回统一格式 DataFrame"""
    period_map = {
        "1个月": "1mo",
        "3个月": "3mo",
        "6个月": "6mo",
        "1年": "1y",
        "2年": "2y",
        "5年": "5y"
    }
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period_map[period_str], interval=interval)
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"{ticker_symbol} 获取数据失败: {e}")
        return None

def resample_ohlcv(df, timeframe="D"):
    """对原始日线数据按周/月聚合为 OHLCV"""
    df = df.set_index("Date")
    ohlc_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }
    df_resampled = df.resample(timeframe).agg(ohlc_dict).dropna().reset_index()
    return df_resampled

# ================= RSI 计算 =================
def calculate_rsi(prices, period=14):
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)

    deltas = np.diff(prices)
    if len(deltas) < period:
        return np.full(n, np.nan)

    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    rsi = np.full(n, np.nan, dtype=float)

    if down == 0:
        rsi[period] = 100.0
    else:
        rs = up / down
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, n):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        if down == 0:
            rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi

# ================= EMA / MACD / Bollinger =================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bbands(close, period=20, mult=2.0):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + mult * std
    lower = ma - mult * std
    return ma, upper, lower

# ================= TD Sequential 计算 =================
def calculate_td_sequential(df, lookback=9, count_threshold=9):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    dates = df["Date"].values

    n = len(df)

    td_setup = np.zeros(n, dtype=int)
    td_setup_direction = np.zeros(n, dtype=int)  # 1: bullish, -1: bearish

    for i in range(lookback, n):
        bullish_setup = True
        bearish_setup = True

        for j in range(9):
            if i - j < 0:
                break
            if i - j - 4 >= 0:
                if close[i - j] <= close[i - j - 4]:
                    bullish_setup = False
                if close[i - j] >= close[i - j - 4]:
                    bearish_setup = False

        if bullish_setup and td_setup[i - 1] < 9:
            td_setup[i] = td_setup[i - 1] + 1
            td_setup_direction[i] = 1
        elif bearish_setup and td_setup[i - 1] > -9:
            td_setup[i] = td_setup[i - 1] - 1
            td_setup_direction[i] = -1
        else:
            td_setup[i] = 0
            td_setup_direction[i] = 0

    td_countdown = np.zeros(n, dtype=int)
    in_countdown = False
    countdown_start = 0

    for i in range(n):
        if not in_countdown:
            if abs(td_setup[i]) == count_threshold:
                in_countdown = True
                countdown_start = i
                td_countdown[i] = 1
        else:
            if td_setup_direction[countdown_start] == 1:
                if i >= 2 and close[i] >= max(high[i - 2], high[i - 1]):
                    td_countdown[i] = td_countdown[i - 1] + 1
                else:
                    td_countdown[i] = td_countdown[i - 1]
            else:
                if i >= 2 and close[i] <= min(low[i - 2], low[i - 1]):
                    td_countdown[i] = td_countdown[i - 1] + 1
                else:
                    td_countdown[i] = td_countdown[i - 1]

            if td_countdown[i] >= 13:
                in_countdown = False
                td_countdown[i] = 0

    signals = []
    for i in range(n):
        if td_setup[i] == count_threshold and td_setup_direction[i] == 1:
            signals.append({
                "date": dates[i],
                "type": "TD Setup Buy",
                "value": close[i],
                "index": i
            })
        elif td_setup[i] == -count_threshold and td_setup_direction[i] == -1:
            signals.append({
                "date": dates[i],
                "type": "TD Setup Sell",
                "value": close[i],
                "index": i
            })
        elif td_countdown[i] == 13:
            if td_setup_direction[countdown_start] == 1:
                signals.append({
                    "date": dates[i],
                    "type": "TD Countdown Buy",
                    "value": close[i],
                    "index": i
                })
            else:
                signals.append({
                    "date": dates[i],
                    "type": "TD Countdown Sell",
                    "value": close[i],
                    "index": i
                })

    return {
        "td_setup": td_setup,
        "td_setup_direction": td_setup_direction,
        "td_countdown": td_countdown,
        "signals": signals
    }

# ================= 策略：SMA 交叉简易回测 =================
def backtest_sma_cross(df, short=20, long=50):
    data = df.copy()
    data["SMA_short"] = data["Close"].rolling(short).mean()
    data["SMA_long"] = data["Close"].rolling(long).mean()
    data["signal"] = 0
    data.loc[data["SMA_short"] > data["SMA_long"], "signal"] = 1
    data.loc[data["SMA_short"] < data["SMA_long"], "signal"] = -1

    data["position"] = data["signal"].shift(1).fillna(0)
    data["ret"] = data["Close"].pct_change().fillna(0)
    data["strategy_ret"] = data["position"] * data["ret"]
    data["equity"] = (1 + data["strategy_ret"]).cumprod()

    trades = data[(data["position"] != data["position"].shift(1)) & (data["position"] != 0)].copy()
    return data, trades

# ================= 数据预处理为 chart_data =================
def prepare_chart_data(
    df,
    timeframe="日线",
    show_sma=False,
    sma_periods=None,
    show_rsi=True,
    show_td=False,
    td_params=None,
    show_bb=False,
    bb_period=20,
    bb_mult=2.0,
    show_macd=False
):
    if sma_periods is None:
        sma_periods = []

    chart_data = {
        "candles": [],
        "volume": [],
        "sma": {},
        "rsi": [],
        "td_signals": [],
        "td_setup": [],
        "td_countdown": [],
        "bb_middle": [],
        "bb_upper": [],
        "bb_lower": [],
        "macd": [],
        "macd_signal": [],
        "macd_hist": []
    }

    # 时间周期处理
    if timeframe == "日线":
        df_tf = df.copy()
    elif timeframe == "周线":
        df_tf = resample_ohlcv(df, "W")
    elif timeframe == "月线":
        df_tf = resample_ohlcv(df, "M")
    else:
        df_tf = df.copy()

    df_tf = df_tf.reset_index(drop=True)

    # K 线
    for _, row in df_tf.iterrows():
        chart_data["candles"].append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })

    # Volume
    for _, row in df_tf.iterrows():
        chart_data["volume"].append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "value": float(row["Volume"]),
            "color": "rgba(239, 83, 80, 0.8)" if row["Close"] < row["Open"] else "rgba(38, 166, 154, 0.8)"
        })

    close = df_tf["Close"]

    # SMA
    if show_sma:
        for period in sma_periods:
            if len(df_tf) >= period:
                sma_values = close.rolling(window=period).mean()
                sma_series = []
                for i, (_, row) in enumerate(df_tf.iterrows()):
                    if not np.isnan(sma_values.iloc[i]):
                        sma_series.append({
                            "time": row["Date"].strftime("%Y-%m-%d"),
                            "value": float(sma_values.iloc[i])
                        })
                chart_data["sma"][f"SMA_{period}"] = sma_series

    # RSI
    if show_rsi and len(df_tf) > 14:
        rsi_values = calculate_rsi(close.values)
        for i, (_, row) in enumerate(df_tf.iterrows()):
            if i < len(rsi_values) and not np.isnan(rsi_values[i]):
                chart_data["rsi"].append({
                    "time": row["Date"].strftime("%Y-%m-%d"),
                    "value": float(rsi_values[i])
                })

    # Bollinger Bands
    if show_bb and len(df_tf) >= bb_period:
        ma, upper, lower = calculate_bbands(close, bb_period, bb_mult)
        for i, (_, row) in enumerate(df_tf.iterrows()):
            t = row["Date"].strftime("%Y-%m-%d")
            if not np.isnan(ma.iloc[i]):
                chart_data["bb_middle"].append({"time": t, "value": float(ma.iloc[i])})
            if not np.isnan(upper.iloc[i]):
                chart_data["bb_upper"].append({"time": t, "value": float(upper.iloc[i])})
            if not np.isnan(lower.iloc[i]):
                chart_data["bb_lower"].append({"time": t, "value": float(lower.iloc[i])})

    # MACD
    if show_macd and len(df_tf) > 35:
        macd_line, signal_line, hist = calculate_macd(close)
        for i, (_, row) in enumerate(df_tf.iterrows()):
            t = row["Date"].strftime("%Y-%m-%d")
            if not np.isnan(macd_line.iloc[i]):
                chart_data["macd"].append({"time": t, "value": float(macd_line.iloc[i])})
            if not np.isnan(signal_line.iloc[i]):
                chart_data["macd_signal"].append({"time": t, "value": float(signal_line.iloc[i])})
            if not np.isnan(hist.iloc[i]):
                chart_data["macd_hist"].append({"time": t, "value": float(hist.iloc[i])})

    # TD Sequential
    if show_td and td_params and len(df_tf) > td_params["lookback"]:
        td_results = calculate_td_sequential(df_tf, td_params["lookback"], td_params["count_threshold"])

        for i, (_, row) in enumerate(df_tf.iterrows()):
            t = row["Date"].strftime("%Y-%m-%d")
            setup_value = td_results["td_setup"][i]
            countdown_value = td_results["td_countdown"][i]

            if i >= td_params["lookback"] and abs(setup_value) >= 5:
                chart_data["td_setup"].append({"time": t, "value": int(setup_value)})

            if countdown_value > 0:
                chart_data["td_countdown"].append({"time": t, "value": int(countdown_value)})

        for signal in td_results["signals"]:
            chart_data["td_signals"].append({
                "time": pd.to_datetime(signal["date"]).strftime("%Y-%m-%d"),
                "type": signal["type"],
                "value": float(signal["value"])
            })

    return chart_data, df_tf

# ================= 构建 charts 列表 =================
def create_charts(
    chart_data,
    show_sma=False,
    sma_periods=None,
    show_rsi=True,
    show_td=False,
    show_bb=False,
    show_macd=False
):
    if sma_periods is None:
        sma_periods = []

    charts = []

    # 主图
    chart_options_main = {
        "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
        "grid": {
            "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
            "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
        },
        "width": 1000,
        "height": 400
    }

    main_series = [{
        "type": "Candlestick",
        "data": chart_data["candles"],
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
            "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01}
        }
    }]

    # SMA
    if show_sma and chart_data["sma"]:
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        for idx, (period, sma_series) in enumerate(chart_data["sma"].items()):
            if sma_series:
                main_series.append({
                    "type": "Line",
                    "data": sma_series,
                    "options": {
                        "color": colors[idx % len(colors)],
                        "lineWidth": 2,
                        "title": f"SMA {period.split('_')[1]}"
                    }
                })

    # Bollinger Bands
    if show_bb:
        if chart_data["bb_middle"]:
            main_series.append({
                "type": "Line",
                "data": chart_data["bb_middle"],
                "options": {"color": "#FFC107", "lineWidth": 1, "title": "BB Mid"}
            })
        if chart_data["bb_upper"]:
            main_series.append({
                "type": "Line",
                "data": chart_data["bb_upper"],
                "options": {"color": "#FF9800", "lineWidth": 1, "title": "BB Upper"}
            })
        if chart_data["bb_lower"]:
            main_series.append({
                "type": "Line",
                "data": chart_data["bb_lower"],
                "options": {"color": "#FF9800", "lineWidth": 1, "title": "BB Lower"}
            })

    # TD 信号 markers
    if show_td and chart_data["td_signals"]:
        markers_data = []
        for signal in chart_data["td_signals"]:
            if "Buy" in signal["type"]:
                marker_color = "#4CAF50"
                shape = "arrowUp"
                position = "belowBar"
            else:
                marker_color = "#F44336"
                shape = "arrowDown"
                position = "aboveBar"
            markers_data.append({
                "time": signal["time"],
                "position": position,
                "color": marker_color,
                "shape": shape,
                "text": signal["type"]
            })
        main_series.append({"type": "Markers", "data": markers_data})

    main_chart = {"chart": chart_options_main, "series": main_series}
    charts.append(main_chart)

    # Volume
    volume_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "width": 1000,
            "height": 120
        },
        "series": [{
            "type": "Histogram",
            "data": chart_data["volume"],
            "options": {"priceFormat": {"type": "volume"}}
        }]
    }
    charts.append(volume_chart)

    # RSI pane
    if show_rsi and chart_data["rsi"]:
        rsi_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 120
            },
            "series": [{
                "type": "Line",
                "data": chart_data["rsi"],
                "options": {"color": "#9C27B0", "lineWidth": 2, "title": "RSI(14)"}
            }]
        }
        charts.append(rsi_chart)

    # MACD pane
    if show_macd and (chart_data["macd"] or chart_data["macd_signal"] or chart_data["macd_hist"]):
        macd_series = []
        if chart_data["macd"]:
            macd_series.append({
                "type": "Line",
                "data": chart_data["macd"],
                "options": {"color": "#03A9F4", "lineWidth": 1, "title": "MACD"}
            })
        if chart_data["macd_signal"]:
            macd_series.append({
                "type": "Line",
                "data": chart_data["macd_signal"],
                "options": {"color": "#FF5722", "lineWidth": 1, "title": "Signal"}
            })
        if chart_data["macd_hist"]:
            macd_series.append({
                "type": "Histogram",
                "data": chart_data["macd_hist"],
                "options": {"color": "#8BC34A", "priceFormat": {"type": "price", "precision": 4}}
            })

        macd_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 120
            },
            "series": macd_series
        }
        charts.append(macd_chart)

    # TD 计数 pane
    if show_td and (chart_data["td_setup"] or chart_data["td_countdown"]):
        td_series = []
        if chart_data["td_setup"]:
            td_series.append({
                "type": "Line",
                "data": chart_data["td_setup"],
                "options": {"color": "#FF9800", "lineWidth": 1, "title": "TD Setup"}
            })
        if chart_data["td_countdown"]:
            td_series.append({
                "type": "Line",
                "data": chart_data["td_countdown"],
                "options": {"color": "#2196F3", "lineWidth": 1, "title": "TD Countdown"}
            })

        td_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 100
            },
            "series": td_series
        }
        charts.append(td_chart)

    return charts

# ================= 侧边栏参数 =================
with st.sidebar:
    st.header("⚙️ 全局参数")

    ticker = st.text_input("主图股票代码", value="AAPL")
    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options)

    timeframe = st.selectbox("周期", ["日线", "周线", "月线"])

    st.subheader("技术指标")
    show_sma = st.checkbox("SMA", value=True)
    show_rsi = st.checkbox("RSI(14)", value=True)
    show_td = st.checkbox("DeMark TD Sequential", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_macd = st.checkbox("MACD", value=True)

    if show_sma:
        sma_periods = st.multiselect("SMA 周期", [5, 10, 20, 50, 100, 200], default=[20, 50])
    else:
        sma_periods = []

    if show_td:
        st.subheader("TD 参数")
        td_lookback = st.slider("回溯周期", 4, 20, 9)
        td_count_threshold = st.slider("计数阈值", 5, 13, 9)
    else:
        td_lookback, td_count_threshold = 9, 9

    st.subheader("Bollinger 参数")
    bb_period = st.slider("BB 周期", 10, 40, 20)
    bb_mult = st.slider("BB 倍数", 1.0, 3.0, 2.0, step=0.1)

    st.subheader("简易 SMA 策略")
    bt_enable = st.checkbox("启用 SMA 交叉回测", value=True)
    bt_short = st.slider("短期 SMA", 5, 50, 20)
    bt_long = st.slider("长期 SMA", 20, 200, 50)

    st.subheader("多标的对比（归一化）")
    multi_tickers = st.text_input("输入多个代码（逗号分隔）", value="AAPL,MSFT,NVDA")

# ================= 主体 Tabs =================
tab1, tab2 = st.tabs(["📊 单标的技术分析 & 策略", "📈 多标的相对强弱"])

# ---------- Tab 1：单标的 ----------
with tab1:
    if st.button("更新图表与回测", type="primary"):
        with st.spinner("正在获取数据并生成图表..."):
            df = get_stock_data(ticker, period)
            if df is not None and not df.empty:
                # 关键指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    current_price = df["Close"].iloc[-1]
                    st.metric("当前价格", f"{current_price:.2f}")
                with col2:
                    prev_price = df["Close"].iloc[-2]
                    change = current_price - prev_price
                    pct_change = (change / prev_price) * 100
                    st.metric("日变化", f"{change:.2f}", f"{pct_change:.2f}%")
                with col3:
                    volume_ratio = df["Volume"].iloc[-1] / df["Volume"].mean()
                    st.metric("成交量比", f"{volume_ratio:.2f}x")
                with col4:
                    if len(df) >= 20:
                        sma_20 = df["Close"].rolling(20).mean().iloc[-1]
                        st.metric("20日SMA", f"{sma_20:.2f}")

                # TD 参数
                td_params = None
                if show_td:
                    td_params = {"lookback": td_lookback, "count_threshold": td_count_threshold}

                # 准备 chart_data + 不同周期 df_tf
                chart_data, df_tf = prepare_chart_data(
                    df,
                    timeframe=timeframe,
                    show_sma=show_sma,
                    sma_periods=sma_periods,
                    show_rsi=show_rsi,
                    show_td=show_td,
                    td_params=td_params,
                    show_bb=show_bb,
                    bb_period=bb_period,
                    bb_mult=bb_mult,
                    show_macd=show_macd
                )

                charts = create_charts(
                    chart_data,
                    show_sma=show_sma,
                    sma_periods=sma_periods,
                    show_rsi=show_rsi,
                    show_td=show_td,
                    show_bb=show_bb,
                    show_macd=show_macd
                )

                st.subheader(f"{ticker} {timeframe} 技术分析图表")
                renderLightweightCharts(charts, key="main_chart")

                # 策略回测
                if bt_enable:
                    st.subheader("📉 简易 SMA 交叉策略回测（基于原始日线）")
                    bt_df, trades = backtest_sma_cross(df, short=bt_short, long=bt_long)
                    eq_chart_data = []
                    for _, row in bt_df.iterrows():
                        eq_chart_data.append({
                            "time": row["Date"].strftime("%Y-%m-%d"),
                            "value": float(row["equity"])
                        })
                    eq_chart = [{
                        "chart": {
                            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                            "width": 1000,
                            "height": 200
                        },
                        "series": [{
                            "type": "Line",
                            "data": eq_chart_data,
                            "options": {"color": "#3F51B5", "lineWidth": 2, "title": "Equity Curve"}
                        }]
                    }]
                    renderLightweightCharts(eq_chart, key="equity_chart")

                    total_return = bt_df["equity"].iloc[-1] - 1
                    max_dd = ((bt_df["equity"].cummax() - bt_df["equity"]) / bt_df["equity"].cummax()).max()
                    st.write(f"策略总收益: {total_return*100:.2f}% | 最大回撤: {max_dd*100:.2f}%")

                    with st.expander("查看交易点（信号变化点）"):
                        st.dataframe(trades[["Date", "position"]].tail(20))

                # TD 信号摘要
                if show_td and chart_data["td_signals"]:
                    st.subheader("📊 TD Sequential 信号（最近 5 个）")
                    recent_signals = chart_data["td_signals"][-5:]
                    for signal in recent_signals:
                        if "Buy" in signal["type"]:
                            st.success(f"📈 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")
                        else:
                            st.error(f"📉 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")

                # 数据表 & 下载
                with st.expander("查看原始数据"):
                    st.dataframe(df.tail(50))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载原始 CSV",
                    data=csv,
                    file_name=f"{ticker}_stock_data.csv",
                    mime="text/csv"
                )

                # 研究笔记
                st.markdown("---")
                st.subheader("📝 交易 / 研究笔记")
                note = st.text_area("在此记录你的盘后总结 / 交易计划 / 研究发现：", height=150)
                if st.button("导出为 Markdown"):
                    buf = StringIO()
                    buf.write(f"# {ticker} 技术分析笔记\n\n")
                    buf.write(f"- 日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    buf.write(f"- 周期：{timeframe}\n")
                    buf.write(f"- 使用指标：SMA={show_sma}, RSI={show_rsi}, TD={show_td}, BB={show_bb}, MACD={show_macd}\n\n")
                    buf.write("## 笔记内容\n\n")
                    buf.write(note if note.strip() else "（尚未填写）\n")
                    md_bytes = buf.getvalue().encode("utf-8")
                    st.download_button(
                        label="📄 下载 Markdown 笔记",
                        data=md_bytes,
                        file_name=f"{ticker}_note_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
            else:
                st.error("无法获取股票数据，请检查代码或时间范围")
    else:
        st.info("👈 点击左上角按钮生成图表与回测")

# ---------- Tab 2：多标的对比 ----------
with tab2:
    st.subheader("多标的归一化收盘价对比（相对强弱）")
    if st.button("更新多标的图表", key="multi_btn"):
        syms = [s.strip().upper() for s in multi_tickers.split(",") if s.strip()]
        if not syms:
            st.error("请输入至少一个代码")
        else:
            series_map = {}
            dates_union = None
            for sym in syms:
                df_sym = get_stock_data(sym, period)
                if df_sym is None or df_sym.empty:
                    st.warning(f"{sym} 数据为空，已跳过")
                    continue
                df_sym = df_sym[["Date", "Close"]].copy()
                df_sym.set_index("Date", inplace=True)
                df_sym[f"{sym}_norm"] = df_sym["Close"] / df_sym["Close"].iloc[0]
                series_map[sym] = df_sym[f"{sym}_norm"]
                if dates_union is None:
                    dates_union = df_sym.index
                else:
                    dates_union = dates_union.union(df_sym.index)

            if not series_map:
                st.error("所有代码数据均为空，无法绘图")
            else:
                multi_series = []
                color_pool = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#3F51B5", "#FF5722"]
                for idx, (sym, s) in enumerate(series_map.items()):
                    s = s.reindex(dates_union).ffill()
                    data = []
                    for date, val in s.items():
                        data.append({"time": date.strftime("%Y-%m-%d"), "value": float(val)})
                    multi_series.append({
                        "type": "Line",
                        "data": data,
                        "options": {
                            "color": color_pool[idx % len(color_pool)],
                            "lineWidth": 2,
                            "title": sym
                        }
                    })

                chart_multi = [{
                    "chart": {
                        "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                        "width": 1000,
                        "height": 500
                    },
                    "series": multi_series
                }]
                renderLightweightCharts(chart_multi, key="multi_chart")
