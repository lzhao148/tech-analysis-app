import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

# ================= 页面配置 =================
st.set_page_config(
    page_title="技术分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 专业级技术分析平台")
st.markdown("使用 Streamlit 和 Lightweight Charts 构建的专业技术分析工具")

# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 参数设置")

    # 股票代码
    ticker = st.text_input("股票代码", value="AAPL")

    # 时间范围
    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options)

    # 技术指标
    st.subheader("技术指标")
    show_sma = st.checkbox("简单移动平均线 (SMA)", value=True)
    show_ema = st.checkbox("指数移动平均线 (EMA)", value=False)  # 占位，暂未使用
    show_rsi = st.checkbox("相对强弱指数 (RSI)", value=True)
    show_td = st.checkbox("DeMark TD Sequential", value=True)

    if show_sma:
        sma_periods = st.multiselect(
            "SMA周期",
            [5, 10, 20, 50, 100, 200],
            default=[20, 50]
        )
    else:
        sma_periods = []

    # TD Sequential 参数
    if show_td:
        st.subheader("TD Sequential参数")
        td_lookback = st.slider("回溯周期", 4, 20, 9, help="TD Sequential的计数回溯周期")
        td_count_threshold = st.slider("计数阈值", 5, 13, 9, help="触发TD Sequential信号的计数阈值")

# ================= 数据获取 =================
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbol, period_str):
    try:
        period_map = {
            "1个月": "1mo",
            "3个月": "3mo",
            "6个月": "6mo",
            "1年": "1y",
            "2年": "2y",
            "5年": "5y"
        }

        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period_map[period_str])

        if df.empty:
            return None

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None

# ================= RSI 计算 =================
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices, dtype=float)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi

# ================= TD Sequential 计算 =================
def calculate_td_sequential(df, lookback=9, count_threshold=9):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    dates = df["Date"].values

    n = len(df)

    # TD Setup
    td_setup = np.zeros(n, dtype=int)
    td_setup_direction = np.zeros(n, dtype=int)  # 1: bullish, -1: bearish

    for i in range(lookback, n):
        bullish_setup = True
        bearish_setup = True

        for j in range(9):
            if i - j < 0:
                break

            if i - j - 4 >= 0:
                # 牛市 Setup
                if close[i - j] <= close[i - j - 4]:
                    bullish_setup = False
                # 熊市 Setup
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

    # TD Countdown
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
            if td_setup_direction[countdown_start] == 1:  # bullish
                if i >= 2 and close[i] >= max(high[i - 2], high[i - 1]):
                    td_countdown[i] = td_countdown[i - 1] + 1
                else:
                    td_countdown[i] = td_countdown[i - 1]
            else:  # bearish
                if i >= 2 and close[i] <= min(low[i - 2], low[i - 1]):
                    td_countdown[i] = td_countdown[i - 1] + 1
                else:
                    td_countdown[i] = td_countdown[i - 1]

            if td_countdown[i] >= 13:
                in_countdown = False
                td_countdown[i] = 0

    # 信号
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

# ================= 数据预处理为 chart_data =================
def prepare_chart_data(df, show_sma=False, sma_periods=None, show_rsi=True, show_td=False, td_params=None):
    if sma_periods is None:
        sma_periods = []

    chart_data = {
        "candles": [],
        "volume": [],
        "sma": {},
        "rsi": [],
        "td_signals": [],
        "td_setup": [],
        "td_countdown": []
    }

    # K 线
    for _, row in df.iterrows():
        chart_data["candles"].append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })

    # 成交量
    for _, row in df.iterrows():
        chart_data["volume"].append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "value": float(row["Volume"]),
            "color": "rgba(239, 83, 80, 0.8)" if row["Close"] < row["Open"] else "rgba(38, 166, 154, 0.8)"
        })

    # SMA
    if show_sma:
        for period in sma_periods:
            if len(df) >= period:
                sma_values = df["Close"].rolling(window=period).mean()
                sma_series = []
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= period - 1:
                        sma_series.append({
                            "time": row["Date"].strftime("%Y-%m-%d"),
                            "value": float(sma_values.iloc[i])
                        })
                chart_data["sma"][f"SMA_{period}"] = sma_series

    # RSI
    if show_rsi and len(df) > 14:
        rsi_values = calculate_rsi(df["Close"].values)
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 14:
                chart_data["rsi"].append({
                    "time": row["Date"].strftime("%Y-%m-%d"),
                    "value": float(rsi_values[i])
                })

    # TD Sequential
    if show_td and td_params and len(df) > td_params["lookback"]:
        td_results = calculate_td_sequential(df, td_params["lookback"], td_params["count_threshold"])

        # TD Setup
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= td_params["lookback"]:
                setup_value = td_results["td_setup"][i]
                if abs(setup_value) >= 5:
                    chart_data["td_setup"].append({
                        "time": row["Date"].strftime("%Y-%m-%d"),
                        "value": int(setup_value)
                    })

        # TD Countdown
        for i, (_, row) in enumerate(df.iterrows()):
            countdown_value = td_results["td_countdown"][i]
            if countdown_value > 0:
                chart_data["td_countdown"].append({
                    "time": row["Date"].strftime("%Y-%m-%d"),
                    "value": int(countdown_value)
                })

        # 信号
        for signal in td_results["signals"]:
            chart_data["td_signals"].append({
                "time": pd.to_datetime(signal["date"]).strftime("%Y-%m-%d"),
                "type": signal["type"],
                "value": float(signal["value"])
            })

    return chart_data

# ================= 构建 charts 列表（符合组件 API） =================
def create_charts(chart_data, show_sma=False, sma_periods=None, show_rsi=True, show_td=False):
    if sma_periods is None:
        sma_periods = []

    # --- 主图 chartOptions ---
    chart_options_main = {
        "layout": {
            "textColor": "black",
            "background": {"type": "solid", "color": "white"}
        },
        "grid": {
            "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
            "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
        },
        "width": 1000,
        "height": 400
    }

    # --- 主图 series：K线 + SMA + TD 标记 ---
    main_series = [{
        "type": "Candlestick",
        "data": chart_data["candles"],
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
            "priceFormat": {
                "type": "price",
                "precision": 2,
                "minMove": 0.01
            }
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

    # TD 信号 Markers
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

        main_series.append({
            "type": "Markers",
            "data": markers_data
        })

    main_chart = {
        "chart": chart_options_main,
        "series": main_series
    }

    # --- 成交量 chart ---
    volume_chart = {
        "chart": {
            "layout": {
                "textColor": "black",
                "background": {"type": "solid", "color": "white"}
            },
            "width": 1000,
            "height": 120
        },
        "series": [{
            "type": "Histogram",
            "data": chart_data["volume"],
            "options": {
                "priceFormat": {"type": "volume"}
            }
        }]
    }

    charts = [main_chart, volume_chart]

    # --- RSI chart ---
    if show_rsi and chart_data["rsi"]:
        rsi_chart = {
            "chart": {
                "layout": {
                    "textColor": "black",
                    "background": {"type": "solid", "color": "white"}
                },
                "width": 1000,
                "height": 120
            },
            "series": [{
                "type": "Line",
                "data": chart_data["rsi"],
                "options": {
                    "color": "#9C27B0",
                    "lineWidth": 2,
                    "title": "RSI(14)"
                }
            }]
        }
        charts.append(rsi_chart)

    # --- TD 计数 chart ---
    if show_td and (chart_data["td_setup"] or chart_data["td_countdown"]):
        td_series = []
        if chart_data["td_setup"]:
            td_series.append({
                "type": "Line",
                "data": chart_data["td_setup"],
                "options": {
                    "color": "#FF9800",
                    "lineWidth": 1,
                    "title": "TD Setup"
                }
            })
        if chart_data["td_countdown"]:
            td_series.append({
                "type": "Line",
                "data": chart_data["td_countdown"],
                "options": {
                    "color": "#2196F3",
                    "lineWidth": 1,
                    "title": "TD Countdown"
                }
            })

        td_chart = {
            "chart": {
                "layout": {
                    "textColor": "black",
                    "background": {"type": "solid", "color": "white"}
                },
                "width": 1000,
                "height": 100
            },
            "series": td_series
        }
        charts.append(td_chart)

    return charts

# ================= 主界面逻辑 =================
if st.sidebar.button("更新图表", type="primary"):
    with st.spinner("正在获取数据..."):
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
                td_params = {
                    "lookback": td_lookback if "td_lookback" in locals() else 9,
                    "count_threshold": td_count_threshold if "td_count_threshold" in locals() else 9
                }

            # 预处理 chart_data
            chart_data = prepare_chart_data(
                df,
                show_sma=show_sma,
                sma_periods=sma_periods,
                show_rsi=show_rsi,
                show_td=show_td,
                td_params=td_params
            )

            # 构建 charts 列表
            charts = create_charts(
                chart_data,
                show_sma=show_sma,
                sma_periods=sma_periods,
                show_rsi=show_rsi,
                show_td=show_td
            )

            st.subheader(f"{ticker} 技术分析图表")
            renderLightweightCharts(charts, key="main_chart")

            # TD 信号摘要
            if show_td and chart_data["td_signals"]:
                st.subheader("📊 TD Sequential信号")
                recent_signals = chart_data["td_signals"][-5:]
                for signal in recent_signals:
                    if "Buy" in signal["type"]:
                        st.success(f"📈 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")
                    else:
                        st.error(f"📉 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")

            # 数据表
            with st.expander("查看原始数据"):
                st.dataframe(df.tail(20))

            # 下载
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下载CSV数据",
                data=csv,
                file_name=f"{ticker}_stock_data.csv",
                mime="text/csv"
            )
        else:
            st.error("无法获取股票数据，请检查股票代码是否正确")
else:
    st.info("👈 请在侧边栏设置参数，然后点击“更新图表”按钮")

st.markdown("---")
st.caption("📊 数据来源: Yahoo Finance | 🚀 部署平台: Streamlit Community Cloud")
