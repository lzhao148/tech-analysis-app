import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="技术分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 专业级技术分析平台")
st.markdown("使用 Streamlit 和 Lightweight Charts 构建的实时技术分析工具")

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("⚙️ 参数设置")

    ticker = st.text_input("股票代码", value="AAPL")

    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options)

    chart_type = st.selectbox("图表类型", ["蜡烛图", "线图", "面积图"])

    st.subheader("技术指标")
    show_sma = st.checkbox("简单移动平均线 (SMA)", value=True)
    show_ema = st.checkbox("指数移动平均线 (EMA)", value=False)  # 先占位，暂未使用
    show_rsi = st.checkbox("相对强弱指数 (RSI)", value=True)

    if show_sma:
        sma_periods = st.multiselect(
            "SMA周期",
            [5, 10, 20, 50, 100, 200],
            default=[20, 50]
        )
    else:
        sma_periods = []

# ---------------- 数据获取 ----------------
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

# ---------------- RSI 计算 ----------------
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

# ---------------- 多窗格图表构建（无 panes 字段） ----------------
def create_multi_pane_charts(df, show_sma=False, sma_periods=None, show_rsi=True):
    if sma_periods is None:
        sma_periods = []

    # K 线数据
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })

    # 成交量数据
    volume_data = []
    for _, row in df.iterrows():
        volume_data.append({
            "time": row["Date"].strftime("%Y-%m-%d"),
            "value": float(row["Volume"]),
            "color": "rgba(239, 83, 80, 0.8)" if row["Close"] < row["Open"] else "rgba(38, 166, 154, 0.8)"
        })

    # SMA
    sma_series = []
    if show_sma:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for idx, p in enumerate(sma_periods):
            if len(df) >= p:
                sma_values = df["Close"].rolling(window=p).mean()
                sma_line = []
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= p - 1:
                        sma_line.append({
                            "time": row["Date"].strftime("%Y-%m-%d"),
                            "value": float(sma_values.iloc[i])
                        })
                sma_series.append({
                    "type": "Line",
                    "data": sma_line,
                    "options": {
                        "color": colors[idx % len(colors)],
                        "lineWidth": 2,
                        "title": f"SMA {p}"
                    }
                })

    # RSI
    rsi_series = []
    if show_rsi and len(df) > 14:
        rsi_values = calculate_rsi(df["Close"].values)
        rsi_line = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 14:
                rsi_line.append({
                    "time": row["Date"].strftime("%Y-%m-%d"),
                    "value": float(rsi_values[i])
                })
        rsi_series.append({
            "type": "Line",
            "data": rsi_line,
            "options": {
                "color": "#9C27B0",
                "lineWidth": 2,
                "title": "RSI(14)"
            }
        })

    # 主图 chartOptions
    chart_options_main = {
        "layout": {
            "textColor": "black",
            "background": {
                "type": "solid",
                "color": "white"
            }
        },
        "grid": {
            "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
            "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
        },
        "width": 900,
        "height": 400
    }

    # 主价格窗格：K线 + SMA
    main_series = [{
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350"
        }
    }]
    main_series.extend(sma_series)

    main_chart = {
        "chart": chart_options_main,
        "series": main_series
    }

    # 成交量窗格（第二个 chart）
    volume_chart = {
        "chart": {
            "layout": {
                "textColor": "black",
                "background": {
                    "type": "solid",
                    "color": "white"
                }
            },
            "width": 900,
            "height": 150
        },
        "series": [{
            "type": "Histogram",
            "data": volume_data,
            "options": {
                "priceFormat": {"type": "volume"}
            }
        }]
    }

    charts = [main_chart, volume_chart]

    # RSI 窗格（第三个 chart，可选）
    if show_rsi and rsi_series:
        rsi_chart = {
            "chart": {
                "layout": {
                    "textColor": "black",
                    "background": {
                        "type": "solid",
                        "color": "white"
                    }
                },
                "width": 900,
                "height": 150
            },
            "series": rsi_series
        }
        charts.append(rsi_chart)

    return charts

# ---------------- 主界面逻辑 ----------------
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

            # 图表
            charts = create_multi_pane_charts(
                df,
                show_sma=show_sma,
                sma_periods=sma_periods,
                show_rsi=show_rsi
            )

            st.subheader(f"{ticker} 技术分析图表")
            renderLightweightCharts(charts, key="main_chart")

            # 数据表格
            with st.expander("查看原始数据"):
                st.dataframe(df.tail(20))

            # CSV 导出
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
