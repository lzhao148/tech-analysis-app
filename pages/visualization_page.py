# visualization_page.py — 股票数据可视化分析
# 支持 A股/港股/美股 K线图、技术指标、成交量分析
# 依赖：streamlit, pandas, numpy, yfinance, akshare, streamlit-lightweight-charts

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

# ============================================================
#  数据获取
# ============================================================

@st.cache_data(ttl=3600, show_spinner=True)
def get_a_stock_data(symbol: str, start_date: str, end_date: str):
    """通过 akshare 获取 A股日线数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq"
        )
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "最高": "High",
            "最低": "Low", "收盘": "Close", "成交量": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"A股数据获取失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=True)
def get_hk_us_stock_data(symbol: str, start_date: str, end_date: str):
    """通过 yfinance 获取 港股/美股 日线数据"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ============================================================
#  技术指标计算
# ============================================================

def calc_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def calc_ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def calc_rsi(close: pd.Series, window=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_bollinger(close: pd.Series, window=20, num_std=2.0):
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

# ============================================================
#  图表构建
# ============================================================

def build_main_chart(df, show_ma, show_boll, show_ema):
    """构建主图：K线 + 均线/布林带"""
    charts = []
    main_series = []

    # K线
    candle_data = []
    for _, row in df.iterrows():
        candle_data.append({
            "time": str(row["Date"])[:10],
            "open": round(float(row["Open"]), 3),
            "high": round(float(row["High"]), 3),
            "low": round(float(row["Low"]), 3),
            "close": round(float(row["Close"]), 3),
        })

    main_series.append({
        "type": "Candlestick",
        "data": candle_data,
        "options": {
            "upColor": "#26a69a", "downColor": "#ef5350",
            "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
            "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    })

    close = df["Close"]

    # 均线
    if show_ma:
        for window, color, label in [(5, "#FF6D00", "MA5"), (10, "#2196F3", "MA10"), (20, "#9C27B0", "MA20"), (60, "#795548", "MA60")]:
            if len(df) < window:
                continue
            ma = calc_ma(close, window)
            ma_data = []
            for i, v in ma.items():
                if pd.notna(v):
                    ma_data.append({"time": str(df.iloc[i]["Date"])[:10], "value": round(float(v), 3)})
            if ma_data:
                main_series.append({
                    "type": "Line",
                    "data": ma_data,
                    "options": {
                        "color": color, "lineWidth": 1,
                        "crosshairMarkerVisible": False,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                        "title": label,
                    },
                })

    # EMA
    if show_ema:
        for window, color, label in [(12, "#00BCD4", "EMA12"), (26, "#FF5722", "EMA26")]:
            if len(df) < window:
                continue
            ema = calc_ema(close, window)
            ema_data = []
            for i, v in ema.items():
                if pd.notna(v):
                    ema_data.append({"time": str(df.iloc[i]["Date"])[:10], "value": round(float(v), 3)})
            if ema_data:
                main_series.append({
                    "type": "Line",
                    "data": ema_data,
                    "options": {
                        "color": color, "lineWidth": 1,
                        "lineStyle": 2,
                        "crosshairMarkerVisible": False,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                        "title": label,
                    },
                })

    # 布林带
    if show_boll:
        mid, upper, lower = calc_bollinger(close)
        for series_data, color, label, style in [
            (mid, "#FF9800", "BOLL-MID", 0),
            (upper, "rgba(255,152,0,0.5)", "BOLL-UP", 2),
            (lower, "rgba(255,152,0,0.5)", "BOLL-DN", 2),
        ]:
            boll_data = []
            for i, v in series_data.items():
                if pd.notna(v):
                    boll_data.append({"time": str(df.iloc[i]["Date"])[:10], "value": round(float(v), 3)})
            if boll_data:
                main_series.append({
                    "type": "Line",
                    "data": boll_data,
                    "options": {
                        "color": color, "lineWidth": 1,
                        "lineStyle": style,
                        "crosshairMarkerVisible": False,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                        "title": label,
                    },
                })

    chart_options = {
        "height": 450,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
        "crosshair": {"mode": 0},
        "timeScale": {"timeVisible": False},
    }
    charts.append({"chart": chart_options, "series": main_series})
    return charts


def build_volume_chart(df):
    """构建成交量图"""
    volume_data = []
    for _, row in df.iterrows():
        volume_data.append({
            "time": str(row["Date"])[:10],
            "value": float(row["Volume"]),
            "color": "#26a69a" if row["Close"] >= row["Open"] else "#ef5350",
        })

    vol_chart = {
        "height": 100,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
    }
    return [{
        "chart": vol_chart,
        "series": [{"type": "Histogram", "data": volume_data, "options": {"priceFormat": {"type": "volume"}}}]
    }]


def build_macd_chart(df, dif, dea, hist):
    """构建MACD指标图"""
    dif_data, dea_data, hist_data = [], [], []
    for i in range(len(dif)):
        if pd.isna(dif.iloc[i]):
            continue
        d = str(df.iloc[i]["Date"])[:10]
        dif_data.append({"time": d, "value": round(float(dif.iloc[i]), 4)})
        dea_data.append({"time": d, "value": round(float(dea.iloc[i]), 4)})
        h = float(hist.iloc[i])
        hist_data.append({
            "time": d,
            "value": round(h, 4),
            "color": "#26a69a" if h >= 0 else "#ef5350",
        })

    macd_chart = {
        "height": 150,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
    }
    return [{
        "chart": macd_chart,
        "series": [
            {"type": "Line", "data": dif_data, "options": {"color": "#2196F3", "lineWidth": 1, "title": "DIF"}},
            {"type": "Line", "data": dea_data, "options": {"color": "#FF9800", "lineWidth": 1, "title": "DEA"}},
            {"type": "Histogram", "data": hist_data, "options": {"title": "MACD"}},
        ]
    }]


def build_rsi_chart(df, rsi):
    """构建RSI指标图"""
    rsi_data = []
    for i in range(len(rsi)):
        if pd.isna(rsi.iloc[i]):
            continue
        rsi_data.append({"time": str(df.iloc[i]["Date"])[:10], "value": round(float(rsi.iloc[i]), 2)})

    rsi_chart = {
        "height": 150,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
    }
    return [{
        "chart": rsi_chart,
        "series": [
            {"type": "Line", "data": rsi_data, "options": {"color": "#9C27B0", "lineWidth": 1, "title": "RSI(14)"}},
        ]
    }]

# ============================================================
#  Streamlit 页面
# ============================================================

st.set_page_config(page_title="数据可视化", page_icon="📊", layout="wide")

st.title("📊 股票数据可视化分析")
st.markdown("K线图 + 技术指标 + 成交量 + MACD + RSI 综合可视化")

# ---- 侧边栏 ----
st.sidebar.header("📈 数据源")
market = st.sidebar.selectbox("市场类型", ["A股", "港股", "美股"], index=0)

if market == "A股":
    default_code = "000001"
    code_hint = "6位数字，如 000001、600519"
elif market == "港股":
    default_code = "0700.HK"
    code_hint = "如 0700.HK、9988.HK"
else:
    default_code = "AAPL"
    code_hint = "如 AAPL、MSFT"

symbol = st.sidebar.text_input("股票代码", value=default_code, help=code_hint)

period_map = {
    "1个月": "1mo", "3个月": "3mo", "6个月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y",
}
period_label = st.sidebar.selectbox("时间范围", list(period_map.keys()), index=3)
period = period_map[period_label]

st.sidebar.header("🔧 技术指标")
show_ma = st.sidebar.checkbox("均线 (MA5/10/20/60)", value=True)
show_ema = st.sidebar.checkbox("指数均线 (EMA12/26)", value=False)
show_boll = st.sidebar.checkbox("布林带 (BOLL20)", value=False)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_rsi = st.sidebar.checkbox("RSI(14)", value=False)

run_button = st.sidebar.button("🚀 开始分析", type="primary", use_container_width=True)

# ---- 主逻辑 ----
if run_button:
    with st.spinner("正在获取数据..."):
        if market == "A股":
            # akshare 用 start/end 日期
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days={"1mo": 35, "3mo": 100, "6mo": 200, "1y": 380, "2y": 760, "5y": 1900}[period])
            df = get_a_stock_data(symbol, str(start_dt.date()), str(end_dt.date()))
        else:
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days={"1mo": 35, "3mo": 100, "6mo": 200, "1y": 380, "2y": 760, "5y": 1900}[period])
            df = get_hk_us_stock_data(symbol, str(start_dt.date()), str(end_dt.date()))

    if df is None or len(df) < 10:
        st.error("数据获取失败或数据量不足，请检查股票代码。")
        st.stop()

    st.success(f"获取到 {len(df)} 根K线数据")

    # 计算指标
    close = df["Close"]
    dif, dea, hist = calc_macd(close)
    rsi = calc_rsi(close) if show_rsi else None

    # 指标卡片
    latest = float(df.iloc[-1]["Close"])
    if len(df) >= 2:
        prev = float(df.iloc[-2]["Close"])
        chg = (latest - prev) / prev * 100
        sign = "🔺" if chg >= 0 else "🔻"
    else:
        chg = 0
        sign = ""

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新价", f"{latest:.2f}", f"{chg:+.2f}%")
    col2.metric("最高价", f"{float(df['High'].max()):.2f}")
    col3.metric("最低价", f"{float(df['Low'].min()):.2f}")
    col4.metric("日均成交额", f"{float(df['Volume'].mean() * latest / 1e8):.2f}亿")

    # K线主图
    st.subheader(f"📈 K线图 — {symbol}")
    main_charts = build_main_chart(df, show_ma, show_boll, show_ema)
    renderLightweightCharts(main_charts)

    # 成交量
    st.subheader("📊 成交量")
    vol_charts = build_volume_chart(df)
    renderLightweightCharts(vol_charts)

    # MACD
    if show_macd:
        st.subheader("📉 MACD")
        macd_charts = build_macd_chart(df, dif, dea, hist)
        renderLightweightCharts(macd_charts)

    # RSI
    if show_rsi and rsi is not None:
        st.subheader("📉 RSI(14)")
        rsi_charts = build_rsi_chart(df, rsi)
        renderLightweightCharts(rsi_charts)

    # 数据预览
    with st.expander("📋 查看原始数据"):
        display_df = df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["涨跌%"] = display_df["Close"].pct_change() * 100
        display_df["涨跌%"] = display_df["涨跌%"].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False)
        st.download_button("📥 下载CSV", csv, f"{symbol}_data.csv", "text/csv")

else:
    st.info("👈 在左侧设置参数后，点击 **🚀 开始分析** 查看可视化结果")

    st.markdown("""
    ### 📊 功能说明

    | 功能 | 说明 |
    |------|------|
    | **K线图** | 真实股票数据K线，支持A股/港股/美股 |
    | **均线** | MA5/10/20/60 多周期均线 |
    | **指数均线** | EMA12/26（MACD基础线） |
    | **布林带** | BOLL20 中轨+上下轨 |
    | **MACD** | DIF + DEA + 柱状图 |
    | **RSI** | 14日相对强弱指标 |
    | **成交量** | 红绿分色柱状图 |

    > 💡 支持 A股(akshare)、港股(yfinance)、美股(yfinance) 三大市场
    """)
