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
        if df.empty:
            return None
            
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
    # Fix HK tickers (e.g., 700.HK -> 0700.HK)
    if ".HK" in symbol.upper():
        parts = symbol.split('.')
        symbol = f"{parts[0].zfill(4)}.HK"
        
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None) # Strip timezone for consistency
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
#  图表构建 (Optimized with List Comprehensions)
# ============================================================

def build_main_chart(df, dates, show_ma, show_boll, show_ema):
    """构建主图：K线 + 均线/布林带"""
    main_series = []

    # Fast K-line generation
    candle_data = [
        {"time": d, "open": round(o, 3), "high": round(h, 3), "low": round(l, 3), "close": round(c, 3)}
        for d, o, h, l, c in zip(dates, df["Open"], df["High"], df["Low"], df["Close"])
    ]

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

    if show_ma:
        for window, color, label in [(5, "#FF6D00", "MA5"), (10, "#2196F3", "MA10"), (20, "#9C27B0", "MA20"), (60, "#795548", "MA60")]:
            ma = calc_ma(close, window)
            ma_data = [{"time": d, "value": round(v, 3)} for d, v in zip(dates, ma) if pd.notna(v)]
            if ma_data:
                main_series.append({
                    "type": "Line", "data": ma_data,
                    "options": {"color": color, "lineWidth": 1, "crosshairMarkerVisible": False, "lastValueVisible": False, "priceLineVisible": False, "title": label},
                })

    if show_ema:
        for window, color, label in [(12, "#00BCD4", "EMA12"), (26, "#FF5722", "EMA26")]:
            ema = calc_ema(close, window)
            ema_data = [{"time": d, "value": round(v, 3)} for d, v in zip(dates, ema) if pd.notna(v)]
            if ema_data:
                main_series.append({
                    "type": "Line", "data": ema_data,
                    "options": {"color": color, "lineWidth": 1, "lineStyle": 2, "crosshairMarkerVisible": False, "lastValueVisible": False, "priceLineVisible": False, "title": label},
                })

    if show_boll:
        mid, upper, lower = calc_bollinger(close)
        for series_data, color, label, style in [(mid, "#FF9800", "BOLL-MID", 0), (upper, "rgba(255,152,0,0.5)", "BOLL-UP", 2), (lower, "rgba(255,152,0,0.5)", "BOLL-DN", 2)]:
            boll_data = [{"time": d, "value": round(v, 3)} for d, v in zip(dates, series_data) if pd.notna(v)]
            if boll_data:
                main_series.append({
                    "type": "Line", "data": boll_data,
                    "options": {"color": color, "lineWidth": 1, "lineStyle": style, "crosshairMarkerVisible": False, "lastValueVisible": False, "priceLineVisible": False, "title": label},
                })

    return {
        "chart": {
            "height": 450,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
            "crosshair": {"mode": 0},
            "rightPriceScale": {"autoScale": True},
            "timeScale": {"timeVisible": False},
        },
        "series": main_series
    }


def build_volume_chart(df, dates):
    """构建成交量图"""
    volume_data = [
        {"time": d, "value": float(v), "color": "#26a69a" if c >= o else "#ef5350"}
        for d, v, c, o in zip(dates, df["Volume"], df["Close"], df["Open"])
    ]

    return {
        "chart": {
            "height": 100,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
            "timeScale": {"visible": False} # Hide timescale to align cleanly with main chart
        },
        "series": [{"type": "Histogram", "data": volume_data, "options": {"priceFormat": {"type": "volume"}}}]
    }


def build_macd_chart(df, dates, dif, dea, hist):
    """构建MACD指标图"""
    dif_data = [{"time": d, "value": round(v, 4)} for d, v in zip(dates, dif) if pd.notna(v)]
    dea_data = [{"time": d, "value": round(v, 4)} for d, v in zip(dates, dea) if pd.notna(v)]
    hist_data = [{"time": d, "value": round(v, 4), "color": "#26a69a" if v >= 0 else "#ef5350"} for d, v in zip(dates, hist) if pd.notna(v)]

    return {
        "chart": {
            "height": 150,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
        },
        "series": [
            {"type": "Line", "data": dif_data, "options": {"color": "#2196F3", "lineWidth": 1, "title": "DIF"}},
            {"type": "Line", "data": dea_data, "options": {"color": "#FF9800", "lineWidth": 1, "title": "DEA"}},
            {"type": "Histogram", "data": hist_data, "options": {"title": "MACD"}},
        ]
    }


def build_rsi_chart(df, dates, rsi):
    """构建RSI指标图"""
    rsi_data = [{"time": d, "value": round(v, 2)} for d, v in zip(dates, rsi) if pd.notna(v)]

    return {
        "chart": {
            "height": 150,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
        },
        "series": [
            {"type": "Line", "data": rsi_data, "options": {"color": "#9C27B0", "lineWidth": 1, "title": "RSI(14)"}},
        ]
    }

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
    default_code, code_hint = "000001", "6位数字，如 000001、600519"
elif market == "港股":
    default_code, code_hint = "0700.HK", "如 0700.HK、9988.HK"
else:
    default_code, code_hint = "AAPL", "如 AAPL、MSFT"

symbol = st.sidebar.text_input("股票代码", value=default_code, help=code_hint)

period_map = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}
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
        end_dt = datetime.today()
        days_offset = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 380, "2y": 760, "5y": 1900}[period]
        start_dt = end_dt - timedelta(days=days_offset)
        
        if market == "A股":
            df = get_a_stock_data(symbol, str(start_dt.date()), str(end_dt.date()))
        else:
            df = get_hk_us_stock_data(symbol, str(start_dt.date()), str(end_dt.date()))

    if df is None or len(df) < 10:
        st.error("数据获取失败或数据量不足，请检查股票代码。")
        st.stop()

    st.success(f"获取到 {len(df)} 根K线数据")

    # 预处理日期字符串，加速图表构建
    dates_str = df["Date"].dt.strftime("%Y-%m-%d").tolist()

    # 计算指标
    close = df["Close"]
    dif, dea, hist = calc_macd(close)
    rsi = calc_rsi(close) if show_rsi else None

    # 指标卡片
    latest = float(df.iloc[-1]["Close"])
    if len(df) >= 2:
        prev = float(df.iloc[-2]["Close"])
        chg = (latest - prev) / prev * 100
    else:
        chg = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新价", f"{latest:.2f}", f"{chg:+.2f}%")
    col2.metric("最高价", f"{float(df['High'].max()):.2f}")
    col3.metric("最低价", f"{float(df['Low'].min()):.2f}")
    
    # 防止 Volume 为空导致的计算错误
    vol_mean = float(df['Volume'].mean()) if not df['Volume'].empty else 0
    col4.metric("日均成交额", f"{vol_mean * latest / 1e8:.2f}亿")

    # ==== 组装并同步渲染所有图表 ====
    st.subheader(f"📈 走势图表 — {symbol}")
    
    # 将所有图表组合到一个列表中进行单次渲染，实现十字光标和缩放的同步
    sync_charts = []
    
    sync_charts.append(build_main_chart(df, dates_str, show_ma, show_boll, show_ema))
    sync_charts.append(build_volume_chart(df, dates_str))
    
    if show_macd:
        sync_charts.append(build_macd_chart(df, dates_str, dif, dea, hist))
    if show_rsi and rsi is not None:
        sync_charts.append(build_rsi_chart(df, dates_str, rsi))

    renderLightweightCharts(sync_charts)

    # 数据预览
    with st.expander("📋 查看原始数据"):
        display_df = df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["涨跌%"] = (display_df["Close"].pct_change() * 100).round(2)
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
