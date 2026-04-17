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
                    "type
