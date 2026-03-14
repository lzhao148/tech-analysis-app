import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
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

st.title("📈 专业级技术分析平台（多周期 / 多标的 / 策略回测 / 南向资金）")
st.markdown("使用 Streamlit + Lightweight Charts + Yahoo Finance + AKShare 构建的专业技术分析与简易回测终端")

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

@st.cache_data(ttl=3600)
def get_hk_stock_data(ticker_symbol, period_str):
    """从 yfinance 获取港股数据"""
    period_map = {
        "1个月": "1mo",
        "3个月": "3mo",
        "6个月": "6mo",
        "1年": "1y",
        "2年": "2y",
        "5年": "5y"
    }
    try:
        # 港股代码格式处理 (如 00700 -> 0700.HK)
        if ticker_symbol.isdigit():
            if len(ticker_symbol) == 5:
                ticker_symbol = ticker_symbol[1:] + ".HK"
            elif len(ticker_symbol) == 4:
                ticker_symbol = ticker_symbol + ".HK"
        
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period_map[period_str], interval="1d")
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"{ticker_symbol} 获取数据失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_southbound_daily_flow():
    """获取每日南向资金流向"""
    try:
        df = ak.stock_em_hsgt_north_net_flow_in(symbol="港股通")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        st.warning(f"获取南向资金流向失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_stock_hk_ggt_holdings(symbol):
    """获取个股港股通持股数据"""
    try:
        df = ak.stock_hk_ggt_holdings_em(symbol=symbol)
        if df is not None and not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期")
        return df
    except Exception as e:
        st.warning(f"获取港股通持股数据失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_hk_ggt_top_stocks(date_str=None):
    """获取港股通十大成交股"""
    try:
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        df = ak.stock_hk_ggt_top_10_em(symbol="港股通", date=date_str)
        return df
    except Exception as e:
        st.warning(f"获取港股通十大成交股失败: {e}")
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

# ================= 南向资金图表数据准备 =================
def prepare_southbound_chart_data(flow_df, stock_df=None):
    """准备南向资金流向图表数据"""
    chart_data = {
        "flow": [],
        "cumulative": [],
        "stock_price": []
    }
    
    if flow_df is None or flow_df.empty:
        return chart_data
    
    # 计算累计净流入
    flow_df["cumulative"] = flow_df["north_net_flow_in"].cumsum()
    
    for _, row in flow_df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        chart_data["flow"].append({
            "time": date_str,
            "value": float(row["north_net_flow_in"]) if pd.notna(row["north_net_flow_in"]) else 0,
            "color": "rgba(38, 166, 154, 0.8)" if row["north_net_flow_in"] > 0 else "rgba(239, 83, 80, 0.8)"
        })
        chart_data["cumulative"].append({
            "time": date_str,
            "value": float(row["cumulative"]) if pd.notna(row["cumulative"]) else 0
        })
    
    # 合并股价数据
    if stock_df is not None and not stock_df.empty:
        for _, row in stock_df.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            chart_data["stock_price"].append({
                "time": date_str,
                "value": float(row["Close"])
            })
    
    return chart_data

def prepare_holdings_chart_data(holdings_df, stock_df=None):
    """准备持股数据图表"""
    chart_data = {
        "holdings_pct": [],
        "holdings_shares": [],
        "stock_price": []
    }
    
    if holdings_df is None or holdings_df.empty:
        return chart_data
    
    for _, row in holdings_df.iterrows():
        if pd.notna(row.get("日期")):
            date_str = row["日期"].strftime("%Y-%m-%d")
            if pd.notna(row.get("持股比例")):
                chart_data["holdings_pct"].append({
                    "time": date_str,
                    "value": float(row["持股比例"])
                })
            if pd.notna(row.get("持股数量")):
                chart_data["holdings_shares"].append({
                    "time": date_str,
                    "value": float(row["持股数量"]) / 1e8  # 转换为亿股
                })
    
    # 合并股价数据
    if stock_df is not None and not stock_df.empty:
        for _, row in stock_df.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d")
            chart_data["stock_price"].append({
                "time": date_str,
                "value": float(row["Close"])
            })
    
    return chart_data

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

def create_southbound_flow_charts(chart_data, show_cumulative=True, show_stock=True):
    """创建南向资金流向图表"""
    charts = []
    
    # 主图：南向资金净流入
    flow_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "grid": {
                "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
                "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
            },
            "width": 1000,
            "height": 300
        },
        "series": [{
            "type": "Histogram",
            "data": chart_data["flow"],
            "options": {
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "title": "南向资金净流入(亿港元)"
            }
        }]
    }
    charts.append(flow_chart)
    
    # 累计净流入
    if show_cumulative and chart_data["cumulative"]:
        cum_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 200
            },
            "series": [{
                "type": "Line",
                "data": chart_data["cumulative"],
                "options": {
                    "color": "#3F51B5",
                    "lineWidth": 2,
                    "title": "累计净流入(亿港元)"
                }
            }]
        }
        charts.append(cum_chart)
    
    # 股价对比
    if show_stock and chart_data["stock_price"]:
        price_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 200
            },
            "series": [{
                "type": "Line",
                "data": chart_data["stock_price"],
                "options": {
                    "color": "#FF5722",
                    "lineWidth": 2,
                    "title": "股价(港元)"
                }
            }]
        }
        charts.append(price_chart)
    
    return charts

def create_holdings_charts(chart_data):
    """创建持股数据图表"""
    charts = []
    
    # 持股比例
    if chart_data["holdings_pct"]:
        pct_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 250
            },
            "series": [{
                "type": "Line",
                "data": chart_data["holdings_pct"],
                "options": {
                    "color": "#4CAF50",
                    "lineWidth": 2,
                    "title": "港股通持股比例(%)"
                }
            }]
        }
        charts.append(pct_chart)
    
    # 持股数量
    if chart_data["holdings_shares"]:
        shares_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 200
            },
            "series": [{
                "type": "Line",
                "data": chart_data["holdings_shares"],
                "options": {
                    "color": "#2196F3",
                    "lineWidth": 2,
                    "title": "持股数量(亿股)"
                }
            }]
        }
        charts.append(shares_chart)
    
    # 股价对比
    if chart_data["stock_price"]:
        price_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1000,
                "height": 200
            },
            "series": [{
                "type": "Line",
                "data": chart_data["stock_price"],
                "options": {
                    "color": "#FF5722",
                    "lineWidth": 2,
                    "title": "股价(港元)"
                }
            }]
        }
        charts.append(price_chart)
    
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
    
    st.markdown("---")
    st.subheader("💹 南向资金参数")
    hk_ticker = st.text_input("港股代码", value="00700", help="如 00700(腾讯)、09988(阿里)")

# ================= 主体 Tabs =================
tab1, tab2, tab3 = st.tabs(["📊 单标的技术分析 & 策略", "📈 多标的相对强弱", "💹 南向资金流向"])

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
                    
                    # 回测统计
                    total_return = (bt_df["equity"].iloc[-1] - 1) * 100
                    buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
                    num_trades = len(trades) // 2
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("策略累计收益率", f"{total_return:.2f}%")
                    with col2:
                        st.metric("买入持有收益率", f"{buy_hold_return:.2f}%")
                    with col3:
                        st.metric("交易次数", f"{num_trades}")
                    
                    # 权益曲线
                    eq_chart_data = []
                    for _, row in bt_df.iterrows():
                        eq_chart_data.append({
                            "time": row["Date"].strftime("%Y-%m-%d"),
                            "value": float(row["equity"])
                        })
                    
                    equity_chart = {
                        "chart": {
                            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                            "width": 1000,
                            "height": 200
                        },
                        "series": [{
                            "type": "Line",
                            "data": eq_chart_data,
                            "options": {"color": "#4CAF50", "lineWidth": 2, "title": "策略权益曲线"}
                        }]
                    }
                    renderLightweightCharts([equity_chart], key="equity_curve")
                    
                    # 交易记录
                    if not trades.empty:
                        st.subheader("📋 交易记录")
                        trade_display = trades[["Date", "Close", "signal", "position"]].copy()
                        trade_display["操作"] = trade_display["position"].apply(lambda x: "买入" if x > 0 else "卖出")
                        trade_display = trade_display.rename(columns={"Date": "日期", "Close": "价格"})
                        st.dataframe(trade_display[["日期", "价格", "操作"]], use_container_width=True)

                # TD 信号汇总
                if show_td and chart_data["td_signals"]:
                    st.subheader("🎯 TD Sequential 信号汇总")
                    td_signals_df = pd.DataFrame(chart_data["td_signals"])
                    td_signals_df["date"] = pd.to_datetime(td_signals_df["date"])
                    td_signals_df = td_signals_df.sort_values("date", ascending=False)
                    td_signals_df = td_signals_df.rename(columns={"date": "日期", "type": "信号类型", "value": "价格"})
                    st.dataframe(td_signals_df.head(20), use_container_width=True)

                # 数据下载
                st.subheader("📥 数据下载")
                csv_buffer = StringIO()
                df_tf.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="下载当前周期数据 (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"{ticker}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("无法获取数据，请检查股票代码或网络连接。")

# ---------- Tab 2：多标的对比 ----------
with tab2:
    st.subheader("📈 多标的相对强弱对比（归一化价格）")
    
    if st.button("更新多标的对比", type="primary"):
        tickers_list = [t.strip().upper() for t in multi_tickers.split(",") if t.strip()]
        
        if len(tickers_list) < 2:
            st.warning("请至少输入两个股票代码")
        else:
            with st.spinner("正在获取多标的数据..."):
                all_data = {}
                for t in tickers_list:
                    df_t = get_stock_data(t, period)
                    if df_t is not None and not df_t.empty:
                        all_data[t] = df_t
                
                if len(all_data) < 2:
                    st.error("成功获取数据的股票数量不足两个")
                else:
                    # 归一化价格
                    normalized_data = {}
                    for t, df_t in all_data.items():
                        first_close = df_t["Close"].iloc[0]
                        df_t["normalized"] = (df_t["Close"] / first_close) * 100
                        normalized_data[t] = df_t
                    
                    # 构建对比图表
                    comparison_series = []
                    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
                    
                    for idx, (t, df_t) in enumerate(normalized_data.items()):
                        series_data = []
                        for _, row in df_t.iterrows():
                            series_data.append({
                                "time": row["Date"].strftime("%Y-%m-%d"),
                                "value": float(row["normalized"])
                            })
                        comparison_series.append({
                            "type": "Line",
                            "data": series_data,
                            "options": {
                                "color": colors[idx % len(colors)],
                                "lineWidth": 2,
                                "title": t
                            }
                        })
                    
                    comparison_chart = {
                        "chart": {
                            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                            "grid": {
                                "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
                                "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
                            },
                            "width": 1000,
                            "height": 400
                        },
                        "series": comparison_series
                    }
                    
                    st.markdown("**基准日 = 100**")
                    renderLightweightCharts([comparison_chart], key="multi_comparison")
                    
                    # 相关性矩阵
                    st.subheader("📊 价格相关性矩阵")
                    close_prices = pd.DataFrame()
                    for t, df_t in all_data.items():
                        close_prices[t] = df_t.set_index("Date")["Close"]
                    
                    corr_matrix = close_prices.corr()
                    st.dataframe(corr_matrix.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
                    
                    # 收益率对比
                    st.subheader("📈 期间收益率对比")
                    returns_data = []
                    for t, df_t in all_data.items():
                        total_return = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0] - 1) * 100
                        returns_data.append({
                            "股票": t,
                            "期间收益率": f"{total_return:.2f}%",
                            "起始价格": f"{df_t['Close'].iloc[0]:.2f}",
                            "结束价格": f"{df_t['Close'].iloc[-1]:.2f}"
                        })
                    
                    returns_df = pd.DataFrame(returns_data)
                    st.dataframe(returns_df, use_container_width=True)

# ---------- Tab 3：南向资金流向 ----------
with tab3:
    st.subheader("💹 南向资金流向分析")
    st.markdown("数据来源：AKShare（东方财富）| 参考网站：[HKEX Stock Connect](https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk)")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("更新南向资金流向", type="primary"):
            with st.spinner("正在获取南向资金数据..."):
                # 获取南向资金流向
                flow_df = get_southbound_daily_flow()
                
                if flow_df is not None and not flow_df.empty:
                    # 获取港股数据用于对比
                    stock_df = None
                    if hk_ticker:
                        stock_df = get_hk_stock_data(hk_ticker, period)
                    
                    # 准备图表数据
                    sb_chart_data = prepare_southbound_chart_data(flow_df, stock_df)
                    
                    # 统计信息
                    latest_flow = flow_df["north_net_flow_in"].iloc[-1]
                    latest_date = flow_df["date"].iloc[-1].strftime("%Y-%m-%d")
                    total_inflow = flow_df["north_net_flow_in"].sum()
                    avg_flow = flow_df["north_net_flow_in"].mean()
                    
                    st.markdown(f"**最新数据日期**: {latest_date}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("当日净流入", f"{latest_flow:.2f}亿港元")
                    with col_b:
                        st.metric("期间累计净流入", f"{total_inflow:.2f}亿港元")
                    with col_c:
                        st.metric("日均净流入", f"{avg_flow:.2f}亿港元")
                    
                    # 港股通十大成交股
                    st.subheader("🏆 港股通十大成交股")
                    top_stocks = get_hk_ggt_top_stocks()
                    if top_stocks is not None and not top_stocks.empty:
                        display_cols = ["股票代码", "股票简称", "成交金额", "净买入金额"]
                        available_cols = [c for c in display_cols if c in top_stocks.columns]
                        st.dataframe(top_stocks[available_cols].head(10), use_container_width=True)
                    
                    # 流向图表
                    st.subheader("📊 南向资金净流入走势")
                    sb_charts = create_southbound_flow_charts(sb_chart_data, show_cumulative=True, show_stock=stock_df is not None)
                    renderLightweightCharts(sb_charts, key="southbound_flow")
                    
                else:
                    st.error("无法获取南向资金数据，请检查网络连接或稍后重试。")
    
    with col2:
        st.subheader("📌 个股港股通持股分析")
        
        if hk_ticker:
            if st.button("查询个股持股数据", key="query_holdings"):
                with st.spinner("正在获取持股数据..."):
                    # 获取持股数据
                    holdings_df = get_stock_hk_ggt_holdings(hk_ticker)
                    
                    if holdings_df is not None and not holdings_df.empty:
                        # 获取股价数据
                        stock_df = get_hk_stock_data(hk_ticker, period)
                        
                        # 准备图表数据
                        holdings_chart_data = prepare_holdings_chart_data(holdings_df, stock_df)
                        
                        # 最新持股信息
                        latest = holdings_df.iloc[-1]
                        st.markdown(f"**股票代码**: {hk_ticker}")
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            if "持股比例" in latest:
                                st.metric("持股比例", f"{latest['持股比例']:.2f}%")
                        with col_y:
                            if "持股数量" in latest:
                                shares = latest['持股数量']
                                st.metric("持股数量", f"{shares/1e8:.2f}亿股")
                        
                        # 持股图表
                        st.subheader("📈 持股变化趋势")
                        holdings_charts = create_holdings_charts(holdings_chart_data)
                        renderLightweightCharts(holdings_charts, key="holdings_chart")
                        
                        # 持股明细表
                        st.subheader("📋 持股明细")
                        display_holdings = holdings_df.copy()
                        if "日期" in display_holdings.columns:
                            display_holdings["日期"] = display_holdings["日期"].dt.strftime("%Y-%m-%d")
                        st.dataframe(display_holdings.tail(20), use_container_width=True)
                        
                    else:
                        st.warning("该股票暂无港股通持股数据，可能不是港股通标的。")
        else:
            st.info("请在左侧输入港股代码以查询持股数据")

# ================= 页脚 =================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>📈 专业级技术分析平台 | 数据来源: Yahoo Finance, AKShare</p>
    <p>技术指标: RSI, MACD, Bollinger Bands, TD Sequential | 策略回测: SMA 交叉</p>
    <p>南向资金数据来源: 东方财富 (通过 AKShare) | 参考: <a href="https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk" target="_blank">HKEX Stock Connect</a></p>
</div>
""", unsafe_allow_html=True)
