# hk_sector_leaders_page.py — 港股行业龙头K线可视化
# 自包含页面：展示港股不同行业板块龙头股票的K线图
# 依赖：streamlit, pandas, numpy, yfinance, plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
#  港股行业龙头股票定义
# ============================================================

HK_SECTORS = {
    "🤖 科技互联网": {
        "desc": "互联网平台、软件服务、消费电子",
        "stocks": {
            "腾讯控股": "0700.HK",
            "美团-W": "3690.HK",
            "小米集团-W": "1810.HK",
            "快手-W": "1024.HK",
            "网易-S": "9999.HK",
            "百度集团-SW": "9888.HK",
        }
    },
    "🏦 金融银行": {
        "desc": "银行、保险、券商",
        "stocks": {
            "汇丰控股": "0005.HK",
            "建设银行": "0939.HK",
            "工商银行": "1398.HK",
            "中国平安": "2318.HK",
            "招商银行": "3968.HK",
            "友邦保险": "1299.HK",
        }
    },
    "🛒 消费零售": {
        "desc": "食品饮料、服装、零售",
        "stocks": {
            "农夫山泉": "9633.HK",
            "李宁": "2331.HK",
            "安踏体育": "2020.HK",
            "蒙牛乳业": "2319.HK",
            "海底捞": "6862.HK",
            "泡泡玛特": "9992.HK",
        }
    },
    "💊 医药生物": {
        "desc": "创新药、生物科技、医疗器械",
        "stocks": {
            "药明生物": "2269.HK",
            "石药集团": "1093.HK",
            "中国生物制药": "1177.HK",
            "百济神州": "6160.HK",
            "信达生物": "1801.HK",
        }
    },
    "🚗 新能源汽车": {
        "desc": "整车制造、电池、智能驾驶",
        "stocks": {
            "比亚迪": "1211.HK",
            "理想汽车-W": "2015.HK",
            "蔚来-SW": "9866.HK",
            "小鹏汽车-W": "9868.HK",
            "吉利汽车": "0175.HK",
        }
    },
    "⚡ 能源资源": {
        "desc": "石油、天然气、新能源",
        "stocks": {
            "中国石油化工": "0386.HK",
            "中国海洋石油": "0883.HK",
            "中国石油股份": "0857.HK",
            "信义光能": "0968.HK",
        }
    },
    "📱 通信运营商": {
        "desc": "电信服务、5G基建",
        "stocks": {
            "中国移动": "0941.HK",
            "中国电信": "0728.HK",
            "中国联通": "0762.HK",
        }
    },
    "🏠 地产基建": {
        "desc": "房地产开发、基建",
        "stocks": {
            "新鸿基地产": "0016.HK",
            "长实集团": "1113.HK",
            "恒基地产": "0012.HK",
            "中国海外发展": "0688.HK",
        }
    },
    "🎰 博彩文旅": {
        "desc": "澳门博彩、旅游酒店",
        "stocks": {
            "银河娱乐": "0027.HK",
            "金沙中国": "1928.HK",
            "永利澳门": "1128.HK",
        }
    },
    "🏗️ 半导体芯片": {
        "desc": "芯片设计、制造、封测",
        "stocks": {
            "中芯国际": "0981.HK",
            "华虹半导体": "1347.HK",
            "上海复旦": "1385.HK",
        }
    },
}

# ============================================================
#  数据获取
# ============================================================

@st.cache_data(ttl=1800, show_spinner=True)
def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    获取港股日线数据
    symbol: yfinance格式代码，如 0700.HK
    period: 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=True)
def fetch_multiple_stocks(symbols: dict, period: str = "1y") -> dict:
    """
    批量获取多只股票数据
    symbols: {股票名: 代码}
    返回: {股票名: DataFrame}（跳过失败的）
    """
    results = {}
    for name, code in symbols.items():
        df = fetch_stock_data(code, period)
        if df is not None and len(df) > 10:
            results[name] = df
    return results

# ============================================================
#  计算技术指标
# ============================================================

def calc_ma(close: pd.Series, window: int) -> pd.Series:
    """计算简单移动平均线"""
    return close.rolling(window=window).mean()


def calc_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """计算布林带"""
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


# ============================================================
#  TD Sequential (DeMark Setup 9 & Countdown 13)
# ============================================================

def calc_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 TD Sequential 指标：
    - Setup 9: 连续9根K线收盘价高于/低于4根前收盘价
    - Countdown 13: Setup 完成后，收盘价 ≥/≤ 2根前最高/最低价
    返回 DataFrame 增加 td_setup / td_countdown / td_signal 列
    """
    df = df.copy()
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    td_setup = [0] * n
    td_countdown = [0] * n
    td_signal = [""] * n

    setup_state = 0
    countdown_buy = 0
    countdown_sell = 0
    buy_setup_done = False
    sell_setup_done = False

    for i in range(4, n):
        # --- Setup 逻辑 ---
        if close[i] < close[i - 4]:
            if setup_state > 0:
                setup_state += 1
            else:
                setup_state = 1
        elif close[i] > close[i - 4]:
            if setup_state < 0:
                setup_state -= 1
            else:
                setup_state = -1
        else:
            setup_state = 0

        td_setup[i] = setup_state

        # Setup 完成信号
        if setup_state == 9:
            buy_setup_done = True
            sell_setup_done = False
            countdown_buy = 0
            countdown_sell = 0
            td_signal[i] = "BS9"
        elif setup_state == -9:
            sell_setup_done = True
            buy_setup_done = False
            countdown_buy = 0
            countdown_sell = 0
            td_signal[i] = "SS9"
        elif abs(setup_state) < 9 and abs(setup_state) >= 1:
            if setup_state > 0:
                td_signal[i] = f"BS{setup_state}"
            else:
                td_signal[i] = f"SS{abs(setup_state)}"

        # --- Countdown 逻辑 ---
        if buy_setup_done and i >= 2:
            if close[i] <= low[i - 2]:
                countdown_buy += 1
                if countdown_buy == 13:
                    td_signal[i] = "BC13"
                    countdown_buy = 0
                    buy_setup_done = False
                else:
                    td_signal[i] = f"BC{countdown_buy}"

        if sell_setup_done and i >= 2:
            if close[i] >= high[i - 2]:
                countdown_sell += 1
                if countdown_sell == 13:
                    td_signal[i] = "SC13"
                    countdown_sell = 0
                    sell_setup_done = False
                else:
                    td_signal[i] = f"SC{countdown_sell}"

        td_countdown[i] = countdown_buy if countdown_buy else -countdown_sell

    df["td_setup"] = td_setup
    df["td_countdown"] = td_countdown
    df["td_signal"] = td_signal
    return df

# ============================================================
#  Plotly 图表构建
# ============================================================

def calc_rrg_series(df: pd.DataFrame, benchmark_df: pd.DataFrame, window: int = 10):
    """
    计算单只股票的 RS-Ratio 和 RS-Momentum 时间序列（标准 JdK）
    返回 (rs_ratio_series, rs_momentum_series)，长度与 df 一致
    """
    n = min(len(df), len(benchmark_df))
    close = df["Close"].values[-n:]
    bench_close = benchmark_df["Close"].values[-n:]

    rs_raw = close / bench_close
    rs_sma = pd.Series(rs_raw).rolling(window=window).mean().values

    rs_ratio_raw = rs_raw / rs_sma
    rs_ratio = pd.Series(rs_ratio_raw).rolling(window=window).mean().values

    rs_ratio_sma = pd.Series(rs_ratio).rolling(window=window).mean().values
    rs_mom_raw = rs_ratio / rs_ratio_sma
    rs_momentum = pd.Series(rs_mom_raw).rolling(window=window).mean().values

    # 前面填充 NaN 使长度与 df 一致
    pad = len(df) - n
    rs_ratio_full = np.concatenate([np.full(pad, np.nan), rs_ratio]) * 100
    rs_momentum_full = np.concatenate([np.full(pad, np.nan), rs_momentum]) * 100

    return rs_ratio_full, rs_momentum_full


def build_kline_chart(df: pd.DataFrame, stock_name: str, symbol: str,
                      show_ma: bool = True, show_boll: bool = False,
                      show_volume: bool = True, show_td: bool = False,
                      show_rrg: bool = False, benchmark_df: pd.DataFrame = None) -> go.Figure:
    """
    构建单只股票的K线图表（Plotly）
    返回 plotly Figure 对象
    """
    # 子图行数与布局
    n_rows = 1  # K线
    if show_volume:
        n_rows += 1
    if show_rrg:
        n_rows += 2  # RS-Ratio + RS-Momentum

    if show_rrg and show_volume:
        row_heights = [0.40, 0.12, 0.12, 0.10]
        row_kline, row_rs_ratio, row_rs_mom, row_vol = 1, 2, 3, 4
    elif show_rrg:
        row_heights = [0.55, 0.15, 0.15]
        row_kline, row_rs_ratio, row_rs_mom = 1, 2, 3
    elif show_volume:
        row_heights = [0.75, 0.25]
        row_kline, row_vol = 1, 2
    else:
        row_heights = [1.0]
        row_kline = 1

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    dates = df["Date"]
    close = df["Close"]

    # 1. K线
    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        showlegend=False,
    ), row=row_kline, col=1)

    # 2. 均线
    if show_ma:
        for window, color in [(5, "#FF6D00"), (10, "#2196F3"), (20, "#9C27B0")]:
            ma = calc_ma(close, window)
            fig.add_trace(go.Scatter(
                x=dates, y=ma, mode="lines",
                line=dict(color=color, width=1),
                showlegend=False, hovertemplate=f"MA{window}: %{{y:.2f}}<extra></extra>",
            ), row=row_kline, col=1)

    # 3. 布林带
    if show_boll:
        mid, upper, lower = calc_bollinger(close)
        fig.add_trace(go.Scatter(
            x=dates, y=upper, mode="lines",
            line=dict(color="rgba(255,152,0,0.5)", width=1, dash="dash"),
            showlegend=False, hovertemplate="BOLL-UP: %{y:.2f}<extra></extra>",
        ), row=row_kline, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=lower, mode="lines",
            line=dict(color="rgba(255,152,0,0.5)", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(255,152,0,0.08)",
            showlegend=False, hovertemplate="BOLL-DN: %{y:.2f}<extra></extra>",
        ), row=row_kline, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=mid, mode="lines",
            line=dict(color="#FF9800", width=1),
            showlegend=False, hovertemplate="BOLL-MID: %{y:.2f}<extra></extra>",
        ), row=row_kline, col=1)

    # 4. TD Sequential — 仅标注 9 和 13
    if show_td:
        df_td = calc_td_sequential(df)
        buy_x, buy_y, buy_text = [], [], []
        sell_x, sell_y, sell_text = [], [], []
        for i in range(len(df_td)):
            sig = df_td.iloc[i]["td_signal"]
            if not sig:
                continue
            is_setup9 = sig in ("BS9", "SS9")
            is_cd13 = sig in ("BC13", "SC13")
            if not is_setup9 and not is_cd13:
                continue

            d = df_td.iloc[i]["Date"]
            label = "9" if is_setup9 else "13"

            if sig.startswith("B"):
                # 买入 — 标在K线下方
                buy_x.append(d)
                buy_y.append(float(df_td.iloc[i]["Low"]))
                buy_text.append(label)
            else:
                # 卖出 — 标在K线上方
                sell_x.append(d)
                sell_y.append(float(df_td.iloc[i]["High"]))
                sell_text.append(label)

        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y, mode="text",
                text=buy_text, textposition="bottom center",
                textfont=dict(size=13, color="#4CAF50", family="Arial Black"),
                showlegend=False, hovertemplate="Buy TD %{text}<extra></extra>",
            ), row=row_kline, col=1)
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y, mode="text",
                text=sell_text, textposition="top center",
                textfont=dict(size=13, color="#F44336", family="Arial Black"),
                showlegend=False, hovertemplate="Sell TD %{text}<extra></extra>",
            ), row=row_kline, col=1)

    # 5. 成交量
    if show_volume:
        colors_vol = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=dates, y=df["Volume"], marker_color=colors_vol,
            showlegend=False, hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=row_vol, col=1)

    # 6. RS-Ratio & RS-Momentum 子图
    if show_rrg and benchmark_df is not None:
        rs_ratio_vals, rs_mom_vals = calc_rrg_series(df, benchmark_df)

        # RS-Ratio
        fig.add_trace(go.Scatter(
            x=dates, y=rs_ratio_vals, mode="lines",
            line=dict(color="#2196F3", width=1.5),
            showlegend=False, hovertemplate="RS-Ratio: %{y:.2f}<extra></extra>",
        ), row=row_rs_ratio, col=1)
        # RS-Ratio 100基准线
        fig.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1,
                       row=row_rs_ratio, col=1)
        # RS-Ratio > 100 区域着色
        fig.add_trace(go.Scatter(
            x=dates, y=np.where(rs_ratio_vals >= 100, rs_ratio_vals, 100),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row_rs_ratio, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=np.full(len(dates), 100.0),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(76,175,80,0.12)",
        ), row=row_rs_ratio, col=1)
        fig.update_yaxes(
            title_text="RS-Ratio", title_font=dict(size=10, color="#2196F3"),
            showgrid=True, gridcolor="#2B2B43",
            row=row_rs_ratio, col=1,
        )

        # RS-Momentum
        fig.add_trace(go.Scatter(
            x=dates, y=rs_mom_vals, mode="lines",
            line=dict(color="#FF9800", width=1.5),
            showlegend=False, hovertemplate="RS-Mom: %{y:.2f}<extra></extra>",
        ), row=row_rs_mom, col=1)
        # RS-Momentum 100基准线
        fig.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1,
                       row=row_rs_mom, col=1)
        # RS-Momentum > 100 区域着色
        fig.add_trace(go.Scatter(
            x=dates, y=np.where(rs_mom_vals >= 100, rs_mom_vals, 100),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row_rs_mom, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=np.full(len(dates), 100.0),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(76,175,80,0.12)",
        ), row=row_rs_mom, col=1)
        fig.update_yaxes(
            title_text="RS-Mom", title_font=dict(size=10, color="#FF9800"),
            showgrid=True, gridcolor="#2B2B43",
            row=row_rs_mom, col=1,
        )

    # 布局
    latest_price = float(df.iloc[-1]["Close"])
    prev_price = float(df.iloc[-2]["Close"]) if len(df) > 1 else latest_price
    chg = (latest_price - prev_price) / prev_price * 100
    sign = "▲" if chg >= 0 else "▼"
    title_text = f"{stock_name} {symbol} — {latest_price:.2f} {sign} {chg:+.2f}%"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#d1d4dc"),
                   x=0.01, xanchor="left"),
        height=550 if show_rrg else 460,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font_color="#d1d4dc",
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        xaxis=dict(type="category", nticks=15, showgrid=True, gridcolor="#2B2B43"),
        yaxis=dict(showgrid=True, gridcolor="#2B2B43", side="right"),
    )
    if show_volume:
        fig.update_yaxes(showgrid=True, gridcolor="#2B2B43", row=row_vol, col=1)

    # 隐藏非交易日的空隙
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig


def build_comparison_chart(data_dict: dict) -> go.Figure:
    """
    构建行业对比归一化走势图（以首日为基准100）
    """
    fig = go.Figure()
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#FFEB3B", "#E91E63"]

    for idx, (name, df) in enumerate(data_dict.items()):
        if df is None or len(df) < 2:
            continue
        close = df["Close"].values
        base = close[0] if close[0] != 0 else 1
        normalized = (close / base) * 100
        fig.add_trace(go.Scatter(
            x=df["Date"], y=normalized, mode="lines",
            name=name, line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        height=450,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font_color="#d1d4dc",
        margin=dict(l=50, r=20, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor="#2B2B43"),
        yaxis=dict(showgrid=True, gridcolor="#2B2B43", side="right",
                   title="归一化指数"),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# ============================================================
#  RRG (Relative Rotation Graph) 计算
# ============================================================

def calc_rrg(data_dict: dict, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算RRG图表数据（标准JdK方法）：
    X轴 = JdK RS-Ratio  = SMA(RS / SMA(RS,N), N)
    Y轴 = JdK RS-Momentum = SMA(RS-Ratio / SMA(RS-Ratio,N), N)
    Momentum与Ratio计算逻辑完全对称，Momentum>1即Ratio在加速上升
    以恒指为基准，计算各股的相对强度
    返回含 sector / stock / rs_ratio / rs_momentum 的 DataFrame
    """
    bench_close = benchmark_df["Close"].values
    results = []

    for sector_tag, stocks_data in data_dict.items():
        for stock_name, df in stocks_data.items():
            if df is None or len(df) < 30:
                continue
            close = df["Close"].values
            n = min(len(close), len(bench_close))

            # 1. 原始相对强度 RS = close / benchmark_close
            rs_raw = close[-n:] / bench_close[-n:]

            # 2. RS 的 10日 SMA
            window = 10
            if len(rs_raw) < window:
                continue
            rs_sma = pd.Series(rs_raw).rolling(window=window).mean().values

            # 3. JdK RS-Ratio = RS / RS的10日SMA 再做10日SMA
            rs_ratio_raw = rs_raw / rs_sma
            rs_ratio = pd.Series(rs_ratio_raw).rolling(window=window).mean().values

            # 4. JdK RS-Momentum = RS-Ratio / SMA(RS-Ratio, N) 再做SMA平滑
            #    与 RS-Ratio 计算逻辑对称：RS-Ratio 是 RS/RS_SMA，Momentum 是 Ratio/Ratio_SMA
            rs_ratio_sma = pd.Series(rs_ratio).rolling(window=window).mean().values
            rs_mom_raw = rs_ratio / rs_ratio_sma
            rs_momentum = pd.Series(rs_mom_raw).rolling(window=window).mean().values

            # 取最后一个有效值
            valid_idx = -1
            for k in range(len(rs_ratio) - 1, -1, -1):
                if not (np.isnan(rs_ratio[k]) or np.isnan(rs_momentum[k])):
                    valid_idx = k
                    break
            if valid_idx < 0:
                continue

            # 标准化：以1.0为中心
            final_rs_ratio = round(float(rs_ratio[valid_idx]) * 100, 2)
            final_rs_momentum = round(float(rs_momentum[valid_idx]) * 100, 2)

            # 提取纯行业名（去掉emoji）
            sector_name = sector_tag.split(" ", 1)[1] if " " in sector_tag else sector_tag

            results.append({
                "sector": sector_name,
                "sector_tag": sector_tag,
                "stock": stock_name,
                "rs_ratio": final_rs_ratio,
                "rs_momentum": final_rs_momentum,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


def build_rrg_chart(rrg_df: pd.DataFrame) -> go.Figure:
    """
    构建RRG图表
    四象限：领先(RS>100, Mom>0) / 加强(RS<100, Mom>0) / 滞后(RS<100, Mom<0) / 减弱(RS>100, Mom<0)
    """
    if rrg_df.empty:
        return None

    fig = go.Figure()

    # 四象限背景色
    quadrant_colors = {
        "领先": "rgba(76,175,80,0.06)",    # 绿 — RS>100, Mom>0
        "加强": "rgba(33,150,243,0.06)",    # 蓝 — RS<100, Mom>0
        "滞后": "rgba(244,67,54,0.06)",     # 红 — RS<100, Mom<0
        "减弱": "rgba(255,152,0,0.06)",     # 橙 — RS>100, Mom<0
    }

    x_min, x_max = rrg_df["rs_ratio"].min() - 2, rrg_df["rs_ratio"].max() + 2
    y_min, y_max = rrg_df["rs_momentum"].min() - 2, rrg_df["rs_momentum"].max() + 2
    x_lo = min(x_min, 99)
    x_hi = max(x_max, 101)
    y_lo = min(y_min, -1)
    y_hi = max(y_max, 1)

    # 四象限矩形
    fig.add_shape(type="rect", x0=100, y0=0, x1=x_hi, y1=y_hi,
                  fillcolor=quadrant_colors["领先"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_lo, y0=0, x1=100, y1=y_hi,
                  fillcolor=quadrant_colors["加强"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_lo, y0=y_lo, x1=100, y1=-0,
                  fillcolor=quadrant_colors["滞后"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=y_lo, x1=x_hi, y1=-0,
                  fillcolor=quadrant_colors["减弱"], line_width=0, layer="below")

    # 十字线
    fig.add_shape(type="line", x0=x_lo, y0=0, x1=x_hi, y1=0,
                  line=dict(color="#555", width=1, dash="dash"))
    fig.add_shape(type="line", x0=100, y0=y_lo, x1=100, y1=y_hi,
                  line=dict(color="#555", width=1, dash="dash"))

    # 象限标签
    fig.add_annotation(x=(100 + x_hi) / 2, y=y_hi - 0.5, text="领先 Leading",
                       font=dict(size=13, color="#4CAF50"), showarrow=False)
    fig.add_annotation(x=(x_lo + 100) / 2, y=y_hi - 0.5, text="加强 Improving",
                       font=dict(size=13, color="#2196F3"), showarrow=False)
    fig.add_annotation(x=(x_lo + 100) / 2, y=y_lo + 0.5, text="滞后 Lagging",
                       font=dict(size=13, color="#F44336"), showarrow=False)
    fig.add_annotation(x=(100 + x_hi) / 2, y=y_lo + 0.5, text="减弱 Weakening",
                       font=dict(size=13, color="#FF9800"), showarrow=False)

    # 按行业分组绘制散点
    sector_colors = {
        "科技互联网": "#2196F3", "金融银行": "#FF9800", "消费零售": "#4CAF50",
        "医药生物": "#9C27B0", "新能源汽车": "#00BCD4", "能源资源": "#FFEB3B",
        "通信运营商": "#E91E63", "地产基建": "#795548", "博彩文旅": "#FF5722",
        "半导体芯片": "#607D8B",
    }

    for sector in rrg_df["sector"].unique():
        sub = rrg_df[rrg_df["sector"] == sector]
        color = sector_colors.get(sector, "#999")
        fig.add_trace(go.Scatter(
            x=sub["rs_ratio"], y=sub["rs_momentum"],
            mode="markers+text",
            text=sub["stock"], textposition="top center",
            textfont=dict(size=10, color=color),
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(width=1, color="#fff")),
            name=sector,
            hovertemplate="<b>%{text}</b><br>行业: " + sector +
                          "<br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        height=550,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font_color="#d1d4dc",
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=dict(
            title="JdK RS-Ratio",
            showgrid=True, gridcolor="#2B2B43",
            range=[x_lo, x_hi],
        ),
        yaxis=dict(
            title="JdK RS-Momentum",
            showgrid=True, gridcolor="#2B2B43",
            range=[y_lo, y_hi],
        ),
        legend=dict(
            font=dict(size=10, color="#d1d4dc"),
            bgcolor="rgba(0,0,0,0.3)",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig

def calc_sector_stats(data_dict: dict) -> pd.DataFrame:
    """计算行业龙头股票的涨跌统计"""
    rows = []
    for name, df in data_dict.items():
        if df is None or len(df) < 2:
            continue
        latest = float(df.iloc[-1]["Close"])
        first = float(df.iloc[0]["Close"])
        high = float(df["High"].max())
        low = float(df["Low"].min())
        change_pct = (latest - first) / first * 100
        avg_vol = float(df["Volume"].mean())

        if len(df) >= 5:
            week_ago = float(df.iloc[-5]["Close"])
            week_chg = (latest - week_ago) / week_ago * 100
        else:
            week_chg = 0

        rows.append({
            "股票": name,
            "最新价": round(latest, 2),
            "区间涨跌%": round(change_pct, 2),
            "近5日涨跌%": round(week_chg, 2),
            "区间最高": round(high, 2),
            "区间最低": round(low, 2),
            "振幅%": round((high - low) / low * 100, 2),
            "日均成交额(亿)": round(avg_vol * latest / 1e8, 2),
        })
    return pd.DataFrame(rows)

# ============================================================
#  Streamlit 页面
# ============================================================

st.set_page_config(page_title="港股行业龙头", page_icon="🇭🇰", layout="wide")

st.title("🇭🇰 港股行业龙头K线可视化")
st.markdown("按行业板块展示港股龙头股票K线走势、技术指标与行业对比")

# ---- 侧边栏 ----
st.sidebar.header("📊 行业选择")
sector_names = list(HK_SECTORS.keys())
selected_sector = st.sidebar.selectbox("选择行业板块", sector_names, index=0)

sector_info = HK_SECTORS[selected_sector]
st.sidebar.caption(f"💡 {sector_info['desc']}")

stock_names = list(sector_info["stocks"].keys())
selected_stocks = st.sidebar.multiselect(
    "选择股票（可多选）", stock_names,
    default=stock_names[:4],
    help="选择要展示K线的股票，最多6只"
)

st.sidebar.header("⏱️ 时间范围")
period_map = {
    "1个月": "1mo", "3个月": "3mo", "6个月": "6mo",
    "1年": "1y", "2年": "2y", "5年": "5y",
}
period_label = st.sidebar.selectbox("K线周期", list(period_map.keys()), index=3)
period = period_map[period_label]

st.sidebar.header("📈 技术指标")
show_ma = st.sidebar.checkbox("均线 (MA5/10/20)", value=True)
show_boll = st.sidebar.checkbox("布林带 (BOLL20)", value=False)
show_td = st.sidebar.checkbox("TD Sequential (Setup9/CD13)", value=False)
show_rrg = st.sidebar.checkbox("RS-Ratio / RS-Momentum", value=False)
show_volume = st.sidebar.checkbox("成交量", value=True)

view_mode = st.sidebar.radio("展示模式", ["个股K线", "行业对比走势"], index=0)

# ---- 获取数据 ----
if not selected_stocks:
    st.warning("👈 请在左侧选择至少一只股票")
    st.stop()

selected_symbols = {name: sector_info["stocks"][name] for name in selected_stocks}

# 获取恒指基准（RS指标需要）
hsi_df = fetch_stock_data("^HSI", period) if show_rrg else None
if show_rrg and hsi_df is None:
    st.sidebar.warning("恒指基准数据获取失败，RS指标无法显示")
    show_rrg = False

with st.spinner(f"正在获取 {len(selected_stocks)} 只股票数据..."):
    data_dict = fetch_multiple_stocks(selected_symbols, period)

if not data_dict:
    st.error("数据获取失败，请检查网络或稍后重试")
    st.stop()

failed = set(selected_stocks) - set(data_dict.keys())
if failed:
    st.warning(f"以下股票数据获取失败，已跳过：{', '.join(failed)}")

# ---- 行业涨跌统计 ----
st.subheader(f"{selected_sector} — 龙头股涨跌概览")
stats_df = calc_sector_stats(data_dict)

def color_change(val):
    if isinstance(val, (int, float)):
        color = "#4CAF50" if val >= 0 else "#F44336"
        return f"color: {color}"
    return ""

styled = stats_df.style.map(color_change, subset=["区间涨跌%", "近5日涨跌%"])
st.dataframe(styled, use_container_width=True, hide_index=True, height=min(len(stats_df) * 45 + 40, 350))

# ---- 图表展示 ----
if view_mode == "个股K线":
    cols_per_row = 2
    stock_list = list(data_dict.items())

    for row_start in range(0, len(stock_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row_start + col_idx
            if idx >= len(stock_list):
                break
            name, df = stock_list[idx]
            symbol = selected_symbols[name]

            with cols[col_idx]:
                fig = build_kline_chart(df, name, symbol, show_ma, show_boll, show_volume, show_td, show_rrg, hsi_df)
                st.plotly_chart(fig, use_container_width=True, config=dict(
                    displayModeBar=False, scrollZoom=True,
                ))

else:
    # 行业对比走势模式
    st.subheader(f"{selected_sector} — 龙头股归一化走势对比")
    st.caption("以区间首日收盘价为基准100，对比各股相对涨跌幅")

    comp_fig = build_comparison_chart(data_dict)
    st.plotly_chart(comp_fig, use_container_width=True, config=dict(
        displayModeBar=False, scrollZoom=True,
    ))

    # 涨跌幅排名柱状图
    st.subheader("涨跌幅排名")
    if not stats_df.empty:
        rank_df = stats_df.sort_values("区间涨跌%", ascending=True)
        bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in rank_df["区间涨跌%"]]
        bar_fig = go.Figure(go.Bar(
            x=rank_df["区间涨跌%"], y=rank_df["股票"],
            orientation="h", marker_color=bar_colors,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        bar_fig.update_layout(
            height=300, plot_bgcolor="#131722", paper_bgcolor="#131722",
            font_color="#d1d4dc", margin=dict(l=80, r=20, t=10, b=30),
            xaxis=dict(showgrid=True, gridcolor="#2B2B43", title="涨跌幅%"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(bar_fig, use_container_width=True, config=dict(displayModeBar=False))

# ---- 全行业 RRG 图表 ----
st.markdown("---")
st.subheader("🔍 全行业龙头 RRG 相对旋转图")
st.caption("以恒生指数为基准，X轴=RS-Ratio(相对强度)，Y轴=RS-Momentum(强度动量)")

with st.spinner("正在获取全行业数据并计算RRG指标..."):
    # 获取恒指作为基准
    hsi_df = fetch_stock_data("^HSI", "3mo")
    if hsi_df is not None and len(hsi_df) > 30:
        # 收集所有行业龙头数据
        all_sector_data = {}
        progress = st.progress(0)
        total_sectors = len(HK_SECTORS)

        for si, (s_name, s_info) in enumerate(HK_SECTORS.items()):
            top_stocks = dict(list(s_info["stocks"].items())[:2])
            s_data = fetch_multiple_stocks(top_stocks, "3mo")
            if s_data:
                all_sector_data[s_name] = s_data
            progress.progress((si + 1) / total_sectors)

        rrg_df = calc_rrg(all_sector_data, hsi_df)

        if not rrg_df.empty:
            rrg_fig = build_rrg_chart(rrg_df)
            if rrg_fig:
                st.plotly_chart(rrg_fig, use_container_width=True, config=dict(
                    displayModeBar=False, scrollZoom=True,
                ))
            # 同时展示数据表
            with st.expander("📊 查看RRG原始数据", expanded=False):
                display_df = rrg_df[["sector", "stock", "rs_ratio", "rs_momentum"]].copy()
                display_df.columns = ["行业", "股票", "RS-Ratio", "RS-Momentum"]
                def _rrg_style(val):
                    if isinstance(val, (int, float)):
                        return "color: #4CAF50" if val >= 100 else "color: #F44336" if val < 100 else ""
                    return ""
                styled_rrg = display_df.style.map(_rrg_style, subset=["RS-Ratio"])
                st.dataframe(styled_rrg, use_container_width=True, hide_index=True)
                st.caption("RS-Ratio > 100 表示跑赢基准，< 100 表示跑输基准")
        else:
            st.warning("数据不足，无法生成RRG图表")
    else:
        st.warning("恒生指数基准数据获取失败，无法生成RRG图表")

# ---- 底部信息 ----
st.markdown("---")
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.caption("📌 数据来源：Yahoo Finance | 仅供参考，不构成投资建议")
with col_info2:
    st.caption(f"📅 数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
