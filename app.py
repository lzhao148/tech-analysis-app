import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from io import StringIO
import akshare as ak
import yfinance as yf
from streamlit_lightweight_charts import renderLightweightCharts

# ================= 页面配置 =================
st.set_page_config(
    page_title="港股通南下资金与股价分析",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💹 港股通南下资金流向与股价走势图")
st.markdown(
    "数据源：AKShare-沪深港通历史数据（南向资金），港股股价来自 Yahoo Finance。[web:108]"
)

# ================= 数据获取函数 =================
@st.cache_data(ttl=3600)
def get_southbound_hist():
    """
    使用 AKShare 获取南向资金历史数据
    symbol=\"南向资金\"，字段含：日期、当日成交净买额、买入成交额、卖出成交额、累计成交净买额等。[web:108]
    单位：亿元人民币。
    """
    try:
        df = ak.stock_hsgt_hist_em(symbol="南向资金")  # choice: 南向资金 / 港股通沪 / 港股通深 等[web:108]
        df = df.rename(columns={
            "日期": "date",
            "当日成交净买额": "net_buy",
            "买入成交额": "buy_amount",
            "卖出成交额": "sell_amount",
            "累计成交净买额": "cum_net_buy"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df[["date", "net_buy", "buy_amount", "sell_amount", "cum_net_buy"]]
    except Exception as e:
        st.error(f"获取南向资金历史数据失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_hk_stock_price(hk_code: str, start: date, end: date):
    """
    获取港股股价：输入 4-5 位数字代码（如 00700、09988），自动转为 yfinance 代码。
    """
    try:
        code = hk_code.strip()
        if code.endswith(".HK"):
            yf_code = code
        else:
            if code.isdigit():
                if len(code) == 5:       # 09988 -> 9988.HK
                    yf_code = code[1:] + ".HK"
                elif len(code) == 4:     # 0700 -> 0700.HK
                    yf_code = code + ".HK"
                else:
                    yf_code = code + ".HK"
            else:
                yf_code = code

        df = yf.download(yf_code, start=start, end=end)
        if df.empty:
            return None
        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.warning(f"获取港股 {hk_code} 股价失败: {e}")
        return None

# ================= 数据预处理 =================
def prepare_southbound_chart_data(df: pd.DataFrame):
    """
    转为 lightweight-charts 结构：
    - flow：当日净买额柱状
    - cumulative：累计净买额折线
    """
    chart_data = {"flow": [], "cumulative": []}
    if df is None or df.empty:
        return chart_data

    for _, row in df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        net = float(row["net_buy"]) if pd.notna(row["net_buy"]) else 0.0
        cum = float(row["cum_net_buy"]) if pd.notna(row["cum_net_buy"]) else 0.0
        chart_data["flow"].append({
            "time": d,
            "value": net,
            "color": "rgba(38,166,154,0.8)" if net >= 0 else "rgba(239,83,80,0.8)"
        })
        chart_data["cumulative"].append({"time": d, "value": cum})
    return chart_data

def prepare_stock_series(df: pd.DataFrame):
    series = []
    if df is None or df.empty:
        return series
    for _, row in df.iterrows():
        series.append({
            "time": row["date"].strftime("%Y-%m-%d"),
            "value": float(row["Close"])
        })
    return series

# ================= 图表构建 =================
def create_southbound_charts(chart_data):
    charts = []

    # 当日净买额
    flow_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "grid": {
                "vertLines": {"color": "rgba(197,203,206,0.3)"},
                "horzLines": {"color": "rgba(197,203,206,0.3)"}
            },
            "width": 1100,
            "height": 320
        },
        "series": [{
            "type": "Histogram",
            "data": chart_data["flow"],
            "options": {
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "title": "南向资金当日成交净买额（亿元）"
            }
        }]
    }
    charts.append(flow_chart)

    # 累计净买额
    cum_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "width": 1100,
            "height": 260
        },
        "series": [{
            "type": "Line",
            "data": chart_data["cumulative"],
            "options": {
                "color": "#3F51B5",
                "lineWidth": 2,
                "title": "南向资金累计成交净买额（亿元）"
            }
        }]
    }
    charts.append(cum_chart)
    return charts

def create_flow_vs_stock_charts(flow_chart_data, stock_series, hk_code: str):
    charts = []

    flow_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "grid": {
                "vertLines": {"color": "rgba(197,203,206,0.3)"},
                "horzLines": {"color": "rgba(197,203,206,0.3)"}
            },
            "width": 1100,
            "height": 280
        },
        "series": [{
            "type": "Histogram",
            "data": flow_chart_data["flow"],
            "options": {
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "title": "南向资金当日成交净买额（亿元）"
            }
        }]
    }
    charts.append(flow_chart)

    if stock_series:
        stock_chart = {
            "chart": {
                "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
                "width": 1100,
                "height": 260
            },
            "series": [{
                "type": "Line",
                "data": stock_series,
                "options": {
                    "color": "#FF5722",
                    "lineWidth": 2,
                    "title": f"{hk_code} 收盘价（港元）"
                }
            }]
        }
        charts.append(stock_chart)

    return charts

# ================= 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 参数设置")

    period_options = ["1个月", "3个月", "6个月", "1年", "全部历史"]
    period_choice = st.selectbox("时间范围", period_options, index=2)

    hk_code = st.text_input(
        "港股代码（可选）",
        value="00700",
        help="4-5 位数字，不带 .HK，如 00700（腾讯）、09988（阿里）"
    )

    st.markdown("---")
    st.markdown("### 数据说明")
    st.markdown(
        "- 南向资金：AKShare `stock_hsgt_hist_em(symbol=\"南向资金\")`，单位为**亿元人民币**。[web:108][web:106]\n"
        "- 股价：Yahoo Finance 港股日 K，单位为港元。"
    )

tab1, tab2 = st.tabs(["📊 南下资金整体流向", "📈 南下资金 vs 港股股价"])

# ================= Tab1：南向资金整体 =================
with tab1:
    st.subheader("南向资金历史净买额与累计净买额")

    if st.button("更新南向资金数据", type="primary", key="btn_south"):
        with st.spinner("正在获取南向资金历史数据..."):
            df_all = get_southbound_hist()

        if df_all is None or df_all.empty:
            st.error("未获取到南向资金数据。")
        else:
            if period_choice != "全部历史":
                if period_choice == "1个月":
                    days = 30
                elif period_choice == "3个月":
                    days = 90
                elif period_choice == "6个月":
                    days = 180
                elif period_choice == "1年":
                    days = 365
                else:
                    days = 365
                cutoff = datetime.today() - timedelta(days=days)
                df = df_all[df_all["date"] >= cutoff].copy()
            else:
                df = df_all.copy()

            if df.empty:
                st.warning("筛选后数据为空，请调整时间范围。")
            else:
                latest_row = df.iloc[-1]
                latest_net = latest_row["net_buy"]
                latest_date = latest_row["date"].strftime("%Y-%m-%d")
                period_total = df["net_buy"].sum()
                avg_net = df["net_buy"].mean()
                max_single = df["net_buy"].max()
                min_single = df["net_buy"].min()

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("最新净买额（亿元）", f"{latest_net:.2f}", help=f"日期：{latest_date}")
                with c2:
                    st.metric("区间累计净买额（亿元）", f"{period_total:.2f}")
                with c3:
                    st.metric("区间日均净买额（亿元）", f"{avg_net:.2f}")
                with c4:
                    st.metric("最大 / 最小单日净买额", f"{max_single:.2f} / {min_single:.2f}")

                chart_data = prepare_southbound_chart_data(df)
                charts = create_southbound_charts(chart_data)
                st.subheader("南向资金流向图表")
                renderLightweightCharts(charts, key="southbound_charts")

                with st.expander("查看原始数据（最近 100 行）"):
                    st.dataframe(df.tail(100))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载南向资金数据 CSV",
                    data=csv,
                    file_name=f"southbound_flow_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("👈 选择时间范围后点击按钮。")

# ================= Tab2：南向资金 vs 港股股价 =================
with tab2:
    st.subheader("港股通南下资金 vs 港股股价")

    if st.button("更新南向资金 + 股价数据", type="primary", key="btn_south_stock"):
        with st.spinner("正在获取南向资金数据..."):
            df_all = get_southbound_hist()

        if df_all is None or df_all.empty:
            st.error("未获取到南向资金数据。")
        else:
            # 时间剪裁
            if period_choice != "全部历史":
                if period_choice == "1个月":
                    days = 30
                elif period_choice == "3个月":
                    days = 90
                elif period_choice == "6个月":
                    days = 180
                elif period_choice == "1年":
                    days = 365
                else:
                    days = 365
                cutoff = datetime.today() - timedelta(days=days)
                df_s = df_all[df_all["date"] >= cutoff].copy()
            else:
                df_s = df_all.copy()

            if df_s.empty:
                st.warning("南向资金筛选后为空。")
            else:
                start_d = df_s["date"].min().date()
                end_d = df_s["date"].max().date() + timedelta(days=1)

                with st.spinner(f"正在获取港股 {hk_code} 股价..."):
                    df_stock = get_hk_stock_price(hk_code, start_d, end_d)

                if df_stock is None or df_stock.empty:
                    st.warning(f"未获取到 {hk_code} 股价数据，只显示南向资金。")
                    stock_series = []
                else:
                    stock_series = prepare_stock_series(df_stock)
                    last_close = df_stock["Close"].iloc[-1]
                    prev_close = df_stock["Close"].iloc[-2] if len(df_stock) > 1 else last_close
                    pct = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(f"{hk_code} 最新收盘价（港元）", f"{last_close:.2f}")
                    with c2:
                        st.metric("日涨跌幅", f"{pct:.2f}%")
                    with c3:
                        st.metric("样本交易日数", f"{len(df_stock)}")

                chart_south = prepare_southbound_chart_data(df_s)
                charts_combo = create_flow_vs_stock_charts(chart_south, stock_series, hk_code)
                st.subheader("南向资金净买额 vs 港股收盘价")
                renderLightweightCharts(charts_combo, key="south_vs_stock")

                with st.expander("查看合并数据（最近 100 行）"):
                    if df_stock is not None and not df_stock.empty:
                        df_merge = pd.merge(
                            df_s,
                            df_stock[["date", "Close"]],
                            on="date",
                            how="left"
                        ).rename(columns={"Close": f"{hk_code}_close"})
                        st.dataframe(df_merge.tail(100))
                    else:
                        st.dataframe(df_s.tail(100))
    else:
        st.info("👈 输入港股代码后点击按钮，查看资金与股价联动。")
