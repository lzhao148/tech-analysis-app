import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from io import StringIO
import akshare as ak
from streamlit_lightweight_charts import renderLightweightCharts

# ================= 页面配置 =================
st.set_page_config(
    page_title="北向资金流向分析平台",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💸 北向资金流向分析平台")
st.markdown("使用 Streamlit + Lightweight Charts + AKShare 构建的北向资金历史分析终端")

# ================= 数据获取函数 =================
@st.cache_data(ttl=3600)
def get_northbound_hist():
    """
    使用 AKShare 获取北向资金历史数据（东方财富-沪深港通资金流向-沪深港通历史数据）[web:108]
    symbol=\"北向资金\" 返回字段含：日期、当日成交净买额、买入成交额、卖出成交额、累计净买额 等。
    """
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")  # [web:108]
        # 标准化字段名，方便后续处理
        df = df.rename(columns={
            "日期": "date",
            "当日成交净买额": "net_buy",
            "买入成交额": "buy_amount",
            "卖出成交额": "sell_amount",
            "累计成交净买额": "cum_net_buy"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 只保留核心字段，单位已经是“亿元人民币”[web:108]
        keep_cols = ["date", "net_buy", "buy_amount", "sell_amount"]
        if "cum_net_buy" in df.columns:
            keep_cols.append("cum_net_buy")
        df = df[keep_cols].copy()

        # 如缺少 cum_net_buy 列，则自算一个
        if "cum_net_buy" not in df.columns:
            df["cum_net_buy"] = df["net_buy"].cumsum()

        return df
    except Exception as e:
        st.error(f"获取北向资金历史数据失败: {e}")
        return None

# ================= 数据预处理为图表结构 =================
def prepare_northbound_chart_data(df: pd.DataFrame):
    """
    将北向资金 DataFrame 转为 lightweight-charts 需要的结构：
    - flow: 当日净买额柱状
    - cumulative: 累计净买额折线
    """
    chart_data = {
        "flow": [],
        "cumulative": []
    }
    if df is None or df.empty:
        return chart_data

    for _, row in df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        net = float(row["net_buy"]) if pd.notna(row["net_buy"]) else 0.0
        cum = float(row["cum_net_buy"]) if pd.notna(row["cum_net_buy"]) else 0.0

        chart_data["flow"].append({
            "time": d,
            "value": net,
            "color": "rgba(38, 166, 154, 0.8)" if net >= 0 else "rgba(239, 83, 80, 0.8)"
        })
        chart_data["cumulative"].append({
            "time": d,
            "value": cum
        })

    return chart_data

def create_northbound_charts(chart_data):
    """
    构建两个 pane：
    1）当日净买额柱状图
    2）累计净买额折线图
    """
    charts = []

    # 当日净买额
    flow_chart = {
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "grid": {
                "vertLines": {"color": "rgba(197, 203, 206, 0.3)"},
                "horzLines": {"color": "rgba(197, 203, 206, 0.3)"}
            },
            "width": 1100,
            "height": 320
        },
        "series": [{
            "type": "Histogram",
            "data": chart_data["flow"],
            "options": {
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "title": "北向资金当日成交净买额（亿元）"
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
                "title": "北向资金累计成交净买额（亿元）"
            }
        }]
    }
    charts.append(cum_chart)

    return charts

# ================= 侧边栏参数 =================
with st.sidebar:
    st.header("⚙️ 参数设置")

    period_options = ["1个月", "3个月", "6个月", "1年", "全部历史", "自定义区间"]
    period_choice = st.selectbox("时间范围", period_options, index=2)

    custom_start, custom_end = None, None
    if period_choice == "自定义区间":
        default_start = date.today() - timedelta(days=365)
        custom_start = st.date_input("开始日期", value=default_start)
        custom_end = st.date_input("结束日期", value=date.today())
        if custom_start > custom_end:
            st.error("开始日期不能晚于结束日期")
    st.markdown("---")
    st.markdown("### 📊 数据说明")
    st.markdown(
        "- 数据源：东方财富-沪深港通资金流向-沪深港通历史数据（AKShare `stock_hsgt_hist_em(symbol=\"北向资金\")`）[web:108]\n"
        "- 单位：成交净买额及买入/卖出额为**亿元人民币**。\n"
    )

# ================= 主体 =================
st.subheader("北向资金历史净买额与累计净买额")

if st.button("更新北向资金数据", type="primary"):
    with st.spinner("正在获取北向资金历史数据..."):
        df_all = get_northbound_hist()

    if df_all is None or df_all.empty:
        st.error("未获取到北向资金数据，请稍后重试或检查网络/AKShare。")
    else:
        # 根据时间范围筛选
        if period_choice != "全部历史":
            if period_choice == "1个月":
                delta = 30
            elif period_choice == "3个月":
                delta = 90
            elif period_choice == "6个月":
                delta = 180
            elif period_choice == "1年":
                delta = 365
            elif period_choice == "自定义区间":
                # 会在下面单独处理
                delta = None
            else:
                delta = None

            if period_choice == "自定义区间" and custom_start and custom_end:
                mask = (df_all["date"].dt.date >= custom_start) & (df_all["date"].dt.date <= custom_end)
                df = df_all[mask].copy()
            elif delta is not None:
                cutoff = datetime.today() - timedelta(days=delta)
                df = df_all[df_all["date"] >= cutoff].copy()
            else:
                df = df_all.copy()
        else:
            df = df_all.copy()

        if df.empty:
            st.warning("筛选后数据为空，请调整时间范围。")
        else:
            # 关键指标
            latest_row = df.iloc[-1]
            latest_net = latest_row["net_buy"]
            latest_date = latest_row["date"].strftime("%Y-%m-%d")
            period_total = df["net_buy"].sum()
            avg_net = df["net_buy"].mean()
            max_single = df["net_buy"].max()
            min_single = df["net_buy"].min()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最新净买额（亿元）", f"{latest_net:.2f}", help=f"日期：{latest_date}")
            with col2:
                st.metric("区间累计净买额（亿元）", f"{period_total:.2f}")
            with col3:
                st.metric("区间日均净买额（亿元）", f"{avg_net:.2f}")
            with col4:
                st.metric("最大 / 最小单日净买额", f"{max_single:.2f} / {min_single:.2f}")

            # 构建图表
            chart_data = prepare_northbound_chart_data(df)
            charts = create_northbound_charts(chart_data)

            st.subheader("北向资金流向图表")
            renderLightweightCharts(charts, key="northbound_charts")

            # 数据表 & 下载
            with st.expander("查看原始数据（最近 100 行）"):
                st.dataframe(df.tail(100))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下载当前区间北向资金数据 CSV",
                data=csv,
                file_name=f"northbound_flow_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
else:
    st.info("👈 在左侧选择时间范围后，点击“更新北向资金数据”按钮。")

# ================= 研究笔记 =================
st.markdown("---")
st.subheader("📝 交易 / 研究笔记")
note = st.text_area("在此记录你基于北向资金的观察、策略想法或盘后总结：", height=150)

if st.button("导出笔记为 Markdown"):
    buf = StringIO()
    buf.write("# 北向资金分析笔记\n\n")
    buf.write(f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    buf.write(f"- 时间范围选择：{period_choice}\n\n")
    buf.write("## 笔记内容\n\n")
    buf.write(note if note.strip() else "（尚未填写）\n")
    md_bytes = buf.getvalue().encode("utf-8")
    st.download_button(
        label="📄 下载 Markdown 笔记",
        data=md_bytes,
        file_name=f"northbound_note_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )
