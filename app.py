import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

BASE_URL = "https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk"

# =============== 抓取单日南向持股（Southbound） ===============
@st.cache_data(ttl=3600)
def fetch_southbound_shareholding(target_date: str) -> pd.DataFrame:
    """
    从 HKEX Southbound Shareholding 页面抓取指定日期的持股数据。
    target_date: 'YYYY/MM/DD' 字符串（注意是斜杠格式，和页面一致）。
    """
    # 这个页面的日期是在 querystring 里用 'sdate' 或类似参数控制的；
    # 但你给的链接是直接当前日，因此这里直接用 read_html 抓当前页面，
    # 并假设你手动改日期后再刷新。想自动化多日的话，可以进一步分析它的 post 提交逻辑。
    url = BASE_URL

    # 直接读取页面所有表格
    tables = pd.read_html(url, header=0)  # header=0: 第一行为表头

    # 页面上第一个真正的数据表通常是持股明细（需要你确认）
    # 根据你贴出的 HTML，表头有: ["Stock Code", "Name", "Shareholding in CCASS", "% of the total number of Issued Shares/Units"]
    df = tables[-1].copy()  # 最后一个表往往是 detail table，如果不对可以改成 tables[0] 或检查列名

    # 清洗列名
    df.columns = [c.strip() for c in df.columns]

    # 只保留有效列
    col_map = {
        "Stock Code": "stock_code",
        "Name": "name",
        "Shareholding in CCASS": "shareholding",
        "% of the total number of Issued Shares/Units": "pct_issued"
    }
    df = df[list(col_map.keys())].rename(columns=col_map)

    # 处理数据类型：去掉千分位逗号等
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["shareholding"] = (
        df["shareholding"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["pct_issued"] = (
        df["pct_issued"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    # 有些行可能是空字符串
    df["pct_issued"] = pd.to_numeric(df["pct_issued"], errors="coerce")

    # 添加日期列便于后续拼多日
    df["date"] = pd.to_datetime(target_date)

    return df

# =============== 把多日持股合成“资金流”近似 ===============
@st.cache_data(ttl=3600)
def build_flow_series(start_date: date, end_date: date):
    """
    循环抓取从 start_date 到 end_date 的每日南向持股，
    用每日日度总持股变化近似“资金流”，返回汇总 DataFrame。
    注意：如果 HKEX 只保留最近 12 个月，太早的数据会抓不到。
    """
    all_days = pd.date_range(start_date, end_date, freq="D")
    records = []

    for d in all_days:
        # HKEX: 如果是周日/公众假期，会显示最近的一个交易日数据
        # 这里简单按日循环；严谨做法是检查实际返回日期或交易日历。
        ds = d.strftime("%Y/%m/%d")
        try:
            df_day = fetch_southbound_shareholding(ds)
            if df_day.empty:
                continue
            total_shares = df_day["shareholding"].sum()
            records.append({"date": df_day["date"].iloc[0], "total_shareholding": total_shares})
        except Exception as e:
            # 避免因为某天失败中断全部循环
            continue

    if not records:
        return pd.DataFrame(columns=["date", "total_shareholding", "flow"])

    df_total = pd.DataFrame(records).drop_duplicates(subset="date").sort_values("date")
    df_total["flow"] = df_total["total_shareholding"].diff()  # 日度变化当成“净流入”
    return df_total

# =============== Streamlit Dashboard ===============
st.set_page_config(page_title="Southbound Shareholding Dashboard", page_icon="💴", layout="wide")

st.title("💴 南向资金持股 / 近似流向 Dashboard")
st.markdown(
    "数据源：HKEX `Stock Connect Southbound Shareholding Search By Date` 页面。"
    " 这里用每日总持股变化近似南向资金流向（非官方成交流量口径）。"
)

# ---- 侧边栏参数 ----
with st.sidebar:
    st.header("⚙️ 参数设置")

    today = date.today()
    default_start = today - timedelta(days=30)

    start = st.date_input("开始日期", default_start)
    end = st.date_input("结束日期", today)

    if start > end:
        st.error("开始日期不能晚于结束日期")
        st.stop()

    view_single_day = st.checkbox("只看单日持股明细", value=True)

    target_day = st.date_input("单日查看日期（用于持股明细）", today)

# ---- 单日持股明细 ----
st.subheader("📄 单日南向持股明细")

with st.spinner("正在抓取单日持股数据..."):
    target_str = target_day.strftime("%Y/%m/%d")
    try:
        df_day = fetch_southbound_shareholding(target_str)
    except Exception as e:
        st.error(f"抓取失败：{e}")
        df_day = pd.DataFrame()

if df_day is None or df_day.empty:
    st.warning("未获取到持股数据，可能是日期超出 HKEX 提供范围（仅过去 12 个月）或页面结构变动。")
else:
    st.markdown(f"**日期：{target_str}**，样本数：{len(df_day)}")
    # 显示 Top N
    top_n = st.slider("显示前 N 大持股标的：", 5, 50, 20)
    st.dataframe(
        df_day.sort_values("shareholding", ascending=False).head(top_n)[
            ["stock_code", "name", "shareholding", "pct_issued"]
        ]
    )

    # 饼图 or bar：用 lightweight-charts 画前几个标的的 shareholding 对比
    # 这里用柱状图简单展示前 N 的持股量
    bar_data = []
    top_df = df_day.sort_values("shareholding", ascending=False).head(top_n)
    for _, row in top_df.iterrows():
        bar_data.append({
            "time": row["stock_code"],  # 对于分类数据，time 也可以用字符串
            "value": float(row["shareholding"])
        })

    top_chart = [{
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "width": 900,
            "height": 400
        },
        "series": [{
            "type": "Histogram",
            "data": bar_data,
            "options": {
                "priceFormat": {"type": "price", "precision": 0},
                "color": "#26a69a"
            }
        }]
    }]
    st.markdown("**前 N 大标的持股量（按 Shareholding in CCASS）**")
    renderLightweightCharts(top_chart, key="top_holdings")

# ---- 多日“资金流”近似 ----
st.subheader("📈 多日总持股变化（资金流近似）")

with st.spinner("正在按日抓取数据并计算总持股变化...（注意 HKEX 限制为近 12 个月内）"):
    df_flow = build_flow_series(start, end)

if df_flow.empty:
    st.warning("多日数据为空，请尝试缩短时间范围或检查是否在 HKEX 保留的 12 个月范围内。")
else:
    st.markdown("**总持股量与日度变化（总持股变化视作资金净流入近似）**")

    # 折线：总持股量
    total_series = []
    for _, row in df_flow.iterrows():
        total_series.append({
            "time": row["date"].strftime("%Y-%m-%d"),
            "value": float(row["total_shareholding"])
        })

    total_chart = [{
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "grid": {
                "vertLines": {"color": "rgba(197,203,206,0.3)"},
                "horzLines": {"color": "rgba(197,203,206,0.3)"}
            },
            "width": 1000,
            "height": 350
        },
        "series": [{
            "type": "Line",
            "data": total_series,
            "options": {"color": "#2962FF", "lineWidth": 2, "title": "总持股量"}
        }]
    }]
    renderLightweightCharts(total_chart, key="total_holdings")

    # 柱状：日度变化 flow
    flow_series = []
    for _, row in df_flow.iterrows():
        flow_series.append({
            "time": row["date"].strftime("%Y-%m-%d"),
            "value": float(row["flow"]) if not np.isnan(row["flow"]) else 0.0
        })

    flow_chart = [{
        "chart": {
            "layout": {"textColor": "black", "background": {"type": "solid", "color": "white"}},
            "width": 1000,
            "height": 250
        },
        "series": [{
            "type": "Histogram",
            "data": flow_series,
            "options": {
                "priceFormat": {"type": "price", "precision": 0},
                "color": "#26a69a"
            }
        }]
    }]
    renderLightweightCharts(flow_chart, key="flow_hist")

    # 数据表 & 下载
    st.markdown("**多日汇总数据表**")
    st.dataframe(df_flow)

    csv_bytes = df_flow.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 下载多日总持股 & 变化 CSV",
        data=csv_bytes,
        file_name=f"southbound_holdings_{start}_{end}.csv",
        mime="text/csv"
    )

