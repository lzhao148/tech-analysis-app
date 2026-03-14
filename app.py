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
    page_title="南向资金流向分析平台",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💹 南向资金流向与股价分析平台")
st.markdown("使用 Streamlit + Lightweight Charts + AKShare 构建的南向资金流向分析终端")

# ================= 工具函数 =================
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
def get_southbound_flow_data(start_date, end_date):
    """获取南向资金流向数据"""
    try:
        # 使用 AKShare 获取港股通资金流向
        df = ak.stock_hk_ggt_components_em()
        return df
    except Exception as e:
        st.warning(f"AKShare 获取数据失败，尝试备用方案: {e}")
        return None

@st.cache_data(ttl=3600)
def get_southbound_daily_flow():
    """获取每日南向资金流向"""
    try:
        # 获取沪深港通资金流向数据
        df = ak.stock_em_hsgt_north_net_flow_in(symbol="港股通")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        st.warning(f"获取南向资金流向失败: {e}")
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

@st.cache_data(ttl=3600)
def get_stock_hk_ggt_holdings(symbol):
    """获取个股港股通持股数据"""
    try:
        # 获取港股通持股明细
        df = ak.stock_hk_ggt_holdings_em(symbol=symbol)
        if df is not None and not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期")
        return df
    except Exception as e:
        st.warning(f"获取港股通持股数据失败: {e}")
        return None

# ================= 数据预处理函数 =================
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

# ================= 构建图表 =================
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
    
    analysis_mode = st.selectbox(
        "分析模式",
        ["南向资金整体流向", "个股港股通持股分析", "港股通十大成交股"]
    )
    
    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options, index=2)
    
    if analysis_mode == "个股港股通持股分析":
        st.subheader("股票参数")
        ticker = st.text_input("港股代码", value="00700", help="输入5位港股代码，如00700(腾讯)、09988(阿里)")
        
        show_stock = st.checkbox("显示股价对比", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 数据说明")
    st.markdown("""
    - **南向资金**: 内地资金通过港股通投资港股
    - **净流入**: 当日买入金额 - 卖出金额
    - **持股比例**: 港股通持股占已发行股份比例
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 数据来源")
    st.markdown("- AKShare (东方财富)")
    st.markdown("- Yahoo Finance")

# ================= 主体 Tabs =================
tab1, tab2, tab3 = st.tabs(["📊 南向资金流向", "📈 个股持股分析", "🏆 十大成交股"])

# ---------- Tab 1：南向资金整体流向 ----------
with tab1:
    st.subheader("南向资金每日净流入")
    
    if st.button("更新南向资金数据", type="primary", key="flow_btn"):
        with st.spinner("正在获取南向资金数据..."):
            flow_df = get_southbound_daily_flow()
            
            if flow_df is not None and not flow_df.empty:
                # 根据时间范围筛选
                period_days = {
                    "1个月": 30,
                    "3个月": 90,
                    "6个月": 180,
                    "1年": 365,
                    "2年": 730,
                    "5年": 1825
                }
                cutoff_date = datetime.now() - timedelta(days=period_days[period])
                flow_df = flow_df[flow_df["date"] >= cutoff_date]
                
                # 关键指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    latest_flow = flow_df["north_net_flow_in"].iloc[-1]
                    st.metric("最新净流入", f"{latest_flow:.2f}亿港元")
                with col2:
                    period_total = flow_df["north_net_flow_in"].sum()
                    st.metric(f"{period}累计净流入", f"{period_total:.2f}亿港元")
                with col3:
                    avg_flow = flow_df["north_net_flow_in"].mean()
                    st.metric("日均净流入", f"{avg_flow:.2f}亿港元")
                with col4:
                    max_inflow = flow_df["north_net_flow_in"].max()
                    st.metric("最大单日净流入", f"{max_inflow:.2f}亿港元")
                
                # 准备图表数据
                chart_data = prepare_southbound_chart_data(flow_df)
                charts = create_southbound_flow_charts(chart_data, show_cumulative=True, show_stock=False)
                
                st.subheader("南向资金流向图表")
                renderLightweightCharts(charts, key="flow_chart")
                
                # 数据表
                with st.expander("查看原始数据"):
                    st.dataframe(flow_df.tail(50))
                
                csv = flow_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载南向资金数据 CSV",
                    data=csv,
                    file_name=f"southbound_flow_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("无法获取南向资金数据，请稍后重试")
    else:
        st.info("👈 点击按钮获取南向资金数据")

# ---------- Tab 2：个股港股通持股分析 ----------
with tab2:
    st.subheader("个股港股通持股变化与股价对比")
    
    if st.button("更新个股数据", type="primary", key="holdings_btn"):
        with st.spinner(f"正在获取 {ticker} 的数据..."):
            # 获取股价数据
            stock_df = get_hk_stock_data(ticker, period)
            
            # 获取港股通持股数据
            holdings_df = get_stock_hk_ggt_holdings(ticker)
            
            if stock_df is not None and not stock_df.empty:
                # 关键指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    current_price = stock_df["Close"].iloc[-1]
                    st.metric("当前股价", f"{current_price:.2f}港元")
                with col2:
                    prev_price = stock_df["Close"].iloc[-2] if len(stock_df) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    st.metric("日涨跌幅", f"{change_pct:.2f}%")
                with col3:
                    avg_volume = stock_df["Volume"].mean()
                    latest_volume = stock_df["Volume"].iloc[-1]
                    volume_ratio = latest_volume / avg_volume
                    st.metric("成交量比", f"{volume_ratio:.2f}x")
                with col4:
                    period_high = stock_df["High"].max()
                    st.metric("区间最高", f"{period_high:.2f}港元")
                
                # 准备图表数据
                chart_data = prepare_holdings_chart_data(holdings_df, stock_df)
                charts = create_holdings_charts(chart_data)
                
                st.subheader(f"{ticker} 港股通持股与股价走势")
                renderLightweightCharts(charts, key="holdings_chart")
                
                # 持股数据表
                if holdings_df is not None and not holdings_df.empty:
                    with st.expander("查看持股数据"):
                        st.dataframe(holdings_df.tail(30))
                    
                    # 持股变化统计
                    st.subheader("持股变化统计")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        latest_pct = holdings_df["持股比例"].iloc[-1] if "持股比例" in holdings_df.columns else 0
                        st.metric("最新持股比例", f"{latest_pct:.2f}%")
                    with col2:
                        if len(holdings_df) > 1 and "持股数量" in holdings_df.columns:
                            shares_change = holdings_df["持股数量"].iloc[-1] - holdings_df["持股数量"].iloc[-2]
                            st.metric("持股数量变化", f"{shares_change/1e8:.2f}亿股")
                        else:
                            st.metric("持股数量变化", "N/A")
                    with col3:
                        if len(holdings_df) > 1 and "持股比例" in holdings_df.columns:
                            pct_change = holdings_df["持股比例"].iloc[-1] - holdings_df["持股比例"].iloc[-2]
                            st.metric("持股比例变化", f"{pct_change:.4f}%")
                        else:
                            st.metric("持股比例变化", "N/A")
                else:
                    st.warning("未获取到港股通持股数据，仅显示股价走势")
                
                # 下载按钮
                col1, col2 = st.columns(2)
                with col1:
                    csv = stock_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 下载股价数据 CSV",
                        data=csv,
                        file_name=f"{ticker}_stock_data.csv",
                        mime="text/csv"
                    )
                with col2:
                    if holdings_df is not None and not holdings_df.empty:
                        csv2 = holdings_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 下载持股数据 CSV",
                            data=csv2,
                            file_name=f"{ticker}_holdings_data.csv",
                            mime="text/csv"
                        )
            else:
                st.error(f"无法获取 {ticker} 的股价数据，请检查股票代码")
    else:
        st.info("👈 点击按钮获取个股数据")

# ---------- Tab 3：港股通十大成交股 ----------
with tab3:
    st.subheader("港股通十大成交股")
    
    date_input = st.date_input("选择日期", value=datetime.now())
    
    if st.button("更新十大成交股", type="primary", key="top10_btn"):
        with st.spinner("正在获取十大成交股数据..."):
            date_str = date_input.strftime("%Y%m%d")
            top_df = get_hk_ggt_top_stocks(date_str)
            
            if top_df is not None and not top_df.empty:
                st.dataframe(top_df, use_container_width=True)
                
                csv = top_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载十大成交股数据 CSV",
                    data=csv,
                    file_name=f"hk_ggt_top10_{date_str}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"未找到 {date_input.strftime('%Y-%m-%d')} 的数据，请尝试其他日期")
    else:
        st.info("👈 选择日期后点击按钮获取数据")

# ================= 页脚 =================
st.markdown("---")
st.markdown("""
### 📝 使用说明
1. **南向资金流向**: 查看每日南向资金净流入和累计流入情况
2. **个股持股分析**: 输入港股代码，查看该股票的港股通持股变化与股价对比
3. **十大成交股**: 查看指定日期的港股通十大成交股

**常用港股代码**:
- 00700: 腾讯控股
- 09988: 阿里巴巴-SW
- 03690: 美团-W
- 09999: 网易-S
- 01810: 小米集团-W
""")

# ================= 研究笔记 =================
st.markdown("---")
st.subheader("📝 交易 / 研究笔记")
note = st.text_area("在此记录你的盘后总结 / 交易计划 / 研究发现：", height=150)
if st.button("导出为 Markdown"):
    buf = StringIO()
    buf.write(f"# 南向资金分析笔记\n\n")
    buf.write(f"- 日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    buf.write(f"- 分析模式：{analysis_mode}\n")
    buf.write(f"- 时间范围：{period}\n\n")
    buf.write("## 笔记内容\n\n")
    buf.write(note if note.strip() else "（尚未填写）\n")
    md_bytes = buf.getvalue().encode("utf-8")
    st.download_button(
        label="📄 下载 Markdown 笔记",
        data=md_bytes,
        file_name=f"southbound_note_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )
