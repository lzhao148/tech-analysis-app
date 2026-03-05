import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置页面配置
st.set_page_config(
    page_title="技术分析平台",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和描述
st.title("�� 专业级技术分析平台")
st.markdown("使用Streamlit和Plotly构建的实时技术分析工具")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 股票选择
    ticker = st.text_input("股票代码", value="AAPL")
    
    # 时间范围选择
    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options)
    
    # 图表类型
    chart_type = st.selectbox("图表类型", ["蜡烛图", "线图", "面积图"])
    
    # 技术指标选择
    st.subheader("技术指标")
    show_sma = st.checkbox("简单移动平均线 (SMA)", value=True)
    show_ema = st.checkbox("指数移动平均线 (EMA)", value=False)
    show_rsi = st.checkbox("相对强弱指数 (RSI)", value=True)
    
    if show_sma:
        sma_periods = st.multiselect(
            "SMA周期", 
            [5, 10, 20, 50, 100, 200],
            default=[20, 50]
        )

# 获取股票数据
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
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None

# 计算RSI
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    
    return rsi

# 创建多窗格图表
def create_multi_pane_chart(df, show_sma=False, sma_periods=[], show_rsi=True):
    # 创建子图
    rows = 2 if not show_rsi else 3
    row_heights = [0.6, 0.2] if not show_rsi else [0.5, 0.2, 0.3]
    titles = ('价格走势', '成交量') if not show_rsi else ('价格走势', '成交量', 'RSI指标')
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=titles,
        row_heights=row_heights
    )
    
    # 添加蜡烛图
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='价格'
        ),
        row=1, col=1
    )
    
    # 添加移动平均线
    if show_sma:
        for period in sma_periods:
            if len(df) >= period:
                sma_values = df['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=sma_values,
                        name=f'SMA {period}',
                        line=dict(width=1, dash='dash')
                    ),
                    row=1, col=1
                )
    
    # 添加成交量
    colors = ['red' if row['Close'] < row['Open'] else 'green' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='成交量',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # 添加RSI
    if show_rsi and len(df) > 14:
        rsi_values = calculate_rsi(df['Close'].values)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=rsi_values,
                name='RSI(14)',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # 添加RSI超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     row=3, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     row=3, col=1, opacity=0.5)
    
    # 更新布局
    fig.update_layout(
        title=f'{ticker} 技术分析',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=800 if show_rsi else 600,
        showlegend=True
    )
    
    if show_rsi:
        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    return fig

# 主界面
if st.sidebar.button("更新图表", type="primary"):
    with st.spinner("正在获取数据..."):
        df = get_stock_data(ticker, period)
        
        if df is not None and not df.empty:
            # 显示关键指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = df['Close'].iloc[-1]
                st.metric("当前价格", f"{current_price:.2f}")
            
            with col2:
                prev_price = df['Close'].iloc[-2]
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100
                st.metric("日变化", f"{change:.2f}", f"{pct_change:.2f}%")
            
            with col3:
                volume_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
                st.metric("成交量比", f"{volume_ratio:.2f}x")
            
            with col4:
                if len(df) >= 20:
                    sma_20 = df['Close'].rolling(20).mean().iloc[-1]
                    st.metric("20日SMA", f"{sma_20:.2f}")
            
            # 显示图表
            fig = create_multi_pane_chart(df, show_sma, sma_periods, show_rsi)
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示数据表格
            with st.expander("查看原始数据"):
                st.dataframe(df.tail(20))
                
            # 数据导出
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="�� 下载CSV数据",
                data=csv,
                file_name=f"{ticker}_stock_data.csv",
                mime="text/csv"
            )
else:
            st.error("无法获取股票数据，请检查股票代码是否正确")
# else:
#     st.info("�� 请在侧边栏设置参数，然后点击'更新图表'按钮")

# 底部信息
st.markdown("---")
st.caption("�� 数据来源: Yahoo Finance | �� 部署平台: Streamlit Community Cloud")
