import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
  st.set_page_config(page_title="数据可视化", page_icon="📊", layout="wide")

# Title
  st.title("📊 股票数据可视化分析")

  # Sample data (replace with actual data loading)
  df = pd.DataFrame({
    '日期': pd.date_range(start='2024-01-01', periods=30),
    '收盘价': [150 + i*2 + np.random.randn()*5 for i in range(30)],
      '成交量': [1000000 + np.random.randint(-100000, 100000) for i in range(30)]
              })

# Create tabs
        tab1, tab2, tab3 = st.tabs(["K线图", "趋势分析", "成交量分析"])

with tab1:
        st.subheader("股票K线图")
        # Add K-line chart here (using plotly or mplfinance)
        st.line_chart(df.set_index('日期')['收盘价'])

with tab2:
        st.subheader("价格趋势分析")
        fig = px.line(df, x='日期', y='收盘价', title='收盘价趋势')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
        st.subheader("成交量分析")
        fig = px.bar(df, x='日期', y='成交量', title='每日成交量')
        st.plotly_chart(fig, use_container_width=True)

# Add data preview
        st.sidebar.subheader("数据预览")
        st.sidebar.dataframe(df.head(10))




























