import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib

st.title("📊 个股技术分析")
st.markdown("---")

# 股票代码输入
stock_code = st.text_input("请输入股票代码（如：AAPL, GOOGL, 600000.SS）", "AAPL")

# 时间范围选择
time_range = st.selectbox(
    "选择时间范围",
    ["1个月", "3个月", "6个月", "1年", "3年", "5年", "全部"]
)

# 转换时间范围为yfinance格式
time_map = {
    "1个月": "1mo",
    "3个月": "3mo", 
    "6个月": "6mo",
    "1年": "1y",
    "3年": "3y",
    "5年": "5y",
    "全部": "max"
}

# 获取股票数据
if st.button("获取数据"):
    with st.spinner("正在获取股票数据..."):
        try:
            stock = yf.Ticker(stock_code)
            hist = stock.history(period=time_map[time_range])
            
            if hist.empty:
                st.error("无法获取股票数据，请检查股票代码是否正确。")
            else:
                st.success(f"成功获取 {stock_code} 的数据！")
                
                # 显示基本信息
                st.subheader("📋 基本信息")
                st.write(f"公司名称: {stock.info.get('longName', '未知')}")
                st.write(f"行业: {stock.info.get('industry', '未知')}")
                st.write(f"当前价格: ${stock.info.get('currentPrice', '未知')}")
                
                # 显示K线图
                st.subheader("📈 K线图")
                st.line_chart(hist[['Close']], use_container_width=True)
                
                # 计算技术指标
                st.subheader("🎯 技术指标")
                
                # MACD
                macd, macdsignal, macdhist = talib.MACD(hist['Close'])
                st.line_chart(pd.DataFrame({
                    'MACD': macd,
                    'Signal': macdsignal
                }).dropna(), use_container_width=True)
                
                # RSI
                rsi = talib.RSI(hist['Close'])
                st.line_chart(rsi.dropna(), use_container_width=True)
                
                # BOLL
                upper, middle, lower = talib.BBANDS(hist['Close'])
                st.line_chart(pd.DataFrame({
                    'Upper': upper,
                    'Middle': middle,
                    'Lower': lower
                }).dropna(), use_container_width=True)
                
                # 显示买卖信号
                st.subheader("📊 买卖信号")
                last_macd = macd.iloc[-1]
                last_signal = macdsignal.iloc[-1]
                last_rsi = rsi.iloc[-1]
                
                if last_macd > last_signal and last_rsi < 30:
                    st.success("✅ 买入信号：金叉且RSI超卖")
                elif last_macd < last_signal and last_rsi > 70:
                    st.error("❌ 卖出信号：死叉且RSI超买")
                else:
                    st.info("ℹ️ 持有信号：无明确买卖信号")
                    
        except Exception as e:
            st.error(f"获取数据时出错：{e}")

















































































