import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

# 设置页面配置
st.set_page_config(
    page_title="技术分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和描述
st.title("📈 专业级技术分析平台")
st.markdown("使用Streamlit和Lightweight Charts构建的专业技术分析工具")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 股票选择
    ticker = st.text_input("股票代码", value="AAPL")
    
    # 时间范围选择
    period_options = ["1个月", "3个月", "6个月", "1年", "2年", "5年"]
    period = st.selectbox("时间范围", period_options)
    
    # 技术指标选择
    st.subheader("技术指标")
    show_sma = st.checkbox("简单移动平均线 (SMA)", value=True)
    show_ema = st.checkbox("指数移动平均线 (EMA)", value=False)
    show_rsi = st.checkbox("相对强弱指数 (RSI)", value=True)
    show_td = st.checkbox("DeMark TD Sequential", value=True)
    
    if show_sma:
        sma_periods = st.multiselect(
            "SMA周期", 
            [5, 10, 20, 50, 100, 200],
            default=[20, 50]
        )
    
    # TD Sequential参数
    if show_td:
        st.subheader("TD Sequential参数")
        td_lookback = st.slider("回溯周期", 4, 20, 9, help="TD Sequential的计数回溯周期")
        td_count_threshold = st.slider("计数阈值", 5, 13, 9, help="触发TD Sequential信号的计数阈值")

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

# 计算DeMark TD Sequential指标
def calculate_td_sequential(df, lookback=9, count_threshold=9):
    """
    计算TD Sequential指标
    - TD Setup: 连续9根K线收盘价高于4根前的收盘价
    - TD Countdown: 在Setup完成后开始计数
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    dates = df['Date'].values
    
    n = len(df)
    
    # TD Setup计数
    td_setup = np.zeros(n, dtype=int)
    td_setup_direction = np.zeros(n, dtype=int)  # 1: bullish, -1: bearish
    
    for i in range(lookback, n):
        # 检查是否满足TD Setup条件
        bullish_setup = True
        bearish_setup = True
        
        for j in range(9):
            if i - j < 0:
                break
                
            # 牛市Setup: 收盘价高于4根K线前的收盘价
            if i - j - 4 >= 0:
                if close[i - j] <= close[i - j - 4]:
                    bullish_setup = False
            
            # 熊市Setup: 收盘价低于4根K线前的收盘价
            if i - j - 4 >= 0:
                if close[i - j] >= close[i - j - 4]:
                    bearish_setup = False
        
        if bullish_setup and td_setup[i-1] < 9:
            td_setup[i] = td_setup[i-1] + 1
            td_setup_direction[i] = 1
        elif bearish_setup and td_setup[i-1] > -9:
            td_setup[i] = td_setup[i-1] - 1
            td_setup_direction[i] = -1
        else:
            td_setup[i] = 0
            td_setup_direction[i] = 0
    
    # TD Countdown计数
    td_countdown = np.zeros(n, dtype=int)
    in_countdown = False
    countdown_start = 0
    
    for i in range(n):
        if not in_countdown:
            # 检查是否完成TD Setup (达到9)
            if abs(td_setup[i]) == 9:
                in_countdown = True
                countdown_start = i
                td_countdown[i] = 1
        else:
            # 在Countdown中
            if td_setup_direction[countdown_start] == 1:  # 牛市Countdown
                # 检查条件: 收盘价高于或等于2根K线前的最高价
                if i >= 2 and close[i] >= max(high[i-2], high[i-1]):
                    td_countdown[i] = td_countdown[i-1] + 1
                else:
                    td_countdown[i] = td_countdown[i-1]
            else:  # 熊市Countdown
                # 检查条件: 收盘价低于或等于2根K线前的最低价
                if i >= 2 and close[i] <= min(low[i-2], low[i-1]):
                    td_countdown[i] = td_countdown[i-1] + 1
                else:
                    td_countdown[i] = td_countdown[i-1]
            
            # 如果Countdown达到13，重置
            if td_countdown[i] >= 13:
                in_countdown = False
                td_countdown[i] = 0
    
    # 识别TD Sequential信号
    signals = []
    for i in range(n):
        if td_setup[i] == 9 and td_setup_direction[i] == 1:
            signals.append({
                'date': dates[i],
                'type': 'TD Setup Buy',
                'value': close[i],
                'index': i
            })
        elif td_setup[i] == -9 and td_setup_direction[i] == -1:
            signals.append({
                'date': dates[i],
                'type': 'TD Setup Sell',
                'value': close[i],
                'index': i
            })
        elif td_countdown[i] == 13:
            if td_setup_direction[countdown_start] == 1:
                signals.append({
                    'date': dates[i],
                    'type': 'TD Countdown Buy',
                    'value': close[i],
                    'index': i
                })
            else:
                signals.append({
                    'date': dates[i],
                    'type': 'TD Countdown Sell',
                    'value': close[i],
                    'index': i
                })
    
    return {
        'td_setup': td_setup,
        'td_setup_direction': td_setup_direction,
        'td_countdown': td_countdown,
        'signals': signals
    }

# 准备Lightweight Charts数据
def prepare_chart_data(df, show_sma=False, sma_periods=[], show_rsi=True, show_td=False, td_params=None):
    chart_data = {
        "candles": [],
        "volume": [],
        "sma": {},
        "rsi": [],
        "td_signals": [],
        "td_setup": [],
        "td_countdown": []
    }
    
    # 准备K线数据
    for _, row in df.iterrows():
        candle = {
            "time": row['Date'].strftime('%Y-%m-%d'),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        }
        chart_data["candles"].append(candle)
    
    # 准备成交量数据
    for _, row in df.iterrows():
        volume = {
            "time": row['Date'].strftime('%Y-%m-%d'),
            "value": float(row['Volume']),
            "color": "rgba(239, 83, 80, 0.8)" if row['Close'] < row['Open'] else "rgba(38, 166, 154, 0.8)"
        }
        chart_data["volume"].append(volume)
    
    # 准备移动平均线数据
    if show_sma:
        for period in sma_periods:
            if len(df) >= period:
                sma_values = df['Close'].rolling(window=period).mean()
                sma_series = []
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= period - 1:
                        sma_series.append({
                            "time": row['Date'].strftime('%Y-%m-%d'),
                            "value": float(sma_values.iloc[i])
                        })
                chart_data["sma"][f"SMA_{period}"] = sma_series
    
    # 准备RSI数据
    if show_rsi and len(df) > 14:
        rsi_values = calculate_rsi(df['Close'].values)
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 14:
                chart_data["rsi"].append({
                    "time": row['Date'].strftime('%Y-%m-%d'),
                    "value": float(rsi_values[i])
                })
    
    # 准备TD Sequential数据
    if show_td and td_params and len(df) > td_params['lookback']:
        td_results = calculate_td_sequential(df, td_params['lookback'], td_params['count_threshold'])
        
        # TD Setup标记
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= td_params['lookback']:
                setup_value = td_results['td_setup'][i]
                if abs(setup_value) >= 5:  # 只显示显著的Setup值
                    chart_data["td_setup"].append({
                        "time": row['Date'].strftime('%Y-%m-%d'),
                        "value": setup_value
                    })
        
        # TD Countdown标记
        for i, (_, row) in enumerate(df.iterrows()):
            countdown_value = td_results['td_countdown'][i]
            if countdown_value > 0:
                chart_data["td_countdown"].append({
                    "time": row['Date'].strftime('%Y-%m-%d'),
                    "value": countdown_value
                })
        
        # TD信号标记
        for signal in td_results['signals']:
            chart_data["td_signals"].append({
                "time": pd.to_datetime(signal['date']).strftime('%Y-%m-%d'),
                "type": signal['type'],
                "value": float(signal['value'])
            })
    
    return chart_data

# 创建Lightweight Charts配置
def create_lightweight_chart_config(chart_data, show_sma=False, sma_periods=[], show_rsi=True, show_td=False):
    # 基础配置
    chart_config = {
        "layout": {
            "textColor": 'black',
            "background": {
                "type": 'solid',
                "color": 'white'
            }
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(197, 203, 206, 0.3)'
            },
            "horzLines": {
                "color": 'rgba(197, 203, 206, 0.3)'
            }
        },
        "width": 1000,
        "height": 600
    }
    
    # 创建系列
    series = []
    
    # 添加K线系列
    series.append({
        "type": "Candlestick",
        "data": chart_data["candles"],
        "options": {
            "upColor": '#26a69a',
            "downColor": '#ef5350',
            "borderVisible": False,
            "wickUpColor": '#26a69a',
            "wickDownColor": '#ef5350',
            "priceFormat": {
                "type": 'price',
                "precision": 2,
                "minMove": 0.01
            }
        }
    })
    
    # 添加移动平均线
    if show_sma and chart_data["sma"]:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for idx, (period, sma_series) in enumerate(chart_data["sma"].items()):
            if sma_series:
                color_idx = idx % len(colors)
                series.append({
                    "type": "Line",
                    "data": sma_series,
                    "options": {
                        "color": colors[color_idx],
                        "lineWidth": 2,
                        "title": f"SMA {period.split('_')[1]}"
                    }
                })
    
    # 添加TD Sequential标记
    if show_td and chart_data["td_signals"]:
        for signal in chart_data["td_signals"]:
            if signal['type'] == 'TD Setup Buy' or signal['type'] == 'TD Countdown Buy':
                marker_color = '#4CAF50'  # 绿色表示买入信号
            else:
                marker_color = '#F44336'  # 红色表示卖出信号
            
            # 在价格上添加标记
            series.append({
                "type": "Markers",
                "data": [{
                    "time": signal['time'],
                    "position": 'aboveBar',
                    "color": marker_color,
                    "shape": 'arrowDown' if 'Sell' in signal['type'] else 'arrowUp',
                    "text": signal['type']
                }],
                "options": {
                    "markers": [{
                        "time": signal['time'],
                        "position": 'aboveBar',
                        "color": marker_color,
                        "shape": 'arrowDown' if 'Sell' in signal['type'] else 'arrowUp',
                        "text": signal['type']
                    }]
                }
            })
    
    # 创建多窗格图表
    panes = []
    
    # 主价格窗格
    price_pane = {
        "series": series
    }
    panes.append(price_pane)
    
    # 成交量窗格
    volume_pane = {
        "height": 100,
        "series": [{
            "type": "Histogram",
            "data": chart_data["volume"],
            "options": {
                "priceFormat": {
                    "type": 'volume'
                },
                "priceScaleId": 'volume'
            }
        }]
    }
    panes.append(volume_pane)
    
    # RSI窗格
    if show_rsi and chart_data["rsi"]:
        rsi_pane = {
            "height": 100,
            "series": [{
                "type": "Line",
                "data": chart_data["rsi"],
                "options": {
                    "color": '#9C27B0',
                    "lineWidth": 2,
                    "title": "RSI(14)"
                }
            }]
        }
        panes.append(rsi_pane)
    
    # TD Sequential计数窗格
    if show_td and (chart_data["td_setup"] or chart_data["td_countdown"]):
        td_pane = {
            "height": 80,
            "series": []
        }
        
        # 添加TD Setup线
        if chart_data["td_setup"]:
            td_pane["series"].append({
                "type": "Line",
                "data": chart_data["td_setup"],
                "options": {
                    "color": '#FF9800',
                    "lineWidth": 1,
                    "title": "TD Setup"
                }
            })
        
        # 添加TD Countdown线
        if chart_data["td_countdown"]:
            td_pane["series"].append({
                "type": "Line",
                "data": chart_data["td_countdown"],
                "options": {
                    "color": '#2196F3',
                    "lineWidth": 1,
                    "title": "TD Countdown"
                }
            })
        
        panes.append(td_pane)
    
    return {
        "chart": chart_config,
        "panes": panes
    }

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
            
            # 准备TD Sequential参数
            td_params = None
            if show_td:
                td_params = {
                    'lookback': td_lookback if 'td_lookback' in locals() else 9,
                    'count_threshold': td_count_threshold if 'td_count_threshold' in locals() else 9
                }
            
            # 准备图表数据
            chart_data = prepare_chart_data(
                df, 
                show_sma, 
                sma_periods if show_sma else [], 
                show_rsi, 
                show_td, 
                td_params
            )
            
            # 创建图表配置
            chart_config = create_lightweight_chart_config(
                chart_data, 
                show_sma, 
                sma_periods if show_sma else [], 
                show_rsi, 
                show_td
            )
            
            # 渲染图表
            st.subheader(f"{ticker} 技术分析图表")
            renderLightweightCharts([chart_config], key='main_chart')
            
            # 显示TD Sequential信号
            if show_td and chart_data["td_signals"]:
                st.subheader("📊 TD Sequential信号")
                recent_signals = chart_data["td_signals"][-5:]  # 显示最近5个信号
                for signal in recent_signals:
                    if 'Buy' in signal['type']:
                        st.success(f"📈 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")
                    else:
                        st.error(f"📉 {signal['time']}: {signal['type']} @ {signal['value']:.2f}")
            
            # 显示数据表格
            with st.expander("查看原始数据"):
                st.dataframe(df.tail(20))
                
            # 数据导出
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载CSV数据",
                data=csv,
                file_name=f"{ticker}_stock_data.csv",
                mime="text/csv"
            )
        else:
            st.error("无法获取股票数据，请检查股票代码是否正确")
else:
    st.info("👈 请在侧边栏设置参数，然后点击'更新图表'按钮")

# 底部信息
st.markdown("---")
st.caption("📊 数据来源: Yahoo Finance | 🚀 部署平台: Streamlit Community Cloud")
