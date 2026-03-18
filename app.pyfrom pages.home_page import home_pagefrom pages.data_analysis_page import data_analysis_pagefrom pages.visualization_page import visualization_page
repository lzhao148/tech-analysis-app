import streamlit as st
import pandas as pd
import numpy as np
from pages.home_page import home_page
from pages.data_analysis_page import data_analysis_page
from pages.visualization_page import visualization_page


# 设置页面配置
st.set_page_config(
    page_title="技术分析应用",
    page_icon="📊",
    layout="wide"
)


pages = [
    st.Page(home_page, title="主页", icon="🏠"),
    st.Page(data_analysis_page, title="数据分析", icon="📊"),
    st.Page(visualization_page, title="可视化", icon="📈"),
    ##st.Page(settings_page, title="设置", icon="⚙️")
]


# 显示导航
pg = st.navigation(pages)
pg.run()
