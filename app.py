import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="技术分析应用",
    page_icon="📊",
    layout="wide"
)

# 创建导航菜单
pages = {
    "主页": st.Page(
        lambda: st.title("主页"),
        title="主页",
        icon="🏠"
    ),
    "数据分析": st.Page(
        lambda: st.title("数据分析"),
        title="数据分析",
        icon="📈"
    ),
    "可视化": st.Page(
        lambda: st.title("可视化图表"),
        title="可视化",
        icon="📊"
    ),
    "设置": st.Page(
        lambda: st.title("应用设置"),
        title="设置",
        icon="⚙️"
    )
}

# 显示导航
pg = st.navigation(pages.values())
pg.run()
