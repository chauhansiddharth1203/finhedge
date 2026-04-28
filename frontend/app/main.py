"""
main.py - FinHedge app router.
"""
import streamlit as st

st.set_page_config(
    page_title="FinHedge AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
section[data-testid="stSidebar"] { border-right: 1px solid #1e293b !important; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/home.py",          title="Home",       default=True),
    st.Page("pages/01_Prediction.py", title="Prediction"),
    st.Page("pages/02_Hedging.py",    title="Hedging"),
    st.Page("pages/03_Pipeline.py",   title="Pipeline"),
    st.Page("pages/04_Monitoring.py", title="Monitoring"),
    st.Page("pages/05_About.py",      title="About"),
])
pg.run()
