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

pg = st.navigation([
    st.Page("pages/home.py",          title="Home",       default=True),
    st.Page("pages/01_Prediction.py", title="Prediction"),
    st.Page("pages/02_Hedging.py",    title="Hedging"),
    st.Page("pages/03_Pipeline.py",   title="Pipeline"),
    st.Page("pages/04_Monitoring.py", title="Monitoring"),
    st.Page("pages/05_About.py",      title="About"),
])
pg.run()
