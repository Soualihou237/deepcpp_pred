import streamlit as st
import time

st.set_page_config(
    page_title="DeepCPP",
    page_icon="icon.ico",
    layout="centered",
    initial_sidebar_state="collapsed"
    
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
        visibility: hidden;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
st.image("page1.png", caption="")
st.image("page2.png", caption="")