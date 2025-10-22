# app.py
import streamlit as st

st.set_page_config(page_title="ELSS Dashboard", layout="wide")
st.image("https://upload.wikimedia.org/wikipedia/commons/5/58/Indian_Rupee_symbol.svg", width=80)  # placeholder logo
st.title("Comparative Financial Performance — ELSS Dashboard")
st.markdown(
    """
    This dashboard contains objective-wise pages.  
    Use the left sidebar (Pages) to open **Objective 1 — Historical Performance**.
    """
)
st.write("Data is expected in `data/nav_prices.csv`. See repository README for details.")