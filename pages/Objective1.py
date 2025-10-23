import pandas as pd
import streamlit as st
import plotly.express as px
st.title("Comparative Analysis of Fund Performance")
st.markdown("""
The objective of this chapter is to conduct an in-depth comparative analysis of five leading ELSS mutual funds in India over the period 2020–2024. 
By examining the interplay of NAV growth, Assets Under Management (AUM) dynamics, and multi-period return performances, 
this chapter seeks to unveil critical insights into fund strategies, management efficiency, and investor behavior patterns. 
Rather than treating performance metrics in isolation, a cross-linked approach is adopted to demonstrate how various factors—such as fund size, risk-taking ability, and sectoral exposure—collectively influence investor outcomes. 
            """)
st.subheader("NAV GROWTH ANALYSIS")
st.markdown("""
The Net Asset Value (NAV) represents the per-unit market value of a mutual fund.
It is the price at which investors buy (purchase price) or sell (redemption price) units of a fund. 
NAV is a critical indicator of a fund's performance, reflecting the market value of all securities held by the fund, after accounting for liabilities and expenses.
A steadily growing NAV over time indicates efficient capital appreciation and good fund management, assuming consistent reinvestments and strong performance of the underlying portfolio assets. Thus, NAV trends provide insights into the fund's ability to generate returns and enhance investor wealth.
""")
st.markdown("""
            The indexed NAVs for each scheme were plotted on a line graph. 
            This visualization enables the analysis of the comparative growth patterns of different funds over the study period. 
            Funds that show a steeper upward trajectory on the graph indicate stronger capital appreciation, while flatter lines suggest relatively slower growth.)
            """)