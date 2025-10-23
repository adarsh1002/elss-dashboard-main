import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("Portfolio Analysis")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Objective 4: Portfolio Composition and Allocation Analysis</h4>

  <p>
  In mutual fund analysis, <b>portfolio composition</b> plays a pivotal role in shaping the 
  <b>risk–return characteristics</b> of a scheme. 
  For <b>Equity Linked Savings Schemes (ELSS)</b>, which primarily invest in equities, 
  understanding the underlying portfolio is essential for interpreting historical performance 
  and anticipating future outcomes. 
  This chapter examines the <b>portfolio strategies</b> adopted by the top five ELSS mutual funds in India, 
  analyzing their allocation patterns across both <b>market capitalization categories</b> 
  (large-cap, mid-cap, and small-cap) and <b>industry sectors</b>.
  </p>

  <p>
  The <b>market capitalization allocation</b> of a fund reflects its strategic positioning and risk appetite. 
  A scheme with higher exposure to <b>small-cap equities</b> indicates an aggressive growth orientation 
  with potentially higher returns but increased volatility. 
  Conversely, funds tilted toward <b>large-cap stocks</b> suggest a more conservative and stable approach. 
  Similarly, <b>sectoral allocation patterns</b> reveal thematic preferences or concentration risks — 
  such as an overexposure to specific industries — that can materially influence a fund’s performance 
  across different phases of the market cycle.
  </p>

  <p>
  Another important aspect of portfolio analysis is the <b>impact of fund management</b>. 
  Changes in fund managers often result in shifts in portfolio construction, investment philosophy, and risk management. 
  Evaluating the <b>consistency, experience, and strategy</b> of fund managers helps in understanding 
  the qualitative dimensions influencing fund behavior and investor confidence.
  </p>

  <p>
  By integrating these factors, this analysis provides a holistic perspective on 
  <b>how fund managers allocate investor capital</b>, 
  how those allocations evolve over time, and what these shifts imply for future performance, 
  risk exposure, and long-term investor outcomes.
  </p>

  <h5>Methodology</h5>

  <p>
  The portfolio composition analysis is based on <b>half-yearly portfolio disclosures</b> 
  obtained from official AMC websites, covering the period <b>2020 to 2024</b>. 
  The study focuses on three core analytical dimensions:
  </p>

  <ul>
    <li><b>1. Market Capitalization Allocation:</b> 
    Data on fund-wise exposure to <b>large-cap</b>, <b>mid-cap</b>, and <b>small-cap equities</b> 
    was compiled and analyzed to understand asset allocation strategies and their evolution over time.</li>

    <li><b>2. Sectoral Allocation:</b> 
    Sector-wise investment data was aggregated to identify <b>concentration patterns</b>, 
    <b>sector rotation</b>, and <b>thematic tilts</b>. 
    Visualization tools such as <b>stacked bar charts</b> and <b>heatmaps</b> were employed 
    to illustrate diversification and dominance across key sectors.</li>

    <li><b>3. Fund Management Trends:</b> 
    Changes in fund management and their possible influence on allocation and performance 
    were qualitatively assessed to understand consistency and strategic alignment.</li>
  </ul>

  <p>
  All data analysis and visualization were conducted using <b>Python</b> libraries 
  such as <b>Pandas</b>, <b>Plotly</b>, and <b>Matplotlib</b>, 
  supplemented with <b>Excel</b> for data compilation and verification.
  </p>

</div>
    """,
    unsafe_allow_html=True
)