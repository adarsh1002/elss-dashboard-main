import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
st.title("2. Risk and Volatility Analysis")
st.markdown(
    """
    <div style="text-align: justify;">
      <h4>Introduction</h4>
      <p>
        Investment decisions in equity-linked instruments, such as Equity Linked Savings Schemes (ELSS), require a careful balance between risk and return.
        While ELSS funds offer investors the dual advantage of potential capital appreciation and tax benefits under Section 80C of the Income Tax Act, 1961,
         their equity-oriented nature exposes them to significant market-linked volatility. Understanding and quantifying this risk is essential for assessing the stability and performance of these schemes over time.
      </p>

      <p>
        In financial terms, risk represents the degree of uncertainty associated with expected returns, reflecting the potential deviation of actual outcomes from anticipated performance.
        For mutual funds, risk arises from both <b>systematic factors</b> (macroeconomic, political, and market-wide influences)
        and <b>unsystematic factors</b> (fund-specific or sector-specific events). Systematic risk cannot be diversified away, whereas unsystematic risk can be mitigated through effective portfolio diversification strategies.
      </p>

      <p>
        This chapter evaluates the <b>risk and volatility characteristics</b> of selected ELSS mutual funds over the five-year period from <b>2020 to 2024</b>,
        using three key statistical and financial measures:
      </p>

      <ul>
        <li><b>Standard Deviation</b>, which quantifies the volatility or variability in fund returns.</li>
        <li><b>Beta</b>, which assesses the fund’s sensitivity to market movements and its relative systematic risk.</li>
        <li><b>Sharpe Ratio</b>, which evaluates the fund’s risk-adjusted performance, indicating how efficiently it compensates investors for the risk undertaken.</li>
      </ul>

      <p>
        The analysis aims to identify patterns of volatility, compare market responsiveness across schemes, and highlight funds demonstrating efficient risk management and superior risk-adjusted returns.
        By combining these measures, the chapter provides a comprehensive understanding of the <b>risk–return trade-off</b> inherent in ELSS investments and supports investors in making informed, objective, and risk-aligned investment decisions.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)