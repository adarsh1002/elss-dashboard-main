# app.py
import streamlit as st

st.set_page_config(page_title="ELSS Dashboard", layout="wide")


st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

<h1 style="text-align:center; color:#003366;">Comparative Financial Performance Analysis of ELSS Mutual Funds</h1>

<h3 style="text-align:center; color:#666666;">An Interactive Analytical Dashboard</h3>

<hr style="border: 1px solid #cccccc;">

<p>
This interactive dashboard presents a comprehensive visualization of the 
<b>comparative financial performance</b> of 
<b>Equity Linked Savings Schemes (ELSS)</b> offered by five leading mutual fund houses in India:
<b>Axis Mutual Fund</b>, <b>SBI Mutual Fund</b>, <b>Mirae Asset Mutual Fund</b>, 
<b>DSP Mutual Fund</b>, and <b>HDFC Mutual Fund</b>.
</p>

<p>
Developed using <b>Streamlit</b> and <b>Python</b>, the dashboard transforms the detailed quantitative 
and qualitative analyses conducted in the research study into an intuitive, interactive format. 
It consolidates all dimensions of evaluation — <b>fund performance, risk and volatility, 
cost-effectiveness, portfolio composition</b>, and <b>benchmark comparison</b> — into 
a unified analytical environment. Users can explore, compare, and interpret 
fund-level insights in real time through interactive controls, filters, and 
data visualizations.
</p>

<h4 style="color:#003366;">Purpose and Features</h4>

<ul>
<li><b>Dynamic Data Exploration:</b> Navigate across key performance dimensions 
(NAV growth, AUM trends, return consistency, risk metrics, expense ratios, and alpha generation).</li>
<li><b>Interactive Visualizations:</b> Analyze trends using hover-enabled charts, 
date sliders, and fund selectors for customized comparisons.</li>
<li><b>Holistic Evaluation:</b> Integrates performance, risk, cost, and portfolio analytics 
to provide a multi-perspective understanding of ELSS fund behavior.</li>
<li><b>Benchmarking Framework:</b> Compare each scheme against its 
official <b>Total Returns Index (TRI)</b> benchmark to evaluate 
active management effectiveness and alpha persistence.</li>
<li><b>Downloadable Data:</b> Export filtered datasets and performance summaries 
for offline reference or extended analysis.</li>
</ul>

<h4 style="color:#003366;">Dashboard Modules</h4>

<p>
The dashboard has been structured into five interactive analytical modules, 
each aligned with the objectives of the study:
</p>

<ol>
<li><b>Comparative Performance Analysis:</b> 
Tracks historical NAV growth, AUM dynamics, and rolling returns (1Y, 3Y, 5Y).</li>

<li><b>Risk and Volatility Analysis:</b> 
Examines Standard Deviation, Beta, and Sharpe Ratio trends to assess 
risk profiles and risk-adjusted efficiency.</li>

<li><b>Cost-Effectiveness Analysis:</b> 
Evaluates Total Expense Ratios (TER) and their relationship with returns 
through box plots and bubble charts.</li>

<li><b>Portfolio Composition Analysis:</b> 
Visualizes fund allocation patterns across market capitalization segments 
(Large-, Mid-, Small-Cap) and key sectors over time.</li>

<li><b>Benchmarking and Alpha Analysis:</b> 
Compares each fund’s rolling returns with its respective benchmark 
and illustrates periods of outperformance or underperformance 
through <b>alpha shading</b> visualization.</li>
</ol>

<h4 style="color:#003366;">Development Framework</h4>

<table style="width:100%; border-collapse: collapse;" border="1" cellpadding="6">
<tr style="background-color:#f2f2f2;">
<th>Category</th>
<th>Tools / Libraries Used</th>
<th>Purpose</th>
</tr>
<tr>
<td>Programming Language</td>
<td>Python</td>
<td>Data analysis and dashboard creation</td>
</tr>
<tr>
<td>Visualization Libraries</td>
<td>Plotly, Matplotlib</td>
<td>Interactive and static charting</td>
</tr>
<tr>
<td>Data Handling</td>
<td>Pandas</td>
<td>Data cleaning, aggregation, and analysis</td>
</tr>
<tr>
<td>Web Application Framework</td>
<td>Streamlit</td>
<td>Dashboard interface and deployment</td>
</tr>
<tr>
<td>Data Sources</td>
<td>AMFI, Fund Fact Sheets</td>
<td>Fund-specific data inputs</td>
</tr>
</table>

<br>
<p>
By combining analytical depth with interactivity, this dashboard bridges the gap 
between <b>quantitative research and practical investment analysis</b>. 
It offers a live demonstration of how empirical financial research can be operationalized 
using modern visualization technologies for <b>real-world decision support</b>.
</p>

<p style="text-align:center; font-style: italic; color:#555555;">
Select a module from the sidebar to begin exploring the comparative analysis.
</p>

</div>
    """,
    unsafe_allow_html=True
)