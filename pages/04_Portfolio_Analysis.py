import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("Portfolio Analysis")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Introduction</h4>

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
<li><b>1. Market Capitalization Allocation:</b>  Data on fund-wise exposure to <b>large-cap</b>, <b>mid-cap</b>, and <b>small-cap equities</b> 
    was compiled and analyzed to understand asset allocation strategies and their evolution over time.</li>

<li><b>2. Sectoral Allocation:</b>  Sector-wise investment data was aggregated to identify <b>concentration patterns</b>, 
    <b>sector rotation</b>, and <b>thematic tilts</b>.  Visualization tools such as <b>stacked bar charts</b> and <b>heatmaps</b> were employed 
to illustrate diversification and dominance across key sectors.</li>

<li><b>3. Fund Management Trends:</b>  Changes in fund management and their possible influence on allocation and performance 
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
st.subheader("Market Capitalization Analysis")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Summary: Market Capitalization Allocation and Risk Profile</h4>

  <p>
  A crucial dimension of a mutual fund’s investment strategy lies in its allocation across different 
  <b>market capitalizations</b> — namely <b>Large-Cap</b>, <b>Mid-Cap</b>, and <b>Small-Cap</b> companies. 
  This distribution provides direct insight into the fund’s <b>risk appetite</b> and its primary sources of expected returns.
  </p>

  <ul>
<li><b>Large-Cap Stocks:</b> Represent well-established, financially stable companies that offer steady but relatively slower growth. 
A fund with a heavy large-cap orientation reflects a <b>conservative and risk-averse strategy</b>, emphasizing capital preservation and consistent long-term performance.</li>

<li><b>Mid-Cap Stocks:</b> Consist of companies in their expansion or high-growth phase. These investments strike a balance between <b>stability and growth potential</b>, 
    though they carry greater risk than large-cap exposures.</li>

<li><b>Small-Cap Stocks:</b> Comprise smaller, often emerging companies with the highest potential for returns — 
    but also the greatest volatility. A higher allocation to small-caps signifies a <b>more aggressive investment stance</b>, 
    aiming for higher alpha but with elevated short-term risk.</li>
</ul>

  <p>
  For <b>Equity Linked Savings Schemes (ELSS)</b>, which feature a mandatory three-year lock-in period, 
  fund managers have the flexibility to invest dynamically across market-cap segments to 
  balance <b>risk and long-term return potential</b>. 
  Examining the market-cap composition of the top five ELSS funds allows investors to 
  look beyond stated objectives and assess the actual <b>risk–return trade-offs</b> reflected in portfolio construction.
  </p>

  <p>
  This analysis highlights whether each fund leans toward the <b>stability of blue-chip large-caps</b> 
  or ventures into <b>mid and small-cap equities</b> in pursuit of higher growth. 
  By understanding these allocation patterns, investors can gauge how each fund’s 
  strategic positioning aligns with their own <b>risk tolerance and investment horizon</b>.
  </p>

</div>
    """,
    unsafe_allow_html=True
)
DATA_PATH = "data/Data_obj4_MarketCap_final.xlsx"  # adjust if needed

@st.cache_data
def load_marketcap_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Parse month_year robustly
    if "month_year" in df.columns:
        # try common formats, fallback to pandas to_datetime
        df["Date"] = pd.to_datetime(df["month_year"], errors="coerce", dayfirst=False)
        # normalize to period start (first of month)
        df["Date"] = pd.to_datetime(df["Date"].dt.to_period("M").dt.to_timestamp())
    else:
        # fallback: try to find a date-like column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            df["Date"] = pd.NaT

    # Ensure columns we need exist
    # Expected columns in your sheet: Fund House (or similar), Categorization (Large/Mid/Small), Contribution (as fraction)
    # Try to detect names
    col_names = { "fund": None, "cat": None, "contrib": None }
    for c in df.columns:
        cl = str(c).lower()
        if ("fund" in cl or "house" in cl or "scheme" in cl) and col_names["fund"] is None:
            col_names["fund"] = c
        if ("categor" in cl or "cap" in cl or "segment" in cl) and col_names["cat"] is None:
            col_names["cat"] = c
        if ("contrib" in cl or "contribution" in cl or "weight" in cl or "allocation" in cl) and col_names["contrib"] is None:
            col_names["contrib"] = c

    if None in col_names.values():
        raise ValueError(f"Could not detect required columns automatically. Found columns: {df.columns.tolist()}")

    df = df.rename(columns={col_names["fund"]: "Fund House",
                            col_names["cat"]: "Categorization",
                            col_names["contrib"]: "Contribution"})

    # Ensure Contribution numeric (fractional or percent)
    df["Contribution"] = (
        df["Contribution"].astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", np.nan)
    )
    df["Contribution"] = pd.to_numeric(df["Contribution"], errors="coerce")
    # if contributions look like fractions (0-1), convert to percent
    if df["Contribution"].max() <= 1.01:
        df["Contribution"] = df["Contribution"] * 100

    # Drop rows without date/fund/category/contribution
    df = df.dropna(subset=["Date", "Fund House", "Categorization", "Contribution"]).copy()

    # Standardize categorization names (optional)
    df["Categorization"] = df["Categorization"].str.strip().replace({
        "Large": "Large Cap", "LargeCap": "Large Cap", "Large-Cap": "Large Cap",
        "Mid": "Mid Cap", "MidCap": "Mid Cap", "Mid-Cap": "Mid Cap",
        "Small": "Small Cap", "SmallCap": "Small Cap", "Small-Cap": "Small Cap"
    })

    return df

# Load data
try:
    df = load_marketcap_data()
except Exception as e:
    st.error(f"Error loading market-cap file: {e}")
    st.stop()

# Build list of available months sorted
available_months = sorted(df["Date"].dropna().unique())
if len(available_months) == 0:
    st.info("No month_year data found in file.")
    st.stop()

# Human-friendly labels
month_labels = [d.strftime("%b %Y") for d in available_months]
month_to_label = dict(zip(available_months, month_labels))
label_to_month = {v: k for k, v in month_to_label.items()}

# Sidebar controls
st.sidebar.header("Month selection")
st.sidebar.markdown("Use the sidebar slider or the select box to choose any available month. Data is half-yearly from Mar 2020 onward.")
# slider index-based for nicer UX
idx = st.sidebar.slider("Select month index", 0, len(available_months)-1, len(available_months)-1)
selected_date = available_months[idx]

# Also show selectbox synced
sel_label = st.sidebar.selectbox("Or choose month", options=month_labels, index=idx)
# sync if user picks selectbox
if sel_label:
    selected_date = label_to_month[sel_label]
    # update slider to reflect selection (display only; cannot programmatically set slider value easily)
# Optionally select funds
all_funds = sorted(df["Fund House"].unique().tolist())
selected_funds = st.sidebar.multiselect("Select Fund Houses (leave empty = all)", options=all_funds, default=all_funds)

# Filter to selected month and funds
df_sel = df[df["Date"] == selected_date].copy()
if selected_funds:
    df_sel = df_sel[df_sel["Fund House"].isin(selected_funds)].copy()

if df_sel.empty:
    st.info("No data for the selected month / funds.")
    st.stop()

# Aggregate contributions per fund & category
agg = df_sel.groupby(["Fund House", "Categorization"], as_index=False)["Contribution"].sum()
# convert to percent already handled
# Pivot
pivot = agg.pivot(index="Fund House", columns="Categorization", values="Contribution").fillna(0)

# Compute Cash & Other Holdings = 100 - sum(equity categories)
pivot["Total Equity"] = pivot.sum(axis=1)
pivot["Cash & Other Holdings"] = (100 - pivot["Total Equity"]).clip(lower=0)  # no negative

# Drop temporary
pivot = pivot.drop(columns=["Total Equity"])

# Reorder columns
cap_order = ["Large Cap", "Mid Cap", "Small Cap", "Cash & Other Holdings"]
present_cols = [c for c in cap_order if c in pivot.columns]
pivot = pivot[present_cols]

# Prepare plotting order for consistent colors
funds_order = pivot.index.tolist()

# Build Plotly stacked bars (one trace per category)
colors = {
    "Large Cap": "#577590",
    "Mid Cap": "#90be6d",
    "Small Cap": "#f4a259",
    "Cash & Other Holdings": "#9a8c98"
}
fig = go.Figure()
for cat in present_cols:
    fig.add_trace(go.Bar(
        name=cat,
        x=funds_order,
        y=pivot[cat].round(2),
        marker_color=colors.get(cat, None),
        hovertemplate="<b>%{x}</b><br>%{y:.2f}% " + f"({cat})<extra></extra>"
    ))

# Layout
fig.update_layout(
    barmode="stack",
    title=f"Market Cap Allocation of Selected ELSS Funds — {selected_date.strftime('%b %Y')}",
    xaxis_title="Fund House",
    yaxis_title="Allocation (%)",
    yaxis=dict(range=[0, 100], ticksuffix="%"),
    legend_title="Category",
    template="plotly_white",
    height=600,
    margin=dict(l=80, r=80, t=100, b=140)
)

# Add percentage labels inside segments (Plotly supports textinfo per bar; we'll add total stacked annotations per bar)
# Create cumulative sums for label positioning
cumulative = np.zeros(len(funds_order))
for cat in present_cols:
    yvals = pivot[cat].round(2).values
    # position where to put text (only if segment > threshold)
    texts = [f"{v:.1f}%" if v >= 3 else "" for v in yvals]  # skip tiny slices
    fig.add_trace(go.Bar(
        x=funds_order,
        y=[0]*len(yvals),  # invisible helper series just to use textposition; skip adding if not desired
        text=texts,
        textposition='inside',
        showlegend=False,
        marker_opacity=0
    ))
    cumulative += yvals

# Show chart
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Analysis Summary: Market Capitalization Allocation of ELSS Funds</h4>

  <p>
  The market capitalization analysis reveals that all selected <b>ELSS funds</b> maintain a strong foundation in 
  <b>large-cap equities</b>, consistent with their objective of achieving long-term capital appreciation 
  while maintaining controlled risk. 
  However, variations in <b>mid- and small-cap exposure</b> across the schemes highlight differing 
  investment philosophies and levels of aggressiveness among fund houses.
  </p>

  <ul>
    <li><b>Axis ELSS Tax Saver Fund:</b> Exhibits a pronounced large-cap bias (≈ 70.7%), 
    moderate mid-cap exposure (≈ 23.8%), and minimal small-cap allocation (≈ 3%). 
    This conservative structure prioritizes <b>stability and reduced volatility</b>.</li>

    <li><b>HDFC ELSS Tax Saver Fund:</b> Follows a predominantly large-cap strategy (≈ 73.7%) 
    complemented by small-cap exposure of around 11.6%, balancing <b>steady growth with limited risk-taking</b>.</li>

    <li><b>DSP and Mirae Asset ELSS Funds:</b> Demonstrate more <b>aggressive allocation patterns</b> 
    with smaller large-cap weights and higher small-cap exposure — 
    14.5% for DSP and 21.9% for Mirae Asset — reflecting a <b>higher risk-reward orientation</b>.</li>

    <li><b>SBI Long Term Equity Fund:</b> Maintains a <b>balanced and diversified allocation</b> 
    across all segments, with slightly higher liquidity (≈ 8.9%) to manage short-term market fluctuations.</li>
  </ul>

  <p>
  Funds with higher <b>mid- and small-cap exposure</b> — such as <b>DSP, Mirae Asset,</b> and <b>SBI</b> — 
  adopted a more aggressive strategy aimed at generating higher alpha. 
  Their superior <b>CAGR performance (≈ 21–25%)</b> over 2020–2024 
  supports the view that mid- and small-cap equities tend to outperform during 
  periods of <b>economic recovery and expansion</b>. 
  Conversely, large-cap heavy funds like <b>Axis ELSS</b>, while stable, 
  posted more modest returns (≈ 14.05%) due to their conservative positioning.
  </p>

  <p>
  Overall, the findings highlight that a <b>balanced market-cap allocation</b> strategy — 
  combining the stability of large-caps with selective mid- and small-cap exposure — 
  delivers the most effective balance between <b>risk and return</b>. 
  Funds such as <b>SBI</b> and <b>HDFC ELSS</b> achieved this equilibrium, 
  reflecting efficient portfolio construction and superior <b>risk-adjusted performance</b>. 
  Meanwhile, the higher volatility observed in <b>Mirae Asset</b> and <b>DSP</b> 
  underscores the trade-off between aggressive growth potential and market sensitivity. 
  This analysis reinforces that <b>diversified market-cap exposure</b> is key to 
  optimizing long-term performance and managing downside risk in ELSS portfolios.
  </p>

</div>
    """,
    unsafe_allow_html=True
)