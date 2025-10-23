import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
st.title("1. Comparative Analysis of Fund Performance")
st.markdown("""
The objective of this chapter is to conduct an in-depth comparative analysis of five leading ELSS mutual funds in India over the period 2020–2024. 
By examining the interplay of NAV growth, Assets Under Management (AUM) dynamics, and multi-period return performances, 
this chapter seeks to unveil critical insights into fund strategies, management efficiency, and investor behavior patterns. 
Rather than treating performance metrics in isolation, a cross-linked approach is adopted to demonstrate how various factors—such as fund size, risk-taking ability, and sectoral exposure—collectively influence investor outcomes. 
            """)
st.subheader("1.1 NAV GROWTH ANALYSIS")
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
st.markdown("**Indexed NAV Trend of ELSS Mututal Funds( Jan 2020-Dec 2024) **")
# Load the data
file_path = 'data/Data_Obj1.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert 'NAV Date' to datetime format
df['NAV Date'] = pd.to_datetime(df['NAV Date'])

# Filter necessary columns
df_nav = df[['Scheme Name', 'NAV Date', 'NAV Direct']]

# Pivot the data: Rows = NAV Date, Columns = Scheme Name, Values = NAV Direct
df_pivot = df_nav.pivot_table(index='NAV Date', columns='Scheme Name', values='NAV Direct')

# Index each scheme's NAV to 100 at start
df_indexed = df_pivot.divide(df_pivot.iloc[0]).multiply(100)

# Plot the trend line
plt.figure(figsize=(14,8))
for scheme in df_indexed.columns:
    plt.plot(df_indexed.index, df_indexed[scheme], label=scheme, linewidth=4)

plt.title('Indexed NAV Trend of ELSS Mutual Funds (Jan 2020 - Dec 2024)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Indexed NAV (Base = 100)', fontsize=14)
plt.grid(True, linestyle='--', alpha=1)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# NAV Growth Analysis Plotly Graph - Interactive

# %%
import pandas as pd
import plotly.graph_objects as go

# Load the data
DATA_PATH = "data/Data_Obj1.xlsx"   # your excel file path (sheet 'Sheet1')

@st.cache_data
def load_nav_data(path=DATA_PATH, sheet_name="Sheet1"):
    """Load NAV data from Excel and return cleaned DataFrame."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    # required columns: 'Scheme Name', 'NAV Date', 'NAV Direct'
    df = df.rename(columns=lambda c: c.strip())
    expected = {"Scheme Name", "NAV Date", "NAV Direct"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Excel sheet must contain columns: {expected}. Found: {set(df.columns)}")
    df["NAV Date"] = pd.to_datetime(df["NAV Date"])
    # drop rows with missing NAVs or scheme name or date
    df = df.dropna(subset=["Scheme Name", "NAV Date", "NAV Direct"])
    # sort
    df = df.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True)
    return df

try:
    df = load_nav_data()
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Please place your file and try again.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Prepare pivot table of NAVs (rows = date, columns = scheme)
df_nav = df[["Scheme Name", "NAV Date", "NAV Direct"]].copy()
df_pivot = df_nav.pivot_table(index="NAV Date", columns="Scheme Name", values="NAV Direct")

# Sidebar controls: choose schemes and date range
st.sidebar.header("Controls")
all_schemes = list(df_pivot.columns)
selected_schemes = st.sidebar.multiselect("Select Scheme(s) to display", options=all_schemes,
                                          default=all_schemes)  # default show all

# Date range picker (bounded by available data)
min_date = df_pivot.index.min().date()
max_date = df_pivot.index.max().date()
start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Filter pivot by date range and selected schemes
df_filtered = df_pivot.loc[(df_pivot.index.date >= start_date) & (df_pivot.index.date <= end_date), selected_schemes]

if df_filtered.empty:
    st.info("No data available for the chosen date range / schemes. Adjust the controls.")
    st.stop()

# Index NAV: make the first available value in the filtered range equal to 100 for each fund
# Use first VALID value (not NaN) in the filtered range for each column
base = df_filtered.apply(lambda col: col[col.first_valid_index()] if col.first_valid_index() is not None else pd.NA)
# Replace columns that have no valid base with NaN so they are ignored
valid_cols = [c for c in df_filtered.columns if pd.notna(base[c])]
if not valid_cols:
    st.info("Selected schemes have no NAV values in the chosen date range.")
    st.stop()

df_indexed = df_filtered[valid_cols].divide(base[valid_cols], axis=1).multiply(100.0)

# Build Plotly figure
fig = go.Figure()
for scheme in df_indexed.columns:
    fig.add_trace(go.Scatter(
        x=df_indexed.index,
        y=df_indexed[scheme],
        mode="lines",
        name=scheme,
        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%d %b %Y}<br>Indexed NAV: %{y:.2f}<extra></extra>"
    ))

fig.update_layout(
    title={
        "text": "Interactive Indexed NAV Trend of ELSS Mutual Funds (2020–2024)",
        "x": 0.05,  # left-align title a bit so it doesn't collide with legend
        "xanchor": "left",
        "yanchor": "top"
    },
    xaxis_title="Date",
    yaxis_title="Indexed NAV (Base = 100)",
    hovermode="x unified",
    template="plotly_white",
    # --- Legend moved to the right ---
    legend=dict(
        orientation="v",          # vertical layout
        yanchor="top",
        y=1,                      # top align with plot
        xanchor="left",
        x=1.02,                   # place just outside the right border
        title_text="Scheme Name",
        bgcolor="rgba(255,255,255,0.8)",  # semi-transparent white box
        bordercolor="lightgray",
        borderwidth=1
    ),
    margin=dict(t=80, b=40, l=40, r=180),  # extra right margin for legend
    height=600
)
fig.update_xaxes(rangeslider_visible=True)   # allow range slider for quick zooming

# Render Plotly figure in Streamlit (DO NOT use fig.show())
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Fund-wise Performance

- **SBI Long Term Equity Fund** has emerged as the top performer among the compared funds, achieving the highest indexed NAV, peaking above 320 during 2024 before a slight decline towards the year-end.  
- **DSP ELSS Tax Saver Fund** and **HDFC ELSS Tax Saver Fund** closely follow SBI's performance, showcasing strong and consistent capital appreciation.  
- **Mirae Asset ELSS Tax Saver Fund** also displays a robust growth pattern but slightly trails behind DSP and HDFC towards the end of the period.  
  *(Disclaimer: 5-year return history for Mirae Asset was partially unavailable; hence comparisons should be viewed accordingly.)*  
- **Axis ELSS Tax Saver Fund**, although starting similarly, significantly lags behind the other funds, with its growth plateauing relative to peers post-2022. This indicates a relatively underwhelming performance compared to the others in the latter part of the analysis period.  

**Consistency and Volatility:**  
Funds like **HDFC** and **DSP** show relatively stable and consistent growth with fewer sharp fluctuations. **SBI** displayed higher volatility but also higher returns, reflecting a slightly more aggressive investment approach. **Axis**, on the other hand, exhibited periods of stagnation, suggesting challenges in outperforming the broader market or peer group.  

A comparative analysis with the **NIFTY 50** returns over the corresponding period, which stood at approximately **14% (NIFTY50 n.d.)**, indicates that all selected ELSS schemes significantly outperformed the benchmark index, with the exception of the **Axis ELSS Tax Saver Fund**. The performance of the Axis ELSS Tax Saver Fund was largely comparable to the NIFTY 50, reflecting a relatively modest return in contrast to its peers.
""")
st.subheader("1.2 ASSETS UNDER MANAGEMENT (AUM) ANALYSIS")
st.markdown("""
            <div style="text-align: justify;">
**Assets Under Management (AUM)** are influenced by (i) net investor inflows and (ii) capital appreciation. However, a large AUM does not inherently guarantee superior returns; 
            it might also lead to challenges in nimble portfolio management.</div>""", unsafe_allow_html=True)

