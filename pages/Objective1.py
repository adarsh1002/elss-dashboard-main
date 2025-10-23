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
<b>Assets Under Management (AUM)</b> are influenced by (i) net investor inflows and (ii) capital appreciation. However, a large AUM does not inherently guarantee superior returns; 
            it might also lead to challenges in nimble portfolio management.</div>""", unsafe_allow_html=True)
#AUM Growth Analysis Plotly Graph - Interactive
# -------------------------
# AUM Growth Analysis (below Indexed NAV chart)
# -------------------------
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# File path (update if needed)
AUM_FILE = "data/Data_Obj1.xlsx"  # your AUM excel file path

@st.cache_data
def load_and_clean_aum(path=AUM_FILE):
    df_a = pd.read_excel(path)
    df_a.columns = [c.strip() for c in df_a.columns]
    # required columns check (try to detect)
    if "NAV Date" not in df_a.columns:
        # try common alternatives
        possible_date_cols = [c for c in df_a.columns if "date" in c.lower()]
        if possible_date_cols:
            df_a = df_a.rename(columns={possible_date_cols[0]: "NAV Date"})
        else:
            raise ValueError("No NAV Date column found in AUM file.")
    if "Scheme Name" not in df_a.columns:
        possible_scheme_cols = [c for c in df_a.columns if "scheme" in c.lower() or "fund" in c.lower()]
        if possible_scheme_cols:
            df_a = df_a.rename(columns={possible_scheme_cols[0]: "Scheme Name"})
        else:
            raise ValueError("No Scheme Name column found in AUM file.")
    # detect AUM column
    aum_candidates = [c for c in df_a.columns if "AUM" in c.upper() or "Aum" in c or "daily aum" in c.lower()]
    if aum_candidates:
        aum_col = aum_candidates[0]
    else:
        # fallback: numeric columns other than NAV if present
        numeric_cols = df_a.select_dtypes(include=[np.number]).columns.tolist()
        # pick the numeric column that is not NAV if possible
        aum_col = None
        for c in numeric_cols:
            if "nav" not in c.lower():
                aum_col = c
                break
        if aum_col is None:
            raise ValueError("Could not find an AUM column. Ensure your AUM file contains an AUM column.")
    # standardize names
    df_a = df_a.rename(columns={aum_col: "AUM", "NAV Date": "NAV Date", "Scheme Name": "Scheme Name"})
    df_a["NAV Date"] = pd.to_datetime(df_a["NAV Date"], errors="coerce")
    # clean AUM strings -> numeric
    df_a["AUM"] = df_a["AUM"].astype(str).fillna("").str.strip()
    df_a["AUM_clean"] = df_a["AUM"].str.replace(r"[^0-9.\-]", "", regex=True)
    df_a["AUM_Cr"] = pd.to_numeric(df_a["AUM_clean"], errors="coerce")
    df_a = df_a.dropna(subset=["NAV Date", "AUM_Cr", "Scheme Name"])
    df_a = df_a.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True)
    return df_a

# Load AUM data (handle errors gracefully)
try:
    aum_all = load_and_clean_aum()
except FileNotFoundError:
    st.error(f"AUM file not found: {AUM_FILE}. Please place it in the data/ folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading AUM file: {e}")
    st.stop()

# Use same selected_schemes, start_date, end_date from top controls
# If your main controls are named differently, update the variable names accordingly.
if "selected_schemes" not in globals():
    # fallback: use all schemes from AUM if selected_schemes not defined
    selected_schemes = sorted(aum_all["Scheme Name"].unique().tolist())

# Filter by date and scheme
mask = (aum_all["NAV Date"].dt.date >= start_date) & (aum_all["NAV Date"].dt.date <= end_date)
mask &= aum_all["Scheme Name"].isin(selected_schemes)
aum_df = aum_all.loc[mask].copy()
if aum_df.empty:
    st.info("No AUM data for selected schemes / date range. Adjust controls.")
else:
    # compute last available AUM per scheme (within filtered range)
    last_aum = aum_df.sort_values("NAV Date").groupby("Scheme Name", as_index=False).tail(1)
    last_sorted = last_aum.sort_values("AUM_Cr", ascending=False)
    ordered_schemes = last_sorted["Scheme Name"].tolist()
    # choose top_n to show panels (limit to avoid too many panels)
    top_n = min(len(ordered_schemes), 6)  # max 6 panels for layout
    plot_schemes = ordered_schemes[:top_n]

    # layout grid for panels
    cols = 2
    rows = (len(plot_schemes) + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=plot_schemes, shared_xaxes=False)

    # add traces: background (all selected schemes) in light grey, highlighted in blue
    for idx, focus in enumerate(plot_schemes):
        r = idx // cols + 1
        c = idx % cols + 1
        # background traces (all selected_schemes)
        for bg in selected_schemes:
            bg_series = aum_df[aum_df["Scheme Name"] == bg]
            if bg_series.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=bg_series["NAV Date"],
                    y=bg_series["AUM_Cr"],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"bg-{bg}"
                ),
                row=r, col=c
            )
        # focused trace
        focus_series = aum_df[aum_df["Scheme Name"] == focus]
        fig.add_trace(
            go.Scatter(
                x=focus_series["NAV Date"],
                y=focus_series["AUM_Cr"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=4),
                name=focus,
                hovertemplate="<b>%{text}</b><br>Date: %{x|%d %b %Y}<br>AUM (Cr): %{y:,.2f}<extra></extra>",
                text=[focus]*len(focus_series),
                showlegend=False
            ),
            row=r, col=c
        )
        # axes titles
        fig.update_xaxes(title_text="Date", row=r, col=c)
        fig.update_yaxes(title_text="AUM (Cr.)", row=r, col=c)

    fig.update_layout(
        title_text=f"AUM Growth of Selected ELSS Funds (highlighted panels; top {top_n} by last AUM)",
        template="plotly_white",
        height=300 * rows,
        margin=dict(t=100, b=60, l=60, r=60)
    )

    # render plot
    st.subheader("AUM Growth (panels sorted by last available AUM)")
    st.plotly_chart(fig, use_container_width=True)
text_Aum="""<div style="text-align: justify;">
The figure shows the Assets Under Management (AUM) growth trajectories of the top 5 selected ELSS mutual fund schemes over the five-year period from 2020 to 2024. Each panel highlights one fund prominently in blue, while showing others in light gray for comparison. The panels are arranged based on the size of each fund’s AUM as of the latest available date, with the largest fund appearing first. 
In mutual fund performance analysis, organizing funds based on the size of their Assets Under Management (AUM) provides a clearer and more meaningful comparison. Larger AUM generally indicates higher investor confidence, better fund stability, and greater management scalability.
Sorting panels by AUM size enables readers to immediately identify which funds dominate the ELSS market in terms of investor assets, making it easier to interpret growth patterns in relation to market leadership. Moreover, this approach aligns the visualization with investment significance, emphasizing funds that have garnered larger market trust and scale over time. Hence, the sorting enhances the analytical rigor and practical relevance of the AUM growth study. Axis ELSS commands the largest AUM, yet it has struggled to translate asset scale into proportionate returns. SBI Long Term Equity Fund, with a significantly lower AUM, has outperformed Axis both in NAV growth and recent returns. Further, during the study period, the mutual fund has increased its AUM by 5 times which shows strong investor confidence in the fund as well as fund has delivered highest returns in the segment.
Mirae Asset, despite having roughly half the AUM size of Axis, delivered superior NAV and returns growth, revealing efficient capital deployment.
 DSP ELSS, operating at almost one-third the AUM size of Axis, still managed competitive returns, indicating lean and focused fund management practices.
While Axis's AUM scale theoretically should have provided diversification benefits, its relative underperformance highlights the diminishing returns of scale beyond a 
threshold—especially when agility and selective sector exposure become key to alpha generation.</div>"""
st.markdown(text_Aum, unsafe_allow_html=True)
st.subheader("1.3 Returns Trend Analysis")
returns_text="""
<div style="text-align: justify;">
The performance of Equity Linked Savings Schemes (ELSS) over different investment horizons provides critical insights into their risk-return behavior and resilience to market cycles.
 To facilitate a comprehensive assessment, the returns of five selected ELSS mutual funds were analyzed across three distinct timeframes — 1-Year, 3-Year, and 5-Year rolling returns — from January 2020 to December 2024.
 The returns were plotted on separate panels for each investment horizon to capture nuances in short-term volatility, medium-term consistency, and long-term wealth creation capabilities. 
 This approach offers a layered understanding of how each fund behaved during different market phases, including the pandemic-driven market crash, the subsequent recovery, and the broader economic cycles.
 Important Note: Due to the shorter history of Mirae Asset ELSS Fund, 5-year returns should be interpreted with caution.
</div>
"""
st.markdown(returns_text, unsafe_allow_html=True)
# Returns Trend Analysis Plotly Graph - Interactive
# -------------------------
# -------------------------
# Returns Trend Analysis (1Y / 3Y / 5Y) - main-body radio to select horizon
# -------------------------
import pandas as pd
import numpy as np
import plotly.express as px

RET_FILE = "data/Data_Obj1.xlsx"  # your returns excel file path


@st.cache_data
def load_and_clean_returns(path=RET_FILE):
    raw = pd.read_excel(path)
    # if pandas returns dict (multiple sheets), take first
    if isinstance(raw, dict):
        raw = list(raw.values())[0].copy()
    dfr = raw.copy()
    # normalize column names
    dfr.columns = [c.strip() if isinstance(c, str) else c for c in dfr.columns]

    # detect NAV Date
    if "NAV Date" not in dfr.columns:
        possible_date = [c for c in dfr.columns if isinstance(c, str) and "date" in c.lower()]
        if possible_date:
            dfr = dfr.rename(columns={possible_date[0]: "NAV Date"})
        else:
            raise ValueError("No NAV Date column found in returns file.")

    # detect Scheme Name
    if "Scheme Name" not in dfr.columns:
        possible_scheme = [c for c in dfr.columns if isinstance(c, str) and ("scheme" in c.lower() or "fund" in c.lower())]
        if possible_scheme:
            dfr = dfr.rename(columns={possible_scheme[0]: "Scheme Name"})
        else:
            raise ValueError("No Scheme Name column found in returns file.")

    # detect return columns and rename consistently
    rename_map = {}
    for col in dfr.columns:
        if not isinstance(col, str):
            continue
        lc = col.lower()
        if "return 1" in lc or "1 year" in lc or "return_1" in lc:
            rename_map[col] = "Return_1Y"
        if "return 3" in lc or "3 year" in lc or "return_3" in lc:
            rename_map[col] = "Return_3Y"
        if "return 5" in lc or "5 year" in lc or "return_5" in lc:
            rename_map[col] = "Return_5Y"
    dfr = dfr.rename(columns=rename_map)

    # ensure the three return cols exist
    for c in ["Return_1Y", "Return_3Y", "Return_5Y"]:
        if c not in dfr.columns:
            dfr[c] = np.nan

    # parse date and coerce types
    dfr["NAV Date"] = pd.to_datetime(dfr["NAV Date"], errors="coerce")
    # ensure scheme name is string and 1d
    dfr["Scheme Name"] = dfr["Scheme Name"].astype(str).str.strip()

    for c in ["Return_1Y", "Return_3Y", "Return_5Y"]:
        dfr[c] = pd.to_numeric(dfr[c], errors="coerce")

    # drop rows without essential info
    dfr = dfr.dropna(subset=["NAV Date", "Scheme Name"]).reset_index(drop=True)
    dfr = dfr.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True)
    return dfr

# Load returns
try:
    returns_all = load_and_clean_returns()
except FileNotFoundError:
    st.error(f"Returns file not found: {RET_FILE}. Please place it in the data/ folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading returns file: {e}")
    st.stop()

# Use existing sidebar controls if present; else create local ones
if "selected_schemes" not in globals():
    schemes_list = sorted(returns_all["Scheme Name"].unique().tolist())
    selected_schemes = st.sidebar.multiselect("Select Schemes (returns)", options=schemes_list, default=schemes_list)
if "start_date" not in globals():
    min_dt = returns_all["NAV Date"].min().date()
    max_dt = returns_all["NAV Date"].max().date()
    start_date = st.sidebar.date_input("Start date (returns)", value=min_dt, min_value=min_dt, max_value=max_dt)
    end_date = st.sidebar.date_input("End date (returns)", value=max_dt, min_value=min_dt, max_value=max_dt)

# smoothing options in sidebar (shared)
st.sidebar.write("Returns smoothing (optional)")
smooth = st.sidebar.checkbox("Apply smoothing (rolling mean)", value=True)
smooth_window = st.sidebar.selectbox("Smoothing window (months)", options=[1, 3, 6], index=1)

# main-body radio to choose horizon
st.subheader("Returns Trend Analysis")
horizon = st.radio("Choose return horizon to display:", options=["1-Year", "3-Year", "5-Year"], index=0, horizontal=True)

# map horizon to column
hcol_map = {"1-Year": "Return_1Y", "3-Year": "Return_3Y", "5-Year": "Return_5Y"}
selected_col = hcol_map[horizon]

# filter data by date & selected schemes
mask = (returns_all["NAV Date"].dt.date >= start_date) & (returns_all["NAV Date"].dt.date <= end_date)
mask &= returns_all["Scheme Name"].isin(selected_schemes)
returns_df = returns_all.loc[mask, ["Scheme Name", "NAV Date", selected_col]].copy()

# rename to general column
returns_df = returns_df.rename(columns={selected_col: "Return_pct"})

# Basic defensive checks
if returns_df.empty:
    st.info("No return data available for the selected horizon / schemes / date range. Adjust the controls.")
else:
    # Ensure columns are 1-D and have correct dtypes
    if not pd.api.types.is_datetime64_any_dtype(returns_df["NAV Date"]):
        try:
            returns_df["NAV Date"] = pd.to_datetime(returns_df["NAV Date"], errors="coerce")
        except Exception as e:
            st.error(f"Could not parse NAV Date column: {e}")
            st.stop()
    # Ensure Scheme Name is string
    returns_df["Scheme Name"] = returns_df["Scheme Name"].astype(str)
    # Ensure Return_pct numeric
    returns_df["Return_pct"] = pd.to_numeric(returns_df["Return_pct"], errors="coerce")

    # Drop any rows with NaN Return_pct
    returns_df = returns_df.dropna(subset=["Return_pct"])
    if returns_df.empty:
        st.info("After filtering, there is no numeric return data for the chosen options.")
    else:
        # optionally apply smoothing: resample monthly and rolling mean
        if smooth and smooth_window > 1:
            tmp_list = []
            for scheme, g in returns_df.groupby("Scheme Name"):
                g = g.set_index("NAV Date").resample("M").last().dropna(subset=["Return_pct"])
                if g.empty:
                    continue
                g["Return_sm"] = g["Return_pct"].rolling(window=smooth_window, min_periods=1).mean()
                g = g.reset_index().rename(columns={"Return_sm": "Return_pct"})
                g["Scheme Name"] = scheme
                tmp_list.append(g[["Scheme Name", "NAV Date", "Return_pct"]])
            if tmp_list:
                plot_df = pd.concat(tmp_list, ignore_index=True)
            else:
                plot_df = returns_df.copy()
        else:
            plot_df = returns_df.copy()

        # Final check that plot_df columns are 1-D and valid
        if plot_df.empty:
            st.info("No data to plot after smoothing/resampling.")
        else:
            # Ensure scheme column is 1-D series with simple strings
            plot_df["Scheme Name"] = plot_df["Scheme Name"].astype(str)
            plot_df = plot_df.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True)

            # Now create interactive Plotly line chart (one line per scheme)
            try:
                fig_ret = px.line(
                    plot_df,
                    x="NAV Date",
                    y="Return_pct",
                    color="Scheme Name",
                    labels={"Return_pct": f"{horizon} Return (%)", "NAV Date": "Date"},
                    template="plotly_white",
                    title=f"{horizon} Returns Trend across Selected Schemes"
                )
                fig_ret.update_traces(mode="lines+markers")
                fig_ret.update_layout(hovermode="x unified", legend=dict(title="Scheme", orientation="v", x=1.02, y=1))
                fig_ret.update_yaxes(ticksuffix="%", tickformat=".2f")
                fig_ret.update_xaxes(rangeslider_visible=True)

                st.plotly_chart(fig_ret, use_container_width=True)
            except Exception as e:
                st.error(f"Plotly error while drawing returns chart: {e}")
                st.write("Debug info (first few rows of plot_df):")
                st.dataframe(plot_df.head(20))

            # Data preview & download
            st.markdown("### Data used for this chart (preview)")
            st.dataframe(plot_df.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True).head(200))

            @st.cache_data
            def to_csv(df_):
                return df_.to_csv(index=False).encode("utf-8")

            st.download_button(f"Download {horizon} returns data (CSV)", data=to_csv(plot_df), file_name=f"returns_{selected_col}.csv", mime="text/csv")

    