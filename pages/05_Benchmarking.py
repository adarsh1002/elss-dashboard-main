import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from plotly.subplots import make_subplots
st.set_page_config("Benchmarking", layout="wide"
                   )
st.title("📊 Benchmarking"
         )
st.markdown(""" <hr style="border: 1px solid #cccccc;"> """, unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  

  <p>
  Benchmarking mutual fund performance against appropriate market indices is a critical component of performance evaluation. 
  For <b>Equity Linked Savings Schemes (ELSS)</b> — which are actively managed and equity-oriented — benchmarking provides 
  an objective basis to measure <b>fund manager effectiveness</b> and the consistency of <b>alpha generation</b>, i.e., 
  the excess return earned over general market movements. 
  </p>

  <p>
  The benchmarks selected for this study align with those officially prescribed by the respective fund houses — 
  primarily the <b>Nifty 500 Total Returns Index (TRI)</b> and the <b>BSE 200/500 Total Returns Index (TRI)</b>. 
  These indices serve as comprehensive market proxies, encompassing large-cap, mid-cap, and small-cap segments of the Indian equity market. 
  The <b>Nifty 500 TRI</b> covers nearly 95% of the market’s free-float capitalization, making it a broad indicator of overall equity performance, 
  while the <b>BSE 200/500 TRI</b> captures more stable large- and mid-cap trends.
  </p>

  <p>
  A key advantage of using <b>Total Return Indices (TRI)</b> lies in their inclusion of both <b>capital appreciation</b> and 
  <b>dividend reinvestments</b>, providing a more accurate and equitable comparison with mutual fund returns. 
  Unlike price-only indices, TRI-based benchmarks ensure that performance evaluations account for the full return potential, 
  making them the <b>industry standard</b> for assessing fund performance.
  </p>

  <p>
  This chapter benchmarks the <b>3-year rolling returns</b> of the top five ELSS schemes selected for this project 
  against their respective TRI benchmarks. Data has been sourced from the 
  <b>Association of Mutual Funds in India (AMFI)</b> and verified through official fund fact sheets. 
  The analysis aims to identify whether active management has resulted in consistent 
  <b>outperformance</b> over benchmarks and to assess the <b>sustainability of alpha generation</b> 
  across varying market conditions between <b>2020 and 2024</b>.
  </p>

  <h5>Objective and Analytical Framework</h5>

  <p>
  The objective of this analysis is to evaluate the 3-year rolling returns of each ELSS fund relative to its benchmark to determine:
  </p>

  <ul>
    <li>Whether the schemes have <b>outperformed or underperformed</b> their respective benchmark indices over time.</li>
    <li>The <b>consistency</b> of alpha generation across different market phases.</li>
  </ul>

  <p>
  The 3-year rolling return is used as the primary performance metric, reflecting the average annualized return 
  over successive 3-year periods — a measure particularly relevant for ELSS funds due to their mandatory 
  <b>three-year lock-in period</b>. Each fund’s performance is plotted against its declared benchmark index 
  (<b>Nifty 500 TRI, BSE 200 TRI,</b> or <b>BSE 500 TRI</b>) using <b>faceted line graphs</b> that visualize 
  instances of outperformance (fund line above benchmark) or underperformance (fund line below benchmark).
  </p>

  <p>
  By systematically comparing fund returns to benchmark trajectories, this benchmarking exercise 
  provides an objective perspective on the <b>value added by active management</b> and the 
  <b>long-term efficiency</b> of each ELSS scheme in generating superior risk-adjusted returns.
  </p>

</div>
    """,
    unsafe_allow_html=True
)
st.subheader("Trend of Alpha Generation Over Time")
# Interactive Benchmarking page with selectable horizon (1Y / 3Y / 5Y)
import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- CONFIG ----------
DATA_PATH = "data/Performance data_main.xlsx"   # <-- update if needed

# ---------- LOAD / PREP ----------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    # Ensure NAV Date exists and parse
    if "NAV Date" not in df.columns:
        raise KeyError("Required column 'NAV Date' not found in data.")
    df["NAV Date"] = pd.to_datetime(df["NAV Date"], errors="coerce")
    df = df.dropna(subset=["NAV Date"]).copy()
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Please upload file or update DATA_PATH.")
    st.stop()
except KeyError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------- HORIZON SELECTION ----------
# Map user-friendly horizon to expected column name patterns in your dataset
horizon_map = {
    "1Y": {
        "fund": "Return 1 Year (%) Direct",
        "bench": "Return 1 Year (%) Benchmark"
    },
    "3Y": {
        "fund": "Return 3 Year (%) Direct",
        "bench": "Return 3 Year (%) Benchmark"
    },
    "5Y": {
        "fund": "Return 5 Year (%) Direct",
        "bench": "Return 5 Year (%) Benchmark"
    }
}

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Chart controls")

# Horizon selector
horizon = st.sidebar.radio("Select return horizon", options=["1Y", "3Y", "5Y"], index=1, key="benchmark_horizon")

fund_col = horizon_map[horizon]["fund"]
bench_col = horizon_map[horizon]["bench"]

# Verify required columns exist
missing_cols = [c for c in [fund_col, bench_col, "Scheme Name", "NAV Date"] if c not in df_raw.columns]
if missing_cols:
    st.error(f"Missing required columns for the {horizon} horizon: {missing_cols}.")
    st.stop()

# Prepare dataset: coerce numerics and sort
df = df_raw[["Scheme Name", "NAV Date", fund_col, bench_col]].copy()
df[fund_col] = pd.to_numeric(df[fund_col], errors="coerce")
df[bench_col] = pd.to_numeric(df[bench_col], errors="coerce")
df = df.sort_values(["Scheme Name", "NAV Date"]).reset_index(drop=True)

# Scheme selection
all_schemes = sorted(df["Scheme Name"].dropna().unique().tolist())
selected_schemes = st.sidebar.multiselect(
    "Select scheme(s) to plot (max 12 recommended):",
    options=all_schemes,
    default=all_schemes,
    key="benchmark_schemes"
)

# Date index-based slider (use index over unique months to avoid widget collisions)
unique_dates = sorted(df["NAV Date"].dt.to_period("M").dt.to_timestamp().unique())
if len(unique_dates) == 0:
    st.error("No NAV Date data available.")
    st.stop()

date_labels = [d.strftime("%Y-%m") for d in unique_dates]
idx_min, idx_max = st.sidebar.slider(
    "Select date range (month index)",
    0,
    len(unique_dates) - 1,
    (0, len(unique_dates) - 1),
    key="benchmark_date_slider"
)
start_date = unique_dates[idx_min]
end_date = unique_dates[idx_max]

# Toggle alpha
show_alpha = st.sidebar.checkbox("Show alpha (Fund - Benchmark) as area", value=True, key="benchmark_alpha")

# Panels per row
cols_per_row = st.sidebar.selectbox("Panels per row", options=[1, 2, 3], index=1, key="benchmark_cols")

st.sidebar.markdown("---")
st.sidebar.caption(f"Available data: {unique_dates[0].date()} → {unique_dates[-1].date()}")

# ---------- FILTER DATA ----------
if not selected_schemes:
    st.info("Please select at least one scheme.")
    st.stop()

mask = (df["Scheme Name"].isin(selected_schemes)) & (df["NAV Date"] >= start_date) & (df["NAV Date"] <= end_date)
df_filt = df.loc[mask].copy()
if df_filt.empty:
    st.warning("No data for the selected schemes and date range.")
    st.stop()



# ---------- BUILD PLOTLY SUBPLOTS ----------
# --- Required imports (if not already imported) ---
from math import ceil
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Layout params (assume these variables already exist) ---
# selected_schemes, df_filt, fund_col, bench_col, horizon, cols_per_row, show_alpha

n = len(selected_schemes)
ncols = cols_per_row
nrows = ceil(n / ncols)
height_per_row = 320
total_height = max(420, nrows * height_per_row)

fig = make_subplots(
    rows=nrows,
    cols=ncols,
    subplot_titles=selected_schemes,
    shared_xaxes=False,
    vertical_spacing=0.12,
    horizontal_spacing=0.06
)

fund_color = "#1f77b4"
bench_color = "#ff7f0e"
pos_alpha_fill = "rgba(31,119,180,0.12)"    # blue translucent
neg_alpha_fill = "rgba(222,45,38,0.12)"     # red translucent

row = col = 1
for i, scheme in enumerate(selected_schemes):
    sub = df_filt[df_filt["Scheme Name"] == scheme].sort_values("NAV Date")

    # If no data, add placeholder empty trace (legend suppressed)
    if sub.empty or (sub[fund_col].dropna().empty and sub[bench_col].dropna().empty):
        fig.add_trace(go.Scatter(x=[], y=[], name="No data", showlegend=False), row=row, col=col)
    else:
        # Show legend items only for first subplot to avoid duplicates
        show_legend_flag = (i == 0)

        # Fund trace (showlegend only once)
        fig.add_trace(
            go.Scatter(
                x=sub["NAV Date"],
                y=sub[fund_col],
                mode="lines+markers",
                name="Fund (" + horizon + ")",
                line=dict(color=fund_color, width=2),
                marker=dict(size=6),
                hovertemplate="<b>%{x|%b %Y}</b><br>Fund: %{y:.2f}%<extra></extra>",
                showlegend=show_legend_flag
            ),
            row=row, col=col
        )

        # Benchmark trace (showlegend only once)
        fig.add_trace(
            go.Scatter(
                x=sub["NAV Date"],
                y=sub[bench_col],
                mode="lines+markers",
                name="Benchmark (" + horizon + ")",
                line=dict(color=bench_color, width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>%{x|%b %Y}</b><br>Benchmark: %{y:.2f}%<extra></extra>",
                showlegend=show_legend_flag
            ),
            row=row, col=col
        )

        # Alpha shading split into positive and negative areas
        if show_alpha:
            # compute alpha (vectorized) and handle NaNs
            alpha_vals = (sub[fund_col].to_numpy(dtype=float) - sub[bench_col].to_numpy(dtype=float))
            x_vals = sub["NAV Date"]

            # positive alpha (alpha > 0), else NaN so Plotly skips points
            pos_alpha = [v if v > 0 else None for v in alpha_vals]
            # negative alpha (alpha < 0), else None
            neg_alpha = [v if v < 0 else None for v in alpha_vals]

            # Positive alpha fill (showlegend only once)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=pos_alpha,
                    mode="lines",
                    name="Positive Alpha",
                    line=dict(color="rgba(0,0,0,0)"),  # invisible line; area shows fill
                    fill="tozeroy",
                    fillcolor=pos_alpha_fill,
                    hovertemplate="<b>%{x|%b %Y}</b><br>Alpha: %{y:.2f}%<extra></extra>",
                    showlegend=show_legend_flag
                ),
                row=row, col=col
            )

            # Negative alpha fill (showlegend only once)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=neg_alpha,
                    mode="lines",
                    name="Negative Alpha",
                    line=dict(color="rgba(0,0,0,0)"),
                    fill="tozeroy",
                    fillcolor=neg_alpha_fill,
                    hovertemplate="<b>%{x|%b %Y}</b><br>Alpha: %{y:.2f}%<extra></extra>",
                    showlegend=show_legend_flag
                ),
                row=row, col=col
            )

    # Axis labels & formatting for the panel
    fig.update_yaxes(title_text=f"{horizon} Return (%)", row=row, col=col, ticksuffix="%")
    fig.update_xaxes(title_text="NAV Date", row=row, col=col, tickformat="%Y-%m", tickangle=0)

    # increment grid position
    col += 1
    if col > ncols:
        col = 1
        row += 1

# Layout / title / legend spacing
fig.update_layout(
    title=dict(
        text=f"{horizon} Rolling Returns: Fund vs Benchmark",
        font=dict(size=20, family="Arial", color="#333"),
        x=0.5,
        xanchor="center",
        y=0.97
    ),
    height=total_height,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.93,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.6)"
    ),
    margin=dict(l=60, r=40, t=120, b=60),
    template="plotly_white"
)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)

# ---------- ALPHA SUMMARY TABLE ----------
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">


<p>
The faceted charts compare each ELSS fund’s <b>3-year rolling return</b> (solid line) with its <b>TRI benchmark</b> (dashed line). 
The shaded regions depict <b>alpha = (Fund 3Y – Benchmark 3Y)</b> — 
where <span style="color:#0b6fa4;"><b>blue shading</b></span> represents outperformance (positive alpha) 
and <span style="color:firebrick;"><b>red shading</b></span> indicates underperformance (negative alpha). 
This visualization highlights not only when funds outperformed their benchmarks, but also how consistently and by what magnitude.
</p>

<h5>Key Observations</h5>

<ul>
<li><b>Timing and Market Phases:</b> All funds exhibit the pandemic-induced dip in early 2020, followed by recovery. 
The <span style="color:#0b6fa4;">blue patches</span> post-2020 represent periods of active alpha generation. 
The broad market rally of early–mid 2023 emerges as a <b>structural turning point</b>, where performance patterns across schemes diverged sharply.</li>

<li><b>Consistency of Alpha (Persistence):</b>
  <ul>
    <li><b>Mirae Asset ELSS:</b> Displays the most persistent positive alpha across 2020–2024, reflecting superior portfolio positioning and sustained active management.</li>
    <li><b>SBI Long Term Equity Fund:</b> Shifted strongly to positive alpha during the 2023 rally and maintained it thereafter, suggesting robust post-rally strength.</li>
    <li><b>DSP ELSS:</b> Mixed but mostly positive alpha during 2021–2023, indicating steady though moderate outperformance.</li>
    <li><b>Axis ELSS:</b> Early parity or mild outperformance turned into extended underperformance post-2023, implying lost momentum in cyclical markets.</li>
    <li><b>HDFC ELSS:</b> Primarily negative alpha pre-2023, followed by partial recovery during the rally, signifying a late-cycle catch-up rather than sustained alpha.</li>
  </ul>
</li>

<li><b>Magnitude of Alpha:</b> 
Mirae Asset and SBI show the <b>widest and most sustained blue regions</b>, denoting strong and consistent outperformance. 
Axis and HDFC, by contrast, exhibit larger <b>red regions</b> post-2023, indicating meaningful underperformance. 
DSP’s narrower blue fills suggest modest but reliable alpha.</li>

<li><b>Strategic Implications:</b> 
The 2023 rally marked a <b>decisive inflection point</b>. 
Funds tilted toward <b>mid- and small-cap</b> exposure or active sector rotation (Mirae, SBI) generated durable alpha. 
Meanwhile, conservative, large-cap-heavy portfolios (Axis, HDFC) underperformed, demonstrating how market positioning shaped alpha persistence.</li>
</ul>

<p>
Overall, <b>Mirae Asset</b> and <b>SBI</b> emerged as the most consistent alpha generators, 
while <b>DSP</b> offered balanced but moderate performance. 
In contrast, <b>Axis</b> and <b>HDFC</b> lost relative strength post-2023, underscoring the importance of adaptive portfolio management 
in sustaining outperformance during shifting market cycles.
</p>

</div>
    """,
    unsafe_allow_html=True
)
