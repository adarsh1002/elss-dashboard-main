import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Cost Effectiveness Analysis", layout="wide")
st.title("Cost Effectiveness Analysis")
st.sidebar.title("Cost Effectiveness Analysis")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Objective 3: Cost Effectiveness of ELSS Schemes</h4>

  <p>
    While performance and risk-adjusted returns often dominate mutual fund evaluation, the <b>expense ratio</b> remains an equally critical yet frequently overlooked metric. 
     The expense ratio represents the annual fee charged to investors to cover a fund’s operating costs, including management fees, administrative expenses, distribution charges, and other associated costs. 
     For <b>Equity Linked Savings Schemes (ELSS)</b>—which already carry a mandatory three-year lock-in period—higher expenses can significantly erode long-term gains, 
     making cost-effectiveness a vital component of comparative analysis.
  </p>

  <p>
    Since ELSS schemes are designed as <b>long-term tax-saving investments</b>, even marginal differences in expense ratios can compound substantially over time, 
    directly impacting investor wealth creation. 
     Fund houses adopt varying cost structures depending on their brand positioning, operational scale, and management philosophy. 
    Some may justify higher fees through active management and proprietary research, while others may employ leaner structures in direct plans to attract retail investors seeking cost efficiency.
  </p>

  <p>
    In this context, analyzing the <b>Total Expense Ratios (TERs)</b> of ELSS schemes across the top five Asset Management Companies (AMCs) in India 
    provides insights into which funds offer the best <b>value for money</b>. 
    The objective is to assess whether higher-cost funds deliver proportionately superior returns or if lower-cost schemes can achieve comparable performance, 
     thereby offering superior cost efficiency.
  </p>

  <p>
    The focus of this analysis is to compare and interpret the <b>TERs of selected ELSS Direct Plans</b> over the five-year period from 2020 to 2024, 
    and evaluate how these costs influence net investor returns. 
    This dimension adds depth to the broader comparative financial performance study by linking fund expenses directly to return outcomes and efficiency.
  </p>

  <h5>Methodology Overview</h5>

  <ul>
    <li><b>Data Sources:</b> Expense ratio data was primarily sourced from official fund fact sheets released monthly by the respective AMCs. 
    Where data gaps existed, official websites were used to extract TER information for Direct Plans.</li>

    <li><b>Time Period:</b> Data was compiled for the five calendar years from 2020 to 2024, aligning with the study's broader temporal scope. When monthly data was available, it was averaged to obtain annualized TERs.</li>

    <li><b>Data Processing:</b> Python scripts were utilized to automate data extraction and compute annual averages for each fund, ensuring consistency and accuracy in analysis.</li>

    <li><b>Comparative Analysis:</b> The following visualizations were employed to illustrate cost dynamics:
         <ul>
            <li>Year-wise TER trends across all funds.</li>
            <li>Average TER comparison over five years.</li>
            <li>Correlation between TER and 5-year CAGR returns to evaluate cost efficiency.</li>
        </ul>
    </li>

    <li><b>Key Metrics Derived:</b>
      <ul>
        <li><b>5-Year Average TER</b> - to compare long-term cost positions.</li>
        <li><b>Return per Expense Ratio Unit</b> - calculated as 5-year CAGR divided by average TER, indicating cost-efficiency of returns.</li>
        <li><b>Fund Ranking</b> - based on cost-effectiveness metrics.</li>
      </ul>
    </li>
  </ul>

  <p>
    Overall, this analysis enables a nuanced understanding of <b>cost-to-return efficiency</b> within ELSS mutual funds, 
     helping investors identify schemes that not only perform well but also optimize expenses for long-term wealth creation.
  </p>

</div>
    """,
    unsafe_allow_html=True
)
st.subheader("Expense Ratio Spread")
# ---------- Robust loader using month_year column ----------
import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser

DATA_PATH = "data/Objective_3_Data.xlsx"

@st.cache_data
def load_expense_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # If 'month_year' exists, parse it to a datetime (first day of month)
    if "month_year" in df.columns:
        # coerce strings, try multiple parse approaches
        def parse_month_year(val):
            if pd.isna(val):
                return pd.NaT
            try:
                # try direct pandas parse (fast)
                return pd.to_datetime(str(val), errors="coerce", dayfirst=False)
            except Exception:
                try:
                    # fallback to dateutil parser
                    return parser.parse(str(val), default=pd.Timestamp("1900-01-01"))
                except Exception:
                    return pd.NaT

        df["Date"] = df["month_year"].apply(parse_month_year)
        # If parse yields only year-month (no day), normalize day to 1
        df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    else:
        # try existing Date-like columns
        date_cols = [c for c in df.columns if "date" in str(c).lower()]
        if date_cols:
            df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            df["Date"] = pd.NaT

    # Derive Year and Month columns
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Ensure Expense Ratio numeric (strip % and other chars)
    if "Expense Ratio" in df.columns:
        df["Expense Ratio"] = (
            df["Expense Ratio"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
        )
        df["Expense Ratio"] = pd.to_numeric(df["Expense Ratio"], errors="coerce")
    else:
        raise ValueError("No 'Expense Ratio' column found in the data file.")

    # Ensure Scheme Name exists (try to auto-detect)
    if "Scheme Name" not in df.columns:
        cand = [c for c in df.columns if "scheme" in str(c).lower() or "fund" in str(c).lower()]
        if cand:
            df = df.rename(columns={cand[0]: "Scheme Name"})
        else:
            raise ValueError("No 'Scheme Name' column found in the data file.")

    # Drop rows missing essential items
    df = df.dropna(subset=["Scheme Name", "Expense Ratio"]).copy()
    return df

# Usage: load the df at top of the page
try:
    df = load_expense_data()
except Exception as e:
    st.error(f"Error loading expense data: {e}")
    st.stop()

# Now build year list for sidebar slider
all_years = sorted(df["Year"].dropna().unique().astype(int).tolist())
try:
    df = load_expense_data()
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Place it in data/ folder or adjust DATA_PATH.")
    st.stop()

# Sidebar controls
st.sidebar.header("Filters")
all_years = sorted(df["Year"].dropna().unique().astype(int).tolist())
if not all_years:
    st.sidebar.warning("No year values detected in the data.")
    year_min, year_max = None, None
else:
    year_min, year_max = all_years[0], all_years[-1]
    year_range = st.sidebar.slider("Select Year or Range", min_value=year_min, max_value=year_max,
                                   value=(year_min, year_max), step=1)

# Scheme selector (optional)
schemes = sorted(df["Scheme Name"].unique().tolist())
selected_schemes = st.sidebar.multiselect("Select Scheme(s)", options=schemes, default=schemes)

# Filter dataframe based on selections
mask = df["Scheme Name"].isin(selected_schemes)
if year_min is not None:
    mask &= df["Year"].between(year_range[0], year_range[1])
df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.info("No data for the selected filters. Adjust the year range / schemes.")
    st.stop()

# Compute summary stats (mean & median) for annotations
summary = df_filtered.groupby("Scheme Name")["Expense Ratio"].agg(["mean", "median"]).reset_index()
summary["mean"] = summary["mean"].round(3)
summary["median"] = summary["median"].round(3)

# Build Plotly box plot with jittered points (points='all')
fig = px.box(
    df_filtered,
    x="Expense Ratio",
    y="Scheme Name",
    color="Scheme Name",
    orientation="h",
    points="all",            # show all points like swarm
    hover_data=["Year"],
    category_orders={"Scheme Name": schemes},
    title=f"Expense Ratio Distribution ({year_range[0]} - {year_range[1]})",
    template="plotly_white",
)

# Remove color legend (optional) and clean layout
fig.update_layout(showlegend=False, height=700, margin=dict(l=180, r=40, t=80, b=60))

# Add mean and median annotations as scatter traces for each scheme
# Create a mapping from scheme to y-position (categories -> numeric positions)
scheme_order = fig.data[0].y if fig.data else df_filtered["Scheme Name"].unique()
# Build per-scheme annotation traces
for _, row in summary.iterrows():
    scheme = row["Scheme Name"]
    mean_val = row["mean"]
    median_val = row["median"]
    # add mean marker
    fig.add_trace(
        go.Scatter(
            x=[mean_val],
            y=[scheme],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12, color="blue"),
            text=[f"Mean: {mean_val:.2f}"],
            textposition="middle right",
            showlegend=False,
            hovertemplate=f"{scheme}<br>Mean: {mean_val:.3f}<extra></extra>",
        )
    )
    # add median marker
    fig.add_trace(
        go.Scatter(
            x=[median_val],
            y=[scheme],
            mode="markers+text",
            marker=dict(symbol="square", size=10, color="green"),
            text=[f"Median: {median_val:.2f}"],
            textposition="bottom right",
            showlegend=False,
            hovertemplate=f"{scheme}<br>Median: {median_val:.3f}<extra></extra>",
        )
    )

fig.update_xaxes(title_text="Expense Ratio (%)")
fig.update_yaxes(title_text="Scheme Name", automargin=True)

# Display
st.plotly_chart(fig, use_container_width=True)