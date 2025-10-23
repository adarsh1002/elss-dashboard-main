import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Cost Effectiveness Analysis", layout="wide")
st.title("Cost Effectiveness Analysis")
st.markdown(""" <hr style="border: 1px solid #cccccc;"> """, unsafe_allow_html=True)
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
</div>""",
    unsafe_allow_html=True)
st.markdown("""
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">
  <h5>Methodology Overview</h5>
    <ul>
    <li><b>Data Sources:</b> Expense ratio data was primarily sourced from official fund fact sheets released monthly by the respective AMCs. 
    Where data gaps existed, official websites were used to extract TER information for Direct Plans.</li>
    <li><b>Time Period:</b> Data was compiled for the five calendar years from 2020 to 2024, aligning with the study's broader temporal scope. 
    When monthly data was available, it was averaged to obtain annualized TERs.</li>
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
        <li><b>5-Year Average TER</b> : to compare long-term cost positions.</li>
        <li><b>Return per Expense Ratio Unit</b> : calculated as 5-year CAGR divided by average TER, indicating cost-efficiency of returns.</li>
        <li><b>Fund Ranking</b> : based on cost-effectiveness metrics.</li>
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
    df = df[(df["Year"] >= 2020) & (df["Year"] <= 2024)]
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
st.sidebar.header("Filters for the Expense Spread Plot")
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
fig.update_layout(showlegend=True, height=700, margin=dict(l=180, r=40, t=80, b=60))

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

st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Interpretation</h4>

  <p>
  Each box in the visualization represents the <b>interquartile range (IQR)</b>, 
  with the horizontal line indicating the <b>median</b>. 
  Blue and green annotations mark the <b>mean</b> and <b>median</b> values respectively, 
  while the overlaid dots correspond to individual monthly Total Expense Ratio (TER) data points, 
  capturing the stability or volatility in each fund’s cost structure over the five-year period.
  </p>

  <h5>Key Highlights</h5>

<ol>
<li><b>Central Tendencies (Mean & Median):</b> 
<p><b>HDFC ELSS Tax Saver Fund</b> records the highest average TER (1.22%) and median (1.23%),signifying a consistently higher cost structure. <b>Mirae Asset ELSS Fund</b> displays the lowest average TER (0.51%) and median (0.55%), 
      making it the most cost-efficient among the compared schemes. For <b>Axis</b>, <b>DSP</b>, and <b>HDFC</b>, the proximity between mean and median suggests symmetrical distributions, indicating an absence of major skewness in their expense data.
</p></li>

<li><b>Dispersion and Variability:</b> 
<p><b>SBI Long Term Equity Fund</b> exhibits the widest IQR, implying higher variability in its monthly expense ratios.<b>Axis ELSS Tax Saver Fund</b> has the narrowest box, showing a highly stable and predictable TER policy. <b>Mirae Asset ELSS Fund</b> also demonstrates a tightly grouped spread, reinforcing its reputation for low and consistent costs.
</p></li>

<li><b>Outliers and Irregularities:</b> 
<p>Occasional outliers appear in <b>DSP</b> and <b>Axis</b> schemes, likely arising from temporary spikes in AUM or 
      extraordinary operational costs.  <b>HDFC</b> and <b>SBI</b>, though high in TER, display limited outliers—suggesting structurally embedded higher costs 
      rather than erratic fee changes.</p>
</li>

<li><b>Relative Positioning of Funds:</b> 
<p>
In the comparative landscape, <b>Mirae</b> and <b>Axis</b> occupy the lower-cost spectrum, 
appealing to cost-conscious investors.  <b>HDFC</b> and <b>SBI</b> represent premium-priced schemes, where higher TERs may need to be justified through 
      superior performance or alpha generation. <b>DSP</b> lies in the mid-range, reflecting moderate costs with some intermittent fluctuations.
</p></li>

<li><b>Investment Implications:</b> 
<p>Funds with consistently low TERs, such as <b>Mirae Asset ELSS Fund</b>, are particularly suitable for long-term SIP investors, 
      as lower expenses enhance compounding benefits over time. 
      Conversely, high-cost schemes like <b>HDFC</b> or <b>SBI</b> should be justified only if they consistently deliver superior risk-adjusted returns. 
      Under the three-year ELSS lock-in, expense ratios play a direct role in determining net investor outcomes, 
      making <b>cost-effectiveness a key selection parameter</b> for long-term wealth creation.</p></li>
</ol>

</div>
    """,
    unsafe_allow_html=True
)
st.subheader("Cost Efficiency Analysis")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Introduction: Cost Efficiency and Performance Alignment in ELSS Funds</h4>

  <p>
  In the realm of mutual fund evaluation, particularly for tax-saving instruments such as 
  <b>Equity Linked Savings Schemes (ELSS)</b>, investors often focus primarily on absolute returns. 
  However, this narrow perspective frequently overlooks the crucial influence of the <b>expense ratio</b> — 
  the annual fee charged by fund houses to manage a scheme. 
  For ELSS funds, where investments are mandatorily locked in for three years, 
  even marginal differences in expense ratios can significantly affect an investor’s net returns over time.
  </p>

  <p>
  Moreover, a scheme that appears superior based on high historical returns may, in reality, exhibit 
  inconsistent or volatile performance. 
  To derive a more accurate and representative understanding, this analysis employs 
  <b>median values</b> for both returns and expense ratios rather than averages. 
  Median values provide a robust measure of central tendency in datasets 
  that are prone to outliers or skewed distributions — a common occurrence in financial markets.
  </p>

  <ul>
    <li>Funds may record one-off extreme gains or losses in certain periods due to market shocks 
    (e.g., the COVID-19 crash or rebound), which distort average return calculations.</li>
    <li>Expense ratios may also temporarily fluctuate because of changes in Assets Under Management (AUM) 
    or variations in operational expenses.</li>
  </ul>

  <p>
  Accordingly, the <b>3-year return</b> has been chosen as the principal performance metric, 
  aligning with the statutory lock-in period of ELSS under <b>Section 80C</b> of the Income Tax Act. 
  ELSS funds, by design, require investors to remain invested for a minimum of three years, 
  making short-term returns less meaningful. 
  Evaluating performance over this horizon filters out market noise, 
  capturing the true capability of fund managers to deliver consistent, sustainable value.
  </p>

  <p>
  By using <b>median 3-year returns</b> alongside <b>median expense ratios</b>, 
  the analysis presents a realistic picture of an investor’s typical experience. 
  This approach removes distortions caused by outliers and improves comparability across schemes, 
  helping assess consistency and cost-effectiveness over a multi-year investment horizon.
  </p>

  <h5>Methodological Overview</h5>
  <p>
  The relationship between cost and performance is examined through a 
  <b>three-dimensional visualization</b>:
  </p>

  <ul>
    <li><b>X-axis:</b> Median Expense Ratio (%) – representing the typical cost borne by the investor.</li>
    <li><b>Y-axis:</b> Median 3-Year Return (%) – reflecting consistent investment performance.</li>
    <li><b>Bubble Size & Label:</b> Average Sharpe Ratio – indicating the risk-adjusted efficiency of each fund.</li>
  </ul>

  <p>
  Together, these dimensions provide a balanced perspective on 
  <b>cost-to-performance efficiency</b> — revealing which ELSS schemes deliver the 
  best blend of cost control, return stability, and risk-adjusted value for investors.
  </p>

</div>
    """,
    unsafe_allow_html=True
)
# Objective 3 — Interactive Cost Efficiency Bubble Chart (Plotly + Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


DATA_PATH = "data/Objective_3_Data.xlsx"  # update path if needed
SHEET_NAME = "Sheet1"

@st.cache_data
def load_data(path=DATA_PATH, sheet_name=SHEET_NAME):
    xls = pd.ExcelFile(path)
    df = xls.parse(sheet_name)
    # strip column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

# Load
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Put the file in data/ or change DATA_PATH.")
    st.stop()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Ensure required columns exist (attempt to detect variants)
col_map = {}
for needed in ["Scheme Name", "Sharpe Ratio", "Expense Ratio", "Return 3 Year"]:
    if needed in df.columns:
        col_map[needed] = needed
    else:
        # fuzzy detection
        cand = [c for c in df.columns if needed.split()[0].lower() in c.lower()]
        col_map[needed] = cand[0] if cand else None

missing = [k for k, v in col_map.items() if v is None]
if missing:
    st.error(f"Missing required columns in the file: {missing}. Columns found: {df.columns.tolist()}")
    st.stop()

# Rename columns to canonical names
df = df.rename(columns={col_map["Scheme Name"]: "Scheme Name",
                        col_map["Sharpe Ratio"]: "Sharpe Ratio",
                        col_map["Expense Ratio"]: "Expense Ratio",
                        col_map["Return 3 Year"]: "Return 3 Year"})

# Coerce numeric
df["Sharpe Ratio"] = pd.to_numeric(df["Sharpe Ratio"], errors="coerce")
df["Expense Ratio"] = pd.to_numeric(df["Expense Ratio"], errors="coerce")
df["Return 3 Year"] = pd.to_numeric(df["Return 3 Year"], errors="coerce")

# Optionally let user filter schemes (sidebar)
st.sidebar.header("Controls for Cost Efficiency Plot")
all_schemes = sorted(df["Scheme Name"].dropna().unique().tolist())
selected_schemes = st.sidebar.multiselect("Select scheme(s) to include", options=all_schemes, default=all_schemes)

# Optionally let user choose aggregation (median/mean for returns)
agg_return_choice = st.sidebar.selectbox("Aggregate 3-Year Return by", options=["median", "mean"], index=0)
agg_expense_choice = st.sidebar.selectbox("Aggregate Expense Ratio by", options=["median", "mean"], index=0)
agg_sharpe_choice = st.sidebar.selectbox("Aggregate Sharpe Ratio by", options=["mean", "median"], index=0)

# Filter
plot_df = df[df["Scheme Name"].isin(selected_schemes)].copy()
if plot_df.empty:
    st.info("No data after filtering. Select at least one scheme.")
    st.stop()

# Aggregate
agg_funcs = {
    "Expense Ratio": (agg_expense_choice),
    "Return 3 Year": (agg_return_choice),
    "Sharpe Ratio": (agg_sharpe_choice)
}

# build agg mapping for pandas
agg_map = {
    "Expense Ratio": agg_funcs["Expense Ratio"],
    "Return 3 Year": agg_funcs["Return 3 Year"],
    "Sharpe Ratio": agg_funcs["Sharpe Ratio"]
}

bubble_df = (
    plot_df
    .groupby("Scheme Name", as_index=False)
    .agg(agg_map)
    .rename(columns={
        "Expense Ratio": "Median Expense Ratio" if agg_expense_choice == "median" else "Avg Expense Ratio",
        "Return 3 Year": "Median 3-Year Return" if agg_return_choice == "median" else "Avg 3-Year Return",
        "Sharpe Ratio": "Avg Sharpe Ratio" if agg_sharpe_choice == "mean" else "Median Sharpe Ratio"
    })
)

# Standardize column names for plotting
expense_col = [c for c in bubble_df.columns if "Expense" in c][0]
return_col = [c for c in bubble_df.columns if "3-Year" in c or "3 Year" in c][0]
sharpe_col = [c for c in bubble_df.columns if "Sharpe" in c][0]

# Drop NA rows
bubble_df = bubble_df.dropna(subset=[expense_col, return_col, sharpe_col]).reset_index(drop=True)
if bubble_df.empty:
    st.info("No aggregated numeric rows to plot after cleaning.")
    st.stop()

# Normalize marker sizes for Plotly
# size is based on sharpe_col; scale to reasonable marker size range
size_vals = bubble_df[sharpe_col].astype(float).fillna(0)
# avoid negative/zero sizes causing issues: shift if necessary
min_size, max_size = 8, 60
if size_vals.max() == size_vals.min():
    marker_sizes = np.clip((size_vals - size_vals.min()) + (max_size+min_size)/2, min_size, max_size)
else:
    marker_sizes = ((size_vals - size_vals.min()) / (size_vals.max() - size_vals.min())) * (max_size - min_size) + min_size

# Create interactive Plotly scatter (bubble) chart
fig = px.scatter(
    bubble_df,
    x=expense_col,
    y=return_col,
    size=marker_sizes,            # pass computed sizes (Plotly will scale again, but OK)
    size_max=max_size,
    color="Scheme Name",
    hover_name="Scheme Name",
    hover_data={expense_col: ':.3f', return_col: ':.3f', sharpe_col: ':.3f'},
    labels={expense_col: "Median Expense Ratio (%)", return_col: "Median 3-Year Return (%)"},
    template="plotly_white",
    title="Cost Efficiency: Median 3-Year Return vs Median Expense Ratio"
)

# Add text labels (scheme + Sharpe) positioned to the right
for i, row in bubble_df.iterrows():
    fig.add_annotation(
        x=float(row[expense_col]),
        y=float(row[return_col]),
        text=f"{row['Scheme Name']}<br>Sharpe: {row[sharpe_col]:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=11),
        opacity=0.9
    )

# Layout tweaks
fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")), selector=dict(mode='markers'))
fig.update_layout(
    legend_title_text="Scheme",
    height=640,
    margin=dict(l=80, r=300, t=80, b=80)
)

# Display
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  <h4>Interpretation</h4>

  <p>
  Each bubble in the chart represents an <b>ELSS scheme</b>, with its size proportional to the <b>average Sharpe Ratio</b>—a measure of risk-adjusted efficiency. 
  The <b>X-axis</b> plots the median <b>Expense Ratio (%)</b>, while the <b>Y-axis</b> shows the median <b>3-Year Return (%)</b>, enabling a direct view of each fund’s 
  cost-to-performance balance.
  </p>

<ul>
<li><b>Mirae Asset ELSS Tax Saver Fund:</b> The most cost-efficient fund, combining the <b>lowest median expense ratio (≈ 0.55%)</b> 
with the <b>highest median 3-year return (≈ 20.1%)</b> and a strong <b>Sharpe Ratio of 0.70</b>. It delivers maximum return per unit of cost and risk, ideal for long-term ELSS investors.</li>

<li><b>DSP ELSS Tax Saver Fund:</b> Offers an excellent risk-return balance with a <b>median 3-year return of ≈ 19.8%</b> 
    and a moderate <b>expense ratio of ≈ 0.82%</b>. Its <b>Sharpe Ratio (0.67)</b> indicates efficient risk management, making it a strong, balanced choice.</li>

<li><b>SBI Long Term Equity Fund:</b> Maintains a higher cost (<b>≈ 1.19%</b>) but compensates with robust returns (<b>≈ 18.6%</b>) and the <b>highest Sharpe Ratio (0.74)</b>. 
    Suitable for investors prioritizing <b>risk-adjusted efficiency</b> even at a slightly higher fee.</li>

<li><b>Axis ELSS Tax Saver Fund:</b> Displays the weakest performance, with <b>median return ≈ 12.9%</b> and <b>Sharpe Ratio 0.45</b>. 
While costs are moderate (<b>≈ 0.79%</b>), its limited return efficiency reduces overall attractiveness.</li>

<li><b>HDFC ELSS Tax Saver Fund:</b> The <b>least cost-efficient</b> option, showing the <b>highest expense ratio (≈ 1.23%)</b> 
and only <b>moderate returns (≈ 15.6%)</b>. 
Its <b>Sharpe Ratio (0.37)</b> is the lowest, indicating poor risk-adjusted performance relative to cost.</li>
</ul>

<p>
The analysis demonstrates that <b>lower-cost schemes</b> such as Mirae Asset and DSP consistently outperform in both absolute and 
risk-adjusted terms, proving that efficient management can yield strong investor outcomes without high fees. 
Conversely, higher-cost options like HDFC and Axis underperformed in cost-return efficiency, while SBI maintained a premium 
position supported by superior risk control.
</p>

  <h3>Summary of Findings</h3>
  <p>
  This chapter evaluated the <b>cost-effectiveness</b> of leading ELSS funds by examining <b>expense ratios</b>, 
  <b>3-year returns</b>, and <b>Sharpe Ratios</b>. 
  The <b>boxplot analysis</b> revealed variation in monthly expense structures (2020–2024), with 
  <b>Mirae Asset</b> as the lowest-cost and most consistent scheme, and <b>HDFC</b> as the highest-cost. 
  The subsequent <b>bubble-chart analysis</b> mapped cost, performance, and risk efficiency simultaneously, confirming that 
  <b>low-cost funds (Mirae Asset and DSP)</b> delivered superior value, while <b>HDFC and Axis</b> lagged in both 
  return efficiency and stability. 
  These insights highlight that investors should evaluate not only returns but also <b>how efficiently those returns are delivered</b>, 
  considering both <b>expenses</b> and <b>risk-adjusted performance</b> to make informed long-term investment decisions in ELSS.
  </p>

</div>
""",
    unsafe_allow_html=True
)
