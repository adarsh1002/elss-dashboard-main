import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(page_title=" Risk & Volatility", layout="wide")
st.title("Risk and Volatility Analysis")
st.markdown(""" <hr style="border: 1px solid #cccccc;"> """, unsafe_allow_html=True)
st.subheader("Introduction")
st.markdown(
    """
    
    <div style="text-align: justify;">
      
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
st.subheader("Risk Profile Analysis")
st.markdown("""
            <div style="text-align: justify;">
             <h4>Standard deviation - Volatility Trend Analysis </h4>
            <p>Standard deviation is a statistical measure that reflects the total volatility of a mutual fund's returns. It captures how much the fund's returns deviate from its average over time. A higher standard deviation implies a more volatile scheme, indicating a greater level of risk and unpredictability in its performance.
To evaluate this aspect for ELSS schemes, a bar chart was constructed showing the average annual standard deviation for five leading ELSS funds from 2020 to 2024.
</p>
            </div>
            
            
            """, unsafe_allow_html=True)


#analysis part
# pages/Chapter2_Risk_Volatility.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



DATA_PATH = "data/Data_Obective2_final.xlsx"  # adjust path if needed

@st.cache_data
def load_sd_data(path=DATA_PATH, sheet_name=0):
    """Load, clean and return a DataFrame for standard deviation analysis."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    # normalize column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    # required columns check
    required = {"Date", "Scheme Name", "Standard Deviation"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input file must contain columns: {required}. Found: {set(df.columns)}")
    # parse date and year
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Scheme Name", "Standard Deviation"]).copy()
    df["Year"] = df["Date"].dt.year
    # remove percentage signs and convert to numeric (defensive)
    df["Standard Deviation"] = (
        df["Standard Deviation"].astype(str)
          .str.replace("%", "", regex=False)
          .str.replace(",", "", regex=False)
          .str.strip()
    )
    df["Standard Deviation"] = pd.to_numeric(df["Standard Deviation"], errors="coerce")
    # drop rows where conversion failed
    df = df.dropna(subset=["Standard Deviation"])
    # exclude 2019 as in original code
    df = df[df["Year"] != 2019].reset_index(drop=True)
    return df

# Load data
try:
    sd_raw = load_sd_data()
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}. Please place the Excel file in the data/ folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading STD data: {e}")
    st.stop()

# Sidebar controls (reuse shared variables if present)
st.sidebar.header("Controls - Risk & Volatility Analysis")
all_schemes = sorted(sd_raw["Scheme Name"].unique().tolist())
if "selected_schemes" in globals():
    # if shared controls exist, use them
    selected_schemes = [s for s in all_schemes if s in selected_schemes] if 'selected_schemes' in globals() else all_schemes
else:
    selected_schemes = st.sidebar.multiselect("Select Scheme(s)", options=all_schemes, default=all_schemes)

years_sorted = sorted(sd_raw["Year"].unique())
min_year, max_year = years_sorted[0], years_sorted[-1]
selected_years = st.sidebar.multiselect("Select Year(s) to display", options=years_sorted, default=years_sorted)

# Aggregation: average standard deviation by Year and Scheme
agg = (
    sd_raw[sd_raw["Scheme Name"].isin(selected_schemes) & sd_raw["Year"].isin(selected_years)]
    .groupby(["Year", "Scheme Name"], as_index=False)["Standard Deviation"]
    .mean()
)

if agg.empty:
    st.info("No data available for the chosen scheme(s) / year(s). Adjust filters.")
    st.stop()

# Round values for display & plotting
agg["Standard Deviation"] = agg["Standard Deviation"].round(2)

# Create Plotly grouped bar chart
fig = px.bar(
    agg,
    x="Year",
    y="Standard Deviation",
    color="Scheme Name",
    barmode="group",
    text="Standard Deviation",
    labels={"Standard Deviation": "Standard Deviation (%)"},
    category_orders={"Year": sorted(agg["Year"].unique())},
    template="plotly_white",
    title="Average Standard Deviation of ELSS Schemes by Year"
)

# Layout polish: legend on right, bigger size, margins for legend
fig.update_layout(
    legend=dict(title="Scheme Name", orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
    margin=dict(t=100, b=80, l=60, r=220),
    height=560,
    width=None,
    hovermode="x unified",
    font=dict(size=13)
)

# Format text annotations and hover
fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0.5)
fig.update_yaxes(title_text="Standard Deviation (%)", showgrid=True, gridcolor="lightgray")
fig.update_xaxes(type="category")  # keep discrete years in order

# Display chart
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div style="text-align: justify; line-height:1.5; font-family: Arial, sans-serif;">
      <h4>Interpretation of the Standard Deviation (Volatility) Bar Plot</h4>
      <p>
        The bar plot reveals that volatility was highest between <b>2020 and 2022</b>, reflecting the market turbulence caused by the COVID-19 shock and related macroeconomic disruptions. 
        Certain schemes — notably <b>DSP Tax Saver</b> and <b>Mirae Asset ELSS</b> — exhibited elevated standard deviations (>22%) in 2021–2022, which coincided with very strong 1-year returns that year. 
        This pattern supports the classical risk–return trade-off: higher volatility in these funds was, during that period, accompanied by higher short-term returns.
      </p>

      <p>
        By contrast, funds such as <b>HDFC ELSS</b> displayed consistently lower volatility across the period, trading off upside potential for stability — a profile that appeals to risk-averse investors. 
        <b>SBI Long Term Equity Fund</b> demonstrated a balanced profile with moderate volatility and robust multi-horizon returns, indicating efficient risk management with growth orientation. 
        <b>Axis ELSS</b> recorded relatively muted volatility by 2024 but simultaneously showed weaker returns, raising questions about its risk-adjusted efficiency.
      </p>

      <p>
        Overall, the bar plot underscores that higher short-term volatility in some ELSS schemes was often associated with higher short-term returns, while lower volatility funds tended to deliver steadier but more moderate performance. 
        Investors should therefore evaluate funds not only on absolute returns but on how returns compensate for the volatility undertaken — i.e., on a risk-adjusted basis.
      </p>

      <h4>Scheme-level Summary</h4>
      <table style="width:100%; border-collapse: collapse; font-size:13px;">
        <thead>
          <tr>
            <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Scheme</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Risk (Std Dev)</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Return Profile</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Inference</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>HDFC ELSS</b></td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Very low</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Low–moderate (11.9%–25.3%)</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Conservative; suitable for stability-seeking investors.</td>
          </tr>
          <tr>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>Mirae Asset ELSS</b></td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">High → moderate</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">High (5Y: ~25.1%; 1Y spikes)</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Strong risk–reward balance; attractive for growth-oriented investors.</td>
          </tr>
          <tr>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>DSP Tax Saver</b></td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">High</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">High (e.g., 1Y: ~32.7%)</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Aggressive, growth-oriented; higher volatility compensated by high short-term returns.</td>
          </tr>
          <tr>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>SBI Long Term Equity Fund</b></td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Moderate</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Strong (1Y ~26.9%; 3Y ~17.4%)</td>
            <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Balanced performer — suitable for moderate-to-high risk investors seeking consistent growth.</td>
          </tr>
          <tr>
            <td style="padding:8px;"><b>Axis ELSS</b></td>
            <td style="padding:8px;">Moderate (lower by 2024)</td>
            <td style="padding:8px;">Low–average (5Y ~13.5%)</td>
            <td style="padding:8px;">Risk-controlled but underperforming; limited reward for risk taken.</td>
          </tr>
        </tbody>
      </table>

      <p style="margin-top:10px;">
        <small><b>Note:</b> Values cited (returns and volatility) are approximate and drawn from the analysis period. Investors should consider multiple metrics (including beta and Sharpe ratio) and their investment horizon before drawing final conclusions.</small>
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Sharpe Ratio – Risk-Adjusted Performance Analysis")
text_sharpe = """
            <div style="text-align: justify;">
            The Sharpe Ratio serves as a critical measure of how effectively a mutual fund delivers returns relative to the risk it undertakes. By evaluating the excess return over the risk-free rate per unit of standard deviation, the Sharpe Ratio allows investors to assess the efficiency of a fund's return generation. T
            The line chart presented below illustrates the year-wise average Sharpe Ratio for selected ELSS schemes from 2020 to 2024.
            </div>
            """
st.markdown(text_sharpe, unsafe_allow_html=True)
# -------------------------
# Sharpe Ratio section — reuse cached data & shared sidebar filters
# -------------------------
# -------------------------
# Sharpe Ratio section (REUSES Standard Deviation filters; no new filters created)
# -------------------------
# -------------------------
# Sharpe Ratio section — reuse SD filters (years and schemes only)
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

DATA_PATH = "data/Data_Obective2_final.xlsx"  # adjust if needed

# Cached loader (only loads once per session / file change)
@st.cache_data
def _load_chapter2_data(path=DATA_PATH, sheet_name=0):
    raw = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(raw, dict):
        raw = list(raw.values())[0].copy()
    df = raw.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Normalize columns
    if "Date" not in df.columns:
        poss = [c for c in df.columns if isinstance(c, str) and "date" in c.lower()]
        if poss:
            df = df.rename(columns={poss[0]: "Date"})
    if "Scheme Name" not in df.columns:
        poss = [c for c in df.columns if isinstance(c, str) and ("scheme" in c.lower() or "fund" in c.lower())]
        if poss:
            df = df.rename(columns={poss[0]: "Scheme Name"})

    # Detect sharpe column (fuzzy)
    sharpe_col = None
    if "Sharpe Ratio" in df.columns:
        sharpe_col = "Sharpe Ratio"
    else:
        cand = [c for c in df.columns if isinstance(c, str) and "sharpe" in c.lower()]
        if cand:
            sharpe_col = cand[0]
    if sharpe_col and sharpe_col != "Sharpe Ratio":
        df = df.rename(columns={sharpe_col: "Sharpe Ratio"})
    if "Sharpe Ratio" not in df.columns:
        df["Sharpe Ratio"] = np.nan

    # Clean types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Scheme Name"] = df["Scheme Name"].astype(str).str.strip()
    df["Sharpe Ratio"] = pd.to_numeric(df["Sharpe Ratio"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df = df.dropna(subset=["Date", "Scheme Name"]).reset_index(drop=True)
    return df

# Ensure data loaded into session_state (so we don't re-read file every interaction)
if "chapter2_df" not in st.session_state:
    try:
        st.session_state["chapter2_df"] = _load_chapter2_data()
    except FileNotFoundError:
        st.error(f"Chapter 2 data file not found: {DATA_PATH}. Please place it in data/ and reload.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading chapter 2 data: {e}")
        st.stop()

df_ch2 = st.session_state["chapter2_df"]

# --- REQUIRE: reuse SD filters 'selected_years' and 'selected_schemes' ---
# Try session_state first, then globals
if "selected_years" in st.session_state:
    sd_years = st.session_state["selected_years"]
elif "selected_years" in globals():
    sd_years = selected_years
else:
    sd_years = None

if "selected_schemes" in st.session_state:
    sd_schemes = st.session_state["selected_schemes"]
elif "selected_schemes" in globals():
    sd_schemes = selected_schemes
else:
    sd_schemes = None

# Validate presence
if not sd_years or not sd_schemes:
    st.error(
        "Sharpe Ratio chart requires the Standard Deviation filters to be applied first. "
        "Please ensure you have selected Scheme(s) and Year(s in SD section)."
    )
else:
    # Ensure sd_years is list of ints
    try:
        sd_years_int = [int(y) for y in sd_years]
    except Exception:
        sd_years_int = [int(y) for y in list(sd_years)]

    # derive date range from selected years: Jan 1 of min year to Dec 31 of max year
    min_year = min(sd_years_int)
    max_year = max(sd_years_int)
    start_date_derived = pd.to_datetime(date(min_year, 1, 1))
    end_date_derived = pd.to_datetime(date(max_year, 12, 31))

    # Apply filters: scheme, year, and derived date bounds (year filter is primary)
    mask = df_ch2["Scheme Name"].isin(sd_schemes)
    mask &= df_ch2["Year"].isin(sd_years_int)
    # additionally ensure date falls within derived range (defensive)
    mask &= (df_ch2["Date"] >= start_date_derived) & (df_ch2["Date"] <= end_date_derived)
    df_sharpe_filtered = df_ch2.loc[mask].copy()

    if df_sharpe_filtered.empty:
        st.info("No Sharpe Ratio data for the selected schemes/years. Try expanding the SD filter selection.")
    else:
        # compute yearly mean Sharpe ratio per scheme
        avg_sharpe = (
            df_sharpe_filtered.groupby(["Year", "Scheme Name"], as_index=False)["Sharpe Ratio"]
            .mean()
            .sort_values(["Scheme Name", "Year"])
        )

        if avg_sharpe["Sharpe Ratio"].isna().all():
            st.warning("Sharpe Ratio column appears empty or non-numeric for the filtered data. Check source file.")
        else:
            # Interactive Plotly line
            fig = px.line(
                avg_sharpe,
                x="Year",
                y="Sharpe Ratio",
                color="Scheme Name",
                markers=True,
                template="plotly_white",
                title="Year-wise Average Sharpe Ratio of Selected ELSS Schemes",
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=7))
            fig.update_layout(
                hovermode="x unified",
                legend=dict(title="Scheme", orientation="v", x=1.02, y=1),
                margin=dict(t=90, b=80, l=60, r=220),
                height=580,
                font=dict(size=13)
            )

            st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">
  

  <p>
  The chart highlights the evolution of <b>risk-adjusted performance</b> across the selected ELSS funds from 2020 to 2024. 
  While <b>Mirae Asset</b>, <b>SBI</b>, and <b>DSP ELSS</b> funds demonstrated consistent improvement, 
  <b>Axis ELSS</b> exhibited a declining efficiency despite maintaining moderate risk levels. 
  This divergence emphasizes the critical importance of evaluating <b>returns relative to risk</b> rather than focusing solely on absolute performance.
  </p>

  <h5>Key Insights</h5>
<ul>
<li><b>Mirae Asset ELSS Fund</b> and <b>SBI Long Term Equity Fund</b> consistently improved their Sharpe Ratios over the five-year period, 
    reaching values above <b>1.0</b> by 2023–2024. 
    This indicates efficient risk management and superior ability to deliver high-quality returns for the volatility undertaken.</li>

<li><b>DSP Tax Saver Fund</b> also showed a healthy trajectory, particularly after 2021, maintaining Sharpe Ratios near or above <b>1.0</b>. 
    This reflects balanced risk-taking behavior and consistent performance under varying market conditions.</li>

 <li><b>HDFC ELSS Tax Saver Fund</b> started with relatively low risk-adjusted efficiency but showed substantial improvement by 2024, 
    indicating effective portfolio realignment and enhanced compensation for the risk undertaken.</li>

<li><b>Axis ELSS Tax Saver Fund</b>, despite moderate volatility, showed inconsistent Sharpe Ratios throughout the study period. 
    Its declining trend after 2022 signals inefficiencies in converting risk exposure into meaningful returns, 
    raising concerns about its ability to sustain performance during shifting market conditions.</li>
</ul>

  <p>
  Overall, this analysis reinforces that <b>risk-adjusted return</b> is a more meaningful measure of performance than raw returns alone. 
  Funds such as <b>Mirae Asset</b>, <b>SBI</b>, and <b>DSP</b> emerged as <b>strong risk-adjusted performers</b>, 
  demonstrating consistent value creation for the risk assumed, while <b>Axis ELSS</b> reflected a potential misalignment between volatility and return generation.
  </p>
</div>
    """,
    unsafe_allow_html=True
)
st.subheader("Beta-Sharpe Trajectory of ELSS Schemes")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">
  

  <p>
  This section employs a <b>visual trajectory-based approach</b> to assess how selected ELSS mutual funds have evolved over the five-year period from <b>2020 to 2024</b>. 
  By combining measures of market sensitivity and risk-adjusted returns, the analysis captures how each scheme’s performance dynamics have shifted in response to changing market conditions.
  </p>

  <p>
  The visualization is constructed using <b>annual median values</b> of <b>Beta</b> and <b>Sharpe Ratio</b>, plotted on a two-dimensional grid, where:
  </p>

  <ul>
    <li><b>X-axis:</b> Represents <b>Beta</b> – a measure of the fund’s market sensitivity (systematic risk).</li>
    <li><b>Y-axis:</b> Represents <b>Sharpe Ratio</b> – a measure of the fund’s risk-adjusted returns.</li>
    <li><b>Directional arrows</b> connect yearly coordinates, illustrating how the fund’s position changes over time.</li>
    <li>Each fund is displayed in a <b>dedicated panel</b>, while other schemes are shown in the background with reduced opacity for contextual comparison.</li>
  </ul>

  <p>
  This dynamic visualization goes beyond static performance snapshots to highlight how actively or passively fund managers have responded to evolving market conditions. 
  It also helps determine whether their strategic decisions have <b>enhanced or weakened the scheme’s overall risk–return efficiency</b> over time.
  </p>
</div>
    """,
    unsafe_allow_html=True
)
# -------------------------
# Beta–Sharpe Trajectory panels (dashboard-ready)
# -------------------------
# ---------- Beta–Sharpe: robust cleaning, aggregation and plotting ----------
# -------------------------
# Beta–Sharpe Trajectory (using Standard Deviation filters)
# -------------------------
# ---------- Robust Beta–Sharpe aggregation + plotting (drop-in replacement) ----------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil

# -------------------------
# Assumptions:
# - df_ch2 is loaded in st.session_state["chapter2_df"] (or will be loaded below)
# - selected_schemes and selected_years exist in st.session_state or globals (SD filters)
# -------------------------

DATA_PATH = "data/Data_Obective2_final.xlsx"

# Helper: load if not already present
@st.cache_data
def _load_df(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    # normalize likely column names
    for c in df.columns:
        if "date" in str(c).lower() and "Date" not in df.columns:
            df = df.rename(columns={c: "Date"})
        if ("scheme" in str(c).lower() or "fund" in str(c).lower()) and "Scheme Name" not in df.columns:
            df = df.rename(columns={c: "Scheme Name"})
        if "sharpe" in str(c).lower() and "Sharpe Ratio" not in df.columns:
            df = df.rename(columns={c: "Sharpe Ratio"})
        if "beta" == str(c).strip().lower() and "Beta" not in df.columns:
            df = df.rename(columns={c: "Beta"})
        if "std" in str(c).lower() or "deviation" in str(c).lower():
            if "Standard Deviation" not in df.columns:
                df = df.rename(columns={c: "Standard Deviation"})
    # ensure columns exist
    for col in ["Date", "Scheme Name"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path}.")
    # ensure numeric candidate columns exist (create if missing)
    for col in ["Sharpe Ratio", "Beta", "Standard Deviation"]:
        if col not in df.columns:
            df[col] = np.nan
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    return df

# Ensure df_ch2
if "chapter2_df" in st.session_state:
    df_ch2 = st.session_state["chapter2_df"]
else:
    try:
        df_ch2 = _load_df()
        st.session_state["chapter2_df"] = df_ch2
    except Exception as e:
        st.error(f"Could not load chapter2 data: {e}")
        st.stop()

# Reuse SD filters (selected_schemes and selected_years)
if "selected_schemes" in st.session_state:
    schemes_filter = st.session_state["selected_schemes"]
elif "selected_schemes" in globals():
    schemes_filter = selected_schemes
else:
    st.error("Missing 'selected_schemes' (create in Standard Deviation section first).")
    st.stop()

if "selected_years" in st.session_state:
    years_filter = st.session_state["selected_years"]
elif "selected_years" in globals():
    years_filter = selected_years
else:
    st.error("Missing 'selected_years' (create in Standard Deviation section first).")
    st.stop()

# Convert years to ints
try:
    years_filter_int = [int(y) for y in years_filter]
except Exception:
    years_filter_int = list(map(int, list(years_filter)))

# Work on a copy
df_work = df_ch2.copy()

# List of columns we want to median-aggregate
num_cols = ["Beta", "Sharpe Ratio", "Standard Deviation"]

# Robust cleaning: convert to string, remove %, commas, currency symbols and anything non-numeric (except dot and minus)
import re
def clean_numeric_series(s):
    # keep NaN as-is
    s_str = s.astype(str).fillna("")
    # remove percent signs, commas, non-digit/dot/minus characters
    s_clean = (
        s_str
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", np.nan)
    )
    # convert to numeric coercing errors to NaN
    s_num = pd.to_numeric(s_clean, errors="coerce")
    return s_num, s_clean

# Apply cleaning and capture problematic entries
problem_rows = []
for col in num_cols:
    if col in df_work.columns:
        converted, cleaned_text = clean_numeric_series(df_work[col])
        df_work[col + "_num"] = converted
        # find entries that could not be converted but were not originally null
        mask_bad = df_work[col].notna() & df_work[col].astype(str).str.strip().ne("") & df_work[col + "_num"].isna()
        if mask_bad.any():
            # capture up to 20 sample offending values
            samples = df_work.loc[mask_bad, ["Date", "Scheme Name", col]].head(20)
            problem_rows.append((col, samples))
    else:
        df_work[col + "_num"] = np.nan
        problem_rows.append((col, pd.DataFrame(columns=["Date", "Scheme Name", col])))

# If there are problematic entries, show a compact debug summary for user's inspection
if problem_rows:
    any_problems = False
    debug_html = "<div style='font-family:Arial,sans-serif;'>"
    for col, dfp in problem_rows:
        if not dfp.empty:
            any_problems = True
            debug_html += f"<h5>Non-numeric / problematic values detected in column <b>{col}</b> (sample up to 20)</h5>"
            debug_html += dfp.to_html(index=False, justify="left")
    debug_html += "</div>"
    if any_problems:
        st.warning("Some cells in numeric columns could not be converted automatically. See sample below — these values were ignored in aggregation.")
        st.markdown(debug_html, unsafe_allow_html=True)

# Now use the cleaned numeric columns (suffix _num) for aggregation
agg_input_cols = [c + "_num" for c in num_cols]
# Filter dataset according to SD filters
mask = df_work["Scheme Name"].isin(schemes_filter)
mask &= df_work["Year"].isin(years_filter_int)
df_filtered = df_work.loc[mask].copy()

if df_filtered.empty:
    st.info("No data after applying Standard Deviation filters.")
    st.stop()

# Before grouping, ensure agg_input_cols exist and are numeric
for c in agg_input_cols:
    if c not in df_filtered.columns:
        df_filtered[c] = np.nan

# Perform groupby median ONLY on numeric columns
# Build mapping to nicer names afterwards
grouped = df_filtered.groupby(["Year", "Scheme Name"], as_index=False)[agg_input_cols].median()

# rename numeric suffix back to original names
rename_map = {f"{col}_num": col for col in num_cols}
grouped = grouped.rename(columns=rename_map)

# Drop rows where Beta or Sharpe Ratio are NaN — cannot plot points without both
grouped = grouped.dropna(subset=["Beta", "Sharpe Ratio"])
if grouped.empty:
    st.info("After cleaning and aggregation there is no numeric Beta/Sharpe data to plot for the selected filters.")
    st.stop()

# --- Proceed to plotting (same approach as prior code) ---
available_schemes = sorted(grouped["Scheme Name"].unique().tolist())
focus_schemes = [s for s in schemes_filter if s in available_schemes]
if not focus_schemes:
    st.info("None of the selected schemes have Beta/Sharpe median data in the chosen years.")
    st.stop()

n = len(focus_schemes)
n_cols = 2
n_rows = ceil(n / n_cols)

subplot_titles = focus_schemes
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
                    horizontal_spacing=0.08, vertical_spacing=0.12)

palette = px.colors.qualitative.Plotly

beta_min, beta_max = grouped["Beta"].min(), grouped["Beta"].max()
sharpe_min, sharpe_max = grouped["Sharpe Ratio"].min(), grouped["Sharpe Ratio"].max()
beta_pad = (beta_max - beta_min) * 0.12 if beta_max > beta_min else 0.5
sharpe_pad = (sharpe_max - sharpe_min) * 0.12 if sharpe_max > sharpe_min else 0.5
x_range = [beta_min - beta_pad, beta_max + beta_pad]
y_range = [sharpe_min - sharpe_pad, sharpe_max + sharpe_pad]

for idx, scheme in enumerate(focus_schemes):
    row = idx // n_cols + 1
    col = idx % n_cols + 1

    # background schemes
    for other in available_schemes:
        other_df = grouped[grouped["Scheme Name"] == other].sort_values("Year")
        if other_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=other_df["Beta"],
                y=other_df["Sharpe Ratio"],
                mode="lines+markers",
                line=dict(color="lightgray", width=1),
                marker=dict(size=6, color="lightgray"),
                name=other,
                showlegend=False,
                opacity=0.35,
                hoverinfo="skip"
            ),
            row=row, col=col
        )

    # focus scheme
    focus_df = grouped[grouped["Scheme Name"] == scheme].sort_values("Year")
    colorscheme = palette[idx % len(palette)]
    fig.add_trace(
        go.Scatter(
            x=focus_df["Beta"],
            y=focus_df["Sharpe Ratio"],
            mode="lines+markers",
            line=dict(color=colorscheme, width=3),
            marker=dict(size=8, color=colorscheme),
            name=scheme,
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>Year: %{customdata[0]}<br>Beta: %{x:.3f}<br>Sharpe: %{y:.3f}<extra></extra>",
            text=[scheme]*len(focus_df),
            customdata=focus_df[["Year"]].values
        ),
        row=row, col=col
    )

    # arrows and year labels
    for i in range(len(focus_df)-1):
        x0, y0 = float(focus_df.iloc[i]["Beta"]), float(focus_df.iloc[i]["Sharpe Ratio"])
        x1, y1 = float(focus_df.iloc[i+1]["Beta"]), float(focus_df.iloc[i+1]["Sharpe Ratio"])
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref=f"x{(idx+1)}" if (idx+1) > 1 else "x",
            yref=f"y{(idx+1)}" if (idx+1) > 1 else "y",
            axref=f"x{(idx+1)}" if (idx+1) > 1 else "x",
            ayref=f"y{(idx+1)}" if (idx+1) > 1 else "y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.6,
            arrowcolor=colorscheme,
            opacity=0.85
        )
    for i, rowdata in focus_df.reset_index(drop=True).iterrows():
        fig.add_annotation(
            x=float(rowdata["Beta"]),
            y=float(rowdata["Sharpe Ratio"]),
            text=str(int(rowdata["Year"])),
            showarrow=False,
            font=dict(size=10, color="black"),
            xref=f"x{(idx+1)}" if (idx+1) > 1 else "x",
            yref=f"y{(idx+1)}" if (idx+1) > 1 else "y",
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.05)",
            opacity=0.9
        )

    fig.update_xaxes(title_text="Beta (Market Sensitivity)", range=x_range, row=row, col=col, zeroline=True)
    fig.update_yaxes(title_text="Sharpe Ratio", range=y_range, row=row, col=col, zeroline=True)

# hide unused subplots
total_subplots = n_rows * n_cols
for empty_idx in range(len(focus_schemes), total_subplots):
    r = empty_idx // n_cols + 1
    c = empty_idx % n_cols + 1
    fig.update_xaxes(visible=False, row=r, col=c)
    fig.update_yaxes(visible=False, row=r, col=c)

fig.update_layout(
    title_text="ELSS Schemes: Beta–Sharpe Trajectories (annual medians)",
    template="plotly_white",
    height=380 * n_rows,
    showlegend=False,
    margin=dict(t=110, b=80, l=80, r=160),
    hovermode="closest",
    font=dict(size=13)
)

st.subheader("Beta–Sharpe Trajectories (annual medians)")
st.plotly_chart(fig, use_container_width=True)


st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">

  

  <p>
  Each panel highlights one scheme’s annual movement, with directional arrows indicating the evolution of its risk–return profile over time. 
  Other schemes are shown in the background to provide contextual comparison. 
  The trajectory of each fund reflects how its market sensitivity (Beta) and risk-adjusted return efficiency (Sharpe Ratio) have evolved 
  through different market cycles between 2020 and 2024.
  </p>

  <h5>Interpretation of Fund-wise Movement Patterns</h5>

  <ul>
<li><b>Mirae Asset ELSS Tax Saver Fund:</b> The trajectory for Mirae Asset shows a strategically favorable shift. 
    It began with a relatively high Beta (~1.03) in 2020, indicating higher exposure to market volatility, 
    and gradually reduced it to ~0.91 by 2024—adopting a more conservative stance. 
    Simultaneously, its Sharpe Ratio improved from around 0.45 to 0.78, reflecting enhanced efficiency in generating returns per unit of risk. 
    This trend suggests active portfolio management through sectoral rebalancing and reduced exposure to high-beta stocks.</li>

<li><b>DSP ELSS Tax Saver Fund:</b> DSP maintained a consistently high Beta (~1.00–1.02), reflecting strong market alignment. 
    However, its Sharpe Ratio fluctuated during the volatile 2022 period before recovering in 2023–2024. 
    The pattern suggests a cyclical, market-sensitive strategy suitable for investors with higher risk tolerance 
    who seek alpha during bullish market phases.</li>

<li><b>Axis ELSS Tax Saver Fund:</b> Axis displayed a non-linear path, with Beta fluctuating between 0.95 and 1.03 and no consistent upward trend in the Sharpe Ratio. 
    This lack of strategic progression implies intermittent portfolio adjustments without lasting impact—possibly due to frequent sectoral shifts or managerial turnover.</li>

<li><b>HDFC ELSS Tax Saver Fund:</b> HDFC demonstrated the most conservative and stable movement, maintaining Beta values between 0.85–0.89 
    and Sharpe Ratios around 0.45–0.55. 
    This consistency indicates a risk-controlled, capital-preserving approach suitable for long-term, risk-averse investors.</li>

<li><b>SBI Long Term Equity Fund:</b> SBI’s trajectory reflects calculated improvement in efficiency. 
    Its Beta remained close to 1.00, maintaining balanced market exposure, while the Sharpe Ratio steadily increased over the years. 
    This indicates a structured, steady approach to improving returns without raising risk exposure, aligning with disciplined fund management practices.</li>
</ul>

  <p>
  Overall, the trajectory visualization allows investors to assess not only where each fund stands today, but how it has evolved over time. 
  It distinguishes between funds with clear, proactive strategies and those showing inconsistent or reactive behavior, 
  enabling better evaluation of consistency, adaptability, and long-term reliability.
  </p>

  <h5>Summary Table: Risk–Return Evolution Highlights (2020–2024)</h5>
  <table style="width:100%; border-collapse: collapse; font-size:13px;">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Scheme Name</th>
        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Beta Trend</th>
        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Sharpe Trend</th>
        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Key Strategic Insight</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>Mirae Asset ELSS</b></td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Decreasing</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Increasing</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">De-risked portfolio while boosting efficiency.</td>
      </tr>
      <tr>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>DSP ELSS</b></td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Stable (~1.0)</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Cyclical</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Market-aligned; performs well in momentum-driven markets.</td>
      </tr>
      <tr>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>Axis ELSS</b></td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Fluctuating</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Inconsistent</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">No clear strategic improvement; limited conversion of risk into returns.</td>
      </tr>
      <tr>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;"><b>HDFC ELSS</b></td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Stable (Low)</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Stable (Moderate)</td>
        <td style="padding:8px; border-bottom:1px solid #f0f0f0;">Conservative and consistent; suitable for stability-focused investors.</td>
      </tr>
      <tr>
        <td style="padding:8px;"><b>SBI Long Term Equity Fund</b></td>
        <td style="padding:8px;">Stable (~1.0)</td>
        <td style="padding:8px;">Improving</td>
        <td style="padding:8px;">Balanced risk and return with steady efficiency gains.</td>
      </tr>
    </tbody>
  </table>

  <p style="margin-top:10px;">
  In conclusion, the <b>trajectory-based analysis</b> offers a dynamic perspective on how each ELSS scheme evolved over five years 
  in terms of its market sensitivity and risk-adjusted performance. 
  By tracking the direction and magnitude of annual changes in Beta and Sharpe Ratio, 
  it provides deeper insight into managerial responsiveness, portfolio discipline, and long-term adaptability — 
  going beyond static metrics to reveal the true behavioral patterns of mutual fund strategies.
  </p>

</div>
    """,
    unsafe_allow_html=True
)

st.subheader("Risk Analysis Summary")
st.markdown(
    """
<div style="text-align: justify; line-height:1.6; font-family: Arial, sans-serif;">


  <p>
  The multi-dimensional risk profile analysis—spanning <b>Standard Deviation</b> (volatility), 
  <b>Beta</b> (market sensitivity), and <b>Sharpe Ratio</b> (risk-adjusted return)—offers a holistic understanding 
  of how the selected ELSS schemes behaved and evolved between <b>2020 and 2024</b>.
  </p>

  <p>
  <b>Volatility patterns</b> revealed that fund risk levels peaked during <b>2020–2022</b> amid macroeconomic disruptions, 
  before gradually stabilizing by 2024. Most schemes converged below the category average volatility of <b>17.03%</b>. 
  <b>HDFC ELSS</b> consistently maintained the lowest volatility, 
  while <b>DSP</b> and <b>Mirae Asset ELSS</b> exhibited higher fluctuations yet delivered stronger returns, 
  reflecting the classic <b>risk–return trade-off</b>.
  </p>

  <p>
  <b>Beta analysis</b> indicated that most funds maintained values around <b>1.0</b>, 
  reflecting broad market alignment. 
  <b>Mirae Asset</b> and <b>HDFC ELSS</b> strategically reduced their betas, signaling a move toward stability, 
  while <b>DSP</b> and <b>SBI</b> retained a balanced, market-linked posture.
  </p>

  <p>
  <b>Sharpe Ratio trends</b> captured fund efficiency in converting risk into returns. 
  Funds like <b>Mirae Asset ELSS</b> showed improving Sharpe Ratios alongside declining volatility, 
  highlighting superior risk-adjusted management. 
  In contrast, <b>Axis ELSS</b> displayed inconsistent Sharpe performance despite moderate risk levels, 
  indicating suboptimal return efficiency.
  </p>

  <p>
  The <b>trajectory-based Beta–Sharpe analysis</b> visually reinforced these patterns by mapping each fund’s annual evolution. 
  It identified managers that actively optimized risk-return efficiency versus those with reactive or inconsistent strategies.
  </p>

  <p>
  <b>Overall,</b> the findings emphasize that effective ELSS evaluation requires more than absolute return comparison. 
  Integrating volatility, market responsiveness, and reward-for-risk metrics provides a more complete framework 
  for informed, long-term investment decisions aligned with investor risk tolerance and performance objectives.
  </p>

</div>
    """,
    unsafe_allow_html=True
)