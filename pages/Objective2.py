import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(page_title=" Risk & Volatility", layout="wide")
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
st.subheader("2.1 Risk Profile Analysis")
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
                title="Year-wise Average Sharpe Ratio (reusing SD years & schemes)"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=7))
            fig.update_layout(
                hovermode="x unified",
                legend=dict(title="Scheme", orientation="v", x=1.02, y=1),
                margin=dict(t=90, b=80, l=60, r=220),
                height=580,
                font=dict(size=13)
            )

            st.subheader("Sharpe Ratio — Yearly Average (using SD filters)")
            st.plotly_chart(fig, use_container_width=True)

           