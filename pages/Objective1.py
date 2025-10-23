import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
st.title("Comparative Analysis of Fund Performance")
st.markdown("""
The objective of this chapter is to conduct an in-depth comparative analysis of five leading ELSS mutual funds in India over the period 2020–2024. 
By examining the interplay of NAV growth, Assets Under Management (AUM) dynamics, and multi-period return performances, 
this chapter seeks to unveil critical insights into fund strategies, management efficiency, and investor behavior patterns. 
Rather than treating performance metrics in isolation, a cross-linked approach is adopted to demonstrate how various factors—such as fund size, risk-taking ability, and sectoral exposure—collectively influence investor outcomes. 
            """)
st.subheader("NAV GROWTH ANALYSIS")
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
file_path = 'Sampled_perf_direct.xlsx'

df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert 'NAV Date' to datetime
df['NAV Date'] = pd.to_datetime(df['NAV Date'])

# Pivot the data: NAV Date as index, Scheme Name as columns
df_nav = df[['Scheme Name', 'NAV Date', 'NAV Direct']]
df_pivot = df_nav.pivot_table(index='NAV Date', columns='Scheme Name', values='NAV Direct')

# Index each fund's NAV to 100 at start
df_indexed = df_pivot.divide(df_pivot.iloc[0]).multiply(100)

# Create the Plotly Figure
fig = go.Figure()

# Add a line for each fund
for scheme in df_indexed.columns:
    fig.add_trace(go.Scatter(
        x=df_indexed.index,
        y=df_indexed[scheme],
        mode='lines',
        name=scheme,
        hovertemplate='%{y:.2f} (Indexed NAV) on %{x|%b %Y}<extra>%{fullData.name}</extra>'
    ))

# Layout customization
fig.update_layout(
    title='Interactive Indexed NAV Trend of ELSS Mutual Funds (2020-2024)',
    xaxis_title='Date',
    yaxis_title='Indexed NAV (Base = 100)',
    hovermode='x unified',
    legend_title='Scheme Name',
    template='plotly_white',
    height=600,
    width=1000
)

fig.show()