"""
Mini Finance Dashboard (single API, single clean chart)

Team: Team 9
Members: Suyog Mainali • Luke Kovats • Bruce Marin • Venessa Broadrup
Story: Do higher policy rates (Federal Funds Rate) line up with lower inflation after a lag?
Data Source: Federal Reserve Economic Data (FRED). Series used: CPIAUCSL, FEDFUNDS.
"""

import os, io, base64, calendar # for month-end day calc
import pandas as pd # data handling

import matplotlib # plotting
matplotlib.use("Agg") # for non-GUI backend (server)
import matplotlib.pyplot as plt #
import seaborn as sns # for nicer plots

from datetime import date # for current date
from dotenv import load_dotenv # for .env file handling
from fredapi import Fred # for FRED API
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context # for Dash app
import dash_bootstrap_components as dbc # for Bootstrap styles

# ---------------- Setup FRED ----------------
load_dotenv() # take environment variables from .env
fred = Fred(api_key=os.getenv("FRED_API_KEY"))   # initialize FRED API client

# ---------------- Helpers ----------------
# month-end day helper (for year/month jumps)
def month_end_day(year: int, month: int) -> int:  # return last day of month
    return calendar.monthrange(year, month)[1]  # (1..28/29/30/31)   

def set_dom_safely(year: int, month: int, day: int) -> pd.Timestamp: # set day-of-month safely
    return pd.Timestamp(year=year, month=month, day=min(day, month_end_day(year, month))) # clamp day

# fetch CPI and FEDFUNDS with a small buffer so transformations work even on short windows
def get_chart_data(start, end):
    # ensure strings -> Timestamps
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp(date.today())

    # buffer so YoY (needs 12m) and MA(3) have history
    buffer_start = (start_dt - pd.DateOffset(months=15)).strftime("%Y-%m-%d")

    # CPI index (monthly) -> year-over-year percent
    cpi = fred.get_series("CPIAUCSL", observation_start=buffer_start, observation_end=end_dt)
    cpi.index = pd.to_datetime(cpi.index)
    cpi_yoy = (cpi.pct_change(12) * 100).rename("Inflation (Year-over-Year Percent)")

    # Fed Funds is daily; aggregate to monthly mean then 3-month moving average
    ff = fred.get_series("FEDFUNDS", observation_start=buffer_start, observation_end=end_dt)
    ff.index = pd.to_datetime(ff.index)
    ff_m = ff.resample("MS").mean()
    ff_ma3 = ff_m.rolling(3).mean().rename("Federal Funds Rate (3-Month Moving Average)")

    # Trim to requested window
    df = pd.concat([cpi_yoy, ff_ma3], axis=1)
    df = df.loc[start_dt:end_dt]
    return df

def render_png(df):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    df.plot(ax=ax)

    # annotate policy-rate peak
    ff_col = "Federal Funds Rate (3-Month Moving Average)"
    if ff_col in df.columns and df[ff_col].notna().any():
        s = df[ff_col].dropna()
        peak_dt, peak_val = s.idxmax(), s.max()
        ax.annotate(f"Peak: {peak_val:.2f}%",
                    xy=(peak_dt, peak_val),
                    xytext=(peak_dt, peak_val + 0.8),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=8)

    ax.set_title("Inflation (Year-over-Year Percent) vs Federal Funds Rate (3-Month Moving Average)",
                 fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend(loc="best")
    fig.text(0.99, 0.01, "Source: Federal Reserve Economic Data (FRED)", ha="right", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

# ---------------- App ----------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# year/month option lists (1990 .. current)
CURRENT_YEAR = date.today().year
YEAR_OPTS  = [{"label": str(y), "value": y} for y in range(1990, CURRENT_YEAR + 1)]
MONTH_OPTS = [{"label": pd.Timestamp(month=m, year=2000, day=1).strftime("%B"), "value": m} for m in range(1, 13)]

app.title = "Team (9) Mini Finance Dashboard" 
app.layout = html.Div([
    # Header
    html.Div(className="app-header", children=[
        html.Div(className="team-line", children=[
            html.Div("Team 9", className="team-title"),
            html.Div("Members: Suyog Mainali • Luke Kovats • Bruce Marin • Venessa Broadrup", className="members"),
            html.Div("Data Source: Federal Reserve Economic Data (FRED)", className="api-badge"),
        ])
    ]),

    # 2-column layout
    html.Div(className="dashboard-row", children=[

        # Sidebar
        html.Div(className="sidebar", children=[
            html.H2("Mini Finance Dashboard"),

            # START controls
            html.Div(className="dates-block", children=[
                html.Label("Start date", className="date-label"),
                dcc.DatePickerSingle(
                    id="start-date",
                    date="2019-01-01",
                    display_format="YYYY-MM-DD",
                    min_date_allowed="1990-01-01",
                    max_date_allowed=date.today()
                ),
                html.Div(className="ym-row", children=[
                    dcc.Dropdown(id="start-year",  options=YEAR_OPTS,  value=2019, clearable=False, className="ym-dd"),
                    dcc.Dropdown(id="start-month", options=MONTH_OPTS, value=1,    clearable=False, className="ym-dd"),
                ])
            ]),

            # END controls
            html.Div(className="dates-block", children=[
                html.Label("End date", className="date-label"),
                dcc.DatePickerSingle(
                    id="end-date",
                    date=date.today(),
                    display_format="YYYY-MM-DD",
                    min_date_allowed="1990-01-01",
                    max_date_allowed=date.today()
                ),
                html.Div(className="ym-row", children=[
                    dcc.Dropdown(id="end-year",  options=YEAR_OPTS,  value=CURRENT_YEAR, clearable=False, className="ym-dd"),
                    dcc.Dropdown(id="end-month", options=MONTH_OPTS, value=date.today().month, clearable=False, className="ym-dd"),
                ])
            ]),

            # Quick presets
            html.Div(className="quick-row", children=[
                html.Button("1 Year",    id="q_1y",  n_clicks=0, className="quick-btn btn btn-outline-secondary"),
                html.Button("3 Years",   id="q_3y",  n_clicks=0, className="quick-btn btn btn-outline-secondary"),
                html.Button("5 Years",   id="q_5y",  n_clicks=0, className="quick-btn btn btn-outline-secondary"),
                html.Button("Full Range", id="q_max", n_clicks=0, className="quick-btn btn btn-outline-secondary"),
            ]),

            html.Button("Export Chart", id="export-btn",
                        n_clicks=0, className="export-btn btn btn-dark"),
            html.Div(id="export-msg", className="export-msg")
        ]),

        # Main (chart + explanation)
        html.Div(className="main", children=[
            html.Div(className="chart-card", children=[
                html.Div("Inflation (Year-over-Year Percent) vs Federal Funds Rate (3-Month Moving Average)",
                         id="chart-title", className="chart-title"),
                html.Img(id="chart-img", className="chart-img", alt="chart will appear here")
            ]),
            html.Div(id="insight", className="insight-card")
        ])
    ])
])

# ------------- Presets + Year/Month jump (sync classes & dropdowns too) -------------
@app.callback(
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("q_1y", "className"),
    Output("q_3y", "className"),
    Output("q_5y", "className"),
    Output("q_max", "className"),
    Output("start-year", "value"),
    Output("start-month", "value"),
    Output("end-year", "value"),
    Output("end-month", "value"),
    Input("q_1y", "n_clicks"),
    Input("q_3y", "n_clicks"),
    Input("q_5y", "n_clicks"),
    Input("q_max", "n_clicks"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("start-year", "value"),
    Input("start-month", "value"),
    Input("end-year", "value"),
    Input("end-month", "value"),
    prevent_initial_call=True
)
def presets_and_jumps(n1, n3, n5, nmax, start_date, end_date, sy, sm, ey, em):
    ctx = callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # baseline output classes (neutral)
    inactive = "quick-btn btn btn-outline-secondary"
    active   = "quick-btn btn btn-secondary"
    b1 = b3 = b5 = bm = inactive

    # current dates
    sd = pd.Timestamp(start_date) if start_date else pd.Timestamp("2019-01-01")
    ed = pd.Timestamp(end_date)   if end_date   else pd.Timestamp(date.today())

    if trig in {"q_1y", "q_3y", "q_5y", "q_max"}:
        ed = pd.Timestamp(date.today())
        if trig == "q_1y":
            sd = ed - pd.DateOffset(years=1)
            b1 = active
        elif trig == "q_3y":
            sd = ed - pd.DateOffset(years=3)
            b3 = active
        elif trig == "q_5y":
            sd = ed - pd.DateOffset(years=5)
            b5 = active
        elif trig == "q_max":
            sd = pd.Timestamp("1990-01-01")
            bm = active
        # sync dropdowns to dates chosen
        return (sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"),
                b1, b3, b5, bm,
                sd.year, sd.month, ed.year, ed.month)

    # Jump via start-year/month dropdowns
    if trig in {"start-year", "start-month"}:
        new_sd = set_dom_safely(int(sy), int(sm), sd.day)
        return (new_sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"),
                b1, b3, b5, bm,
                new_sd.year, new_sd.month, ed.year, ed.month)

    # Jump via end-year/month dropdowns
    if trig in {"end-year", "end-month"}:
        new_ed = set_dom_safely(int(ey), int(em), ed.day)
        return (sd.strftime("%Y-%m-%d"), new_ed.strftime("%Y-%m-%d"),
                b1, b3, b5, bm,
                sd.year, sd.month, new_ed.year, new_ed.month)

    # Manual date clicks: keep dates, clear highlight
    return (sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"),
            b1, b3, b5, bm,
            sd.year, sd.month, ed.year, ed.month)

# ------------- Chart update -------------
@app.callback(
    Output("chart-img", "src"),
    Output("insight", "children"),
    Input("start-date", "date"),
    Input("end-date", "date"),
)
def update_chart(start, end):
    df = get_chart_data(start, end)
    if df.dropna(how="all").empty:
        return no_update, "No data for this range."

    img = render_png(df)

    # simple explanation from visible window (no abbreviations)
    last = df.dropna(how="all").index.max().strftime("%Y-%m")
    infl = df["Inflation (Year-over-Year Percent)"].dropna()
    ff   = df["Federal Funds Rate (3-Month Moving Average)"].dropna()
    expl = [
        html.Div("What this shows", className="insight-title"),
        html.Ul(className="insight-list", children=[
            html.Li(f"As of {last}, inflation is {infl.iloc[-1]:.2f} percent (year-over-year)."),
            html.Li(f"The Federal Funds Rate (3-Month Moving Average) is {ff.iloc[-1]:.2f} percent "
                    f"(peak about {ff.max():.2f} percent in this window)."),
            html.Li("Typical pattern: higher policy rates are followed by easing inflation with a lag.")
        ])
    ]
    return img, expl

# ------------- Export PNG -------------
@app.callback(
    Output("export-msg", "children"),
    Input("export-btn", "n_clicks"),
    State("start-date", "date"),
    State("end-date", "date"),
    prevent_initial_call=True
)
def export_png(n, start, end):
    df = get_chart_data(start, end)
    if df.dropna(how="all").empty:
        return "Nothing to export."
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    df.plot(ax=ax)
    ax.set_title("Inflation (Year-over-Year Percent) vs Federal Funds Rate (3-Month Moving Average)",
                 fontsize=16, weight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Percent")
    ax.legend(loc="best")
    fig.text(0.99, 0.01, "Source: Federal Reserve Economic Data (FRED)", ha="right", fontsize=9)
    fig.tight_layout()
    file_name = "SUBMISSION_inflation_vs_federal_funds.png"
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)
    return f"Saved {file_name} in this folder."

if __name__ == "__main__":
    app.run_server(debug=False)