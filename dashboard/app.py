"""FM Energy Dashboard — historical hourly electricity usage for three meters.

Reads the `meter_readings_source` table from Neon over its HTTP SQL API
(outbound port 5432 is blocked by the local firewall, so we use https://<host>/sql
with the connection string in the `Neon-Connection-String` header, after stripping
`&channel_binding=require`). The corporate TLS proxy's CA is not in certifi but
is in the Windows certificate store, so requests go through an adapter that
trusts the OS store.

Layout:
  * Top:    three per-meter hourly charts, side by side.
  * Middle: total energy usage, summed by timestamp across all meters.
  * Bottom: placeholders for Prediction and LLM Analysis (not implemented yet).

The data is historical (static 2016 readings); no live replay yet.

Run from the project root (fm-energy-workshop):
    ../vibe/Scripts/streamlit run dashboard/app.py --server.headless true
"""
import os
import ssl
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

REPO = Path(__file__).resolve().parent.parent  # fm-energy-workshop/
TABLE = "meter_readings_source"

st.set_page_config(page_title="FM Energy Dashboard", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
#  Neon HTTP SQL API connection
# --------------------------------------------------------------------------- #
class _WindowsTrustStoreAdapter(HTTPAdapter):
    """Trust the OS certificate store so the corporate TLS proxy is accepted."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.load_default_certs()  # on Windows this enumerates the Windows stores
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


@st.cache_resource
def _session():
    s = requests.Session()
    s.mount("https://", _WindowsTrustStoreAdapter())
    return s


@st.cache_resource
def _endpoint():
    """Return (sql_endpoint_url, connection_string) from DATABASE_URL in .env."""
    load_dotenv(REPO / ".env")
    url = os.environ["DATABASE_URL"]
    # Neon's HTTP SQL API rejects channel_binding; strip it in any position.
    url = url.replace("&channel_binding=require", "")
    url = url.replace("?channel_binding=require&", "?")
    url = url.replace("?channel_binding=require", "")
    return f"https://{urlparse(url).hostname}/sql", url


def run_sql(query, params=None):
    """Run one SQL statement via the Neon HTTP SQL API; return its rows."""
    endpoint, conn = _endpoint()
    body = {"query": query}
    if params is not None:
        body["params"] = params
    resp = _session().post(
        endpoint,
        headers={"Neon-Connection-String": conn, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Neon API error (HTTP {resp.status_code}): {resp.text[:300]}"
        )
    return resp.json()["rows"]


@st.cache_data(ttl=300, show_spinner="Loading readings from Neon…")
def load_readings():
    rows = run_sql(
        f"SELECT meter_id, building_id, timestamp, electricity_kwh "
        f"FROM {TABLE} ORDER BY timestamp;"
    )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚡ FM Energy Dashboard")
    st.caption("Historical energy readings from Neon.")
    st.divider()
    st.markdown("#### Run from the project root")
    st.code(
        "../vibe/Scripts/streamlit run dashboard/app.py --server.headless true",
        language="bash",
    )
    st.caption("`--server.headless true` skips Streamlit's email prompt.")


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.title("⚡ FM Energy Dashboard")
st.markdown(
    f"🕒 **Historical data** — static 2016 hourly meter readings from the "
    f"`{TABLE}` table on Neon. No live replay yet."
)

try:
    df = load_readings()
except Exception as e:  # noqa: BLE001 - surface any failure to the user
    st.error(f"Could not load readings from Neon:\n\n{e}")
    st.stop()

meters = sorted(df["meter_id"].unique())
date_min = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
date_max = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")

m1, m2, m3 = st.columns(3)
m1.metric("Readings", f"{len(df):,}")
m2.metric("Meters", len(meters))
m3.metric("Period", f"{date_min}  →  {date_max}")

st.divider()

# --------------------------------------------------------------------------- #
#  Per-meter charts (side by side)
# --------------------------------------------------------------------------- #
st.subheader("Hourly electricity usage by meter")
st.caption("One chart per meter; hourly readings in kWh.")

meter_colors = {"METER_01": "#1f77b4", "METER_02": "#ff7f0e", "METER_03": "#2ca02c"}
cols = st.columns(3)
for col, meter in zip(cols, meters):
    mdf = df[df["meter_id"] == meter].sort_values("timestamp")
    building = mdf["building_id"].iloc[0]
    fig = px.line(
        mdf, x="timestamp", y="electricity_kwh",
        title=f"{meter} · {building}",
    )
    fig.update_traces(line_color=meter_colors.get(meter), line_width=1.3)
    fig.update_layout(
        height=320,
        xaxis_title=None,
        yaxis_title="kWh",
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
    )
    col.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
#  Total energy usage (summed by timestamp)
# --------------------------------------------------------------------------- #
st.subheader("Total energy usage — all meters")
st.caption("Sum of the three meters at each timestamp, in kWh.")

total = df.groupby("timestamp", as_index=False)["electricity_kwh"].sum()
total_fig = px.area(total, x="timestamp", y="electricity_kwh")
total_fig.update_traces(line_color="#444", line_width=1.4)
total_fig.update_layout(
    height=360,
    xaxis_title=None,
    yaxis_title="kWh",
    margin=dict(l=10, r=10, t=20, b=10),
    hovermode="x unified",
)
st.plotly_chart(total_fig, width="stretch")

st.divider()

# --------------------------------------------------------------------------- #
#  Placeholders for coming features
# --------------------------------------------------------------------------- #
st.subheader("Coming next")
ph1, ph2 = st.columns(2)
with ph1:
    st.markdown("#### 🔮 Prediction")
    st.info("Placeholder — next-hour usage prediction will appear here.")
with ph2:
    st.markdown("#### 🧠 LLM Analysis")
    st.info("Placeholder — natural-language energy analysis will appear here.")
