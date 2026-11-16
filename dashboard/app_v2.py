"""FM Energy Dashboard v2 — simulated live playback of historical readings.

Reads `meter_readings_source` from Neon over the HTTP SQL API (outbound port
5432 is blocked, so we use https://<host>/sql with the connection string in the
`Neon-Connection-String` header, after stripping `&channel_binding=require`; the
corporate TLS proxy is trusted via the Windows certificate store).

This version replays the 2016 readings as a simulated live feed:
  * At startup, load only 1-31 January 2016 for all three meters.
  * Every real minute, fetch the next recorded hour (one reading per meter)
    from Neon and append it to the display.
  * Update the three per-meter charts and the combined total chart.
  * A Restart button resets to the January starting data and restarts the
    one-minute timer.
  * When the last record is retrieved, refreshing stops.

Original timestamps and hourly kWh values are preserved throughout.

Run from the project root (fm-energy-workshop):
    ../vibe/Scripts/streamlit run dashboard/app_v2.py --server.headless true
"""
import os
import ssl
import time
from datetime import timedelta
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
TICK_SECONDS = 60
JAN_END = "2016-02-01 00:00:00"  # exclusive upper bound for the January start set
COLS = ["meter_id", "building_id", "timestamp", "electricity_kwh"]
METER_COLORS = {"METER_01": "#1f77b4", "METER_02": "#ff7f0e", "METER_03": "#2ca02c"}

st.set_page_config(page_title="FM Energy Dashboard v2", page_icon="⚡", layout="wide")


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


_SESSION = None
_ENDPOINT = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.mount("https://", _WindowsTrustStoreAdapter())
        _SESSION = s
    return _SESSION


def _get_endpoint():
    global _ENDPOINT
    if _ENDPOINT is None:
        load_dotenv(REPO / ".env")
        url = os.environ["DATABASE_URL"]
        # Neon's HTTP SQL API rejects channel_binding; strip it in any position.
        url = url.replace("&channel_binding=require", "")
        url = url.replace("?channel_binding=require&", "?")
        url = url.replace("?channel_binding=require", "")
        _ENDPOINT = (f"https://{urlparse(url).hostname}/sql", url)
    return _ENDPOINT


def run_sql(query, params=None):
    """Run one SQL statement via the Neon HTTP SQL API; return its rows."""
    endpoint, conn = _get_endpoint()
    body = {"query": query}
    if params is not None:
        body["params"] = params
    resp = _get_session().post(
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


# --------------------------------------------------------------------------- #
#  Data access
# --------------------------------------------------------------------------- #
def _to_df(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"])
    return df.sort_values(["timestamp", "meter_id"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading January readings from Neon…")
def load_january():
    rows = run_sql(
        f"SELECT meter_id, building_id, timestamp, electricity_kwh "
        f"FROM {TABLE} WHERE timestamp < $1 ORDER BY timestamp, meter_id;",
        params=[JAN_END],
    )
    return _to_df(rows)


@st.cache_data(show_spinner=False)
def get_final_timestamp():
    rows = run_sql(f"SELECT MAX(timestamp) AS mx FROM {TABLE};")
    return pd.to_datetime(rows[0]["mx"])


def fetch_next_hour(after_ts):
    """Return the next recorded hour's three readings (one per meter)."""
    rows = run_sql(
        f"SELECT meter_id, building_id, timestamp, electricity_kwh "
        f"FROM {TABLE} WHERE timestamp > $1 "
        f"ORDER BY timestamp, meter_id LIMIT 3;",
        params=[after_ts.strftime("%Y-%m-%d %H:%M:%S")],
    )
    return _to_df(rows) if rows else pd.DataFrame(columns=COLS)


# --------------------------------------------------------------------------- #
#  Static UI
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚡ FM Energy Dashboard v2")
    st.caption("Simulated live playback from Neon.")
    st.divider()
    st.markdown("#### Run from the project root")
    st.code(
        "../vibe/Scripts/streamlit run dashboard/app_v2.py --server.headless true",
        language="bash",
    )
    st.caption("`--server.headless true` skips Streamlit's email prompt.")

st.title("⚡ FM Energy Dashboard v2")
st.markdown(
    "🔁 **Simulated live playback** — starts with 1–31 January 2016, then adds the "
    "next recorded hour (one reading per meter) every real minute until the last "
    "record. Original timestamps and hourly kWh values are preserved."
)


# --------------------------------------------------------------------------- #
#  Simulation (auto-reruns every minute via a Streamlit fragment)
# --------------------------------------------------------------------------- #
def _reset_to_january():
    jan = load_january().copy()
    st.session_state.display_df = jan
    st.session_state.last_ts = jan["timestamp"].max()
    st.session_state.final_ts = get_final_timestamp()
    st.session_state.total_to_fetch = int(
        (st.session_state.final_ts - st.session_state.last_ts).total_seconds() // 3600
    )
    st.session_state.fetched = 0
    st.session_state.done = False
    st.session_state.next_tick_at = time.time() + TICK_SECONDS


def _render(df):
    meters = sorted(df["meter_id"].unique())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Readings shown", f"{len(df):,}")
    k2.metric("Meters", len(meters))
    k3.metric("Simulated time", st.session_state.last_ts.strftime("%Y-%m-%d %H:%M"))
    k4.metric("Hours added", f"{st.session_state.fetched} / {st.session_state.total_to_fetch}")

    st.subheader("Hourly electricity usage by meter")
    cols = st.columns(3)
    for col, meter in zip(cols, meters):
        mdf = df[df["meter_id"] == meter].sort_values("timestamp")
        building = mdf["building_id"].iloc[0]
        fig = px.line(mdf, x="timestamp", y="electricity_kwh", title=f"{meter} · {building}")
        fig.update_traces(line_color=METER_COLORS.get(meter), line_width=1.3)
        fig.update_layout(height=320, xaxis_title=None, yaxis_title="kWh",
                          margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified")
        col.plotly_chart(fig, width="stretch")

    st.subheader("Total energy usage — all meters")
    total = df.groupby("timestamp", as_index=False)["electricity_kwh"].sum()
    tfig = px.area(total, x="timestamp", y="electricity_kwh")
    tfig.update_traces(line_color="#444", line_width=1.4)
    tfig.update_layout(height=360, xaxis_title=None, yaxis_title="kWh",
                       margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(tfig, width="stretch")


@st.fragment(run_every=timedelta(seconds=TICK_SECONDS))
def simulation():
    # Initialise once with the January starting data.
    if "display_df" not in st.session_state:
        try:
            _reset_to_january()
        except Exception as e:  # noqa: BLE001 - surface any load failure
            st.error(f"Could not load readings from Neon:\n\n{e}")
            return

    # Restart control (checked before advancing so a reset takes precedence).
    col_btn, col_status = st.columns([1, 3])
    if col_btn.button("🔄 Restart simulation"):
        _reset_to_january()

    # Advance one hour when the minute timer is due.
    if not st.session_state.done and time.time() >= st.session_state.next_tick_at:
        new_rows = fetch_next_hour(st.session_state.last_ts)
        if new_rows.empty:
            st.session_state.done = True
        else:
            st.session_state.display_df = pd.concat(
                [st.session_state.display_df, new_rows], ignore_index=True
            )
            st.session_state.last_ts = new_rows["timestamp"].max()
            st.session_state.fetched += 1
            if st.session_state.last_ts >= st.session_state.final_ts:
                st.session_state.done = True
        st.session_state.next_tick_at = time.time() + TICK_SECONDS

    # Status + progress (reflects the current state).
    with col_status:
        if st.session_state.done:
            st.success("✅ Simulation complete — all records retrieved. "
                       "Refreshing stopped.")
        else:
            st.info(f"⏱ New hour added every {TICK_SECONDS}s — "
                    f"{st.session_state.fetched} / {st.session_state.total_to_fetch} "
                    f"hours retrieved so far.")
        st.progress(min(st.session_state.fetched / max(st.session_state.total_to_fetch, 1), 1.0))

    _render(st.session_state.display_df)


simulation()
