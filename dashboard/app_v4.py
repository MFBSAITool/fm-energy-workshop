"""FM Energy Dashboard v4 — live playback + prediction + LLM analysis.

Builds on v3 (simulated live playback of `meter_readings_source` from Neon, plus
per-meter next-hour prediction) and adds an LLM Analysis area:

  * A fixed, read-only prompt: "Give me a summary of the energy usage for
    METER_02 for the past 2 days" and a Send button.
  * When Send is pressed, Python prepares the data: the latest 48 hours currently
    revealed for METER_02, with total / average / minimum / maximum kWh, packed
    into a JSON string (meter name, time range, statistics, 48 hourly readings).
    The JSON is attached to the fixed prompt as data context in the user message
    sent to OpenAI.
  * The API is called ONLY when Send is pressed. The response is kept in session
    state, so refreshes and the one-minute replay never send new requests.
  * The response is shown with "Based on readings up to <latest time>", a count
    of LLM requests this session, and the JSON in a collapsed section.
  * OPENAI_API_KEY and OPENAI_MODEL come from .env. The key is never displayed.

Run from the project root (fm-energy-workshop):
    ../vibe/Scripts/streamlit run dashboard/app_v4.py --server.headless true
"""
import json
import os
import ssl
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

REPO = Path(__file__).resolve().parent.parent  # fm-energy-workshop/
TABLE = "meter_readings_source"
MODEL_PATH = REPO / "models" / "gbr_model.joblib"
FEATURE_ORDER_PATH = REPO / "models" / "feature_order.json"
TICK_SECONDS = 60
JAN_END = "2016-02-01 00:00:00"  # exclusive upper bound for the January start set
COLS = ["meter_id", "building_id", "timestamp", "electricity_kwh"]
METER_COLORS = {"METER_01": "#1f77b4", "METER_02": "#ff7f0e", "METER_03": "#2ca02c"}

LLM_METER = "METER_02"
LLM_HOURS = 48
LLM_PROMPT = "Give me a summary of the energy usage for METER_02 for the past 2 days"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

st.set_page_config(page_title="FM Energy Dashboard v4", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
#  HTTP session + Neon HTTP SQL API connection
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
    """Return the next recorded hour's three readings (one reading per meter)."""
    rows = run_sql(
        f"SELECT meter_id, building_id, timestamp, electricity_kwh "
        f"FROM {TABLE} WHERE timestamp > $1 "
        f"ORDER BY timestamp, meter_id LIMIT 3;",
        params=[after_ts.strftime("%Y-%m-%d %H:%M:%S")],
    )
    return _to_df(rows) if rows else pd.DataFrame(columns=COLS)


# --------------------------------------------------------------------------- #
#  Saved model + feature order (loaded, never retrained)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_order():
    with open(FEATURE_ORDER_PATH) as f:
        return json.load(f)


def _meter_series(meter_df):
    """Return a timestamp-indexed Series of kWh for one meter (deduped, sorted)."""
    s = (meter_df.sort_values("timestamp")
         .set_index("timestamp")["electricity_kwh"])
    return s[~s.index.duplicated(keep="last")].sort_index()


def predict_next_hour(display_df):
    """Predict each meter's next recorded hour from its revealed readings.

    Returns a list of dicts: {meter, target, missing(bool), pred|message}.
    """
    model = load_model()
    feature_order = load_feature_order()
    results = []
    for meter in sorted(display_df["meter_id"].unique()):
        mdf = display_df[display_df["meter_id"] == meter]
        if mdf.empty:
            results.append({"meter": meter, "target": None, "missing": True,
                            "message": "no readings revealed yet"})
            continue

        s = _meter_series(mdf)
        t_last = s.index.max()
        t_next = t_last + pd.Timedelta(hours=1)  # the next recorded hour

        # Lags reference readings BEFORE the target hour (t_next - k hours).
        needed = {
            "lag_1": t_last,                       # t_next - 1h
            "lag_2": t_last - pd.Timedelta(hours=1),
            "lag_3": t_last - pd.Timedelta(hours=2),
            "lag_24": t_last - pd.Timedelta(hours=23),
            "lag_168": t_last - pd.Timedelta(hours=167),
        }
        lags, missing = {}, []
        for name, ts in needed.items():
            if ts in s.index:
                lags[name] = float(s.loc[ts])
            else:
                missing.append(name)

        if missing:
            results.append({
                "meter": meter,
                "target": t_next,
                "missing": True,
                "message": (f"cannot predict {t_next.strftime('%Y-%m-%d %H:%M')} — "
                            f"missing required history ({', '.join(missing)}); "
                            f"{len(s)} readings revealed, need at least 168 hours "
                            f"for lag_168"),
            })
            continue

        features = {
            "hour_of_day": int(t_next.hour),
            "day_of_week": int(t_next.day_of_week),
            **lags,
        }
        X = pd.DataFrame([features], columns=feature_order)
        pred = float(model.predict(X)[0])
        results.append({"meter": meter, "target": t_next, "missing": False,
                        "pred": pred})
    return results


# --------------------------------------------------------------------------- #
#  LLM analysis (OpenAI) — data prepared in Python, called only on Send
# --------------------------------------------------------------------------- #
def build_llm_context(display_df, meter=LLM_METER, hours=LLM_HOURS):
    """Build the JSON data context from the latest `hours` revealed readings.

    Returns (json_string, as_of_timestamp), or (None, None) if no readings.
    """
    mdf = display_df[display_df["meter_id"] == meter]
    if mdf.empty:
        return None, None
    s = _meter_series(mdf).tail(hours)  # latest 48 hours currently revealed
    payload = {
        "meter": meter,
        "time_range": {
            "start": s.index.min().strftime("%Y-%m-%d %H:%M"),
            "end": s.index.max().strftime("%Y-%m-%d %H:%M"),
            "hours": int(len(s)),
        },
        "statistics_kwh": {
            "total": round(float(s.sum()), 2),
            "average": round(float(s.mean()), 2),
            "minimum": round(float(s.min()), 2),
            "maximum": round(float(s.max()), 2),
        },
        "hourly_readings_kwh": [
            {"timestamp": ts.strftime("%Y-%m-%d %H:%M"), "kwh": round(float(v), 2)}
            for ts, v in s.items()
        ],
    }
    return json.dumps(payload, indent=2), s.index.max()


def call_openai(json_context):
    """Send the fixed prompt + JSON data context to OpenAI; return the reply text.

    The API key is read from .env and used only in the Authorization header; it
    is never displayed or included in any message shown to the user.
    """
    load_dotenv(REPO / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env.")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set in .env.")

    user_message = (
        f"{LLM_PROMPT}\n\n"
        f"Data context (JSON):\n{json_context}"
    )
    resp = _get_session().post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": user_message}]},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error (HTTP {resp.status_code}): "
                           f"{resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
#  Static UI
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚡ FM Energy Dashboard v4")
    st.caption("Live playback, prediction and LLM analysis from Neon.")
    st.divider()
    st.markdown("#### Run from the project root")
    st.code(
        "../vibe/Scripts/streamlit run dashboard/app_v4.py --server.headless true",
        language="bash",
    )
    st.caption("`--server.headless true` skips Streamlit's email prompt.")

st.title("⚡ FM Energy Dashboard v4")
st.markdown(
    "🔁 **Simulated live playback, prediction and LLM analysis** — starts with "
    "1–31 January 2016, adds the next recorded hour every real minute, predicts "
    "each meter's next hour, and can ask an LLM to summarise METER_02's last 2 days."
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
    # Clear results tied to the old replay position; the request count is kept.
    st.session_state.pop("predictions", None)
    st.session_state.pop("llm_result", None)


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


def _render_prediction():
    st.divider()
    st.subheader("🔮 Prediction")
    st.caption("Saved model (models/gbr_model.joblib) applied to each meter's "
               "revealed readings to forecast its next recorded hour. The model is "
               "loaded, not retrained.")

    if st.button("Predict next hour", type="primary"):
        try:
            st.session_state.predictions = predict_next_hour(st.session_state.display_df)
        except Exception as e:  # noqa: BLE001 - surface model/predict failures
            st.error(f"Prediction failed:\n\n{e}")

    preds = st.session_state.get("predictions")
    if preds is None:
        st.info("Click **Predict next hour** to forecast each meter's next "
                "recorded hour from the readings revealed so far.")
    else:
        st.caption(f"Based on readings revealed through "
                   f"{st.session_state.last_ts.strftime('%Y-%m-%d %H:%M')}.")
        pcols = st.columns(3)
        for col, p in zip(pcols, preds):
            with col:
                if p["missing"]:
                    st.warning(f"**{p['meter']}** — {p['message']}")
                else:
                    st.metric(p["meter"], f"{p['pred']:.2f} kWh")
                    st.caption(f"target {p['target'].strftime('%Y-%m-%d %H:%M')} "
                                f"(next recorded hour)")


def _render_llm():
    st.divider()
    st.subheader("🧠 LLM Analysis")
    st.caption("Fixed prompt plus a JSON summary of METER_02's latest 48 revealed "
               "hours. The API is called only when you press Send.")

    st.text_area("Prompt (read-only)", value=LLM_PROMPT, disabled=True, height=70)

    if st.button("Send", type="primary"):
        context_json, as_of = build_llm_context(st.session_state.display_df)
        if context_json is None:
            st.warning(f"No {LLM_METER} readings revealed yet — nothing to send.")
        else:
            st.session_state.llm_requests = st.session_state.get("llm_requests", 0) + 1
            try:
                with st.spinner("Asking the LLM…"):
                    text = call_openai(context_json)
                st.session_state.llm_result = {
                    "text": text, "as_of": as_of, "json": context_json, "error": None,
                }
            except Exception as e:  # noqa: BLE001 - show the failure, not the key
                st.session_state.llm_result = {
                    "text": None, "as_of": as_of, "json": context_json,
                    "error": str(e),
                }

    st.caption(f"LLM requests this session: "
               f"**{st.session_state.get('llm_requests', 0)}**")

    result = st.session_state.get("llm_result")
    if result is None:
        st.info("Click **Send** to get an LLM summary of METER_02's past 2 days.")
        return
    if result["error"]:
        st.error(f"LLM request failed:\n\n{result['error']}")
    else:
        st.markdown(f"**Based on readings up to "
                    f"{result['as_of'].strftime('%Y-%m-%d %H:%M')}**")
        st.markdown(result["text"])
    with st.expander("Data sent to the LLM", expanded=False):
        st.code(result["json"], language="json")


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
    _render_prediction()
    _render_llm()


simulation()
