"""Load Data/three_meters.csv into Neon's `meter_readings_source` table.

The workshop's Postgres is on Neon, and outbound port 5432 is blocked by the
local firewall, so we talk to Neon over its HTTP SQL API
(https://<host>/sql) with the `requests` library instead of a libpq driver.

The connection string is read from DATABASE_URL in .env. Neon's HTTP API does
not support `channel_binding=require`, so that parameter is stripped before the
string is sent in the `Neon-Connection-String` request header.

TLS: the corporate network intercepts HTTPS with a proxy whose CA is not in
certifi but IS in the Windows certificate store, so requests are made through an
adapter that trusts the OS store (no verification is disabled).

Table (one row per meter reading, four CSV columns):
    timestamp        TIMESTAMPTZ NOT NULL
    meter_id         TEXT        NOT NULL
    building_id      TEXT        NOT NULL
    electricity_kwh  NUMERIC     NOT NULL
    UNIQUE (meter_id, timestamp)

Uploads are batched multi-row INSERTs with ON CONFLICT DO NOTHING, so the
script is safe to re-run: it never duplicates a (meter_id, timestamp) pair.
"""
import csv
import os
import re
import ssl
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "Data" / "three_meters.csv"
TABLE = "meter_readings_source"
BATCH_ROWS = 500  # rows per INSERT request (4 params each -> 2000 params max)


# --------------------------------------------------------------------------- #
#  Connection setup
# --------------------------------------------------------------------------- #
def build_api() -> tuple[str, dict]:
    """Return (sql_endpoint_url, headers) from DATABASE_URL in .env."""
    load_dotenv(REPO / ".env")
    url = os.environ["DATABASE_URL"]
    # Neon's HTTP SQL API rejects channel_binding; strip it in any position.
    url = url.replace("&channel_binding=require", "")
    url = url.replace("?channel_binding=require&", "?")
    url = url.replace("?channel_binding=require", "")
    host = urlparse(url).hostname
    endpoint = f"https://{host}/sql"
    headers = {
        "Neon-Connection-String": url,
        "Content-Type": "application/json",
    }
    return endpoint, headers


class WindowsTrustStoreAdapter(HTTPAdapter):
    """Use the OS certificate store so the corporate TLS proxy is trusted."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.load_default_certs()  # on Windows, enumerates the Windows stores
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def make_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", WindowsTrustStoreAdapter())
    return s


# --------------------------------------------------------------------------- #
#  HTTP SQL helper
# --------------------------------------------------------------------------- #
def sql(session, endpoint, headers, query, params=None):
    """Run one SQL statement via the Neon HTTP SQL API; return parsed JSON."""
    body = {"query": query}
    if params is not None:
        body["params"] = params
    resp = session.post(endpoint, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Neon API error (HTTP {resp.status_code}) for query:\n{query}\n"
            f"Response: {resp.text[:500]}"
        )
    return resp.json()


# --------------------------------------------------------------------------- #
#  CSV reading + validation
# --------------------------------------------------------------------------- #
def read_csv(path):
    """Return list[dict] of rows, validating the four columns are present."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = ["timestamp", "meter_id", "building_id", "electricity_kwh"]
    for i, r in enumerate(rows, start=2):  # row 1 is the header
        missing = [c for c in expected if c not in r or r[c] == "" or r[c] is None]
        if missing:
            raise ValueError(f"line {i}: missing/empty column(s) {missing}: {r}")
    return rows


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 64)
    print("Upload three_meters.csv -> Neon meter_readings_source")
    print("=" * 64)

    endpoint, headers = build_api()
    session = make_session()
    print(f"[API] {endpoint}")

    # --- read + validate CSV ------------------------------------------------ #
    rows = read_csv(CSV_PATH)
    print(f"[CSV] {CSV_PATH.name}: {len(rows):,} data rows, "
          f"{len(rows[0])} columns -> {list(rows[0].keys())}")
    print(f"[CSV] preview: {dict(rows[0])}")

    meters = sorted({r["meter_id"] for r in rows})
    print(f"[CSV] meters: {meters}")

    pairs = {(r["meter_id"], r["timestamp"]) for r in rows}
    if len(pairs) != len(rows):
        dup = len(rows) - len(pairs)
        print(f"[CSV] WARNING: {dup} duplicate (meter_id, timestamp) pairs "
              f"will be de-duplicated by the table's UNIQUE constraint.")
    else:
        print(f"[CSV] {len(pairs):,} unique (meter_id, timestamp) pairs "
              f"(no duplicates in source)")

    # --- create table (idempotent) ----------------------------------------- #
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        timestamp        TIMESTAMPTZ NOT NULL,
        meter_id         TEXT        NOT NULL,
        building_id      TEXT        NOT NULL,
        electricity_kwh  NUMERIC     NOT NULL,
        CONSTRAINT {TABLE}_meter_ts_key UNIQUE (meter_id, timestamp)
    );
    """
    sql(session, endpoint, headers, ddl)
    print(f"[DB] table '{TABLE}' ready (created if not present)")

    # --- upload in batches -------------------------------------------------- #
    cols = ["timestamp", "meter_id", "building_id", "electricity_kwh"]
    n_batches = (len(rows) + BATCH_ROWS - 1) // BATCH_ROWS
    inserted = 0
    for b in range(n_batches):
        chunk = rows[b * BATCH_ROWS:(b + 1) * BATCH_ROWS]
        placeholders, params = [], []
        for r in range(len(chunk)):
            base = r * 4 + 1
            placeholders.append(f"(${base},${base + 1},${base + 2},${base + 3})")
            row = chunk[r]
            params.extend([row["timestamp"], row["meter_id"],
                           row["building_id"], row["electricity_kwh"]])
        q = (f"INSERT INTO {TABLE} ({', '.join(cols)}) "
             f"VALUES {', '.join(placeholders)} "
             f"ON CONFLICT (meter_id, timestamp) DO NOTHING")
        out = sql(session, endpoint, headers, q, params)
        inserted += int(out.get("rowCount") or 0)
        print(f"[UP] batch {b + 1}/{n_batches}: {len(chunk)} rows sent, "
              f"{out.get('rowCount')} affected (cumulative {inserted:,})")

    print(f"[UP] done: {inserted:,} rows affected this run "
          f"(ON CONFLICT DO NOTHING skips pairs already present)")

    # --- summary from the database ------------------------------------------ #
    summary = sql(session, endpoint, headers, f"""
        SELECT meter_id,
               COUNT(*)        AS row_count,
               MIN(timestamp) AS earliest,
               MAX(timestamp) AS latest
        FROM {TABLE}
        GROUP BY meter_id
        ORDER BY meter_id;
    """)
    print("\n[meter_readings_source] per-meter summary:")
    print(f"  {'meter_id':<12}{'row_count':>10}{'earliest':>23}{'latest':>23}")
    print(f"  {'-'*12}{'-'*10}{'-'*23}{'-'*23}")
    total = 0
    for row in summary["rows"]:
        print(f"  {row['meter_id']:<12}{int(row['row_count']):>10,}"
              f"{row['earliest']:>23}{row['latest']:>23}")
        total += int(row["row_count"])
    print(f"  {'-'*12}{'-'*10}{'-'*23}{'-'*23}")
    print(f"  {'TOTAL':<12}{total:>10,}")
    print("\nUpload complete.")


if __name__ == "__main__":
    sys.exit(main())
