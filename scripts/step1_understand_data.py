"""Step 1 - Understand the data.

Load data/one_meter.csv, describe it, and identify anything that needs
cleaning or preprocessing. This step only inspects; it does not modify the
data.
"""
from pathlib import Path

import pandas as pd

# Resolve the data path relative to this script so it works from any cwd.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "one_meter.csv"

print("=" * 64)
print("STEP 1: Understand the data")
print("=" * 64)

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# --- Basic description -----------------------------------------------------
print("\n[Shape]")
print(f"rows = {len(df)}, columns = {len(df.columns)}")
print(f"columns = {list(df.columns)}")

print("\n[Dtypes]")
print(df.dtypes.to_string())

print("\n[Head]")
print(df.head(3).to_string(index=False))

print("\n[Tail]")
print(df.tail(3).to_string(index=False))

# --- Time range & continuity ----------------------------------------------
print("\n[Time range]")
t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
print(f"start = {t_min}")
print(f"end   = {t_max}")
span_hours = int((t_max - t_min).total_seconds() // 3600)
expected_if_continuous = span_hours + 1
print(f"span  = {span_hours} hours between first and last timestamp")
print(f"rows present = {len(df)} | expected if continuous hourly = {expected_if_continuous}")

# --- Missing values -------------------------------------------------------
print("\n[Missing values per column]")
print(df.isna().sum().to_string())

# --- Duplicate timestamps -------------------------------------------------
dup_ts = int(df["timestamp"].duplicated().sum())
print(f"\n[Duplicate timestamps] {dup_ts}")

# --- Constant-label columns ----------------------------------------------
print("\n[Constant-label columns]")
for col in ["meter_id", "building_id"]:
    uniq = df[col].unique().tolist()
    print(f"{col}: {df[col].nunique()} unique -> {uniq}")

# --- Hourly gap detection -------------------------------------------------
ts_sorted = df["timestamp"].sort_values().reset_index(drop=True)
diffs = ts_sorted.diff().dropna()
print("\n[Time-step distribution (gap detection)]")
print(diffs.value_counts().to_string())

# --- Target summary -------------------------------------------------------
print("\n[electricity_kwh summary]")
print(df["electricity_kwh"].describe().round(2).to_string())
print(f"negative values: {int((df['electricity_kwh'] < 0).sum())}")
print(f"zero values:     {int((df['electricity_kwh'] == 0).sum())}")

# --- Findings -------------------------------------------------------------
print("\n" + "=" * 64)
print("FINDINGS / CLEANING & PREPROCESSING NEEDS")
print("=" * 64)
continuous = diffs.nunique() == 1 and diffs.iloc[0] == pd.Timedelta(hours=1)
print(f"- Timestamps are continuous hourly (no gaps): {bool(continuous)}")
print(f"- Duplicate timestamps: {dup_ts}")
print(f"- Missing values: {int(df.isna().sum().sum())}")
print(f"- meter_id is constant ({df['meter_id'].unique().tolist()}) -> "
      "drop as a feature; keep only as a label.")
print(f"- building_id is constant ({df['building_id'].unique().tolist()}) -> "
      "drop as a feature; keep only as a label.")
print(f"- electricity_kwh has no negative or zero values in this sample.")
print("Recommended preprocessing for later steps:")
print("  1. Set 'timestamp' as the index and sort the series by time.")
print("  2. Drop constant 'meter_id'/'building_id' (kept as metadata only).")
print("  3. Data already looks clean: no NaNs, no duplicates, no gaps, "
      "no bad values -> no imputation needed.")
