"""Step 2 - Prepare the data: build features and drop incomplete rows."""
import pandas as pd

from prepare_data import build_feature_frame, FEATURES, TARGET

print("=" * 64)
print("STEP 2: Prepare the data")
print("=" * 64)

frame = build_feature_frame()

print("\n[Feature set] (order will be saved later as feature_order.json)")
for f in FEATURES:
    print(f"  - {f}")
print(f"  - {TARGET}  (target)")

print("\n[Prepared frame]")
print(f"rows = {len(frame)}  (original 1440 - 168 dropped = 1272)")
print(f"columns = {len(frame.columns)}")
print(f"index type = {type(frame.index).__name__}")
print(f"index first = {frame.index.min()}")
print(f"index last  = {frame.index.max()}")

print("\n[Dtypes]")
print(frame.dtypes.to_string())

print("\n[First 5 rows]")
print(frame.head().to_string())

print("\n[Quality checks]")
print(f"total NaNs in frame: {int(frame.isna().sum().sum())}")
print(f"hour_of_day range: {int(frame['hour_of_day'].min())}-{int(frame['hour_of_day'].max())}")
print(f"day_of_week range: {int(frame['day_of_week'].min())}-{int(frame['day_of_week'].max())}")

# Sanity: lags must equal earlier targets (within the prepared frame).
print("\n[Sanity checks]")
i1 = 1
print(f"lag_1 @ {frame.index[i1]} = {frame.iloc[i1]['lag_1']} | "
      f"target @ {frame.index[i1-1]} = {frame.iloc[i1-1][TARGET]} | "
      f"match: {bool(frame.iloc[i1]['lag_1'] == frame.iloc[i1-1][TARGET])}")
k = 168
print(f"lag_168 @ {frame.index[k]} = {frame.iloc[k]['lag_168']} | "
      f"target @ {frame.index[k-168]} = {frame.iloc[k-168][TARGET]} | "
      f"match: {bool(frame.iloc[k]['lag_168'] == frame.iloc[k-168][TARGET])}")

print("\n[Interpretation]")
print("- Each row predicts the consumption at its own hour (t) using only")
print("  calendar info at t and readings from 1/2/3/24/168 hours earlier.")
print("- The first 168 hours (no full week of history yet) are dropped, so")
print(f"  {len(frame)} usable rows remain, all with complete features.")
