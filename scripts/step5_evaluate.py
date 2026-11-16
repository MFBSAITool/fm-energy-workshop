"""Step 5 - Evaluate the model.

Compare the model's MAE and RMSE (in kWh) against a persistence baseline that
predicts the previous hour's consumption, on the same held-out test rows.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prepare_data import build_feature_frame, split_train_test
from model import train_gbr

print("=" * 64)
print("STEP 5: Evaluate the model")
print("=" * 64)

frame = build_feature_frame()
X_train, y_train, X_test, y_test = split_train_test(frame)

model = train_gbr(X_train, y_train)
model_pred = model.predict(X_test)

# Baseline: previous hour's consumption = lag_1 (an actual past reading,
# available at prediction time).
baseline_pred = X_test["lag_1"]

# Sanity: baseline at the first test row equals the last train actual (the
# true previous-hour reading right at the train/test boundary).
print("\n[Sanity] baseline = previous hour's consumption (lag_1)")
print(f"  last train actual @ {y_train.index[-1]}: {y_train.iloc[-1]:.2f} kWh")
print(f"  baseline pred   @ {baseline_pred.index[0]}: {baseline_pred.iloc[0]:.2f} kWh")
print(f"  match: {bool(y_train.iloc[-1] == baseline_pred.iloc[0])}")

model_mae = mean_absolute_error(y_test, model_pred)
model_rmse = root_mean_squared_error(y_test, model_pred)
base_mae = mean_absolute_error(y_test, baseline_pred)
base_rmse = root_mean_squared_error(y_test, baseline_pred)

print("\n[Context]")
print(f"  test target mean: {y_test.mean():.2f} kWh | std: {y_test.std():.2f} kWh")

print("\n[Results on the held-out 2-week test set] (lower is better)")
print("  Metric | Model    | Baseline(lag_1) | Improvement")
print(f"  MAE    | {model_mae:7.3f} | {base_mae:14.3f} | {base_mae - model_mae:+.3f}")
print(f"  RMSE   | {model_rmse:7.3f} | {base_rmse:14.3f} | {base_rmse - model_rmse:+.3f}")

print("\n[Relative improvement vs baseline]")
print(f"  MAE : {(base_mae - model_mae) / base_mae * 100:+.1f}%")
print(f"  RMSE: {(base_rmse - model_rmse) / base_rmse * 100:+.1f}%")

print("\n[Sample test predictions (first 8 rows)]")
samp = pd.DataFrame({
    "actual": y_test.values[:8],
    "model": model_pred[:8],
    "baseline_lag1": baseline_pred.values[:8],
}, index=y_test.index[:8])
print(samp.round(2).to_string())

print("\n[Interpretation]")
print("- Baseline = 'next hour = last hour's reading'. It is realistic and")
print("  valid because the previous hour is known at prediction time.")
beats = model_mae < base_mae and model_rmse < base_rmse
print(f"- Model MAE {model_mae:.3f} vs baseline {base_mae:.3f}: the model "
      f"{'BEATS' if beats else 'does NOT beat'} the baseline on both metrics.")
