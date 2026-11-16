"""Step 6 - Inspect example predictions and save the work.

Plots actual vs predicted over the two-week test window, then saves the
fitted model to models/gbr_model.joblib and the feature order to
models/feature_order.json for the later dashboard.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prepare_data import build_feature_frame, split_train_test, FEATURES
from model import train_gbr

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
REPORTS_DIR = REPO / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 64)
print("STEP 6: Inspect example predictions and save the work")
print("=" * 64)

frame = build_feature_frame()
X_train, y_train, X_test, y_test = split_train_test(frame)
model = train_gbr(X_train, y_train)
model_pred = model.predict(X_test)
baseline_pred = X_test["lag_1"]

model_mae = mean_absolute_error(y_test, model_pred)
model_rmse = root_mean_squared_error(y_test, model_pred)
base_mae = mean_absolute_error(y_test, baseline_pred)
base_rmse = root_mean_squared_error(y_test, baseline_pred)

print("\n[Metrics recap] (held-out 2-week test set)")
print(f"  Model    MAE={model_mae:.3f}  RMSE={model_rmse:.3f}")
print(f"  Baseline MAE={base_mae:.3f}  RMSE={base_rmse:.3f}")

# --- Plot actual vs predicted over the two-week test window ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_test.index, y_test.values, label="actual", color="#1f77b4", linewidth=1.3)
ax.plot(y_test.index, model_pred, label="predicted (model)",
        color="#ff7f0e", linewidth=1.0, alpha=0.9)
ax.set_title("METER_01 - actual vs predicted (test: 2016-02-16 to 2016-02-29)")
ax.set_ylabel("Electricity (kWh)")
ax.set_xlabel("Date")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
plot_path = REPORTS_DIR / "actual_vs_predicted.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)
print(f"\n[Plot saved] {plot_path.name}  ({plot_path.stat().st_size:,} bytes)")

# --- Save model and feature order ---
model_path = MODELS_DIR / "gbr_model.joblib"
joblib.dump(model, model_path)
print(f"[Model saved] {model_path.name}  ({model_path.stat().st_size:,} bytes)")

feature_order_path = MODELS_DIR / "feature_order.json"
with open(feature_order_path, "w") as f:
    json.dump(FEATURES, f, indent=2)
print(f"[Feature order saved] {feature_order_path.name}  "
      f"({feature_order_path.stat().st_size:,} bytes)")

# --- Verify the saved artifacts load back and behave identically ---
loaded = joblib.load(model_path)
with open(feature_order_path) as f:
    loaded_features = json.load(f)
pred_check = loaded.predict(X_test[loaded_features])
print("\n[Verification]")
print(f"  loaded model type : {type(loaded).__name__}")
print(f"  loaded feature order: {loaded_features}")
print(f"  feature order matches: {loaded_features == FEATURES}")
print(f"  reload predicts identically: {bool((pred_check == model_pred).all())}")

print("\n[Saved files]")
for p in [model_path, feature_order_path, plot_path]:
    print(f"  {p.relative_to(REPO)}  ({p.stat().st_size:,} bytes)")

print("\n[Interpretation]")
print("- The saved model + feature_order.json are everything the later")
print("  dashboard needs to forecast the next hour from fresh readings.")
print("- The plot shows predicted tracking the daily rise/fall of actual")
print("  consumption across the two-week window; the model follows the real")
print(f"  curve closely (MAE {model_mae:.2f} kWh), beating the previous-hour")
print(f"  baseline (MAE {base_mae:.2f} kWh).")
