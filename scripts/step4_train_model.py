"""Step 4 - Train a simple boosted-tree model.

Trains scikit-learn GradientBoostingRegressor (random_state=42) on the train
set. Test-set evaluation is deliberately left to Step 5.
"""
import time

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prepare_data import build_feature_frame, split_train_test
from model import train_gbr, RANDOM_STATE

print("=" * 64)
print("STEP 4: Train a simple boosted-tree model")
print("=" * 64)

frame = build_feature_frame()
X_train, y_train, X_test, y_test = split_train_test(frame)

print(f"\n[Inputs] train rows: {len(X_train)}  features: {list(X_train.columns)}")

t0 = time.perf_counter()
model = train_gbr(X_train, y_train)
dt = time.perf_counter() - t0
print(f"\n[Model] {type(model).__name__} trained in {dt:.2f}s (random_state={RANDOM_STATE})")

print("\n[Key parameters]")
for k in ["loss", "learning_rate", "n_estimators", "max_depth", "subsample",
          "min_samples_leaf", "random_state"]:
    print(f"  {k}: {model.get_params()[k]}")

# In-sample fit (training set only) just to confirm the model learned.
pred_train = model.predict(X_train)
train_mae = mean_absolute_error(y_train, pred_train)
train_rmse = root_mean_squared_error(y_train, pred_train)
y_mean = y_train.mean()

print("\n[In-sample fit on TRAIN set (not the test set)]")
print(f"  train target mean: {y_mean:.2f} kWh")
print(f"  train MAE : {train_mae:.3f} kWh")
print(f"  train RMSE: {train_rmse:.3f} kWh")

print("\n[Sample training predictions (first 5 rows)]")
import pandas as pd
samp = pd.DataFrame({"actual": y_train.values[:5], "predicted": pred_train[:5]},
                    index=y_train.index[:5])
print(samp.round(2).to_string())

print("\n[Interpretation]")
print("- The boosted tree fits the training data well (low in-sample error).")
print("- In-sample error is optimistic because the model has seen these rows;")
print("  the honest assessment comes in Step 5 on the held-out test set.")
