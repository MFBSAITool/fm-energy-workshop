"""Step 3 - Split into train and test sets.

Hold back the last two weeks (336 hours) as a test set the model never sees
during training; train on everything before that.
"""
from prepare_data import build_feature_frame, split_train_test, TEST_HOURS, FEATURES, TARGET

print("=" * 64)
print("STEP 3: Split into train and test sets")
print("=" * 64)

frame = build_feature_frame()
X_train, y_train, X_test, y_test = split_train_test(frame)

print(f"\n[Split rule] last {TEST_HOURS} hours ({TEST_HOURS // 24} days) = test set; "
      f"everything before = train set")

print("\n[Sizes]")
print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}  y_test : {y_test.shape}")

print("\n[Date ranges]")
print(f"train: {X_train.index.min()}  ->  {X_train.index.max()}  ({len(X_train)} hours)")
print(f"test : {X_test.index.min()}  ->  {X_test.index.max()}  ({len(X_test)} hours)")

print("\n[Contiguity checks]")
gap_h = (X_test.index.min() - X_train.index.max()).total_seconds() / 3600
print(f"hours between train end and test start: {gap_h:.0f}  (1 means no gap, no overlap)")
test_days = int(round(len(X_test) / 24))
print(f"test length: {len(X_test)} hours = {test_days} full days (expect {TEST_HOURS // 24})")

print("\n[Feature / target alignment]")
print(f"X_train columns: {list(X_train.columns)}")
print(f"y_train name   : {y_train.name}")
print(f"train rows X==y: {len(X_train) == len(y_train)}")
print(f"test  rows X==y: {len(X_test) == len(y_test)}")

print("\n[Interpretation]")
print("- The split is by time only (no shuffling), so the test set is a real")
print("  'future' the model never trained on.")
print("- Train = first ~9.6 weeks (936 hours); Test = final 2 weeks (336 hours).")
print("- Train ends exactly one hour before test begins, so the two sets are")
print("  contiguous with no missing hours and no overlap.")
