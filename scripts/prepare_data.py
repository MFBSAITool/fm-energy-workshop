"""Shared data preparation for the FM Energy Workshop.

Loads data/one_meter.csv, builds the prediction-time-safe feature set, and
returns the prepared frame (features + target) with the first 168 rows that
lack full history removed.

Feature definitions (predict electricity_kwh at hour t):
  - hour_of_day : 0-23, from the timestamp at t  (known in advance)
  - day_of_week : Monday=0 .. Sunday=6, from the timestamp at t (known in advance)
  - lag_1  : electricity_kwh 1 hour earlier   (y[t-1])
  - lag_2  : electricity_kwh 2 hours earlier  (y[t-2])
  - lag_3  : electricity_kwh 3 hours earlier   (y[t-3])
  - lag_24 : electricity_kwh 24 hours earlier (y[t-24])
  - lag_168: electricity_kwh 168 hours earlier = 1 week earlier (y[t-168])
All lags reference readings BEFORE hour t, so nothing from the target hour or
the future is used -> no leakage.
"""
from pathlib import Path

import pandas as pd

FEATURES = ["hour_of_day", "day_of_week", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET = "electricity_kwh"

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "one_meter.csv"


def build_feature_frame(data_path=None) -> pd.DataFrame:
    """Return a time-indexed DataFrame of FEATURES + TARGET.

    The first 168 rows are dropped because the largest lag (lag_168) needs a
    full week of prior history before it is defined.
    """
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # meter_id / building_id are constant labels (no predictive signal); keep
    # only the target series to build lags from.
    series = df[TARGET]

    frame = pd.DataFrame(index=df.index)
    frame["hour_of_day"] = df.index.hour          # 0-23
    frame["day_of_week"] = df.index.dayofweek      # Monday=0 .. Sunday=6
    frame["lag_1"] = series.shift(1)
    frame["lag_2"] = series.shift(2)
    frame["lag_3"] = series.shift(3)
    frame["lag_24"] = series.shift(24)
    frame["lag_168"] = series.shift(168)
    frame[TARGET] = series                         # value to predict (at hour t)

    # Drop the first 168 rows that lack full history (largest lag is 168).
    frame = frame.iloc[168:].copy()
    return frame


# Two weeks of hourly data held back as the unseen test set.
TEST_HOURS = 14 * 24  # 336


def split_train_test(frame, test_hours=TEST_HOURS):
    """Chronological split: the last `test_hours` rows are the test set the
    model never sees during training; everything before is the train set.

    Returns (X_train, y_train, X_test, y_test), preserving the DatetimeIndex.
    """
    train_df = frame.iloc[:-test_hours]
    test_df = frame.iloc[-test_hours:]
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    return X_train, y_train, X_test, y_test
