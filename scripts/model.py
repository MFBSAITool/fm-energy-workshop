"""Model training helper for the FM Energy Workshop.

Trains a simple boosted-tree model (GradientBoostingRegressor) with a fixed
random_state so results are reproducible.
"""
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_STATE = 42


def train_gbr(X_train, y_train, random_state=RANDOM_STATE) -> GradientBoostingRegressor:
    """Fit and return a GradientBoostingRegressor on the training data."""
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model
