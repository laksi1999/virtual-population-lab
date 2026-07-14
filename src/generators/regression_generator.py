import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def generate(train_df, base_df, features, n_samples=1000):
    """
    For each feature, fits one RandomForestRegressor (once, on train_df)
    predicting it from the other features. Synthetic rows are built by
    taking a base apple's other features from base_df — held out from
    training, so predictions aren't leaking from rows the model has seen —
    and adding noise bootstrapped from that model's real training residuals,
    so outputs vary instead of collapsing to a deterministic point estimate.
    """
    models = {}
    residuals = {}

    for target in features:
        inputs = [f for f in features if f != target]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(train_df[inputs], train_df[target])

        residuals[target] = train_df[target].values - model.predict(train_df[inputs])
        models[target] = (model, inputs)

    base_rows = base_df.sample(n_samples, replace=True).reset_index(drop=True)

    synthetic = {}
    for target, (model, inputs) in models.items():
        point_estimate = model.predict(base_rows[inputs])
        noise = np.random.choice(residuals[target], size=n_samples, replace=True)
        synthetic[target] = point_estimate + noise

    return pd.DataFrame(synthetic)[features]
