import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from train_decision_tree import (
    safe_load_listings,
    prepare_features,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
)


def run_experiment(test_size=0.8, random_state=42):
    df = safe_load_listings('data/listings_cleaned_2025.csv')
    df_prepared, X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Same hyperparameters as the main model, so this experiment isolates the
    # effect of the train/test ratio rather than of tree complexity.
    model = DecisionTreeRegressor(
        random_state=random_state, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    rmse_log = math.sqrt(mean_squared_error(y_test, pred_test))

    # Convert to price
    # get corresponding price_clean if available
    test_idx = X_test.index
    test_df = df_prepared.loc[test_idx].copy()
    test_df['predicted_log_price'] = pred_test
    test_df['predicted_price'] = np.exp(test_df['predicted_log_price'])

    if 'price_clean' in test_df.columns and test_df['price_clean'].notna().sum() > 0:
        rmse_price = math.sqrt(mean_squared_error(test_df['price_clean'].fillna(0), test_df['predicted_price'].fillna(0)))
    else:
        rmse_price = float('nan')

    print(f"Experiment: train on {100*(1-test_size):.0f}% | test on {100*test_size:.0f}%")
    print(f"RMSE(log_price) = {rmse_log:.4f}")
    print(f"RMSE(price) = {rmse_price:.2f}")
    print(f"Multiplicative error ~ e^RMSE_log = {math.exp(rmse_log):.3f}x")


if __name__ == '__main__':
    run_experiment(test_size=0.8)
