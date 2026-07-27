import os
import sys
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def safe_load_listings(path):
    if not os.path.exists(path):
        print(f"ERROR: input file not found: {path}")
        sys.exit(2)
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    df = df.copy()

    # Ensure label exists
    if 'log_price' not in df.columns:
        if 'price_clean' in df.columns:
            df['log_price'] = np.log(df['price_clean'].replace(0, np.nan))
        else:
            raise ValueError('No label column `log_price` or `price_clean` present')

    # Feature candidates
    numeric_feats = [
        'minimum_nights',
        'number_of_reviews',
        'reviews_per_month',
        'calculated_host_listings_count',
        'availability_365',
        'accommodates',
        'review_scores_rating'
    ]

    for c in numeric_feats:
        if c not in df.columns:
            # create if missing
            df[c] = np.nan

    # Boolean-like fields: normalize to 0/1
    bool_map = {
        'instant_bookable': ['t', 'true', 'True', 'TRUE', True],
        'is_location_exact': ['t', 'true', 'True', True],
        'host_is_superhost': ['t', 'true', 'True', True]
    }
    for col, truthy in bool_map.items():
        if col in df.columns:
            df[col] = df[col].isin(truthy).astype(int)
        else:
            df[col] = 0

    # Categorical features to one-hot
    cat_feats = [c for c in ['room_type', 'neighbourhood'] if c in df.columns]
    for c in cat_feats:
        df[c] = df[c].fillna('Unknown')

    # Fill numeric NaNs
    df[numeric_feats] = df[numeric_feats].fillna(0)

    # One-hot encode categories (drop_first to avoid collinearity)
    if cat_feats:
        df = pd.get_dummies(df, columns=cat_feats, drop_first=True)

    # Build final feature matrix
    feature_cols = [c for c in df.columns if c not in (
        ['id', 'price', 'price_clean', 'log_price', 'host_name', 'name', 'description']
    ) and (not c.startswith('Unnamed:'))]

    # Make sure label not in features
    feature_cols = [c for c in feature_cols if c != 'log_price']

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    # If there are non-numeric columns left, drop them
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = X.drop(columns=non_numeric)

    y = df['log_price'].astype(float)
    return df, X, y


def train_and_save(df, X, y, output_csv='data/listings_model_predictions.csv', model_path='models/decision_tree_regressor.joblib'):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    rmse_log = math.sqrt(mean_squared_error(y_test, pred_test))

    # Report RMSE in price units as well
    # Convert log predictions back to price
    test_df = df.loc[idx_test].copy()
    test_df['predicted_log_price'] = pred_test
    test_df['predicted_price'] = np.exp(test_df['predicted_log_price'])
    test_df['price_clean'] = test_df.get('price_clean', np.nan)
    # Where we have price_clean, compute rmse on price
    if test_df['price_clean'].notna().sum() > 0:
        rmse_price = math.sqrt(mean_squared_error(test_df['price_clean'].fillna(0), test_df['predicted_price'].fillna(0)))
    else:
        rmse_price = float('nan')

    print(f"RMSE (log_price): {rmse_log:.4f}")
    print(f"RMSE (price): {rmse_price:.2f}")

    # Predict for full dataset
    X_full = X
    preds_full = model.predict(X_full)
    out = df.copy()
    out['predicted_log_price'] = preds_full
    out['predicted_price'] = np.exp(out['predicted_log_price'])
    out['residual'] = out.get('price_clean', np.nan) - out['predicted_price']

    # Select output columns
    cols = ['id']
    if 'price_clean' in out.columns:
        cols.append('price_clean')
    cols += ['log_price', 'predicted_log_price', 'predicted_price', 'residual']

    out[cols].to_csv(output_csv, index=False)
    joblib.dump(model, model_path)
    print(f"Saved predictions to {output_csv}")
    print(f"Saved model to {model_path}")


def main():
    input_path = os.path.join('data', 'listings_cleaned_2025.csv')
    df = safe_load_listings(input_path)
    df_prepared, X, y = prepare_features(df)
    train_and_save(df_prepared, X, y)


if __name__ == '__main__':
    main()
