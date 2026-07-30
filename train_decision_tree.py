import os
import sys
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# The model predicts a "fair price" from what a listing physically *is*, not from
# how well it is doing. This matters because the gap between the actual price and
# the predicted one is later used as a Hidden Gem criterion: if the model already
# knew how popular a listing is, that criterion would be contaminated by the very
# signal we are trying to hold out.
NUMERIC_FEATURES = [
    # Capacity and layout
    'accommodates',
    'bedrooms',
    'beds',
    'bathrooms_num',
    # Amenities
    'amenities_count',
    'has_wifi',
    'has_kitchen',
    'has_air_conditioning',
    'has_tv',
    'has_elevator',
    'has_washer',
    'has_dishwasher',
    'has_free_parking',
    # Location
    'latitude',
    'longitude',
    # Booking policy
    'minimum_nights',
    'maximum_nights',
    'instant_bookable_flag',
    # Host
    'host_years',
    'host_is_superhost_flag',
    'calculated_host_listings_count',
]

CATEGORICAL_FEATURES = ['room_type', 'neighbourhood_cleansed', 'property_type']

# property_type has 63 levels with a very long tail; keep the common ones and
# fold the rest into "Other" so one-hot encoding does not explode.
PROPERTY_TYPE_TOP_N = 15

# Chosen by 5-fold cross-validation on the training split alone. An unconstrained
# tree grew 17,619 leaves for 18,354 training rows - one leaf per listing - which
# drove train RMSE to 0.01 while test RMSE stayed at 0.50. These limits give the
# best CV score (0.4114) with almost no train/test gap, and keep the tree small
# enough to actually interpret.
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 20

# Never features. Reviews, ratings and occupancy are popularity signals, and
# estimated_revenue_l365d is roughly price x nights booked, i.e. direct leakage
# of the target. Listed explicitly so the assertion below can enforce it.
EXCLUDED_BY_DESIGN = [
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'number_of_reviews',
    'number_of_reviews_ltm',
    'number_of_reviews_l30d',
    'number_of_reviews_ly',
    'reviews_per_month',
    'first_review',
    'last_review',
    'estimated_occupancy_l365d',
    'estimated_revenue_l365d',
    'availability_30',
    'availability_60',
    'availability_90',
    'availability_365',
    'price',
    'price_clean',
    'log_price',
]


def safe_load_listings(path):
    if not os.path.exists(path):
        print(f"ERROR: input file not found: {path}")
        sys.exit(2)
    # id must stay a string: 18-19 digit Airbnb ids do not survive float64.
    return pd.read_csv(path, dtype={'id': str}, low_memory=False, encoding='utf-8')


def prepare_features(df):
    df = df.copy()

    # Ensure label exists
    if 'log_price' not in df.columns:
        if 'price_clean' in df.columns:
            df['log_price'] = np.log(df['price_clean'].replace(0, np.nan))
        else:
            raise ValueError('No label column `log_price` or `price_clean` present')

    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected feature columns: {missing}')

    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Median imputation for the few structural fields that have gaps
    # (bedrooms ~2%, beds ~11%); filling with 0 would claim "no bedrooms".
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())

    top_types = df['property_type'].value_counts().head(PROPERTY_TYPE_TOP_N).index
    df['property_type'] = df['property_type'].where(df['property_type'].isin(top_types), 'Other')
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna('Unknown')

    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True, dtype=int)
    X = pd.concat([df[NUMERIC_FEATURES], dummies], axis=1)

    # Guard the whole premise of the project: no popularity or price-derived
    # column may reach the feature matrix, however the code above evolves.
    leaked = [c for c in X.columns if c in EXCLUDED_BY_DESIGN]
    if leaked:
        raise AssertionError(f'Excluded columns leaked into the feature matrix: {leaked}')

    y = df['log_price'].astype(float)
    return df, X, y


def report_metrics(label, actual_log, predicted_log, actual_price):
    predicted_price = np.exp(predicted_log)
    rmse_log = math.sqrt(mean_squared_error(actual_log, predicted_log))
    rmse_price = math.sqrt(mean_squared_error(actual_price, predicted_price))
    mae_price = np.abs(actual_price - predicted_price).mean()
    mape = (np.abs(actual_price - predicted_price) / actual_price).mean() * 100

    print(f"\n--- {label} (n={len(actual_log)}) ---")
    print(f"  RMSE (log_price) : {rmse_log:.4f}   (multiplicative error ~ {math.exp(rmse_log):.2f}x)")
    print(f"  RMSE (price)     : {rmse_price:.2f}")
    print(f"  MAE  (price)     : {mae_price:.2f}")
    print(f"  MAPE             : {mape:.2f}%")
    return rmse_log


def train_and_save(df, X, y, output_csv='data/listings_model_predictions.csv', model_path='models/decision_tree_regressor.joblib'):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(
        random_state=42, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF
    )
    model.fit(X_train, y_train)

    print(f"\nFeatures used: {X.shape[1]} ({len(NUMERIC_FEATURES)} numeric + {X.shape[1] - len(NUMERIC_FEATURES)} one-hot)")
    print(f"Tree: max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, leaves={model.get_n_leaves()}")

    rmse_log_test = report_metrics(
        'TEST SET (held out, the honest estimate)',
        y_test,
        model.predict(X_test),
        df.loc[idx_test, 'price_clean'],
    )

    preds_full = model.predict(X)
    rmse_log_full = report_metrics(
        'ALL DATA (includes training rows, optimistic)',
        y,
        preds_full,
        df['price_clean'],
    )
    print(f"\nOverfitting gap in RMSE(log_price): {rmse_log_test - rmse_log_full:+.4f}")

    print("\n--- Top 15 feature importances ---")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for name, value in importances.head(15).items():
        print(f"  {value:6.3f}  {name}")

    out = df.copy()
    out['predicted_log_price'] = preds_full
    out['predicted_price'] = np.exp(out['predicted_log_price'])
    out['residual'] = out['price_clean'] - out['predicted_price']
    # Lets downstream steps check whether a listing's prediction came from data
    # the tree had already seen, which matters for the Hidden Gem price criterion.
    out['in_test_set'] = out.index.isin(idx_test).astype(int)

    cols = ['id', 'price_clean', 'log_price', 'predicted_log_price', 'predicted_price', 'residual', 'in_test_set']
    out[cols].to_csv(output_csv, index=False)
    joblib.dump(model, model_path)
    print(f"\nSaved predictions to {output_csv}")
    print(f"Saved model to {model_path}")


def main():
    input_path = os.path.join('data', 'listings_cleaned_2025.csv')
    df = safe_load_listings(input_path)
    df_prepared, X, y = prepare_features(df)
    train_and_save(df_prepared, X, y)


if __name__ == '__main__':
    main()
