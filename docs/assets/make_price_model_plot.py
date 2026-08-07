import sys
sys.path.insert(0, "../..")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from train_decision_tree import (
    safe_load_listings,
    prepare_features,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
)

df = safe_load_listings("../../data/listings_cleaned_2025.csv")
df_prepared, X, y = prepare_features(df)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df_prepared.index, test_size=0.8, random_state=42
)

# Must match the model in train_decision_tree.py, or the plot shows a different
# model than the one the report describes.
model = DecisionTreeRegressor(
    random_state=42, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF
)
model.fit(X_train, y_train)
pred_test_log = model.predict(X_test)

actual_price = np.exp(y_test)
predicted_price = np.exp(pred_test_log)

# listing_id as a string: 18-19 digit ids do not survive a float64 dtype.
hidden_gems = pd.read_csv("../../data/hidden_gems.csv", dtype={"listing_id": str})
hidden_ids = set(hidden_gems["listing_id"])
test_ids = df_prepared.loc[idx_test, "id"]
is_hidden = test_ids.isin(hidden_ids).values

fig, ax = plt.subplots(figsize=(7.5, 7))

lims = [5, actual_price.max() * 1.2]
ax.plot(lims, lims, color="#9a9a9a", linewidth=1.5, linestyle="--", zorder=1, label="Perfect prediction (y = x)")

ax.scatter(
    actual_price[~is_hidden], predicted_price[~is_hidden],
    s=22, color="#0072B2", alpha=0.45, edgecolor="none", label=f"Other listings (n={(~is_hidden).sum()})", zorder=2,
)
ax.scatter(
    actual_price[is_hidden], predicted_price[is_hidden],
    s=70, color="#E69F00", edgecolor="black", linewidth=0.6, label=f"Hidden gems (n={is_hidden.sum()})", zorder=3,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual price (EUR, log scale)")
ax.set_ylabel("Predicted price (EUR, log scale)")
ax.set_title(
    f"Decision Tree: Actual vs. Predicted Price\n(held-out test set, n={len(actual_price):,})",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="upper left", frameon=True, fontsize=9)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig("price_model_actual_vs_predicted.png", dpi=150)
print("Saved price_model_actual_vs_predicted.png")
print(f"Test set size: {len(actual_price)}")
