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

# Shared limits for both axes: only then does y = x sit at 45 degrees and does
# "above/below the line" read as over-/under-prediction.
#
# The view is clipped to 20-2000 EUR. A handful of listings priced up to 9,999
# EUR would otherwise stretch the square over three orders of magnitude and
# leave most of the panel empty. Clipping is display-only -- the model is
# trained and evaluated on the full data -- so the count of hidden points is
# printed below and belongs in the figure caption.
VIEW_MIN, VIEW_MAX = 20, 2000
lims = [VIEW_MIN, VIEW_MAX]
off_view = ((actual_price < VIEW_MIN) | (actual_price > VIEW_MAX)
            | (predicted_price < VIEW_MIN) | (predicted_price > VIEW_MAX))

ax.grid(True, which="major", color="#C7CDD3", linewidth=0.6, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

ax.plot(lims, lims, color="#B8BEC4", linewidth=1.0, linestyle="-", zorder=1, label="Perfect prediction (y = x)")

ax.scatter(
    actual_price[~is_hidden], predicted_price[~is_hidden],
    s=18, color="#A6B8C7", alpha=0.4, edgecolor="none", label=f"Other listings (n={(~is_hidden).sum():,})", zorder=2,
)
ax.scatter(
    actual_price[is_hidden], predicted_price[is_hidden],
    s=60, color="#800020", edgecolor="white", linewidth=0.6, label=f"Hidden gems (n={is_hidden.sum():,})", zorder=3,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)

# Default log ticks label only 10^2 and 10^3, which makes euro values hard to
# read off the axis. Label a 1-2-5 sequence as plain numbers instead, and drop
# the minor ticks so the grid stays quiet.
ticks = [20, 50, 100, 200, 500, 1000, 2000]
for axis, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
    setter(ticks)
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("Actual price (EUR, log scale)")
ax.set_ylabel("Predicted price (EUR, log scale)")
ax.set_title(
    f"Decision Tree: Actual vs. Predicted Price\n(held-out test set, n={len(actual_price):,})",
    fontsize=13,
    fontweight="bold",
)
leg = ax.legend(loc="upper left", frameon=True, fontsize=9)
leg.get_frame().set_edgecolor("#D7DBDF")
leg.get_frame().set_linewidth(0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#9AA2AA")
    ax.spines[spine].set_linewidth(0.8)
ax.tick_params(colors="#5A6169", labelcolor="#33383D", length=3, width=0.8)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig("price_model_actual_vs_predicted.png", dpi=150)
print("Saved price_model_actual_vs_predicted.png")
print(f"Test set size: {len(actual_price)}")
print(
    f"Outside the {VIEW_MIN}-{VIEW_MAX} EUR view: {off_view.sum()} listings "
    f"({off_view.mean() * 100:.2f}%) -- state this in the figure caption"
)
