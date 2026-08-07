import pandas as pd

HIDDEN_FLAG_PATH = "data/listing_hidden_flag.csv"
POSITIVE_RATE_PATH = "data/listing_positive_rate.csv"
PREDICTIONS_PATH = "data/listings_model_predictions.csv"
OUTPUT_PATH = "data/hidden_gems.csv"

POSITIVE_RATE_THRESHOLD = 0.75
PRICE_DISCOUNT_THRESHOLD = 0.15

# Airbnb ids are up to 19 digits and must stay strings; a numeric dtype rounds
# them past float64 precision and silently breaks these joins.
hidden_flag = pd.read_csv(HIDDEN_FLAG_PATH, dtype={"listing_id": str})
positive_rate = pd.read_csv(POSITIVE_RATE_PATH, dtype={"listing_id": str})
predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"id": str}).rename(columns={"id": "listing_id"})

merged = hidden_flag.merge(positive_rate, on="listing_id", how="inner")
merged = merged.merge(predictions, on="listing_id", how="inner")

is_hidden = merged["is_hidden"] == 1
is_well_reviewed = merged["positive_rate"] > POSITIVE_RATE_THRESHOLD
is_underpriced = merged["price_clean"] <= merged["predicted_price"] * (1 - PRICE_DISCOUNT_THRESHOLD)
# A listing's predicted price is only an honest, out-of-sample estimate if it
# was in the price model's held-out test set; a training-set listing's price
# was already seen by the tree, so it cannot be judged "underpriced" at all.
is_test_set = merged["in_test_set"] == 1

underpriced_candidates = is_hidden & is_well_reviewed & is_underpriced

hidden_gems = merged[underpriced_candidates & is_test_set].copy()
hidden_gems["discount_pct"] = 1 - hidden_gems["price_clean"] / hidden_gems["predicted_price"]

hidden_gems = hidden_gems[
    ["listing_id", "is_hidden", "positive_rate", "price_clean", "predicted_price", "discount_pct", "in_test_set"]
].sort_values("discount_pct", ascending=False)

hidden_gems.to_csv(OUTPUT_PATH, index=False)


def pct(n):
    return f"{100 * n / len(merged):.1f}%"


print("--- Hidden gem funnel ---")
print(f"Listings evaluable (in all three sources): {len(merged)}")
print(f"  1. low exposure (reviews below median)  : {is_hidden.sum():6d}  ({pct(is_hidden.sum())})")
print(f"  2. + positive rate > {POSITIVE_RATE_THRESHOLD}                : {(is_hidden & is_well_reviewed).sum():6d}  ({pct((is_hidden & is_well_reviewed).sum())})")
print(f"  3. + priced >= {PRICE_DISCOUNT_THRESHOLD:.0%} below prediction  : {underpriced_candidates.sum():6d}  ({pct(underpriced_candidates.sum())})")
print(f"  4. + in the price model's test set       : {len(hidden_gems):6d}  ({pct(len(hidden_gems))})")
print(f"\nHidden gems found: {len(hidden_gems)}")
print(f"Saved to: {OUTPUT_PATH}")
print()
print(hidden_gems.head(10).to_string(index=False))
