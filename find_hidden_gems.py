import pandas as pd

HIDDEN_FLAG_PATH = "data/listing_hidden_flag.csv"
POSITIVE_RATE_PATH = "data/listing_positive_rate.csv"
PREDICTIONS_PATH = "data/listings_model_predictions.csv"
OUTPUT_PATH = "data/hidden_gems.csv"

POSITIVE_RATE_THRESHOLD = 0.75
PRICE_DISCOUNT_THRESHOLD = 0.15

hidden_flag = pd.read_csv(HIDDEN_FLAG_PATH)
positive_rate = pd.read_csv(POSITIVE_RATE_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH).rename(columns={"id": "listing_id"})

merged = hidden_flag.merge(positive_rate, on="listing_id", how="inner")
merged = merged.merge(predictions, on="listing_id", how="inner")

is_hidden = merged["is_hidden"] == 1
is_well_reviewed = merged["positive_rate"] > POSITIVE_RATE_THRESHOLD
is_underpriced = merged["price_clean"] <= merged["predicted_price"] * (1 - PRICE_DISCOUNT_THRESHOLD)

hidden_gems = merged[is_hidden & is_well_reviewed & is_underpriced].copy()
hidden_gems["discount_pct"] = 1 - hidden_gems["price_clean"] / hidden_gems["predicted_price"]

hidden_gems = hidden_gems[
    ["listing_id", "is_hidden", "positive_rate", "price_clean", "predicted_price", "discount_pct"]
].sort_values("discount_pct", ascending=False)

hidden_gems.to_csv(OUTPUT_PATH, index=False)

print(f"Listings evaluated: {len(merged)}")
print(f"Hidden gems found: {len(hidden_gems)}")
print(f"Saved to: {OUTPUT_PATH}")
print(hidden_gems.head(10).to_string(index=False))
