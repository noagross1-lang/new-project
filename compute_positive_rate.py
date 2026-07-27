import pandas as pd

INPUT_PATH = "data/reviews_cleaned_2025_sentiment.csv"
OUTPUT_PATH = "data/listing_positive_rate.csv"
POSITIVE_THRESHOLD = 0.5

df = pd.read_csv(INPUT_PATH, low_memory=False)

# A handful of rows inherited corrupted listing_id values from upstream CSV
# parsing issues in the raw data; drop rows where listing_id isn't a valid id.
valid_listing_id = df["listing_id"].astype(str).str.match(r"^\d+\.?0?$")
n_dropped = (~valid_listing_id).sum()
if n_dropped:
    print(f"Dropping {n_dropped} rows with corrupted listing_id.")
df = df[valid_listing_id].copy()
df["listing_id"] = df["listing_id"].astype(float).astype(int)

df["is_positive"] = (df["sentiment_vader"] >= POSITIVE_THRESHOLD).astype(int)

positive_rate = (
    df.groupby("listing_id")["is_positive"]
    .mean()
    .reset_index(name="positive_rate")
)

positive_rate.to_csv(OUTPUT_PATH, index=False)

print(f"Listings: {len(positive_rate)}")
print(f"Saved to: {OUTPUT_PATH}")
print(positive_rate.head(10).to_string(index=False))
