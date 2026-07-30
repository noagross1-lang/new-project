import pandas as pd

INPUT_PATH = "data/reviews_cleaned_2025_sentiment.csv"
OUTPUT_PATH = "data/listing_positive_rate.csv"
POSITIVE_THRESHOLD = 0.5

# listing_id must be read as a string: Airbnb's newer ids are 18-19 digits, which
# exceeds float64 precision, so letting pandas infer a numeric dtype silently
# rounds them and breaks every downstream join.
df = pd.read_csv(
    INPUT_PATH,
    dtype={"listing_id": str},
    low_memory=False,
    encoding="utf-8",
    encoding_errors="replace",
)

# A handful of rows inherited corrupted listing_id values from upstream CSV
# parsing issues in the raw data; drop rows where listing_id isn't a valid id.
df["listing_id"] = df["listing_id"].astype(str).str.strip().str.removesuffix(".0")
valid_listing_id = df["listing_id"].str.fullmatch(r"\d+")
n_dropped = (~valid_listing_id).sum()
if n_dropped:
    print(f"Dropping {n_dropped} rows with corrupted listing_id.")
df = df[valid_listing_id].copy()

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
