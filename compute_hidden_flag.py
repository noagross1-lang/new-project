import pandas as pd

INPUT_PATH = "data/reviews_cleaned_2025_sentiment.csv"
OUTPUT_PATH = "data/listing_hidden_flag.csv"

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

review_count = df.groupby("listing_id").size().reset_index(name="review_count")

median_reviews = review_count["review_count"].median()
print(f"Median review count per listing: {median_reviews}")

review_count["is_hidden"] = (review_count["review_count"] < median_reviews).astype(int)

hidden_flag = review_count[["listing_id", "is_hidden"]]
hidden_flag.to_csv(OUTPUT_PATH, index=False)

print(f"Listings: {len(hidden_flag)}")
print(f"Hidden: {hidden_flag['is_hidden'].sum()}  Popular: {(hidden_flag['is_hidden'] == 0).sum()}")
print(f"Saved to: {OUTPUT_PATH}")
print(hidden_flag.head(10).to_string(index=False))
