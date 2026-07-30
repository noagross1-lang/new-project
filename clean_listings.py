import argparse
import json
import re

import numpy as np
import pandas as pd

# The raw export stores amenities as a JSON list of free-text labels. We turn the
# most common ones into binary flags so the price model can use them directly.
# Each value is a regex matched case-insensitively against a single amenity label;
# `washer` uses word boundaries so that "Dishwasher" does not match it.
AMENITY_FLAGS = {
    "has_wifi": r"wifi",
    "has_kitchen": r"kitchen",
    "has_air_conditioning": r"air conditioning",
    "has_tv": r"\btv\b|hdtv",
    "has_elevator": r"elevator",
    "has_washer": r"\bwasher\b",
    "has_dishwasher": r"dishwasher",
    "has_free_parking": r"free .*parking",
}

AMENITY_PATTERNS = {name: re.compile(p, re.IGNORECASE) for name, p in AMENITY_FLAGS.items()}

LEADING_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)")

TRUTHY = {"t", "true", "yes", "1"}

# Columns kept in the cleaned output. The raw file has 79 columns, many of them
# long free text (description, host_about) or URLs that nothing downstream reads.
IDENTITY_COLS = ["id", "name", "host_id", "host_name"]
LOCATION_COLS = ["latitude", "longitude", "neighbourhood_cleansed"]
PROPERTY_COLS = [
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "bathrooms_num",
    "amenities_count",
] + list(AMENITY_FLAGS)
POLICY_COLS = ["minimum_nights", "maximum_nights", "instant_bookable", "instant_bookable_flag"]
HOST_COLS = [
    "host_since",
    "host_years",
    "host_is_superhost",
    "host_is_superhost_flag",
    "host_identity_verified",
    "host_total_listings_count",
    "calculated_host_listings_count",
]
PRICE_COLS = ["price", "price_clean", "log_price"]
# Popularity / review fields. These are deliberately NOT model features (see
# EXCLUDED_BY_DESIGN in train_decision_tree.py) but are kept in the cleaned file
# because the clustering profile and the report describe listings with them.
POPULARITY_COLS = [
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "first_review",
    "last_review",
    "availability_365",
]

OUTPUT_COLS = (
    IDENTITY_COLS + LOCATION_COLS + PROPERTY_COLS + POLICY_COLS + HOST_COLS + PRICE_COLS + POPULARITY_COLS
)

NUMERIC_COLS = [
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "maximum_nights",
    "host_total_listings_count",
    "calculated_host_listings_count",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "availability_365",
]


def parse_price(value: str) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return np.nan
    cleaned = text.replace("€", "").replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_amenities(value: str) -> list:
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_bathrooms(bathrooms: str, bathrooms_text: str) -> float:
    """`bathrooms` is missing for ~10% of rows but `bathrooms_text` almost never
    is, so fall back to parsing the text ("1.5 baths", "Half-bath")."""
    value = parse_price(bathrooms)
    if not pd.isna(value):
        return value
    if pd.isna(bathrooms_text):
        return np.nan
    text = str(bathrooms_text)
    match = LEADING_NUMBER.match(text)
    if match:
        return float(match.group(1))
    if "half" in text.lower():
        return 0.5
    return np.nan


def to_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUTHY).astype(int)


def load_listing_ids_with_2025_reviews(reviews_path: str) -> set:
    """Listing ids that have at least one cleaned 2025 review.

    The raw reviews export is not kept in the repo, so we read the cleaned 2025
    file instead. It is already restricted to 2025 and to English-language
    reviews, which matches the population that the sentiment stage scores.
    """
    ids = set()
    reader = pd.read_csv(
        reviews_path,
        dtype=str,
        usecols=["listing_id"],
        chunksize=500000,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )
    for chunk in reader:
        ids.update(chunk["listing_id"].dropna())
    # A few rows carry corrupted listing ids from upstream CSV parsing issues.
    return {i for i in ids if i.isdigit()}


def clean_listings(input_path: str, reviews_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, dtype=str, low_memory=False, encoding="utf-8")
    print(f"Raw listings: {len(df)} rows, {len(df.columns)} columns")

    # Remove duplicate listings, keeping the first occurrence.
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    # Convert price to numeric and compute log_price.
    df["price_clean"] = pd.to_numeric(df["price"].apply(parse_price), errors="coerce")
    df["log_price"] = np.log(df["price_clean"].replace(0, np.nan))

    # Keep only rows with a valid numeric price.
    n_before = len(df)
    df = df[df["price_clean"].notna() & (df["price_clean"] > 0)].copy()
    print(f"Dropped {n_before - len(df)} rows without a usable price.")

    # Keep only listings that have at least one 2025 review.
    valid_listing_ids = load_listing_ids_with_2025_reviews(reviews_path)
    print(f"Listings with a cleaned 2025 review: {len(valid_listing_ids)}")
    n_before = len(df)
    df = df[df["id"].isin(valid_listing_ids)].copy()
    print(f"Dropped {n_before - len(df)} rows without a 2025 review.")

    # Derive property features from the amenities list.
    amenities = df["amenities"].apply(parse_amenities)
    df["amenities_count"] = amenities.apply(len)
    for name, pattern in AMENITY_PATTERNS.items():
        df[name] = amenities.apply(
            lambda items, p=pattern: int(any(p.search(item) for item in items))
        )

    # Bathrooms, with bathrooms_text as a fallback for the missing numeric field.
    df["bathrooms_num"] = [
        parse_bathrooms(b, t) for b, t in zip(df["bathrooms"], df["bathrooms_text"])
    ]

    # Host tenure in years, relative to the scrape date of this export.
    scraped = pd.to_datetime(df["last_scraped"], errors="coerce")
    host_since = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_years"] = (scraped - host_since).dt.days / 365.25

    # Boolean-like fields normalized to 0/1.
    df["host_is_superhost_flag"] = to_flag(df["host_is_superhost"])
    df["instant_bookable_flag"] = to_flag(df["instant_bookable"])

    # Date columns.
    for col in ("first_review", "last_review"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[OUTPUT_COLS].to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nCleaned listings saved to: {output_path}")
    print(f"Rows: {len(df)}  Columns: {len(OUTPUT_COLS)}")
    print(f"Neighbourhoods: {df['neighbourhood_cleansed'].nunique()}")
    print("\nMissing values in derived features:")
    derived = ["amenities_count", "bathrooms_num", "host_years"] + list(AMENITY_FLAGS)
    for col in derived:
        print(f"  {col:26s} {df[col].isna().mean() * 100:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Airbnb listings for Rome and keep only listings with prices and reviews in 2025."
    )
    parser.add_argument("--input", default="data/listings.csv", help="Path to the raw listings CSV.")
    parser.add_argument(
        "--reviews",
        default="data/reviews_cleaned_2025.csv",
        help="Path to the cleaned 2025 reviews CSV, used to keep only listings reviewed in 2025.",
    )
    parser.add_argument("--output", default="data/listings_cleaned_2025.csv", help="Path to write the cleaned listings CSV.")
    args = parser.parse_args()
    clean_listings(args.input, args.reviews, args.output)
