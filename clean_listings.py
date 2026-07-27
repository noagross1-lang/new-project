import argparse
import numpy as np
import pandas as pd


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


def clean_listings(input_path: str, reviews_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, dtype=str)

    # Remove duplicate listings, keeping the first occurrence.
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    # Convert price to numeric and compute log_price.
    df["price_clean"] = df["price"].apply(parse_price)
    df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
    df["log_price"] = np.log(df["price_clean"].replace(0, np.nan))

    # Convert date columns in listings if available.
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Keep only rows with a valid numeric price.
    df = df[df["price_clean"].notna()]
    df = df[df["price_clean"] > 0]

    # Load reviews and keep only listings that have at least one 2025 review.
    reviews = pd.read_csv("data/reviews.csv", dtype=str)
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews_2025 = reviews[reviews["date"].dt.year == 2025]
    valid_listing_ids = set(reviews_2025["listing_id"].dropna().astype(str))
    df = df[df["id"].astype(str).isin(valid_listing_ids)].copy()

    # Convert a few numeric columns for later use, but do not drop rows for missing non-critical fields.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce")
    df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
    df["availability_365"] = pd.to_numeric(df["availability_365"], errors="coerce")
    df["number_of_reviews_ltm"] = pd.to_numeric(df["number_of_reviews_ltm"], errors="coerce")

    # Save cleaned dataset.
    df.to_csv(output_path, index=False)

    print(f"Cleaned listings saved to: {output_path}")
    print(f"Original rows: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb listings for Rome and keep only listings with prices and reviews in 2025.")
    parser.add_argument("--input", default="data/listings.csv", help="Path to the raw listings CSV.")
    parser.add_argument("--reviews", default="data/reviews.csv", help="Path to the raw reviews CSV.")
    parser.add_argument("--output", default="data/listings_cleaned_2025.csv", help="Path to write the cleaned listings CSV.")
    args = parser.parse_args()
    clean_listings(args.input, args.reviews, args.output)
