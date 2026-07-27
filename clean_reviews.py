import argparse
import csv
import re
import unicodedata

import pandas as pd

COMMON_ENGLISH_WORDS = {
    "the",
    "and",
    "is",
    "in",
    "it",
    "this",
    "that",
    "was",
    "for",
    "with",
    "room",
    "place",
    "great",
    "host",
    "nice",
    "good",
    "very",
    "clean",
    "helpful",
    "stay",
    "location",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WORD_PATTERN = re.compile(r"[A-Za-z']+")
LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")
NON_LATIN_LETTER_PATTERN = re.compile(r"[^\u0000-\u007F]")
CLEAN_CHARS_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_meaningful_text(text: str) -> bool:
    if not text:
        return False

    # Remove whitespace and punctuation-only cases.
    stripped = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'Z' and unicodedata.category(ch)[0] != 'P' and unicodedata.category(ch)[0] != 'S')
    if not stripped:
        return False

    # Keep comments with at least one letter-based word.
    words = WORD_PATTERN.findall(text)
    if len(words) == 0:
        return False

    # The text should contain at least one Latin character.
    if LATIN_LETTER_PATTERN.search(text) is None:
        return False

    return True


def looks_english(text: str) -> bool:
    if not text:
        return False

    text = text.lower()
    words = WORD_PATTERN.findall(text)
    if not words:
        return False

    latin_letters = sum(1 for ch in text if 'a' <= ch <= 'z')
    all_letters = sum(1 for ch in text if ch.isalpha())
    if all_letters > 0 and latin_letters / all_letters < 0.8:
        return False

    common_count = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    if common_count >= 1:
        return True

    if len(words) >= 5 and all('a' <= w[0] <= 'z' for w in words[:5]):
        return True

    return False


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk = chunk[chunk["date"].dt.year == 2025]
    if chunk.empty:
        return chunk

    chunk["comments_clean"] = chunk["comments"].fillna("").map(normalize_text)
    chunk = chunk[chunk["comments_clean"].map(has_meaningful_text)]
    chunk = chunk[chunk["comments_clean"].map(looks_english)]
    return chunk


def clean_reviews(input_path: str, output_path: str, chunksize: int = 100000) -> None:
    first_chunk = True
    total_rows = 0
    kept_rows = 0

    csv.field_size_limit(2**25)

    for chunk in pd.read_csv(input_path, dtype=str, chunksize=chunksize):
        cleaned = process_chunk(chunk)
        total_rows += len(chunk)
        kept_rows += len(cleaned)

        if cleaned.empty:
            continue

        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        first_chunk = False

    print(f"Processed {total_rows} review rows.")
    print(f"Kept {kept_rows} reviews from 2025 with English content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb reviews for 2025 and English VADER-ready text.")
    parser.add_argument("--input", default="data/reviews.csv", help="Path to the raw reviews CSV.")
    parser.add_argument("--output", default="data/reviews_cleaned_2025.csv", help="Path to write the cleaned reviews CSV.")
    parser.add_argument("--chunksize", type=int, default=100000, help="Number of rows to process per chunk.")
    args = parser.parse_args()
    clean_reviews(args.input, args.output, args.chunksize)
