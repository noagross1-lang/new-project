import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_PATH = "data/reviews_cleaned_2025.csv"
OUTPUT_PATH = "data/reviews_cleaned_2025_sentiment.csv"
TEXT_COLUMN = "comments_clean"

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8",
    encoding_errors="replace",
    engine="python",
    on_bad_lines="warn",
)

analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

comments_idx = df.columns.get_loc(TEXT_COLUMN)
df.insert(comments_idx + 1, "sentiment_vader", df[TEXT_COLUMN].apply(vader_score))

df.to_csv(OUTPUT_PATH, index=False)

print(f"Rows processed: {len(df)}")
print(f"Saved to: {OUTPUT_PATH}")
print(df[["comments_clean", "sentiment_vader"]].head(5).to_string())
