import re
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag_sents
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

REVIEWS_PATH = "data/reviews_cleaned_2025.csv"
HIDDEN_GEMS_PATH = "data/hidden_gems.csv"
TEXT_COLUMN = "comments_clean"

# v2 outputs: kept separate from the original wordcloud_*.png / wordcloud_phrases_*.csv
# on purpose, so this rerun (2-3 word adj+noun phrases instead of adj+noun bigrams
# only) does not change the assets the existing writeup.md/writeup.docx embed.
PHRASES_HIDDEN_CSV = "data/wordcloud_phrases_hidden_gems_v2.csv"
PHRASES_OTHER_CSV = "data/wordcloud_phrases_other_listings_v2.csv"
WORDCLOUD_HIDDEN_PNG = "data/wordcloud_hidden_gems_v2.png"
WORDCLOUD_OTHER_PNG = "data/wordcloud_other_listings_v2.png"

# Words that are technically nouns/adjectives but carry no descriptive
# meaning in this domain (every Airbnb review mentions them).
DOMAIN_NOISE_WORDS = {
    "rome", "roma", "italy", "italian",
    "apartment", "apt", "flat",
    "host", "hosts",
    "stay", "stayed",
    "airbnb",
    "room", "rooms",
    "place",
    "trip", "vacation", "holiday",
}

STOPWORDS = set(stopwords.words("english")) | DOMAIN_NOISE_WORDS

WORD_PATTERN = re.compile(r"^[a-z]+$")
TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[.,!?;:]")

ADJ_TAGS = {"JJ", "JJR", "JJS"}
NOUN_TAGS = {"NN", "NNS"}

lemmatizer = WordNetLemmatizer()


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def filter_english(texts, n_jobs=None):
    n_jobs = n_jobs or max(cpu_count() - 1, 1)
    with Pool(n_jobs) as pool:
        flags = pool.map(is_english, texts, chunksize=500)
    return [t for t, ok in zip(texts, flags) if ok]


def is_valid_word(word: str) -> bool:
    return bool(WORD_PATTERN.match(word)) and len(word) > 2 and word not in STOPWORDS


PHRASE_TAGS = ADJ_TAGS | NOUN_TAGS


def extract_phrases(tagged_tokens):
    """Adjacent 2- or 3-word windows built only from adjective/noun tokens,
    each containing at least one adjective and at least one noun (e.g.
    "great location", "great quiet neighborhood"). Every word is lowercased
    and lemmatized before it is counted, so "Great Location" and "great
    location" collapse into a single key instead of being counted (and
    rendered in the cloud) as two different phrases. Spans longer than 3
    words are never extracted.
    """
    phrases = []
    n = len(tagged_tokens)
    for length in (2, 3):
        for i in range(n - length + 1):
            window = tagged_tokens[i:i + length]
            tags = [tag for _, tag in window]
            if not all(tag in PHRASE_TAGS for tag in tags):
                continue
            if not (any(tag in ADJ_TAGS for tag in tags) and any(tag in NOUN_TAGS for tag in tags)):
                continue
            words = [word.lower() for word, _ in window]
            if not all(is_valid_word(w) for w in words):
                continue
            lemmas = [
                lemmatizer.lemmatize(w, pos="a" if tag in ADJ_TAGS else "n")
                for w, tag in zip(words, tags)
            ]
            phrases.append(" ".join(lemmas))
    return phrases


def build_phrase_counts(texts, batch_size=20000):
    counter = Counter()
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokenized = [TOKEN_PATTERN.findall(t) for t in batch]
        tagged_batch = pos_tag_sents(tokenized)
        for tagged in tagged_batch:
            counter.update(extract_phrases(tagged))
        print(f"  processed {min(start + batch_size, len(texts))}/{len(texts)} reviews")
    return counter


def make_wordcloud(freqs: Counter, title: str, output_path: str):
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=100,
        prefer_horizontal=0.95,
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(freqs)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=22, fontweight="bold", pad=20)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved word cloud: {output_path}")


def main():
    # listing_id stays a string throughout: 18-19 digit Airbnb ids exceed float64
    # precision, so a numeric dtype would round them and break this membership test.
    hidden_gems = pd.read_csv(HIDDEN_GEMS_PATH, dtype={"listing_id": str})
    hidden_ids = set(hidden_gems["listing_id"])
    print(f"Hidden gem listings: {len(hidden_ids)}")

    reviews = pd.read_csv(
        REVIEWS_PATH,
        dtype={"listing_id": str},
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="warn",
    )

    reviews["listing_id"] = reviews["listing_id"].astype(str).str.strip().str.removesuffix(".0")
    valid_listing_id = reviews["listing_id"].str.fullmatch(r"\d+")
    reviews = reviews[valid_listing_id].copy()

    is_hidden = reviews["listing_id"].isin(hidden_ids)
    hidden_reviews = reviews.loc[is_hidden, TEXT_COLUMN]
    other_reviews = reviews.loc[~is_hidden, TEXT_COLUMN]

    print(f"Hidden gem reviews: {len(hidden_reviews)}")
    print(f"Other reviews: {len(other_reviews)}")

    print("Language-filtering hidden gem reviews...")
    hidden_texts = filter_english(hidden_reviews.tolist())
    print(f"  kept {len(hidden_texts)}/{len(hidden_reviews)} as English")

    print("Language-filtering other listings' reviews...")
    other_texts = filter_english(other_reviews.tolist())
    print(f"  kept {len(other_texts)}/{len(other_reviews)} as English")

    print("Processing hidden gem reviews...")
    hidden_phrases = build_phrase_counts(hidden_texts)
    print(f"Unique phrases (hidden gems): {len(hidden_phrases)}")

    print("Processing other listings' reviews...")
    other_phrases = build_phrase_counts(other_texts)
    print(f"Unique phrases (other listings): {len(other_phrases)}")

    pd.DataFrame(hidden_phrases.most_common(10), columns=["phrase", "count"]).to_csv(
        PHRASES_HIDDEN_CSV, index=False
    )
    pd.DataFrame(other_phrases.most_common(10), columns=["phrase", "count"]).to_csv(
        PHRASES_OTHER_CSV, index=False
    )

    make_wordcloud(
        hidden_phrases,
        "Hidden Gems - Adjective + Noun Phrases",
        WORDCLOUD_HIDDEN_PNG,
    )
    make_wordcloud(
        other_phrases,
        "Other Listings - Adjective + Noun Phrases",
        WORDCLOUD_OTHER_PNG,
    )

    print("\nTop 10 phrases - Hidden Gems:")
    for phrase, count in hidden_phrases.most_common(10):
        print(f"  {phrase}: {count}")

    print("\nTop 10 phrases - Other Listings:")
    for phrase, count in other_phrases.most_common(10):
        print(f"  {phrase}: {count}")


if __name__ == "__main__":
    main()
