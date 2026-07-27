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


def extract_adj_noun_phrases(tagged_tokens):
    phrases = []
    for (word1, tag1), (word2, tag2) in zip(tagged_tokens, tagged_tokens[1:]):
        if tag1 in ADJ_TAGS and tag2 in NOUN_TAGS:
            w1, w2 = word1.lower(), word2.lower()
            if not is_valid_word(w1) or not is_valid_word(w2):
                continue
            adj = lemmatizer.lemmatize(w1, pos="a")
            noun = lemmatizer.lemmatize(w2, pos="n")
            phrases.append(f"{adj} {noun}")
    return phrases


def build_phrase_counts(texts, batch_size=20000):
    counter = Counter()
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokenized = [TOKEN_PATTERN.findall(t) for t in batch]
        tagged_batch = pos_tag_sents(tokenized)
        for tagged in tagged_batch:
            counter.update(extract_adj_noun_phrases(tagged))
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
    hidden_gems = pd.read_csv(HIDDEN_GEMS_PATH)
    hidden_ids = set(hidden_gems["listing_id"].astype(int))
    print(f"Hidden gem listings: {len(hidden_ids)}")

    reviews = pd.read_csv(
        REVIEWS_PATH,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="warn",
    )

    valid_listing_id = reviews["listing_id"].astype(str).str.match(r"^\d+\.?0?$")
    reviews = reviews[valid_listing_id].copy()
    reviews["listing_id"] = reviews["listing_id"].astype(float).astype(int)

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
    print(f"Unique adj+noun phrases (hidden gems): {len(hidden_phrases)}")

    print("Processing other listings' reviews...")
    other_phrases = build_phrase_counts(other_texts)
    print(f"Unique adj+noun phrases (other listings): {len(other_phrases)}")

    pd.DataFrame(hidden_phrases.most_common(50), columns=["phrase", "count"]).to_csv(
        "data/wordcloud_phrases_hidden_gems.csv", index=False
    )
    pd.DataFrame(other_phrases.most_common(50), columns=["phrase", "count"]).to_csv(
        "data/wordcloud_phrases_other_listings.csv", index=False
    )

    make_wordcloud(
        hidden_phrases,
        "Hidden Gems - Adjective + Noun Phrases",
        "data/wordcloud_hidden_gems.png",
    )
    make_wordcloud(
        other_phrases,
        "Other Listings - Adjective + Noun Phrases",
        "data/wordcloud_other_listings.png",
    )

    print("\nTop 20 phrases - Hidden Gems:")
    for phrase, count in hidden_phrases.most_common(20):
        print(f"  {phrase}: {count}")

    print("\nTop 20 phrases - Other Listings:")
    for phrase, count in other_phrases.most_common(20):
        print(f"  {phrase}: {count}")


if __name__ == "__main__":
    main()
