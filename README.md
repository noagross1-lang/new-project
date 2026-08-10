# Hidden Gems in Rome's Airbnb Market

Finding listings that are **underpriced**, **well reviewed**, and **not yet popular**,
using the Inside Airbnb export for Rome (scraped 2025-09-15).

## Setup

```bash
pip install -r requirements.txt
```

`build_wordclouds.py` also needs NLTK data:

```python
import nltk
nltk.download("stopwords"); nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")
```

## Input data (`data/`, not in version control)

| File | Description |
| --- | --- |
| `listings.csv` | Full Inside Airbnb export — one row per listing, 79 columns |
| `reviews.csv` | One row per review |

Neighbourhoods come from the `neighbourhood_cleansed` column of `listings.csv`
(15 Rome *municipi*, no missing values), not from a separate file.

## Pipeline

Run from the repository root, in this order:

```bash
python clean_listings.py          # -> data/listings_cleaned_2025.csv
python clean_reviews.py           # -> data/reviews_cleaned_2025.csv
python compute_sentiment.py       # -> data/reviews_cleaned_2025_sentiment.csv  (VADER)
python compute_positive_rate.py   # -> data/listing_positive_rate.csv
python compute_hidden_flag.py     # -> data/listing_hidden_flag.csv
python train_decision_tree.py     # -> data/listings_model_predictions.csv, models/*.joblib
                                  #    also prints the RMSE / MAE / MAPE table
python find_hidden_gems.py        # -> data/hidden_gems.csv

cd clustering
python build_clusters.py          # -> cluster_assignments_k*.csv, elbow_*.png,
                                  #    external_validation_k4.csv, final_cluster_profile_k4.csv
cd ..

python build_wordclouds.py        # -> data/wordcloud_*_v2.png, data/wordcloud_phrases_*_v2.csv
```

The write-up figures are built from `docs/assets/make_*.py`, each run from the
`docs/assets/` directory.

`clean_listings.py` reads `data/reviews_cleaned_2025.csv` to keep only listings
reviewed in 2025, so `clean_reviews.py` must have been run at least once before it.

## Method

**Price model** (`train_decision_tree.py`) — a `DecisionTreeRegressor` on `log(price)`.
It uses only *structural* features: capacity, layout, amenities, location, booking
policy and host attributes. Review counts, review scores, occupancy and estimated
revenue are excluded by design and enforced by an assertion, because the gap between
actual and predicted price is later used as a Hidden Gem criterion — a model that
already knew how popular a listing is would contaminate that criterion.

**Hidden Gem** — a listing that satisfies all three:

1. fewer 2025 reviews than the median,
2. more than 75% of its reviews classified positive by VADER,
3. priced at least 15% below the model's prediction.

**Clustering** (`clustering/build_clusters.py`) — K-means over location plus
property characteristics (`accommodates`, `bedrooms`, `amenities_count`).

**Word clouds** (`build_wordclouds.py`) — adjective+noun phrase frequencies in the
reviews of Hidden Gems versus all other listings.

## Report

`docs/writeup_lastlyupdated08.08.docx` is the submission write-up, and
`docs/writeup_lastlyupdated08.08_no_photos_no_graphs.docx` is the same document
without the figures. Both are edited directly in Word.

`docs/writeup.md` plus `docs/build_docx.py` are the older Markdown-to-Word
pipeline, still in use for drafting; run `python build_docx.py` from `docs/`.
