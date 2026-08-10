import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

HIDDEN_GEMS_PATH = "../data/hidden_gems.csv"
LISTINGS_PATH = "../data/listings_cleaned_2025.csv"
PREDICTIONS_PATH = "../data/listings_model_predictions.csv"

# Location plus the property characteristics that describe what a guest actually
# books. The expanded Inside Airbnb export finally makes the last three available.
CLUSTER_VARS = ["latitude", "longitude", "accommodates", "bedrooms", "amenities_count"]

# Columns used only to describe the resulting clusters, never to build them.
PROFILE_VARS = ["room_type", "neighbourhood_cleansed", "number_of_reviews", "property_type"]

# k starts at 1 because the elbow is read against the no-clustering baseline:
# the SSE at k=1 is the total spread of the data, and every later value is the
# share of it that clustering removes. Silhouette could not be computed at k=1
# at all, which is one reason the elbow is the criterion here.
K_RANGE = range(1, 8)

# The elbow itself sits at k=3: the curvature of the SSE curve peaks there
# (second difference 250.6, roughly four times any other point). k=4 is kept
# anyway, and the write-up says so outright rather than implying the elbow
# picked it. The override is cheap and buys the one distinction the section is
# about: the fourth cluster still removes 16.2% of the remaining SSE, which is
# not negligible, and it is what splits the compact listings along amenity
# level (22.1 vs 44.2 amenities). At k=3 that axis disappears -- both mainland
# groups land mid-range (31.6 vs 39.1) and the solution collapses to
# "large vs small vs coastal".
FINAL_K = 4

# External validation. None of these entered the clustering, and each is here
# for a different reason:
#   price     - positive control: an economic axis the algorithm never saw
#   room_type - positive control: an independent property-type classification
#   discount  - NEGATIVE control. The whole claim of the section is that the
#               clusters describe property type rather than cheapness, so
#               agreement here should be no better than chance. A validation
#               that is expected to fail, and does, is worth more than two
#               that succeed.
EXTERNAL_VARS = ["price_tertile", "room_type", "discount_tertile"]
N_BASELINE_SHUFFLES = 1000

# Figure style shared with the rest of the write-up's plots.
BASE_COLOR = "#A6B8C7"
HIGHLIGHT_COLOR = "#800020"
GRID_COLOR = "#C7CDD3"
SPINE_COLOR = "#9AA2AA"
TICK_COLOR = "#5A6169"
LABEL_INK = "#33383D"
MUTED_INK = "#6B7280"
RULE_COLOR = "#B8BEC4"


def load_data():
    hidden_gems = pd.read_csv(HIDDEN_GEMS_PATH, dtype={"listing_id": str})
    listings = pd.read_csv(
        LISTINGS_PATH, dtype={"id": str}, low_memory=False, encoding="utf-8"
    ).rename(columns={"id": "listing_id"})
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"id": str}).rename(columns={"id": "listing_id"})

    df = hidden_gems.merge(
        listings[["listing_id"] + CLUSTER_VARS + PROFILE_VARS],
        on="listing_id",
        how="left",
    )
    df = df.merge(predictions[["listing_id", "residual"]], on="listing_id", how="left")
    return df


def check_and_handle_missing(df):
    print("Missing values in clustering variables:")
    print(df[CLUSTER_VARS].isna().sum())
    n_before = len(df)
    df = df.dropna(subset=CLUSTER_VARS).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with missing clustering variables.")
    else:
        print("No missing values found - nothing dropped.")
    return df


# Every one of these is sklearn's default, but spelling them out is the point:
# the write-up has to be able to state which initialisation was used and when
# the loop was considered converged, and "we relied on the library default" is
# not an answer. K-means++ seeds each new centroid with probability
# proportional to its squared distance from the centroids already chosen, which
# is what keeps the run from being at the mercy of a bad random start.
INIT_METHOD = "k-means++"
N_INIT = 10          # independent restarts; the lowest-SSE one is kept
MAX_ITER = 300       # hard cap on assign/update cycles per restart
TOL = 1e-4           # centroid shift below which the run counts as converged
RANDOM_STATE = 42


def run_kmeans(X_scaled, k, random_state=RANDOM_STATE):
    """One K-means run: seed, then alternate assignment and update to convergence."""
    model = KMeans(
        n_clusters=k,
        init=INIT_METHOD,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


def summarize_solution(df, labels, scaler, model, X_scaled, k):
    df = df.copy()
    df["cluster"] = labels

    print(f"\n--- k={k} cluster sizes ---")
    print(df["cluster"].value_counts().sort_index().to_string())

    centroids_df = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=CLUSTER_VARS
    )
    centroids_df.index.name = "cluster"
    print(f"\n--- k={k} centroids (original scale) ---")
    print(centroids_df.round(2).to_string())

    means_df = df.groupby("cluster")[CLUSTER_VARS].mean()
    # inertia_ is the within-cluster sum of squared distances to the centroid --
    # the SSE the elbow method is read off. n_iter_ is how many assign/update
    # cycles the best restart needed before the centroids stopped moving.
    print(f"\nSSE / inertia (k={k}): {model.inertia_:.2f}")
    print(f"Converged after {model.n_iter_} iterations "
          f"(cap {MAX_ITER}, tol {TOL:g}, init {INIT_METHOD}, {N_INIT} restarts)")

    return df, centroids_df, means_df, model.inertia_


# The elbow (peak curvature of the SSE curve) and the k actually used are not
# the same value, and the figures say so rather than quietly marking only one.
ELBOW_K = 3


def marginal_gains(ks, sse):
    """Share of the previous SSE that each additional cluster removes."""
    return [(sse[i - 1] - sse[i]) / sse[i - 1] * 100 for i in range(1, len(ks))]


def _style_axis(ax):
    ax.grid(True, axis="y", which="major", color=GRID_COLOR, linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TICK_COLOR, labelcolor=LABEL_INK, length=3, width=0.8)


def plot_sse_curve(sse_by_k, output_path):
    """The canonical elbow plot: SSE against k, nothing else on the canvas.

    Two points are called out, because they are not the same point: the elbow
    the method identifies (k=3) and the solution actually taken forward (k=4).
    They are distinguished by fill rather than by hue alone -- hollow for the
    elbow, solid for the chosen k -- so the distinction survives greyscale.
    """
    ks = sorted(sse_by_k)
    sse = [sse_by_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(ks, sse, color=BASE_COLOR, linewidth=2.0, zorder=2)
    ax.scatter(ks, sse, s=60, color=BASE_COLOR, edgecolor="white", linewidth=0.6, zorder=3)

    span = sse[0] - sse[-1]
    ax.scatter([ELBOW_K], [sse_by_k[ELBOW_K]], s=130, facecolor="white",
               edgecolor=HIGHLIGHT_COLOR, linewidth=2.0, zorder=4)
    ax.annotate(f"elbow (k = {ELBOW_K})", xy=(ELBOW_K, sse_by_k[ELBOW_K]),
                xytext=(ELBOW_K - 0.08, sse_by_k[ELBOW_K] + span * 0.30),
                fontsize=10, color=LABEL_INK, ha="center", fontweight="bold")
    ax.scatter([FINAL_K], [sse_by_k[FINAL_K]], s=130, color=HIGHLIGHT_COLOR,
               edgecolor="white", linewidth=0.9, zorder=4)
    ax.annotate(f"chosen (k = {FINAL_K})", xy=(FINAL_K, sse_by_k[FINAL_K]),
                xytext=(FINAL_K + 0.75, sse_by_k[FINAL_K] + span * 0.20),
                fontsize=10, color=LABEL_INK, ha="center", fontweight="bold")

    for k, value in zip(ks, sse):
        ax.annotate(f"{value:,.0f}", (k, value), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=8.5, color=MUTED_INK)

    ax.set_xlabel("Number of clusters (k)", fontsize=10, color=LABEL_INK)
    ax.set_ylabel("SSE (within-cluster sum of squares)", fontsize=10, color=LABEL_INK)
    ax.set_xticks(ks)
    ax.set_ylim(0, sse[0] * 1.10)
    _style_axis(ax)

    fig.suptitle("Choosing k: total within-cluster error", fontsize=13,
                 fontweight="bold", x=0.015, ha="left", y=0.97)
    fig.subplots_adjust(left=0.125, right=0.975, top=0.87, bottom=0.13)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_marginal_gain(sse_by_k, output_path):
    """How much each additional cluster actually buys.

    Reading the elbow off the SSE curve alone is an eyeball judgement, and this
    curve has no dramatic corner. Restating each step as the percentage of SSE
    it removes turns "where does it flatten" into a quotable number, and it is
    the figure the k=4 decision rests on: 16.2% is visibly not yet negligible.
    """
    ks = sorted(sse_by_k)
    sse = [sse_by_k[k] for k in ks]
    gains = marginal_gains(ks, sse)
    steps = ks[1:]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colors = [HIGHLIGHT_COLOR if k in (ELBOW_K, FINAL_K) else BASE_COLOR for k in steps]
    ax.bar(steps, gains, color=colors, width=0.5)
    for k, gain in zip(steps, gains):
        ax.text(k, gain + 0.8, f"{gain:.1f}%", ha="center", fontsize=9.5,
                color=LABEL_INK, fontweight="bold")

    # The drop the argument turns on: the last large gain, and the first small one.
    ax.annotate("last large gain", xy=(ELBOW_K, gains[ELBOW_K - 2] / 2),
                fontsize=9, color="white", ha="center", va="center", rotation=90)
    ax.annotate("still not negligible", xy=(FINAL_K, gains[FINAL_K - 2] / 2),
                fontsize=9, color="white", ha="center", va="center", rotation=90)

    ax.set_xlabel("Number of clusters (k)", fontsize=10, color=LABEL_INK)
    ax.set_ylabel("SSE removed vs. previous k", fontsize=10, color=LABEL_INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(steps)
    ax.set_ylim(0, max(gains) * 1.22)
    _style_axis(ax)

    fig.suptitle("Choosing k: gain from each additional cluster", fontsize=13,
                 fontweight="bold", x=0.015, ha="left", y=0.97)
    fig.subplots_adjust(left=0.125, right=0.975, top=0.87, bottom=0.14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_elbow_table(sse_by_k):
    """Console only -- the write-up carries these numbers as figures and prose."""
    ks = sorted(sse_by_k)
    sse = [sse_by_k[k] for k in ks]
    gains = marginal_gains(ks, sse)
    print("\n=== Elbow method ===")
    print(f"{'k':>3}  {'SSE':>10}  {'gain vs k-1':>12}  {'% of k=1 SSE removed':>22}")
    for i, k in enumerate(ks):
        gain = f"{gains[i - 1]:.1f}%" if i else "-"
        removed = (sse[0] - sse[i]) / sse[0] * 100
        print(f"{k:>3}  {sse[i]:>10,.1f}  {gain:>12}  {removed:>21.1f}%")


def profile_final_clusters(df):
    """Characterize clusters using variables NOT used for clustering."""
    profile_rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        profile_rows.append(
            {
                "cluster": cluster_id,
                "n_listings": len(sub),
                "mean_latitude": sub["latitude"].mean(),
                "mean_longitude": sub["longitude"].mean(),
                "mean_accommodates": sub["accommodates"].mean(),
                "mean_bedrooms": sub["bedrooms"].mean(),
                "mean_amenities_count": sub["amenities_count"].mean(),
                "mean_price_clean": sub["price_clean"].mean(),
                "mean_predicted_price": sub["predicted_price"].mean(),
                "mean_residual_price": sub["residual"].mean(),
                "mean_discount_pct": sub["discount_pct"].mean() * 100,
                "mean_positive_rate_pct": sub["positive_rate"].mean() * 100,
                "mean_number_of_reviews": sub["number_of_reviews"].mean(),
                "room_type_distribution_pct": (
                    sub["room_type"].value_counts(normalize=True) * 100
                ).round(1).to_dict(),
                "top_neighbourhoods": sub["neighbourhood_cleansed"].value_counts().head(3).to_dict(),
            }
        )
    return pd.DataFrame(profile_rows)


def purity(labels, classes):
    """Share of listings sitting in their cluster's majority class."""
    table = pd.crosstab(labels, classes)
    return table.max(axis=1).sum() / table.to_numpy().sum()


def normalized_entropy(labels, classes):
    """Size-weighted entropy of the class mix inside each cluster, scaled to 0-1.

    0 means every cluster is pure, 1 means every cluster mirrors the overall
    class distribution. Dividing by log(number of classes) is what makes the
    three external variables comparable to each other.
    """
    table = pd.crosstab(labels, classes)
    counts = table.to_numpy()
    sizes = counts.sum(axis=1, keepdims=True)
    proportions = counts / sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(proportions > 0, -proportions * np.log(proportions), 0.0)
    per_cluster = terms.sum(axis=1)
    weighted = (per_cluster * sizes.ravel()).sum() / counts.sum()
    return weighted / np.log(counts.shape[1])


def shuffled_baseline(labels, classes, n_shuffles=N_BASELINE_SHUFFLES, seed=RANDOM_STATE):
    """What purity and entropy a meaningless clustering of the same shape scores.

    Purity is inflated by imbalance -- 77% of Hidden Gems are entire homes, so
    a room_type purity near 0.77 is what random assignment already achieves.
    Permuting the labels keeps the cluster sizes and the class distribution
    exactly as they are and destroys only the association between them, which
    is precisely the null the reported numbers need to be read against.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(labels)
    purities, entropies = [], []
    for _ in range(n_shuffles):
        permuted = pd.Series(rng.permutation(values), index=classes.index)
        purities.append(purity(permuted, classes))
        entropies.append(normalized_entropy(permuted, classes))
    return float(np.mean(purities)), float(np.mean(entropies))


def add_external_variables(df):
    """Bin the two continuous external variables into balanced tertiles.

    Tertiles rather than fixed cut-offs so the null purity is a flat 1/3 and
    the imbalance problem that afflicts room_type does not arise here too.
    """
    df = df.copy()
    labels = ["low", "mid", "high"]
    df["price_tertile"] = pd.qcut(df["price_clean"], 3, labels=labels)
    df["discount_tertile"] = pd.qcut(df["discount_pct"], 3, labels=labels)
    return df


def evaluate_externally(df, k):
    """Purity and entropy of the chosen partition against variables it never saw."""
    print(f"\n=== External validation of the k={k} solution ===")
    rows = []
    for var in EXTERNAL_VARS:
        classes = df[var]
        observed_purity = purity(df["cluster"], classes)
        observed_entropy = normalized_entropy(df["cluster"], classes)
        base_purity, base_entropy = shuffled_baseline(df["cluster"], classes)
        rows.append(
            {
                "external_variable": var,
                "n_classes": classes.nunique(),
                "purity": observed_purity,
                "purity_random_baseline": base_purity,
                "purity_lift": observed_purity - base_purity,
                "normalized_entropy": observed_entropy,
                "entropy_random_baseline": base_entropy,
            }
        )

    result = pd.DataFrame(rows)
    print(
        result.rename(
            columns={
                "external_variable": "variable",
                "purity_random_baseline": "purity(random)",
                "normalized_entropy": "entropy",
                "entropy_random_baseline": "entropy(random)",
            }
        ).round(3).to_string(index=False)
    )
    print(f"\n(baselines averaged over {N_BASELINE_SHUFFLES:,} label permutations "
          f"holding cluster sizes fixed)")

    for var in EXTERNAL_VARS:
        print(f"\n--- cluster x {var} ---")
        print(pd.crosstab(df["cluster"], df[var]).to_string())

    return result


def main():
    df = load_data()
    print(f"Loaded {len(df)} hidden gem listings.")

    df = check_and_handle_missing(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_VARS])

    results = {}
    for k in K_RANGE:
        model, labels = run_kmeans(X_scaled, k)
        labeled_df, centroids_df, means_df, sse = summarize_solution(
            df, labels, scaler, model, X_scaled, k
        )
        # k=1 is run for its SSE alone -- it is the elbow's baseline. Its
        # "clusters" are the whole dataset, so per-listing assignments would be
        # noise on disk.
        if k > 1:
            labeled_df.to_csv(f"cluster_assignments_k{k}.csv", index=False)
            centroids_df.to_csv(f"cluster_centroids_k{k}.csv")
            means_df.to_csv(f"cluster_means_k{k}.csv")

        results[k] = {"df": labeled_df, "centroids": centroids_df, "sse": sse}

    print("\n=== Cluster sizes by k ===")
    for k in K_RANGE:
        sizes = results[k]["df"]["cluster"].value_counts().sort_index().tolist()
        print(f"k={k}: SSE = {results[k]['sse']:9.2f}   sizes = {sizes}")

    # Two separate figures rather than a two-panel one: each carries its own
    # paragraph in the write-up, and the SSE curve on its own is the form the
    # elbow method is normally presented in.
    sse_by_k = {k: results[k]["sse"] for k in K_RANGE}
    plot_sse_curve(sse_by_k, "elbow_sse.png")
    plot_marginal_gain(sse_by_k, "elbow_marginal_gain.png")
    print_elbow_table(sse_by_k)

    final_df = add_external_variables(results[FINAL_K]["df"])
    external = evaluate_externally(final_df, FINAL_K)
    external.to_csv(f"external_validation_k{FINAL_K}.csv", index=False)

    profile = profile_final_clusters(results[FINAL_K]["df"])
    profile.to_csv(f"final_cluster_profile_k{FINAL_K}.csv", index=False)
    print(f"\n=== Final chosen solution: k={FINAL_K} ===")
    print(profile.drop(columns=["room_type_distribution_pct", "top_neighbourhoods"]).round(2).to_string(index=False))
    for row in profile.itertuples():
        print(f"\ncluster {row.cluster} (n={row.n_listings})")
        print(f"  room types    : {row.room_type_distribution_pct}")
        print(f"  neighbourhoods: {row.top_neighbourhoods}")

    return results


if __name__ == "__main__":
    main()
