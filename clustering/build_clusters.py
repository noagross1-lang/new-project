import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

HIDDEN_GEMS_PATH = "../data/hidden_gems.csv"
LISTINGS_PATH = "../data/listings_cleaned_2025.csv"
PREDICTIONS_PATH = "../data/listings_model_predictions.csv"

# Location plus the property characteristics that describe what a guest actually
# books. The expanded Inside Airbnb export finally makes the last three available.
CLUSTER_VARS = ["latitude", "longitude", "accommodates", "bedrooms", "amenities_count"]

# Columns used only to describe the resulting clusters, never to build them.
PROFILE_VARS = ["room_type", "neighbourhood_cleansed", "number_of_reviews", "property_type"]

K_RANGE = range(2, 7)

# Silhouette falls monotonically across this range (0.31 at k=2 down to 0.22 at
# k=6), which is what happens when the data is one continuous cloud rather than
# well-separated blobs - it rewards merging, not meaningful structure. k=4 is the
# smallest k at which all three property archetypes separate (compact/basic,
# well-equipped mid-size, large group) alongside the coastal Ostia group. k=3
# merges the first two; k=5 and k=6 only split on amenity count without adding
# an interpretable distinction.
FINAL_K = 4

CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


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


def run_kmeans(X_scaled, k, random_state=42):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
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
    sil = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette score (k={k}): {sil:.4f}")

    return df, centroids_df, means_df, sil


def plot_clusters(df, k, centroids_df, output_path):
    """Two views: geography, and the property features that geography cannot show."""
    fig, (ax_geo, ax_prop) = plt.subplots(1, 2, figsize=(16, 7))

    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        ax_geo.scatter(
            sub["longitude"], sub["latitude"], s=45, color=color,
            edgecolor="black", linewidth=0.4, alpha=0.85,
            label=f"Cluster {cluster_id} (n={len(sub)})",
        )
        ax_prop.scatter(
            sub["accommodates"], sub["amenities_count"], s=45, color=color,
            edgecolor="black", linewidth=0.4, alpha=0.85,
            label=f"Cluster {cluster_id} (n={len(sub)})",
        )

    ax_geo.scatter(
        centroids_df["longitude"], centroids_df["latitude"], marker="X", s=260,
        color="black", edgecolor="white", linewidth=1.2, label="Centroids", zorder=5,
    )
    ax_geo.set_xlabel("Longitude")
    ax_geo.set_ylabel("Latitude")
    ax_geo.set_title("Location")
    ax_geo.set_aspect("equal", adjustable="datalim")
    ax_geo.legend(loc="best", frameon=True, fontsize=8)

    ax_prop.scatter(
        centroids_df["accommodates"], centroids_df["amenities_count"], marker="X", s=260,
        color="black", edgecolor="white", linewidth=1.2, label="Centroids", zorder=5,
    )
    ax_prop.set_xlabel("Accommodates (guests)")
    ax_prop.set_ylabel("Number of amenities")
    ax_prop.set_title("Property characteristics")
    ax_prop.legend(loc="best", frameon=True, fontsize=8)

    fig.suptitle(f"Hidden Gems - K-Means Clusters (k={k})", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_silhouette_curve(scores, output_path):
    ks = sorted(scores)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, [scores[k] for k in ks], marker="o", color="#0072B2")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Choosing k for the Hidden Gems clustering", fontweight="bold")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


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


def main():
    df = load_data()
    print(f"Loaded {len(df)} hidden gem listings.")

    df = check_and_handle_missing(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_VARS])

    results = {}
    for k in K_RANGE:
        model, labels = run_kmeans(X_scaled, k)
        labeled_df, centroids_df, means_df, sil = summarize_solution(
            df, labels, scaler, model, X_scaled, k
        )
        plot_clusters(labeled_df, k, centroids_df, f"scatter_k{k}.png")

        labeled_df.to_csv(f"cluster_assignments_k{k}.csv", index=False)
        centroids_df.to_csv(f"cluster_centroids_k{k}.csv")
        means_df.to_csv(f"cluster_means_k{k}.csv")

        results[k] = {"df": labeled_df, "centroids": centroids_df, "silhouette": sil}

    print("\n=== Silhouette comparison ===")
    for k in K_RANGE:
        sizes = results[k]["df"]["cluster"].value_counts().sort_index().tolist()
        print(f"k={k}: silhouette = {results[k]['silhouette']:.4f}   sizes = {sizes}")

    plot_silhouette_curve({k: results[k]["silhouette"] for k in K_RANGE}, "silhouette_by_k.png")

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
