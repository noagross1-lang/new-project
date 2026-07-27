import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

HIDDEN_GEMS_PATH = "../data/hidden_gems.csv"
LISTINGS_PATH = "../data/listings.csv"
PREDICTIONS_PATH = "../data/listings_model_predictions.csv"

# Requested clustering variables were latitude, longitude, accommodates,
# amenities_count. accommodates/amenities_count do not exist in any available
# data file (confirmed: neither data/listings.csv nor the raw/cleaned variants
# contain these fields) - proceeding with the two that do exist, per user
# instruction.
CLUSTER_VARS = ["latitude", "longitude"]

CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def load_data():
    hidden_gems = pd.read_csv(HIDDEN_GEMS_PATH)
    listings = pd.read_csv(LISTINGS_PATH).rename(columns={"id": "listing_id"})
    predictions = pd.read_csv(PREDICTIONS_PATH).rename(columns={"id": "listing_id"})

    df = hidden_gems.merge(
        listings[["listing_id", "latitude", "longitude", "room_type", "neighbourhood", "number_of_reviews"]],
        on="listing_id",
        how="left",
    )
    df = df.merge(predictions[["listing_id", "residual"]], on="listing_id", how="left")
    return df


def check_and_handle_missing(df):
    print("Missing values in clustering variables:")
    missing = df[CLUSTER_VARS].isna().sum()
    print(missing)
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


def summarize_solution(df, labels, scaler, model, k):
    df = df.copy()
    df["cluster"] = labels

    sizes = df["cluster"].value_counts().sort_index()
    print(f"\n--- k={k} cluster sizes ---")
    print(sizes)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=CLUSTER_VARS)
    centroids_df.index.name = "cluster"
    print(f"\n--- k={k} centroids (original scale) ---")
    print(centroids_df)

    means_df = df.groupby("cluster")[CLUSTER_VARS].mean()
    print(f"\n--- k={k} per-cluster means of clustering variables ---")
    print(means_df)

    sil = silhouette_score(scaler.transform(df[CLUSTER_VARS]), labels)
    print(f"\nSilhouette score (k={k}): {sil:.4f}")

    return df, centroids_df, means_df, sil


def plot_clusters(df, k, centroids_df, output_path):
    plt.figure(figsize=(9, 8))
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        plt.scatter(
            sub["longitude"],
            sub["latitude"],
            s=90,
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolor="black",
            linewidth=0.6,
            label=f"Cluster {cluster_id} (n={len(sub)})",
            alpha=0.9,
        )
    plt.scatter(
        centroids_df["longitude"],
        centroids_df["latitude"],
        marker="X",
        s=260,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        label="Centroids",
        zorder=5,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Hidden Gems - K-Means Clusters (k={k})", fontsize=15, fontweight="bold")
    plt.legend(loc="best", frameon=True, fontsize=9)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def profile_final_clusters(df, k):
    """Characterize clusters using variables NOT used for clustering."""
    profile_rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        room_type_dist = (sub["room_type"].value_counts(normalize=True) * 100).round(1).to_dict()
        top_neighbourhoods = sub["neighbourhood"].value_counts().head(3).to_dict()
        profile_rows.append(
            {
                "cluster": cluster_id,
                "n_listings": len(sub),
                "mean_latitude": sub["latitude"].mean(),
                "mean_longitude": sub["longitude"].mean(),
                "mean_price_clean": sub["price_clean"].mean(),
                "mean_predicted_price": sub["predicted_price"].mean(),
                "mean_residual_price": sub["residual"].mean(),
                "mean_discount_pct": sub["discount_pct"].mean() * 100,
                "mean_positive_rate_pct": sub["positive_rate"].mean() * 100,
                "mean_number_of_reviews": sub["number_of_reviews"].mean(),
                "room_type_distribution_pct": room_type_dist,
                "top_neighbourhoods": top_neighbourhoods,
            }
        )
    profile_df = pd.DataFrame(profile_rows)
    return profile_df


def main():
    df = load_data()
    print(f"Loaded {len(df)} hidden gem listings.")

    df = check_and_handle_missing(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_VARS])

    results = {}
    for k in (3, 4):
        model, labels = run_kmeans(X_scaled, k)
        labeled_df, centroids_df, means_df, sil = summarize_solution(df, labels, scaler, model, k)
        plot_clusters(labeled_df, k, centroids_df, f"scatter_k{k}.png")

        labeled_df.to_csv(f"cluster_assignments_k{k}.csv", index=False)
        centroids_df.to_csv(f"cluster_centroids_k{k}.csv")
        means_df.to_csv(f"cluster_means_k{k}.csv")

        results[k] = {
            "df": labeled_df,
            "centroids": centroids_df,
            "means": means_df,
            "silhouette": sil,
        }

    print("\n=== Silhouette comparison ===")
    for k in (3, 4):
        print(f"k={k}: silhouette = {results[k]['silhouette']:.4f}")

    # k=4 is chosen over k=3: k=3 collapses 60/63 listings (95%) into one
    # undifferentiated cluster, leaving only outlier detection as signal.
    # k=4 splits out a genuine, geographically coherent secondary group of 7
    # listings (San Giovanni/Cinecitta + Appia Antica), which is materially
    # more useful for interpretation even though its silhouette score is lower
    # (silhouette rewards tight outlier clusters, not useful group structure).
    final_k = 4
    final_df = results[final_k]["df"]
    profile = profile_final_clusters(final_df, final_k)
    profile.to_csv("final_cluster_profile_k4.csv", index=False)
    print(f"\n=== Final chosen solution: k={final_k} ===")
    print(profile.drop(columns=["room_type_distribution_pct", "top_neighbourhoods"]).round(2))

    return results


if __name__ == "__main__":
    main()
