import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer

CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

TO_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def plot_on_map(assignments_path, centroids_path, k, output_path):
    df = pd.read_csv(assignments_path)
    centroids_df = pd.read_csv(centroids_path, index_col=0)

    df["x"], df["y"] = TO_WEB_MERCATOR.transform(df["longitude"].values, df["latitude"].values)
    centroids_df["x"], centroids_df["y"] = TO_WEB_MERCATOR.transform(
        centroids_df["longitude"].values, centroids_df["latitude"].values
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=90,
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolor="black",
            linewidth=0.6,
            label=f"Cluster {cluster_id} (n={len(sub)})",
            alpha=0.9,
            zorder=4,
        )
    ax.scatter(
        centroids_df["x"],
        centroids_df["y"],
        marker="X",
        s=260,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        label="Centroids",
        zorder=5,
    )

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Hidden Gems - K-Means Clusters on Rome map (k={k})", fontsize=15, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    plot_on_map("cluster_assignments_k4.csv", "cluster_centroids_k4.csv", 4, "map_k4.png")
    plot_on_map("cluster_assignments_k3.csv", "cluster_centroids_k3.csv", 3, "map_k3.png")


if __name__ == "__main__":
    main()
