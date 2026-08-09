"""Cluster profile heatmap for Problem 3 (k=4 Hidden Gems clustering).

The map this replaces spent its whole canvas on location -- the one dimension
that does NOT separate the clusters -- so three of the four groups landed on top
of each other in central Rome and the figure argued against the finding. This
one encodes each cluster's centroid as a distance from the average Hidden Gem,
which makes the two claims of the section visible instead of merely asserted:

  1. the location columns sit at ~0 for all three mainland clusters and blow out
     only for the coastal one, so location separates exactly one group;
  2. the discount column is near-flat, so the split captures property TYPE
     rather than "how cheap" -- the sanity check that the clustering added
     information the Hidden Gem criterion did not already contain.

The clustering variables and the profiling variables are drawn as two blocks
with a gap, because the distinction matters for the defence: price, discount,
positive rate and review count never entered the algorithm.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENTS = ROOT / "clustering" / "cluster_assignments_k4.csv"
OUTPUT = Path(__file__).resolve().parent / "cluster_profile_heatmap.png"

BASE_COLOR = "#A6B8C7"
HIGHLIGHT_COLOR = "#800020"
GRID_COLOR = "#C7CDD3"
SPINE_COLOR = "#9AA2AA"
LABEL_INK = "#33383D"
MUTED_INK = "#6B7280"
RULE_COLOR = "#B8BEC4"

# The house palette's two poles are wildly asymmetric in lightness (#A6B8C7 is
# pale, #800020 is nearly black), which would make negative deviations read as
# "nothing" next to positive ones of the same magnitude. The cool pole is
# therefore deepened to a dark slate in the same family, so the ramp is balanced
# either side of a neutral midpoint -- diverging needs one warm and one cool
# pole of comparable weight, and a grey (never a hue) in the middle.
COOL_POLE = "#3F6180"
NEUTRAL = "#F4F5F6"
DIVERGING = LinearSegmentedColormap.from_list(
    "hg_diverging", [COOL_POLE, BASE_COLOR, NEUTRAL, "#C08090", HIGHLIGHT_COLOR]
)

# Colour saturates at +/- 1.5 SD. Without a cap the coastal cluster's latitude
# (-3.8 SD) and longitude (-3.3 SD) would compress every other cell to white and
# hide the size/amenity contrast that is the actual finding. Nothing is lost:
# every cell is annotated with its value in original units, so the figure is its
# own table view -- required for a continuous colour scale.
Z_CAP = 1.5

# (column, header, unit formatter). Order is deliberate: the two location
# variables lead, so the eye hits the flat pair first.
CLUSTER_COLUMNS = [
    ("latitude", "Latitude", lambda v: f"{v:.2f}"),
    ("longitude", "Longitude", lambda v: f"{v:.2f}"),
    ("accommodates", "Guests", lambda v: f"{v:.1f}"),
    ("bedrooms", "Bedrooms", lambda v: f"{v:.1f}"),
    ("amenities_count", "Amenities", lambda v: f"{v:.1f}"),
]
PROFILE_COLUMNS = [
    ("price_clean", "Price", lambda v: f"€{v:.0f}"),
    ("discount_pct", "Discount vs.\npredicted", lambda v: f"{v:.1f}%"),
    ("positive_rate", "Positive\nreviews", lambda v: f"{v:.1f}%"),
    ("number_of_reviews", "Review\ncount", lambda v: f"{v:.1f}"),
]

# Rows run compact -> equipped -> large, so the size/amenity block reads as a
# gradient, with the geographic exception held back to last behind a rule.
ROW_ORDER = [
    (1, "Compact & basic"),
    (3, "Compact, well equipped"),
    (2, "Large, for groups"),
    (0, "Quiet coastal gems"),
]
GEOGRAPHIC_ROW = 3  # index in ROW_ORDER, i.e. where the separating rule goes

BLOCK_GAP = 0.6  # in column widths, between the two column blocks


def load_profile():
    df = pd.read_csv(ASSIGNMENTS)
    # Stored as fractions; the figure speaks in percent throughout.
    df["discount_pct"] = df["discount_pct"] * 100
    df["positive_rate"] = df["positive_rate"] * 100

    columns = [c for c, _, _ in CLUSTER_COLUMNS + PROFILE_COLUMNS]
    means = df.groupby("cluster")[columns].mean()
    # Deviation is measured against the Hidden Gems themselves, not against all
    # Rome listings: the question is how the sub-groups differ from a typical
    # Hidden Gem, which is the population the clustering was run on.
    z = (means - df[columns].mean()) / df[columns].std()
    sizes = df["cluster"].value_counts()
    return means, z, sizes, len(df)


def x_position(col_index):
    """Column centre, with the profiling block pushed right by the gap."""
    return col_index + (BLOCK_GAP if col_index >= len(CLUSTER_COLUMNS) else 0)


def draw_cells(ax, means, z):
    columns = CLUSTER_COLUMNS + PROFILE_COLUMNS
    clipped = 0
    for row, (cluster_id, _) in enumerate(ROW_ORDER):
        for col, (name, _, fmt) in enumerate(columns):
            value = means.loc[cluster_id, name]
            z_value = z.loc[cluster_id, name]
            if abs(z_value) > Z_CAP:
                clipped += 1
            shade = np.clip(z_value, -Z_CAP, Z_CAP) / Z_CAP  # -1..1
            x = x_position(col)

            # A 2px-equivalent surface gap between cells rather than a border
            # drawn around each one.
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.47, row - 0.44), 0.94, 0.88,
                    facecolor=DIVERGING((shade + 1) / 2), edgecolor="none",
                )
            )
            # Ink flips to white only where the fill is dark enough to need it;
            # the value never wears the series colour.
            dark = abs(shade) > 0.62
            ax.text(
                x, row, fmt(value), ha="center", va="center", fontsize=10,
                color="white" if dark else LABEL_INK,
                fontweight="bold" if dark else "normal",
            )
    return clipped


def draw_row_labels(ax, sizes):
    for row, (cluster_id, name) in enumerate(ROW_ORDER):
        ax.text(-0.75, row - 0.15, name, ha="right", va="center",
                fontsize=10.5, color=LABEL_INK, fontweight="bold")
        ax.text(-0.75, row + 0.19, f"Cluster {cluster_id}  ·  n = {sizes[cluster_id]:,}",
                ha="right", va="center", fontsize=8.5, color=MUTED_INK)


def draw_column_headers(ax):
    for col, (_, header, _) in enumerate(CLUSTER_COLUMNS + PROFILE_COLUMNS):
        ax.text(x_position(col), -0.62, header, ha="center", va="bottom",
                fontsize=9.5, color=LABEL_INK)


def draw_block_headers(ax):
    """Naming the two blocks is the point: only the left one built the clusters."""
    blocks = [
        (0, len(CLUSTER_COLUMNS) - 1, "CLUSTERING VARIABLES", "these built the clusters"),
        (len(CLUSTER_COLUMNS), len(CLUSTER_COLUMNS) + len(PROFILE_COLUMNS) - 1,
         "PROFILING VARIABLES", "never seen by the algorithm"),
    ]
    for first, last, title, subtitle in blocks:
        left, right = x_position(first) - 0.47, x_position(last) + 0.47
        centre = (left + right) / 2
        ax.plot([left, right], [-1.42, -1.42], color=RULE_COLOR, linewidth=1.0)
        # Spaced out by hand: matplotlib has no letter-spacing, and the small
        # caps need the extra air to read as a section marker rather than a label.
        ax.text(centre, -1.52, " ".join(title), ha="center", va="bottom",
                fontsize=9, color=MUTED_INK, fontweight="bold")
        ax.text(centre, -1.30, subtitle, ha="center", va="top",
                fontsize=8.5, color=MUTED_INK, style="italic")


def draw_callouts(ax):
    """Two brackets marking the cells that carry the section's argument."""
    callouts = [
        # "no separation" would misread against the dark coastal cells directly
        # above the bracket, so the exception is named rather than contradicted.
        (0, 1, "identical, except for the coastal group"),        # latitude, longitude
        (len(CLUSTER_COLUMNS) + 1, len(CLUSTER_COLUMNS) + 1, "flat"),  # discount
    ]
    y = len(ROW_ORDER) - 0.42
    for first, last, label in callouts:
        left, right = x_position(first) - 0.47, x_position(last) + 0.47
        ax.plot([left, left, right, right], [y + 0.10, y + 0.20, y + 0.20, y + 0.10],
                color=RULE_COLOR, linewidth=1.0, clip_on=False)
        # va="top" hangs the label below the bracket: the axis is inverted, so
        # "bottom" would push the text back up over the line it belongs under.
        ax.text((left + right) / 2, y + 0.26, label, ha="center", va="top",
                fontsize=9, color=MUTED_INK, style="italic", clip_on=False)


def draw_geographic_rule(ax):
    """Separates the three property archetypes from the one geographic group."""
    n_cols = len(CLUSTER_COLUMNS) + len(PROFILE_COLUMNS)
    ax.plot(
        [-0.6, x_position(n_cols - 1) + 0.47], [GEOGRAPHIC_ROW - 0.5] * 2,
        color=RULE_COLOR, linewidth=1.0, clip_on=False,
    )


def add_colorbar(fig, ax):
    mappable = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-Z_CAP, Z_CAP))
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.02, pad=0.015, aspect=18)
    cbar.set_ticks([-Z_CAP, 0, Z_CAP])
    cbar.set_ticklabels([f"−{Z_CAP:g} SD", "average\nHidden Gem", f"+{Z_CAP:g} SD"])
    cbar.ax.tick_params(colors=MUTED_INK, labelcolor=LABEL_INK, length=0, labelsize=8.5)
    cbar.outline.set_visible(False)
    return cbar


def main():
    means, z, sizes, n_listings = load_profile()

    fig, ax = plt.subplots(figsize=(13.0, 6.1))

    clipped = draw_cells(ax, means, z)
    draw_row_labels(ax, sizes)
    draw_column_headers(ax)
    draw_block_headers(ax)
    draw_callouts(ax)
    draw_geographic_rule(ax)

    n_cols = len(CLUSTER_COLUMNS) + len(PROFILE_COLUMNS)
    ax.set_xlim(-0.55, x_position(n_cols - 1) + 0.5)
    # Bottom limit leaves room for the callout brackets, which sit below the
    # last row of cells.
    ax.set_ylim(len(ROW_ORDER) + 0.32, -1.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_colorbar(fig, ax)

    fig.suptitle(
        f"What separates the four Hidden Gem clusters (k = 4, n = {n_listings:,})",
        fontsize=14, fontweight="bold", x=0.012, ha="left", y=0.975,
    )
    fig.text(
        0.012, 0.905,
        "Cluster averages, as distance from the average Hidden Gem. Size and amenity level carry the split; "
        "location separates only the coastal group,\nand the discount rate is near-constant across all four — "
        "so the clusters describe property type, not how cheap a listing is.",
        fontsize=9.5, color=MUTED_INK, ha="left", va="top",
    )
    # Set explicitly rather than via tight_layout: the row labels are long and
    # the colourbar's midpoint tick wraps onto two lines, neither of which
    # tight_layout can reserve space for alongside a suptitle.
    fig.subplots_adjust(left=0.175, right=0.905, top=0.80, bottom=0.055)

    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT.name}")
    total = len(ROW_ORDER) * n_cols
    print(f"  cells beyond the ±{Z_CAP:g} SD colour cap: {clipped} of {total} "
          f"({clipped / total * 100:.1f}%) - values still printed in the cells")
    for cluster_id, name in ROW_ORDER:
        lat, lon = z.loc[cluster_id, "latitude"], z.loc[cluster_id, "longitude"]
        print(f"  {name:<24} lat {lat:+.2f} SD, lon {lon:+.2f} SD, "
              f"discount {z.loc[cluster_id, 'discount_pct']:+.2f} SD")


if __name__ == "__main__":
    main()
