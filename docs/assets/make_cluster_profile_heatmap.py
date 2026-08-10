"""Cluster profile heatmap for Problem 3 (k=4 Hidden Gems clustering).

This figure REPLACES the results table in the write-up rather than illustrating
it, which drives two of its design choices.

Orientation: variables run down the rows and clusters across the columns,
matching the table it replaces. The earlier landscape version was 13 inches
wide, which on an A4 page in a Hebrew document shrank by half and left the cell
values at roughly 5pt. Portrait keeps the type legible at print size, and it
also suits the right-to-left reading of the surrounding text: the argument now
reads along rows ("the latitude and longitude rows are pale, the size and
amenity rows are not") with the variable names anchored on the right-hand side.

Column order therefore runs coastal-first left to right, so that read from the
right it goes compact -> well equipped -> large -> coastal, the same order the
document's prose uses.

The map this ultimately replaces spent its whole canvas on location -- the one
dimension that does NOT separate the clusters -- so three of the four groups
landed on top of each other in central Rome and the figure argued against the
finding. This one encodes each cluster's centroid as a distance from the
average Hidden Gem, which makes the two claims of the section visible instead
of merely asserted:

  1. the location rows sit at ~0 for all three mainland clusters and blow out
     only for the coastal one, so location separates exactly one group;
  2. the discount row is near-flat, so the split captures property TYPE rather
     than "how cheap" -- the sanity check that the clustering added information
     the Hidden Gem criterion did not already contain.

The clustering variables and the profiling variables are drawn as two blocks
with a gap, because the distinction matters for the defence: price, predicted
price, discount, positive rate and review count never entered the algorithm.
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
# own table view -- required for a continuous colour scale, and doubly so here
# because this figure IS the table.
Z_CAP = 1.5

# (column, row label, unit formatter). The two location variables lead, so the
# eye hits the flat pair first.
CLUSTER_ROWS = [
    ("latitude", "Latitude", lambda v: f"{v:.2f}"),
    ("longitude", "Longitude", lambda v: f"{v:.2f}"),
    ("accommodates", "Guests", lambda v: f"{v:.1f}"),
    ("bedrooms", "Bedrooms", lambda v: f"{v:.1f}"),
    ("amenities_count", "Amenities", lambda v: f"{v:.1f}"),
]
PROFILE_ROWS = [
    ("price_clean", "Price per night", lambda v: f"€{v:.0f}"),
    ("predicted_price", "Predicted price", lambda v: f"€{v:.0f}"),
    ("discount_pct", "Discount vs. predicted", lambda v: f"{v:.1f}%"),
    ("positive_rate", "Positive reviews", lambda v: f"{v:.1f}%"),
    ("number_of_reviews", "Review count", lambda v: f"{v:.1f}"),
]

# Left to right, so that read right-to-left it runs compact -> equipped ->
# large -> coastal. The cluster ids are the labels K-means happened to emit and
# carry no meaning of their own; they are printed only so a reader can trace a
# column back to cluster_assignments_k4.csv.
COLUMN_ORDER = [
    (0, "Quiet coastal\ngems"),
    (2, "Large,\nfor groups"),
    (1, "Compact,\nwell equipped"),
    (3, "Compact\n& basic"),
]
GEOGRAPHIC_COL = 1  # rule goes to the left of this index, isolating the coastal column

BLOCK_GAP = 0.55  # in row heights, between the two row blocks
LABEL_X = 3.62    # right edge of the grid, where the row labels start


def load_profile():
    df = pd.read_csv(ASSIGNMENTS)
    # Stored as fractions; the figure speaks in percent throughout.
    df["discount_pct"] = df["discount_pct"] * 100
    df["positive_rate"] = df["positive_rate"] * 100

    columns = [c for c, _, _ in CLUSTER_ROWS + PROFILE_ROWS]
    means = df.groupby("cluster")[columns].mean()
    # Deviation is measured against the Hidden Gems themselves, not against all
    # Rome listings: the question is how the sub-groups differ from a typical
    # Hidden Gem, which is the population the clustering was run on.
    z = (means - df[columns].mean()) / df[columns].std()
    sizes = df["cluster"].value_counts()
    return means, z, sizes, len(df)


def y_position(row_index):
    """Row centre, with the profiling block pushed down by the gap."""
    return row_index + (BLOCK_GAP if row_index >= len(CLUSTER_ROWS) else 0)


def draw_cells(ax, means, z):
    rows = CLUSTER_ROWS + PROFILE_ROWS
    clipped = 0
    for row, (name, _, fmt) in enumerate(rows):
        y = y_position(row)
        for col, (cluster_id, _) in enumerate(COLUMN_ORDER):
            value = means.loc[cluster_id, name]
            z_value = z.loc[cluster_id, name]
            if abs(z_value) > Z_CAP:
                clipped += 1
            shade = np.clip(z_value, -Z_CAP, Z_CAP) / Z_CAP  # -1..1

            # A surface gap between cells rather than a border drawn round each.
            ax.add_patch(
                plt.Rectangle(
                    (col - 0.46, y - 0.42), 0.92, 0.84,
                    facecolor=DIVERGING((shade + 1) / 2), edgecolor="none",
                )
            )
            # Ink flips to white only where the fill is dark enough to need it;
            # the value never wears the series colour.
            dark = abs(shade) > 0.62
            ax.text(
                col, y, fmt(value), ha="center", va="center", fontsize=11,
                color="white" if dark else LABEL_INK,
                fontweight="bold" if dark else "normal",
            )
    return clipped


def draw_row_labels(ax):
    """Anchored on the right-hand side, where a Hebrew reader starts the row."""
    for row, (_, label, _) in enumerate(CLUSTER_ROWS + PROFILE_ROWS):
        ax.text(LABEL_X, y_position(row), label, ha="left", va="center",
                fontsize=11, color=LABEL_INK)


def draw_block_labels(ax):
    """Naming the two blocks is the point: only the first built the clusters."""
    blocks = [
        (0, "THESE BUILT THE CLUSTERS"),
        (len(CLUSTER_ROWS), "NEVER SEEN BY THE ALGORITHM"),
    ]
    for first_row, title in blocks:
        y = y_position(first_row) - 0.62
        ax.text(LABEL_X, y, title, ha="left", va="center",
                fontsize=8.5, color=MUTED_INK, fontweight="bold")
        ax.plot([-0.46, 3.46], [y + 0.20] * 2, color=RULE_COLOR,
                linewidth=1.0, clip_on=False)


def draw_column_headers(ax, sizes):
    for col, (cluster_id, label) in enumerate(COLUMN_ORDER):
        ax.text(col, -1.30, label, ha="center", va="bottom",
                fontsize=10.5, color=LABEL_INK, fontweight="bold")
        # Two lines: on one line these run into each other at column width.
        ax.text(col, -1.20, f"Cluster {cluster_id}\nn = {sizes[cluster_id]:,}",
                ha="center", va="top", fontsize=8.5, color=MUTED_INK,
                linespacing=1.4)


def draw_geographic_rule(ax):
    """Separates the one geographically defined group from the property types."""
    x = GEOGRAPHIC_COL - 0.5
    bottom = y_position(len(CLUSTER_ROWS) + len(PROFILE_ROWS) - 1) + 0.42
    ax.plot([x, x], [-1.35, bottom], color=RULE_COLOR, linewidth=1.0, clip_on=False)


def draw_callouts(ax):
    """Brackets marking the rows that carry the section's argument."""
    callouts = [
        # "no separation" would misread against the dark coastal cells at the
        # end of these rows, so the exception is named rather than contradicted.
        (0, 1, "identical, except for the coastal group"),
        (len(CLUSTER_ROWS) + 2, len(CLUSTER_ROWS) + 2, "flat"),
    ]
    for first, last, label in callouts:
        top, bottom = y_position(first) - 0.42, y_position(last) + 0.42
        x = -0.60
        ax.plot([x + 0.10, x, x, x + 0.10], [top, top, bottom, bottom],
                color=RULE_COLOR, linewidth=1.0, clip_on=False)
        ax.text(x - 0.10, (top + bottom) / 2, label, ha="center", va="center",
                fontsize=9, color=MUTED_INK, style="italic", rotation=90,
                clip_on=False)


def draw_key(fig):
    """A legend that explains BOTH encodings, in words rather than statistics.

    Every cell holds two channels -- a value in its own units and a deviation
    shown as colour -- so the key has to name both, and it says "typical Hidden
    Gem" rather than "0 SD" because that is the comparison actually being made.
    """
    left, width = 0.085, 0.535
    fig.text(left, 0.088, "How to read a cell", fontsize=10.5,
             fontweight="bold", color=LABEL_INK, ha="left", va="top")
    fig.text(left, 0.062,
             "Number = the cluster's average, in its own units.\n"
             "Colour = how far that sits from a typical Hidden Gem:",
             fontsize=9.5, color=MUTED_INK, ha="left", va="top")

    # A gradient strip rather than a colourbar, so it carries no numeric ticks
    # to be misread as the cell values.
    bar = fig.add_axes([left + width + 0.045, 0.043, 0.285, 0.019])
    bar.imshow(np.linspace(-1, 1, 256).reshape(1, -1), aspect="auto",
               cmap=DIVERGING, vmin=-1, vmax=1)
    bar.set_xticks([])
    bar.set_yticks([])
    for spine in bar.spines.values():
        spine.set_visible(False)
    for x, label, align in [
        (left + width + 0.045, "much lower", "left"),
        (left + width + 0.188, "typical", "center"),
        (left + width + 0.330, "much higher", "right"),
    ]:
        fig.text(x, 0.036, label, fontsize=9, color=LABEL_INK, ha=align, va="top")


def main():
    means, z, sizes, n_listings = load_profile()

    fig, ax = plt.subplots(figsize=(7.8, 8.9))

    clipped = draw_cells(ax, means, z)
    draw_row_labels(ax)
    draw_block_labels(ax)
    draw_column_headers(ax, sizes)
    draw_geographic_rule(ax)
    draw_callouts(ax)

    n_rows = len(CLUSTER_ROWS) + len(PROFILE_ROWS)
    # Right limit leaves room for the longest row label and the block headings,
    # which sit outside the grid rather than as tick labels.
    ax.set_xlim(-0.95, 5.85)
    ax.set_ylim(y_position(n_rows - 1) + 0.55, -1.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_key(fig)

    fig.suptitle(
        f"What separates the four Hidden Gem clusters (k = 4, n = {n_listings:,})",
        fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.982,
    )
    fig.text(
        0.012, 0.955,
        "Size and amenity level carry the split; location separates only the coastal group.",
        fontsize=9.5, color=MUTED_INK, ha="left", va="top",
    )
    # Set explicitly rather than via tight_layout: the row labels sit outside
    # the axes on the right and the key is placed in figure coordinates,
    # neither of which tight_layout can reserve space for.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.915, bottom=0.115)

    fig.savefig(OUTPUT, dpi=200)
    print(f"Saved {OUTPUT.name}")
    total = n_rows * len(COLUMN_ORDER)
    print(f"  cells beyond the ±{Z_CAP:g} SD colour cap: {clipped} of {total} "
          f"({clipped / total * 100:.1f}%) - values still printed in the cells")
    for cluster_id, name in COLUMN_ORDER:
        label = name.replace("\n", " ")
        print(f"  {label:<24} lat {z.loc[cluster_id, 'latitude']:+.2f} SD, "
              f"lon {z.loc[cluster_id, 'longitude']:+.2f} SD, "
              f"discount {z.loc[cluster_id, 'discount_pct']:+.2f} SD")


if __name__ == "__main__":
    main()
