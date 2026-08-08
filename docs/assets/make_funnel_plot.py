import matplotlib.pyplot as plt

# Printed by find_hidden_gems.py.
counts = [22943, 10898, 2766, 564, 453]

# Each stage is labelled as (construct, how it is operationalised). The second
# line matters: "below-median review count" alone would read as lifetime
# reviews, but the criterion counts 2025 reviews only, and "positive rate" is a
# VADER sentiment share rather than a star rating.
stage_rows = [
    ("Evaluable listings", "price, review & sentiment data"),
    ("+ Low exposure", "fewer 2025 reviews than the median"),
    ("+ Well reviewed", ">75% positive review sentiment"),
    ("+ Underpriced", "≥ 15% below predicted price"),
    ("+ Out-of-sample prediction", "in the price model's test set"),
]
# Panel B describes the transitions between stages, so it has one fewer row --
# and no "+" prefix, because each rate stands alone rather than accumulating.
step_rows = [(head.removeprefix("+ "), detail) for head, detail in stage_rows[1:]]

# The step-wise pass rate is the quantity the funnel itself cannot show: the
# cumulative bars are dominated by the first split, which removes ~50% purely
# because it is a median cut.
retention = [counts[i + 1] / counts[i] * 100 for i in range(len(counts) - 1)]

# Burgundy denotes one entity across the whole writeup -- the Hidden Gems --
# so it marks the final bar here and nothing else. Panel B's sharpest criterion
# is called out in words instead: reusing the colour for "look here" would make
# it mean two different things between this figure and the price scatter.
HIGHLIGHT_STAGE = len(counts) - 1
SHARPEST_STEP = 2

BASE_COLOR = "#A6B8C7"
HIGHLIGHT_COLOR = "#800020"
GRID_COLOR = "#C7CDD3"
SPINE_COLOR = "#9AA2AA"
TICK_COLOR = "#5A6169"
LABEL_INK = "#33383D"
MUTED_INK = "#6B7280"

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.5, 7.0), gridspec_kw={"height_ratios": [5, 4]})


def style(ax, rows):
    ax.grid(True, axis="x", which="major", color=GRID_COLOR, linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=TICK_COLOR, labelcolor=LABEL_INK, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()  # a funnel reads top to bottom

    # Drawn by hand rather than as tick labels, so the construct and the way it
    # is measured can carry different weight.
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([""] * len(rows))
    trans = ax.get_yaxis_transform()
    for i, (head, detail) in enumerate(rows):
        ax.text(-0.015, i - 0.17, head, transform=trans, ha="right", va="center",
                fontsize=10, color=LABEL_INK, clip_on=False)
        ax.text(-0.015, i + 0.19, detail, transform=trans, ha="right", va="center",
                fontsize=8.5, color=MUTED_INK, clip_on=False)


# --- Panel A: how many listings survive each filter -------------------------
stage_colors = [HIGHLIGHT_COLOR if i == HIGHLIGHT_STAGE else BASE_COLOR for i in range(len(counts))]
ax_top.barh(range(len(counts)), counts, color=stage_colors, height=0.5)
for i, c in enumerate(counts):
    text = f"{c:,}" if i == 0 else f"{c:,}   ({c / counts[0] * 100:.1f}% of base)"
    ax_top.text(c + counts[0] * 0.012, i, text, va="center", fontsize=10, color=LABEL_INK)
ax_top.text(
    counts[-1] + counts[0] * 0.41, len(counts) - 1, "= Hidden Gems",
    va="center", fontsize=10, fontweight="bold", color=LABEL_INK,
)

ax_top.set_xlim(0, counts[0] * 1.32)
ax_top.set_xticks([0, 5000, 10000, 15000, 20000])
ax_top.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax_top.set_xlabel("Number of listings", fontsize=10, color=LABEL_INK)
ax_top.set_title("A.  Listings remaining after each criterion", fontsize=11.5, fontweight="bold", loc="left", pad=8)
style(ax_top, stage_rows)

# --- Panel B: how selective each filter actually is --------------------------
ax_bot.barh(range(len(retention)), retention, color=BASE_COLOR, height=0.5)

notes = {
    SHARPEST_STEP: "sharpest criterion",
    len(retention) - 1: "≈ the 80% test split",
}
for i, r in enumerate(retention):
    ax_bot.text(r + 1.5, i, f"{r:.1f}%", va="center", fontsize=10, color=LABEL_INK, fontweight="bold")
    if i in notes:
        ax_bot.text(r + 13, i, notes[i], va="center", fontsize=9, color=MUTED_INK, style="italic")

# Ticks stop at 100%, but the limit runs past it so the annotations beside the
# longest bar have room instead of spilling out of the axes.
ax_bot.set_xlim(0, 128)
ax_bot.set_xticks([0, 20, 40, 60, 80, 100])
ax_bot.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax_bot.set_xlabel("Share of the previous stage that passes", fontsize=10, color=LABEL_INK)
ax_bot.set_title(
    "B.  Selectivity of each criterion, conditional on the previous stage",
    fontsize=11.5, fontweight="bold", loc="left", pad=8,
)
style(ax_bot, step_rows)

fig.suptitle(
    "Identifying Hidden Gems in Rome's Airbnb Market",
    fontsize=14, fontweight="bold", x=0.015, ha="left", y=0.98,
)
# Set explicitly rather than via tight_layout: the stage labels are long, and
# tight_layout cannot reserve the left margin they need alongside a suptitle.
fig.subplots_adjust(left=0.30, right=0.985, top=0.885, bottom=0.085, hspace=0.5)
fig.savefig("hidden_gem_funnel.png", dpi=150)
print("Saved hidden_gem_funnel.png")
for (head, _), r in zip(step_rows, retention):
    print(f"  {head:<28} {r:5.1f}%")
