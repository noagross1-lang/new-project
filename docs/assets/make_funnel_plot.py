import matplotlib.pyplot as plt

stages = [
    "Listings evaluated\n(in listings, positive_rate\n& price-model files)",
    "+ Below-median\nreview count",
    "+ Positive rate\n> 75%",
    "+ Price >= 15% below\npredicted price\n= Hidden Gem",
]
# Printed by find_hidden_gems.py.
counts = [22943, 10898, 2766, 562]
color = "#0072B2"
highlight_color = "#E69F00"

fig, ax = plt.subplots(figsize=(8, 5.5))
colors = [color, color, color, highlight_color]
bars = ax.bar(range(len(stages)), counts, color=colors, edgecolor="black", linewidth=0.6, width=0.6)

for i, (bar, c) in enumerate(zip(bars, counts)):
    pct = c / counts[0] * 100
    label = f"{c:,}" if i == 0 else f"{c:,}  ({pct:.1f}%)"
    ax.text(bar.get_x() + bar.get_width() / 2, c + counts[0] * 0.02, label, ha="center", fontsize=12, fontweight="bold")

ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9.5)
ax.set_ylabel("Number of listings")
ax.set_title("Hidden Gem Identification Funnel", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("hidden_gem_funnel.png", dpi=150)
print("Saved hidden_gem_funnel.png")
