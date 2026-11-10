import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_results(results: dict):
    """
    Plot a styled bar chart comparing model accuracies.
    results = {"KNN": 85.0, "Decision Tree": 90.0, "Naive Bayes": 88.5}
    """
    names  = list(results.keys())
    values = list(results.values())

    colors = ["#2beac4", "#c2da24", "#6211ed"]
    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0d1829")

    bars = ax.bar(names, values, color=colors, width=0.5, zorder=3)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 110)
    ax.set_title("Model Accuracy Comparison — SmartClinic AI",
                 color="white", fontsize=14, pad=15)
    ax.set_xlabel("Models", color="#5d8ab0", fontsize=11)
    ax.set_ylabel("Accuracy (%)", color="#5d8ab0", fontsize=11)

    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#1e3a5f")
    ax.spines["left"].set_color("#1e3a5f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1e3a5f", linestyle="--", alpha=0.7, zorder=0)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0f1e")
    plt.show()
    print("✅ Plot saved as model_comparison.png")
