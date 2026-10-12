import numpy as np
import pandas as pd

from make_figures import AXIS, C1, INK2, RESULTS, plt, save, style


def main():
    d = pd.read_csv(RESULTS / "rq4_ablation.csv")
    t = d[d["kind"] == "text"].set_index("setting")["test_macro_f1"]
    full = t["full text"]
    order = [
        "full text",
        "single words only (no word pairs)",
        "no formatting (words only, lowercase)",
        "first sentence removed",
        "last sentence removed",
        "first sentence and last sentence removed",
        "content words only",
        "first 20 words only",
        "formatting only (words replaced by w)",
        "function words only",
    ]
    labels = {
        "full text": "Full text",
        "single words only (no word pairs)": "Single words only, no word pairs",
        "no formatting (words only, lowercase)": "Formatting removed, words kept",
        "first sentence removed": "First sentence removed",
        "last sentence removed": "Last sentence removed",
        "first sentence and last sentence removed": "First and last sentence removed",
        "content words only": "Content words only",
        "first 20 words only": "First 20 words only",
        "formatting only (words replaced by w)": "Formatting only, words hidden",
        "function words only": "Function words only",
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    style(ax, grid_axis="x")
    y = np.arange(len(order))
    ax.axvline(0.20, color=AXIS, linewidth=1)
    for i, k in enumerate(order):
        ax.barh(i, t[k], height=0.4, color=C1, linewidth=0)
        ax.text(t[k] + 0.012, i, f"{t[k]:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order])
    ax.set_ylim(len(order) - 0.5, -0.8)
    ax.text(0.205, -0.6, "chance, 0.20", fontsize=8, color=INK2, va="center")
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("Macro F1 on the test set")
    save(fig, "fig_rq4_ablation.png")


if __name__ == "__main__":
    main()
