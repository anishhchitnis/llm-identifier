from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullLocator

RESULTS = Path("results")
FIGURES = Path("figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
FAMILIES = ["bard", "falcon", "gpt", "llama2", "mpt"]
NAMES = ["Bard", "Falcon", "GPT", "Llama 2", "MPT"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "text.color": INK2,
        "axes.labelcolor": INK2,
    }
)


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, labelsize=8.5)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_truncation():
    table = pd.read_csv(RESULTS / "summary_table.csv")
    table = table[table["mode"] == "output"]
    base = pd.read_csv(RESULTS / "baselines.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    style(ax)
    ax.set_xscale("log", base=2)
    ax.axhline(0.20, color=AXIS, linewidth=1)
    ax.axhline(base.loc["tfidf_word_output", "test_macro_f1"], color=AXIS, linewidth=1)
    ax.text(14, 0.208, "chance, 0.20", color=INK2, fontsize=8, va="bottom")
    ax.text(14, 0.78, "TF IDF words, 0.773", color=INK2, fontsize=8, va="bottom")
    for model, color, name, offset in [("cnn", C1, "CNN", 7), ("lstm", C2, "BiLSTM", -11)]:
        d = table[table["model"] == model].sort_values("max_len")
        x, y, s = d["max_len"].values, d["f1_mean"].values, d["f1_std"].values
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name,
        )
        ax.annotate(
            f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(9, 0), textcoords="offset points", va="center",
            fontsize=8, color=INK2,
        )
    ax.set_xticks([16, 32, 64, 128, 256, 384])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "384"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(13, 520)
    ax.set_ylim(0.1, 0.86)
    ax.set_xlabel("Tokens kept from each answer (log scale)")
    ax.set_ylabel("Macro F1 on the test set")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    save(fig, "fig_truncation.png")


def fig_models():
    base = pd.read_csv(RESULTS / "baselines.csv").set_index("model")
    s = pd.read_csv(RESULTS / "summary_table.csv").set_index("run")
    items = [
        ("TF IDF words + logistic regression", base.loc["tfidf_word_output", "test_macro_f1"], 0.0),
        ("TF IDF characters + logistic regression", base.loc["tfidf_char_output", "test_macro_f1"], 0.0),
        ("CNN (3 seeds)", s.loc["cnn_output", "f1_mean"], s.loc["cnn_output", "f1_std"]),
        ("BiLSTM (3 seeds)", s.loc["lstm_output", "f1_mean"], s.loc["lstm_output", "f1_std"]),
        ("Majority class", base.loc["majority_class", "test_macro_f1"], 0.0),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    style(ax, grid_axis="x")
    y = np.arange(len(items))
    ax.axvline(0.20, color=AXIS, linewidth=1)
    for i, (name, value, std) in enumerate(items):
        ax.barh(i, value, height=0.36, color=C1, linewidth=0)
        if std > 0:
            ax.plot([value - std, value + std], [i, i], color=INK2, linewidth=1.2)
        ax.text(value + std + 0.012, i, f"{value:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.text(0.205, -0.6, "chance, 0.20", fontsize=8, color=INK2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items])
    ax.set_ylim(len(items) - 0.5, -0.8)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("Macro F1 on the test set (thin line shows plus or minus one standard deviation)")
    save(fig, "fig_models.png")


def fig_recall():
    s = pd.read_csv(RESULTS / "summary_table.csv").set_index("run")
    char = pd.read_csv(RESULTS / "baseline_char_report.csv").set_index("family")
    series = [
        ("CNN", C1, [s.loc["cnn_output", f"recall_{f}"] for f in FAMILIES]),
        ("BiLSTM", C2, [s.loc["lstm_output", f"recall_{f}"] for f in FAMILIES]),
        ("TF IDF characters", C3, [char.loc[f, "recall"] for f in FAMILIES]),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    style(ax)
    x = np.arange(len(FAMILIES))
    w = 0.25
    for i, (name, color, values) in enumerate(series):
        ax.bar(x + (i - 1) * w, values, w - 0.04, color=color, linewidth=0, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall on the test set")
    ax.set_xlabel("True family")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2, ncol=3)
    save(fig, "fig_recall.png")


def fig_confusion():
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    mats = [
        ("CNN", pd.read_csv(RESULTS / "cnn_output_confusion.csv", index_col=0).values.astype(float)),
        ("BiLSTM", pd.read_csv(RESULTS / "lstm_output_confusion.csv", index_col=0).values.astype(float)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for ax, (title, m) in zip(axes, mats):
        pct = m / m.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                v = pct[i, j]
                ax.text(
                    j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 45 else INK,
                )
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(NAMES, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(NAMES if ax is axes[0] else [], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0, labelcolor=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Predicted family")
    axes[0].set_ylabel("True family")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Percent of the true family", color=INK2, fontsize=8.5)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=2)
    cbar.outline.set_visible(False)
    save(fig, "fig_confusion.png")


def fig_rq2():
    s = pd.read_csv(RESULTS / "summary_table.csv").set_index("run")
    groups = ["Output only", "Input only", "Input + output"]
    runs = {"CNN": ["cnn_output", "cnn_input", "cnn_both"], "BiLSTM": ["lstm_output", "lstm_input", "lstm_both"]}
    colors = {"CNN": C1, "BiLSTM": C2}
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    style(ax)
    ax.axhline(0.20, color=AXIS, linewidth=1)
    x = np.arange(len(groups))
    w = 0.3
    for i, (name, keys) in enumerate(runs.items()):
        means = [s.loc[k, "f1_mean"] for k in keys]
        stds = [s.loc[k, "f1_std"] for k in keys]
        pos = x + (i - 0.5) * w
        ax.bar(pos, means, w - 0.04, color=colors[name], linewidth=0, label=name)
        for p, m, sd in zip(pos, means, stds):
            ax.plot([p, p], [m - sd, m + sd], color=INK2, linewidth=1.2)
            ax.text(p, m + sd + 0.015, f"{m:.3f}", ha="center", fontsize=8, color=INK2)
    ax.text(-0.78, 0.208, "chance, 0.20", ha="left", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.8, 2.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 on the test set")
    ax.legend(frameon=False, loc="upper center", fontsize=9, labelcolor=INK2, ncol=2)
    save(fig, "fig_rq2.png")


def fig_bard_flan():
    q = pd.read_csv(RESULTS / "flan_prompt_length_quartiles.csv")
    labels = ["5 to 55 words", "56 to 114 words", "115 to 216 words", "217 to 1134 words"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    style(ax)
    overall = 701 / 4701
    ax.axhline(overall, color=AXIS, linewidth=1)
    ax.bar(range(4), q["bard"].values * 100, 0.42, color=C1, linewidth=0)
    for i, v in enumerate(q["bard"].values * 100):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    ax.text(1.5, overall * 100 + 0.8, "Bard share of all Flan rows, 15%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 32)
    ax.set_xlabel("Prompt length group (each group holds a quarter of the Flan prompts)")
    ax.set_ylabel("Share of rows from Bard (%)")
    save(fig, "fig_bard_flan.png")


def fig_length():
    d = pd.read_csv(RESULTS / "dataset_stats.csv").set_index("llm_name")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    style(ax)
    x = np.arange(len(FAMILIES))
    w = 0.3
    for i, (col, name, color) in enumerate([("mean_words", "Mean", C1), ("median_words", "Median", C2)]):
        vals = [d.loc[f, col] for f in FAMILIES]
        pos = x + (i - 0.5) * w
        ax.bar(pos, vals, w - 0.04, color=color, linewidth=0, label=name)
        for p, v in zip(pos, vals):
            ax.text(p, v + 5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylim(0, 330)
    ax.set_ylabel("Words per answer")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2, ncol=2)
    save(fig, "fig_length.png")


def main():
    fig_truncation()
    fig_models()
    fig_recall()
    fig_confusion()
    fig_rq2()
    fig_bard_flan()
    fig_length()
    print(sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
