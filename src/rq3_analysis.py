import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, recall_score

from make_figures import (
    AXIS, BLUE_RAMP, C1, C2, FAMILIES, INK, INK2, NAMES, RESULTS, SURFACE, plt, save, style,
)

DOMAINS = ["evol_instruct", "flan", "sharegpt", "ultrachat"]
DOMAIN_NAMES = ["Evol Instruct", "Flan", "ShareGPT", "UltraChat"]
MODELS = [("cnn", "CNN"), ("lstm", "BiLSTM")]


def in_domain_scores(model, domain):
    preds = pd.read_csv(RESULTS / f"{model}_output_preds.csv")
    d = preds[preds["domain"] == domain]
    f1s, accs = [], []
    recalls = []
    for _, g in d.groupby("seed"):
        f1s.append(f1_score(g["true"], g["pred"], average="macro", labels=FAMILIES))
        accs.append((g["true"] == g["pred"]).mean())
        recalls.append(recall_score(g["true"], g["pred"], average=None, labels=FAMILIES, zero_division=0))
    return np.mean(f1s), np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0, np.mean(accs), np.mean(recalls, axis=0)


def held_out_scores(model, domain):
    path = RESULTS / f"{model}_output_holdout-{domain}.json"
    with open(path) as f:
        r = json.load(f)
    recalls = [r["test_recall_mean"][fam] for fam in FAMILIES]
    return r["test_macro_f1_mean"], r["test_macro_f1_std"], r["test_acc_mean"], np.array(recalls)


def build_tables():
    rows, recall_rows = [], []
    for model, label in MODELS:
        for domain in DOMAINS:
            if not (RESULTS / f"{model}_output_holdout-{domain}.json").exists():
                continue
            in_f1, in_sd, in_acc, in_rec = in_domain_scores(model, domain)
            out_f1, out_sd, out_acc, out_rec = held_out_scores(model, domain)
            rows.append(
                {
                    "model": label, "domain": domain, "in_domain_f1": in_f1, "in_domain_f1_std": in_sd,
                    "held_out_f1": out_f1, "held_out_f1_std": out_sd, "drop": in_f1 - out_f1,
                    "in_domain_acc": in_acc, "held_out_acc": out_acc,
                }
            )
            for i, fam in enumerate(FAMILIES):
                recall_rows.append(
                    {
                        "model": label, "domain": domain, "family": fam,
                        "in_domain_recall": in_rec[i], "held_out_recall": out_rec[i],
                    }
                )
    table = pd.DataFrame(rows)
    recall = pd.DataFrame(recall_rows)
    table.to_csv(RESULTS / "rq3_table.csv", index=False)
    recall.to_csv(RESULTS / "rq3_recall.csv", index=False)
    return table, recall


def fig_rq3_f1(table):
    models = [label for _, label in MODELS if label in set(table["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(6.8, 3.2), sharey=True, squeeze=False)
    for ax, label in zip(axes[0], models):
        style(ax)
        d = table[table["model"] == label].set_index("domain")
        x = np.arange(len(DOMAINS))
        w = 0.32
        for k, (col, sd, name, color) in enumerate(
            [("in_domain_f1", "in_domain_f1_std", "Seen domain", C1), ("held_out_f1", "held_out_f1_std", "Held out domain", C2)]
        ):
            vals = [d.loc[dom, col] if dom in d.index else np.nan for dom in DOMAINS]
            errs = [d.loc[dom, sd] if dom in d.index else 0 for dom in DOMAINS]
            pos = x + (k - 0.5) * w
            ax.bar(pos, vals, w - 0.04, color=color, linewidth=0, label=name)
            for p, v, e in zip(pos, vals, errs):
                if np.isnan(v):
                    continue
                ax.plot([p, p], [v - e, v + e], color=INK2, linewidth=1.1)
                ax.text(p, v + e + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.axhline(0.20, color=AXIS, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(DOMAIN_NAMES, fontsize=8, rotation=20, ha="right")
        ax.set_title(label, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Domain used for testing")
    axes[0][0].set_ylabel("Macro F1")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.84, wspace=0.08)
    save(fig, "fig_rq3_f1.png")


def fig_rq3_recall(recall):
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    models = [label for _, label in MODELS if label in set(recall["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(6.8, 3.3), squeeze=False)
    for ax, label in zip(axes[0], models):
        d = recall[recall["model"] == label]
        m = np.full((len(FAMILIES), len(DOMAINS)), np.nan)
        for i, fam in enumerate(FAMILIES):
            for j, dom in enumerate(DOMAINS):
                v = d[(d["family"] == fam) & (d["domain"] == dom)]["held_out_recall"]
                if len(v):
                    m[i, j] = v.iloc[0] * 100
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(FAMILIES)):
            for j in range(len(DOMAINS)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > 45 else INK)
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels(DOMAIN_NAMES, fontsize=8, rotation=25, ha="right")
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(NAMES if ax is axes[0][0] else [], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0, labelcolor=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(label, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Held out domain")
    axes[0][0].set_ylabel("True family")
    cbar = fig.colorbar(im, ax=list(axes[0]), shrink=0.82, pad=0.02)
    cbar.set_label("Recall on the held out domain (percent)", color=INK2, fontsize=8.5)
    cbar.ax.tick_params(labelcolor=INK2, labelsize=8, length=2)
    cbar.outline.set_visible(False)
    save(fig, "fig_rq3_recall.png")


def main():
    table, recall = build_tables()
    print(table.round(3).to_string(index=False))
    fig_rq3_f1(table)
    fig_rq3_recall(recall)
    print(sorted(p.name for p in Path("figures").iterdir() if "rq3" in p.name))


if __name__ == "__main__":
    main()
