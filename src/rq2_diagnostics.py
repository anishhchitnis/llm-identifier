import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from build_dataset import FAMILY, domain_of, load_raw


def raw_counts(ds):
    rows = []
    for ex in ds:
        dom = domain_of(ex["source"])
        if dom is None:
            continue
        for c in ex["completions"]:
            fam = FAMILY.get(c["model"])
            if fam is None:
                continue
            n = len((c.get("response") or "").strip())
            rows.append((fam, dom, n == 0, 0 < n < 20, n > 8000))
    df = pd.DataFrame(rows, columns=["family", "domain", "empty", "short", "long"])
    out = df.groupby(["family", "domain"]).agg(
        raw=("empty", "size"), empty=("empty", "sum"), short=("short", "sum"), long=("long", "sum")
    )
    out["kept_by_length_rules"] = out["raw"] - out["empty"] - out["short"] - out["long"]
    return out


def length_vs_family(df):
    rows = []
    for dom, g in df.groupby("domain"):
        words = g["input"].str.split().str.len()
        bins = pd.qcut(words, 4, duplicates="drop")
        table = pd.crosstab(bins, g["llm_name"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            rows.append({"domain": dom, "chi2": np.nan, "dof": 0, "p_value": np.nan})
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        rows.append({"domain": dom, "chi2": chi2, "dof": dof, "p_value": p})
    return pd.DataFrame(rows)


def family_share_by_length(df, domain):
    g = df[df["domain"] == domain]
    words = g["input"].str.split().str.len()
    bins = pd.qcut(words, 4, duplicates="drop")
    return pd.crosstab(bins, g["llm_name"], normalize="index")


def prompt_features(df, domain_ids):
    return pd.DataFrame(
        {
            "n_words": df["input"].str.split().str.len().values,
            "n_chars": df["input"].str.len().values,
            "n_newlines": df["input"].str.count("\n").values,
            "domain": df["domain"].map(domain_ids).values,
        }
    )


def prompt_only_baselines(train, val, test):
    domain_ids = {d: i for i, d in enumerate(sorted(train["domain"].unique()))}
    rows = []

    def add(name, y_pred_val, y_pred_test):
        rows.append(
            {
                "classifier": name,
                "val_acc": accuracy_score(val["llm_name"], y_pred_val),
                "test_acc": accuracy_score(test["llm_name"], y_pred_test),
                "test_macro_f1": f1_score(test["llm_name"], y_pred_test, average="macro"),
            }
        )

    top = train["llm_name"].value_counts().idxmax()
    add("majority_class", np.full(len(val), top), np.full(len(test), top))

    per_domain = train.groupby("domain")["llm_name"].agg(lambda s: s.value_counts().idxmax())
    add(
        "domain_only",
        val["domain"].map(per_domain).values,
        test["domain"].map(per_domain).values,
    )

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=0
    )
    clf.fit(prompt_features(train, domain_ids), train["llm_name"])
    add(
        "prompt_length_and_domain",
        clf.predict(prompt_features(val, domain_ids)),
        clf.predict(prompt_features(test, domain_ids)),
    )
    return pd.DataFrame(rows)


def main():
    print("1. Raw responses per family and domain, before any filtering")
    counts = raw_counts(load_raw())
    print(counts.to_string())
    counts.to_csv("results/raw_counts.csv")

    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet")
    test = pd.read_parquet("data/test.parquet")
    full = pd.concat([train, val, test], ignore_index=True)

    print("\n2. Does prompt length relate to family? (chi-square per domain, small p means yes)")
    chi = length_vs_family(full)
    print(chi.round(4).to_string(index=False))
    chi.to_csv("results/prompt_length_chi2.csv", index=False)

    print("\nFamily share by prompt-length quartile in the flan domain:")
    print(family_share_by_length(full, "flan").round(3))

    print("\n3. Prompt-only classifiers (compare with neural input-only: acc 0.226 / 0.240, macro-F1 0.219 / 0.217)")
    base = prompt_only_baselines(train, val, test)
    print(base.round(4).to_string(index=False))
    base.to_csv("results/prompt_only_baselines.csv", index=False)


if __name__ == "__main__":
    main()
