import json
from pathlib import Path

import pandas as pd


def main():
    rows = []
    for path in sorted(Path("results").glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if "per_seed" not in data:
            continue
        row = {
            "run": path.stem,
            "model": data["model"],
            "mode": data["mode"],
            "holdout": data["holdout_domain"] or "-",
            "max_len": data["max_len"],
            "seeds": data["seeds"],
            "acc_mean": data["test_acc_mean"],
            "acc_std": data["test_acc_std"],
            "f1_mean": data["test_macro_f1_mean"],
            "f1_std": data["test_macro_f1_std"],
        }
        for family, value in data["test_recall_mean"].items():
            row[f"recall_{family}"] = value
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv("results/summary_table.csv", index=False)
    print(table.drop(columns=["seeds"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
