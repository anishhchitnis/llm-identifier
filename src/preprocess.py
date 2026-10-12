from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def split_by_prompt(df, seed=SEED):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=df["input"]))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(inner.split(rest, groups=rest["input"]))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


def check_no_prompt_overlap(train, val, test):
    sets = {"train": set(train["input"]), "val": set(val["input"]), "test": set(test["input"])}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = len(sets[a] & sets[b])
        assert shared == 0, f"{shared} prompts shared between {a} and {b}"
    print("prompt overlap between splits: 0")


def main():
    df = pd.read_parquet("data/llm_dataset.parquet")
    print("rows before leak removal:", len(df))
    print("\nSelf-naming rows per family:")
    print(df.groupby("llm_name")["leak"].agg(rows="sum", rate="mean").round(4))

    leaked = df[df["leak"]]
    clean = df[~df["leak"]].reset_index(drop=True)
    print("\nrows after leak removal:", len(clean))

    train, val, test = split_by_prompt(clean)
    check_no_prompt_overlap(train, val, test)

    out = Path("data")
    out.mkdir(exist_ok=True)
    leaked.to_parquet(out / "leak_rows.parquet", index=False)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    tagged = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    print("\nRows per family x split:")
    print(pd.crosstab(tagged["llm_name"], tagged["split"], margins=True)[["train", "val", "test", "All"]])
    print("\nRows per domain x split:")
    print(pd.crosstab(tagged["domain"], tagged["split"], margins=True)[["train", "val", "test", "All"]])
    print("\nTest-set share per family (should be near 20% each):")
    print(test["llm_name"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
