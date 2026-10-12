import re
from pathlib import Path

import pandas as pd

SEED = 42
CAP_PER_CELL = 1000

DOMAINS = {"evol_instruct", "flan", "sharegpt", "ultrachat"}

FAMILY = {
    "gpt-4": "gpt",
    "gpt-3.5-turbo": "gpt",
    "llama-2-7b-chat": "llama2",
    "llama-2-13b-chat": "llama2",
    "llama-2-70b-chat": "llama2",
    "bard": "bard",
    "falcon-40b-instruct": "falcon",
    "mpt-30b-chat": "mpt",
}

LEAK = re.compile(
    r"\b(?:openai|chatgpt|gpt-?3\.?5?|gpt-?4|bard|google ai|llama|falcon|mosaicml|"
    r"developed by|trained by|created by)\b",
    re.I,
)


def domain_of(source):
    if source.startswith("flan"):
        return "flan"
    return source if source in DOMAINS else None


def load_raw():
    from datasets import load_dataset

    try:
        return load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        print("Default load failed, trying parquet branch:", e)
        return load_dataset(
            "openbmb/UltraFeedback", split="train", revision="refs/convert/parquet"
        )


def build(ds, cap=CAP_PER_CELL, seed=SEED):
    rows = []
    for i, ex in enumerate(ds):
        dom = domain_of(ex["source"])
        if dom is None:
            continue
        pid = f'{ex["source"]}:{ex.get("id") or i}'
        prompt = (ex["instruction"] or "").strip()
        if not prompt:
            continue
        for c in ex["completions"]:
            fam = FAMILY.get(c["model"])
            resp = (c.get("response") or "").strip()
            if fam is None or not resp:
                continue
            rows.append(
                dict(
                    llm_name=fam,
                    llm_model=c["model"],
                    domain=dom,
                    prompt_id=pid,
                    input=prompt,
                    output=resp,
                    system_prompt=c.get("custom_system_prompt"),
                    principle=c.get("principle"),
                )
            )
    df = pd.DataFrame(rows)
    n0 = len(df)

    df = df.drop_duplicates(subset=["input", "output"])
    df = df[df["output"].str.len().between(20, 8000)]
    print(f"cleaning: {n0} -> {len(df)} rows")

    df = (
        df.sample(frac=1, random_state=seed)
        .groupby(["llm_name", "domain"])
        .head(cap)
        .reset_index(drop=True)
    )
    df["leak"] = df["output"].str.contains(LEAK)
    return df


def report(df):
    print("\nRows per family x domain:")
    print(pd.crosstab(df["llm_name"], df["domain"], margins=True))
    df = df.assign(n_words=df["output"].str.split().str.len())
    stats = df.groupby("llm_name").agg(
        rows=("output", "size"),
        mean_words=("n_words", "mean"),
        median_words=("n_words", "median"),
        leak_rate=("leak", "mean"),
    )
    print("\nPer-family stats:")
    print(stats.round(2))
    return stats


def main():
    ds = load_raw()
    df = build(ds)
    stats = report(df)
    out = Path("data")
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "llm_dataset.parquet", index=False)
    stats.to_csv(out / "dataset_stats.csv")
    print(f"\nSaved {len(df)} rows -> data/llm_dataset.parquet")


if __name__ == "__main__":
    main()
