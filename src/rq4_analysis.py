import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA = Path("data")
RESULTS = Path("results")
WORD = re.compile(r"[A-Za-z0-9']+")
SENT = re.compile(r"[.!?]+(?:\s|$)")


def features(text):
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    n = max(len(words), 1)
    lines = text.split("\n")
    nonempty = [l for l in lines if l.strip()]
    first100 = lower[:100]
    sentences = max(len(SENT.findall(text)), 1)
    return {
        "words": len(words),
        "chars": len(text),
        "sentences": sentences,
        "words_per_sentence": len(words) / sentences,
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "newlines": text.count("\n"),
        "paragraphs": len(re.findall(r"\n\s*\n", text)) + 1,
        "bullets": sum(1 for l in nonempty if re.match(r"\s*([*\-•]|\d+[.)])\s", l)),
        "bold_markers": text.count("**"),
        "code_ticks": text.count("`"),
        "headers": sum(1 for l in nonempty if l.lstrip().startswith("#")),
        "ttr_first100": len(set(first100)) / max(len(first100), 1),
        "hapax_first100": sum(1 for w in set(first100) if first100.count(w) == 1) / max(len(first100), 1),
        "unique_words": len(set(lower)),
        "exclaim_per_word": text.count("!") / n,
        "question_per_word": text.count("?") / n,
        "comma_per_word": text.count(",") / n,
        "colon_per_word": text.count(":") / n,
        "digit_share": sum(c.isdigit() for c in text) / max(len(text), 1),
        "upper_share": sum(c.isupper() for c in text) / max(len(text), 1),
        "starts_with_sure": float(lower[:1] in (["sure"], ["certainly"], ["of"], ["as"], ["i"])),
        "first_person": sum(1 for w in lower if w in ("i", "my", "me")) / n,
    }


GROUPS = {
    "length": ["words", "chars", "sentences", "words_per_sentence", "avg_word_length", "unique_words"],
    "layout": ["newlines", "paragraphs", "bullets", "bold_markers", "code_ticks", "headers"],
    "diversity": ["ttr_first100", "hapax_first100"],
    "punctuation": ["exclaim_per_word", "question_per_word", "comma_per_word", "colon_per_word", "digit_share", "upper_share"],
    "voice": ["starts_with_sure", "first_person"],
}


def feature_frame(df):
    return pd.DataFrame([features(t) for t in df["output"]])


def strip_format(text):
    return " ".join(WORD.findall(text.lower()))


def skeleton(text):
    text = text.replace("\n", " NL ")
    return re.sub(r"[A-Za-z0-9']+", "w", text)


def only_function_words(text):
    return " ".join(w for w in WORD.findall(text.lower()) if w in ENGLISH_STOP_WORDS)


def only_content_words(text):
    return " ".join(w for w in WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS)


def drop_first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def drop_last_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:-1]) if len(parts) > 1 else ""


def first_n_words(n):
    return lambda text: " ".join(text.split()[:n])


TOKENS = r"\n|\w+|[^\w\s]"


def text_model(train, test, transform, label, ngram=(1, 2), pattern=TOKENS):
    tr = train["output"].map(transform)
    te = test["output"].map(transform)
    vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, max_features=200000, lowercase=False, token_pattern=pattern)
    xtr = vec.fit_transform(tr)
    xte = vec.transform(te)
    clf = LogisticRegression(C=10, max_iter=1000, class_weight="balanced")
    clf.fit(xtr, train["llm_name"])
    pred = clf.predict(xte)
    return {
        "setting": label,
        "test_acc": accuracy_score(test["llm_name"], pred),
        "test_macro_f1": f1_score(test["llm_name"], pred, average="macro"),
    }


def structure_model(ftr, fte, ytr, yte, cols, label):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=0)
    clf.fit(ftr[cols], ytr)
    pred = clf.predict(fte[cols])
    return {
        "setting": label,
        "test_acc": accuracy_score(yte, pred),
        "test_macro_f1": f1_score(yte, pred, average="macro"),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    ftr = feature_frame(train)
    fte = feature_frame(test)
    ftr["llm_name"] = train["llm_name"].values
    summary = ftr.groupby("llm_name").agg(["mean", "median"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(RESULTS / "rq4_feature_means.csv")
    means = ftr.groupby("llm_name").mean().round(3)
    means.to_csv(RESULTS / "rq4_feature_means_only.csv")
    print(means.T.to_string())
    ftr = ftr.drop(columns="llm_name")

    rows = []
    all_cols = [c for cols in GROUPS.values() for c in cols]
    rows.append({"kind": "structure", **structure_model(ftr, fte, train["llm_name"], test["llm_name"], all_cols, "all structure features")})
    for name, cols in GROUPS.items():
        rows.append({"kind": "structure", **structure_model(ftr, fte, train["llm_name"], test["llm_name"], cols, f"only {name}")})
        rest = [c for c in all_cols if c not in cols]
        rows.append({"kind": "structure", **structure_model(ftr, fte, train["llm_name"], test["llm_name"], rest, f"all except {name}")})
        print(rows[-1], flush=True)

    settings = [
        ("full text", lambda t: t, (1, 2)),
        ("no formatting (words only, lowercase)", strip_format, (1, 2)),
        ("formatting only (words replaced by w)", skeleton, (1, 3), r"\S+"),
        ("function words only", only_function_words, (1, 2)),
        ("content words only", only_content_words, (1, 2)),
        ("first sentence removed", drop_first_sentence, (1, 2)),
        ("last sentence removed", drop_last_sentence, (1, 2)),
        ("first 20 words only", first_n_words(20), (1, 2)),
        ("first sentence and last sentence removed", lambda t: drop_last_sentence(drop_first_sentence(t)), (1, 2)),
        ("single words only (no word pairs)", lambda t: t, (1, 1)),
    ]
    for item in settings:
        label, fn, ngram = item[:3]
        pattern = item[3] if len(item) > 3 else TOKENS
        rows.append({"kind": "text", **text_model(train, test, fn, label, ngram, pattern)})
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rq4_ablation.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
