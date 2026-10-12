import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

RESULTS = Path("results")
C_GRID = [1, 10]


def word_prep(text):
    return text.replace("\n", " ¶ ")


def char_prep(text):
    return text.replace("\n", "¶")


def word_vectorizer():
    return TfidfVectorizer(
        preprocessor=word_prep,
        token_pattern=r"\w+|[^\w\s]",
        ngram_range=(1, 2),
        min_df=2,
        max_features=300000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def char_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        preprocessor=char_prep,
        ngram_range=(2, 5),
        min_df=3,
        max_features=300000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def text_of(df, mode):
    if mode == "output":
        return df["output"]
    if mode == "input":
        return df["input"]
    return df["input"] + " ¶¶ " + df["output"]


def fit_logreg(x_train, y_train, x_val, y_val):
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=300, class_weight="balanced")
        clf.fit(x_train, y_train)
        score = f1_score(y_val, clf.predict(x_val), average="macro")
        print(f"  C={c}: val macro-F1 {score:.4f}")
        if best is None or score > best[0]:
            best = (score, c, clf)
    return best[1], best[2]


def top_features(vec, clf, k=15):
    names = np.array(vec.get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[::-1][:k]
        for rank, j in enumerate(idx, 1):
            rows.append(
                {"llm": cls, "rank": rank, "feature": names[j], "weight": float(clf.coef_[i][j])}
            )
    return pd.DataFrame(rows)


def main():
    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet")
    test = pd.read_parquet("data/test.parquet")
    y_train, y_val, y_test = train["llm_name"].values, val["llm_name"].values, test["llm_name"].values
    labels = sorted(set(y_train))
    RESULTS.mkdir(exist_ok=True)

    rows = []
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y_train)
    p_val = dummy.predict(np.zeros((len(val), 1)))
    p_test = dummy.predict(np.zeros((len(test), 1)))
    rows.append(
        {
            "model": "majority_class",
            "C": None,
            "val_acc": accuracy_score(y_val, p_val),
            "val_macro_f1": f1_score(y_val, p_val, average="macro"),
            "test_acc": accuracy_score(y_test, p_test),
            "test_macro_f1": f1_score(y_test, p_test, average="macro"),
        }
    )

    setups = [
        ("tfidf_word_output", word_vectorizer, "output"),
        ("tfidf_char_output", char_vectorizer, "output"),
        ("tfidf_word_input_only", word_vectorizer, "input"),
        ("tfidf_word_input_plus_output", word_vectorizer, "both"),
    ]
    artifacts = {}
    for name, factory, mode in setups:
        print(f"\n{name}")
        vec = factory()
        x_train = vec.fit_transform(text_of(train, mode))
        x_val = vec.transform(text_of(val, mode))
        x_test = vec.transform(text_of(test, mode))
        c, clf = fit_logreg(x_train, y_train, x_val, y_val)
        p_val = clf.predict(x_val)
        p_test = clf.predict(x_test)
        rows.append(
            {
                "model": name,
                "C": c,
                "val_acc": accuracy_score(y_val, p_val),
                "val_macro_f1": f1_score(y_val, p_val, average="macro"),
                "test_acc": accuracy_score(y_test, p_test),
                "test_macro_f1": f1_score(y_test, p_test, average="macro"),
            }
        )
        artifacts[name] = (vec, clf, p_test)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "baselines.csv", index=False)
    print("\nBaseline results (chance level with 5 families is about 0.20):")
    print(table.round(4).to_string(index=False))

    best = max(["tfidf_word_output", "tfidf_char_output"], key=lambda n: table.set_index("model").loc[n, "val_macro_f1"])
    vec, clf, p_test = artifacts[best]
    print(f"\nBest output-only baseline by validation: {best}")
    print(classification_report(y_test, p_test, labels=labels, digits=3))
    cm = pd.DataFrame(confusion_matrix(y_test, p_test, labels=labels), index=labels, columns=labels)
    cm.to_csv(RESULTS / "baseline_confusion.csv")
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm)

    with open(RESULTS / "baseline_report.json", "w") as f:
        json.dump(classification_report(y_test, p_test, labels=labels, output_dict=True), f, indent=2)

    feats = top_features(artifacts["tfidf_word_output"][0], artifacts["tfidf_word_output"][1])
    feats.to_csv(RESULTS / "baseline_top_features.csv", index=False)
    print("\nTop word features per family (word TF-IDF, output only):")
    for cls, g in feats.groupby("llm"):
        print(f"{cls}: " + " | ".join(g.sort_values("rank")["feature"].head(12)))


if __name__ == "__main__":
    main()
