import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from common import build_vocab, encode_frame, predict, set_seed
from models import BiLSTM, TextCNN

RESULTS = Path("results")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "lstm"], required=True)
    p.add_argument("--mode", choices=["output", "input", "both"], default="output")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout_domain", default=None)
    p.add_argument("--tag", default="")
    return p.parse_args()


def load_splits(holdout):
    train = pd.read_parquet("data/train.parquet")
    val = pd.read_parquet("data/val.parquet")
    test = pd.read_parquet("data/test.parquet")
    if holdout is not None:
        full = pd.concat([train, val, test], ignore_index=True)
        train = train[train["domain"] != holdout]
        val = val[val["domain"] != holdout]
        test = full[full["domain"] == holdout]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_loader(x, lengths, y, batch, shuffle):
    return DataLoader(TensorDataset(x, lengths, y), batch_size=batch, shuffle=shuffle)


def build_model(name, vocab_size, n_classes):
    if name == "cnn":
        return TextCNN(vocab_size, n_classes)
    return BiLSTM(vocab_size, n_classes)


def run_seed(seed, args, data, device):
    set_seed(seed)
    (xtr, ltr, ytr), (xva, lva, yva), (xte, lte, yte), vocab_size, n_classes = data
    model = build_model(args.model, vocab_size, n_classes).to(device)
    counts = torch.bincount(ytr, minlength=n_classes).float()
    weights = (counts.sum() / (n_classes * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(xtr, ltr, ytr, args.batch, True)
    val_loader = make_loader(xva, lva, yva, 256, False)
    test_loader = make_loader(xte, lte, yte, 256, False)
    labels = list(range(n_classes))

    best_f1, best_state, bad = -1.0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        p_val = predict(model, val_loader, device)
        val_f1 = f1_score(yva.numpy(), p_val, labels=labels, average="macro")
        val_acc = accuracy_score(yva.numpy(), p_val)
        train_loss = total / len(ytr)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "val_macro_f1": val_f1}
        )
        print(f"seed {seed} epoch {epoch:2d} loss {train_loss:.4f} val acc {val_acc:.4f} val F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    p_test = predict(model, test_loader, device)
    y_true = yte.numpy()
    recall = {}
    report = classification_report(y_true, p_test, labels=labels, output_dict=True, zero_division=0)
    for i in labels:
        recall[i] = report[str(i)]["recall"]
    metrics = {
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
        "train_seconds": time.time() - start,
        "test_acc": accuracy_score(y_true, p_test),
        "test_macro_f1": f1_score(y_true, p_test, labels=labels, average="macro"),
        "test_recall": recall,
        "history": history,
    }
    n_params = sum(p.numel() for p in model.parameters())
    return metrics, p_test, n_params


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = load_splits(args.holdout_domain)
    labels = sorted(train["llm_name"].unique())
    to_id = {name: i for i, name in enumerate(labels)}
    vocab = build_vocab(train, args.mode)

    def prepare(df):
        x, lengths = encode_frame(df, vocab, args.mode, args.max_len)
        y = torch.tensor(df["llm_name"].map(to_id).values, dtype=torch.long)
        return x, lengths, y

    data = (prepare(train), prepare(val), prepare(test), len(vocab) + 3, len(labels))
    print(f"device {device} | train {len(train)} val {len(val)} test {len(test)} | vocab {len(vocab) + 3}")
    print(f"sequence width train {data[0][0].shape[1]}")

    runs, all_preds = [], []
    n_params = 0
    for seed in range(args.seeds):
        metrics, p_test, n_params = run_seed(seed, args, data, device)
        runs.append(metrics)
        all_preds.append(p_test)
        print(
            f"seed {seed} done: test acc {metrics['test_acc']:.4f} test macro-F1 {metrics['test_macro_f1']:.4f}"
        )

    ddof = 1 if len(runs) > 1 else 0
    accs = np.array([r["test_acc"] for r in runs])
    f1s = np.array([r["test_macro_f1"] for r in runs])
    recall_mean = {
        labels[i]: float(np.mean([r["test_recall"][i] for r in runs])) for i in range(len(labels))
    }
    summary = {
        "model": args.model,
        "mode": args.mode,
        "holdout_domain": args.holdout_domain,
        "max_len": args.max_len,
        "seeds": len(runs),
        "n_params": n_params,
        "vocab_size": len(vocab) + 3,
        "labels": labels,
        "test_acc_mean": float(accs.mean()),
        "test_acc_std": float(accs.std(ddof=ddof)),
        "test_macro_f1_mean": float(f1s.mean()),
        "test_macro_f1_std": float(f1s.std(ddof=ddof)),
        "test_recall_mean": recall_mean,
        "per_seed": runs,
    }

    name = f"{args.model}_{args.mode}"
    if args.holdout_domain:
        name += f"_holdout-{args.holdout_domain}"
    if args.tag:
        name += f"_{args.tag}"
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    y_true = test["llm_name"].map(to_id).values
    cm = sum(confusion_matrix(y_true, p, labels=list(range(len(labels)))) for p in all_preds)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(RESULTS / f"{name}_confusion.csv")
    pred_rows = []
    for seed, p in enumerate(all_preds):
        pred_rows.append(
            pd.DataFrame(
                {
                    "row": np.arange(len(test)),
                    "seed": seed,
                    "domain": test["domain"].values,
                    "true": test["llm_name"].values,
                    "pred": [labels[i] for i in p],
                }
            )
        )
    pd.concat(pred_rows).to_csv(RESULTS / f"{name}_preds.csv", index=False)

    print(f"\n{name}: parameters {n_params:,}")
    print(f"test accuracy  {accs.mean():.4f} +/- {accs.std(ddof=ddof):.4f}")
    print(f"test macro-F1  {f1s.mean():.4f} +/- {f1s.std(ddof=ddof):.4f}")
    print("per-family recall (mean over seeds):")
    for k, v in recall_mean.items():
        print(f"  {k:8s} {v:.3f}")
    print("confusion matrix summed over seeds (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))


if __name__ == "__main__":
    main()
