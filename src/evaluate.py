"""Phase 3: evaluate every trained model on the held-out TEST split and compare them.

Run from the repo root:
    python -m src.evaluate --runs /content/drive/MyDrive/plant-disease/runs   # Colab, real checkpoints
    python -m src.evaluate --runs runs/smoke --smoke                          # quick local pipeline check

Writes to reports/:
    comparison.csv          one row per model: accuracy, precision, recall, macro-F1 (best model first)
    comparison.png          bar chart of the same numbers
    confusion_<model>.png   38x38 confusion matrix, each row shows where that class's images went
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # draw straight to PNG files, no window needed
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import MODELS, REPORTS_DIR, RUNS_DIR
from src.data import make_loader
from src.models import load_checkpoint, normalization

METRICS = ["accuracy", "precision", "recall", "macro_f1"]


@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, predicted labels) for every image in the loader."""
    targets, preds = [], []
    for images, labels in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device))
        preds.append(logits.argmax(1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def scores(y_true, y_pred) -> dict:
    """Accuracy plus macro-averaged precision/recall/F1 (every class counts equally, however many images it has)."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "macro_f1": f1}


def plot_confusion(y_true, y_pred, classes, title, path):
    """Row-normalised confusion matrix: cell (i, j) = share of class i images predicted as class j.
    A perfect model is a bright diagonal; off-diagonal spots show which diseases get confused."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    short = [c.replace("___", ": ").replace("_", " ") for c in classes]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), short, fontsize=7)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(rows, path):
    """Grouped bar chart: one group per model, one bar per metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    for i, metric in enumerate(METRICS):
        xs = [m + (i - 1.5) * width for m in range(len(rows))]
        ax.bar(xs, [r[metric] for r in rows], width, label=metric)
    ax.set_xticks(range(len(rows)), [r["model"] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score on test split")
    ax.set_title("Model comparison (test split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(runs: Path, out: Path, smoke: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in MODELS:
        ckpt_path = runs / name / "best.pt"
        if not ckpt_path.exists():
            print(f"skip {name}: no {ckpt_path}")
            continue
        model, classes = load_checkpoint(ckpt_path, device)
        loader = make_loader("test", *normalization(model), smoke=smoke)
        y_true, y_pred = predict(model, loader, device)
        row = {"model": name, **{k: round(v, 4) for k, v in scores(y_true, y_pred).items()},
               "test_images": len(y_true)}
        print(row)
        rows.append(row)
        plot_confusion(y_true, y_pred, classes, f"{name} - test confusion matrix", out / f"confusion_{name}.png")

    assert rows, f"no best.pt found under {runs}"
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)          # best model first
    with open(out / "comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, out / "comparison.png")
    print(f"best model: {rows[0]['model']} (test macro-F1 {rows[0]['macro_f1']:.4f}) - reports in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder with <model>/best.pt from train.py")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    parser.add_argument("--smoke", action="store_true", help="tiny test subset, for checking the pipeline")
    args = parser.parse_args()
    main(args.runs, args.out, args.smoke)
