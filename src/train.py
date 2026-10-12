"""Phase 2: train a model with mixed precision, AdamW, warmup + cosine LR, early stopping and optional W&B.

Run from the repo root:
    python -m src.train --model custom_cnn
    python -m src.train --model all --smoke     # 1 epoch on a tiny subset: checks the pipeline in ~minutes
    python -m src.train --model vit_small --out /content/drive/MyDrive/plant-disease/runs

Each model gets its own folder <out>/<model>/ with:
    best.pt      weights with the best validation macro-F1 (used later for evaluation and the app)
    last.pt      everything needed to resume after a Colab disconnect (just run the same command again)
    history.csv  one row of metrics per epoch
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.config import LABEL_SMOOTHING, MODELS, RUNS_DIR, SEED, WARMUP_EPOCHS, WEIGHT_DECAY
from src.data import class_names, make_loader
from src.models import build_model, load_checkpoint, normalization
from src.utils import set_seed


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Mixed precision: run the forward pass in float16 on the GPU (about 2x faster, half the memory).
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        # The scaler multiplies the loss before backward so tiny float16 gradients don't round to zero.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()                                  # LR changes every batch, not every epoch
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Loss, accuracy and macro-F1 on a validation/test loader (no gradients, no augmentation)."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        total_loss += criterion(logits.float(), labels).item() * len(labels)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)
    accuracy = (preds == targets).float().mean().item()
    # Macro-F1 averages F1 over classes equally, so small classes count as much as big ones.
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return total_loss / len(targets), accuracy, macro_f1


def train(model_name: str, out_dir: Path, smoke: bool = False, epochs: int | None = None) -> None:
    set_seed(SEED)
    cfg = MODELS[model_name]
    epochs = 1 if smoke else (epochs or cfg["epochs"])
    run_dir = out_dir / ("smoke" if smoke else "") / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = class_names()
    model = build_model(model_name, len(classes)).to(device)
    mean, std = normalization(model)
    train_loader = make_loader("train", mean, std, smoke)
    val_loader = make_loader("val", mean, std, smoke)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    # LR schedule: linear warmup (avoids wrecking pretrained weights with big early updates), then cosine decay.
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, milestones=[warmup_steps], schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * steps_per_epoch - warmup_steps)),
    ])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # Resume after a disconnect: reload model, optimizer, scheduler and early-stopping state from last.pt.
    state = {"epoch": 0, "best_f1": -1.0, "bad_epochs": 0, "done": False, "wandb_id": None}
    last_path = run_dir / "last.pt"
    if last_path.exists() and not smoke:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        for name, obj in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            obj.load_state_dict(ckpt[name])
        state = ckpt["state"]
        print(f"resumed {model_name} after epoch {state['epoch']}")
    if state["done"]:
        print(f"{model_name} already finished (best val macro-F1 {state['best_f1']:.4f}) - skipping")
        return

    # W&B is optional: it only switches on when a WANDB_API_KEY is set.
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        state["wandb_id"] = state["wandb_id"] or wandb.util.generate_id()
        wandb_run = wandb.init(project="plant-disease", name=model_name + ("-smoke" if smoke else ""),
                               id=state["wandb_id"], resume="allow", config={**cfg, "model": model_name})
    else:
        print("W&B off (no WANDB_API_KEY) - metrics go to history.csv only")

    print(f"training {model_name} on {device} | {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val images | up to {epochs} epochs")
    for epoch in range(state["epoch"] + 1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
               "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start)}
        print(row)

        history = run_dir / "history.csv"
        new_file = not history.exists()
        with open(history, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if wandb_run:
            wandb_run.log(row)

        # Early stopping on validation macro-F1: keep the best weights, stop when it stops improving.
        if val_f1 > state["best_f1"]:
            state["best_f1"], state["bad_epochs"] = val_f1, 0
            torch.save({"model_name": model_name, "state_dict": model.state_dict(), "classes": classes,
                        "epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}, run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch
        state["done"] = state["bad_epochs"] >= cfg["patience"] or epoch == epochs
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state}, last_path)
        if state["done"]:
            break

    print(f"{model_name} done: best val macro-F1 {state['best_f1']:.4f} -> {run_dir / 'best.pt'}")
    if wandb_run:
        wandb_run.finish()
    if smoke:
        check_reload(run_dir / "best.pt", val_loader, device)


@torch.no_grad()
def check_reload(path: Path, loader, device) -> None:
    """Smoke test: load best.pt into a fresh model (no pretrained download) and check it predicts."""
    model, _ = load_checkpoint(path, device)
    saved_acc = torch.load(path, map_location="cpu", weights_only=False)["val_acc"]
    _, val_acc, _ = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    assert abs(val_acc - saved_acc) < 1e-6, f"reloaded acc {val_acc} != saved {saved_acc}"
    print(f"reload OK: {path.name} gives the same val acc ({val_acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=[*MODELS, "all"])
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="where checkpoints and logs go")
    parser.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset to test the pipeline")
    parser.add_argument("--epochs", type=int, help="override the epoch count from config.py")
    args = parser.parse_args()

    for name in (MODELS if args.model == "all" else [args.model]):   # dict order = training order
        train(name, args.out, args.smoke, args.epochs)
