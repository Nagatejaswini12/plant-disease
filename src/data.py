"""Phase 1: download PlantVillage and make a 70/15/15 split.

The split is stratified by class AND grouped by physical leaf: PlantVillage often has several
photos of the same leaf (up to 33), so a plain random split would put near-identical photos in
both train and test and inflate the test accuracy. Here every leaf goes to exactly one split.

Run from the repo root:
    python -m src.data            # small metadata files + data/splits.csv (works on a laptop)
    python -m src.data --images   # also download the 2.2 GB image zip (do this on Colab)
"""

import argparse
import csv
import json
import os
import random
import zipfile
from collections import Counter, defaultdict

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.config import (BATCH_SIZE, DATA_DIR, HF_DATASET_REPO, IMG_SIZE, SEED, SMOKE_IMAGES,
                        SPLITS_CSV, TEST_RATIO, VAL_RATIO)


def download(filename: str):
    """Download one file from the PlantVillage dataset repo into data/ (skipped if already there)."""
    return hf_hub_download(HF_DATASET_REPO, filename, repo_type="dataset", local_dir=DATA_DIR)


def leaf_id_for(path: str, class_name: str, leaf_map: dict) -> str:
    """Find which physical leaf an image shows.

    Same logic as the dataset's official loader (plant_village.py on the Hub): turn the file name
    into an image id, then look it up in leaf-map.json. Images not in the map count as their own leaf.
    """
    name = path.split("/")[-1].replace("_final_masked", "")
    name = name.split("___")[-1].split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        name = name.replace(ext, "")
    name = name.strip()

    candidates = leaf_map.get(name.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:                  # several leaves share this id: pick the one of our class
        if class_name in c:
            return c
    return f"fallback_{name}"


def build_splits() -> list[dict]:
    """Return one row per image: path, label, label_idx, leaf_id, split."""
    # The author's file lists cover all 54,305 colour images; we merge them and re-split 70/15/15.
    paths = []
    for f in ("splits/color_train.txt", "splits/color_test.txt"):
        with open(download(f)) as fh:
            paths += [line.strip() for line in fh if line.strip()]
    with open(download("leaf_grouping/leaf-map.json")) as fh:
        leaf_map = json.load(fh)

    # Path format: raw/color/<class>/<file>.  Class index = position in the sorted class list.
    classes = sorted({p.split("/")[2] for p in paths})
    rows = []
    for p in sorted(paths):
        label = p.split("/")[2]
        rows.append({"path": p, "label": label, "label_idx": classes.index(label),
                     "leaf_id": f"{label}|{leaf_id_for(p, label, leaf_map)}"})

    # For each class: shuffle its leaves with a fixed seed, then 15% -> test, 15% -> val, rest -> train.
    rng = random.Random(SEED)
    leaves_per_class = defaultdict(set)
    for r in rows:
        leaves_per_class[r["label"]].add(r["leaf_id"])
    split_of_leaf = {}
    for label in classes:
        leaves = sorted(leaves_per_class[label])      # sort first so the shuffle is reproducible
        rng.shuffle(leaves)
        n_test = round(len(leaves) * TEST_RATIO)
        n_val = round(len(leaves) * VAL_RATIO)
        for i, leaf in enumerate(leaves):
            split_of_leaf[leaf] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    for r in rows:
        r["split"] = split_of_leaf[r["leaf_id"]]
    return rows


def check_splits(rows: list[dict]) -> None:
    """Sanity checks that fail loudly if the split is wrong."""
    splits_of_leaf = defaultdict(set)
    for r in rows:
        splits_of_leaf[r["leaf_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_of_leaf.values()), "a leaf appears in more than one split"

    n_classes = len({r["label"] for r in rows})
    for s in ("train", "val", "test"):
        assert len({r["label"] for r in rows if r["split"] == s}) == n_classes, f"{s} is missing a class"

    counts = Counter(r["split"] for r in rows)
    for s, target in (("train", 1 - VAL_RATIO - TEST_RATIO), ("val", VAL_RATIO), ("test", TEST_RATIO)):
        share = counts[s] / len(rows)
        assert abs(share - target) < 0.02, f"{s} has {share:.1%} of images, expected ~{target:.0%}"
        print(f"{s:>5}: {counts[s]:>6} images ({share:.1%})")
    print(f"total: {len(rows)} images, {len(splits_of_leaf)} leaves, {n_classes} classes - all checks passed")


def download_images() -> None:
    """Download data.zip (2.2 GB: colour, grayscale, segmented) and extract only the colour images."""
    zip_path = download("data.zip")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith("raw/color/") and not m.endswith("/")]  # files only
        z.extractall(DATA_DIR, members=members)
    print(f"extracted {len(members)} colour images to {DATA_DIR / 'raw' / 'color'}")


# ---------------- Loading images for training (Phase 2) ----------------

def read_split(split: str, smoke: bool = False) -> list[dict]:
    """Rows of splits.csv for one split. Smoke mode keeps a tiny fixed random sample (64 train / 32 val)."""
    with open(SPLITS_CSV, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    if smoke:
        rows = random.Random(SEED).sample(rows, SMOKE_IMAGES[split])
    return rows


def class_names() -> list[str]:
    """The 38 class names, in label_idx order (index 0 = class 0)."""
    with open(SPLITS_CSV, newline="") as fh:
        return sorted({r["label"] for r in csv.DictReader(fh)})


def build_transforms(train: bool, mean, std) -> v2.Compose:
    """Training: mild random augmentation. Eval: deterministic resize. Both normalise like the pretrained model."""
    if train:
        steps = [
            v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # random zoom + crop
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),                           # a leaf has no natural "up"
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [v2.Resize(IMG_SIZE), v2.CenterCrop(IMG_SIZE)]
    return v2.Compose([v2.ToImage(), *steps, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


class LeafDataset(Dataset):
    """Reads (image, label) pairs from rows of splits.csv."""

    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(DATA_DIR / row["path"]).convert("RGB")
        return self.transform(image), int(row["label_idx"])


def make_loader(split: str, mean, std, smoke: bool = False) -> DataLoader:
    """DataLoader for one split. Only the train split is shuffled and augmented."""
    train = split == "train"
    dataset = LeafDataset(read_split(split, smoke), build_transforms(train, mean, std))
    # Windows starts each worker as a fresh Python process (slow), so load in the main process there.
    # Linux/Colab forks workers cheaply; Colab free tier has 2 CPU cores.
    workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        # Seeded generator -> same shuffle order every run. Workers derive their torch seed from it,
        # and our transforms only use torch's random numbers, so augmentation is reproducible too.
        generator=torch.Generator().manual_seed(SEED),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", action="store_true", help="also download and extract the images")
    args = parser.parse_args()

    rows = build_splits()
    check_splits(rows)
    with open(SPLITS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {SPLITS_CSV}")

    if args.images:
        download_images()
