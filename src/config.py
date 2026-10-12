"""Project-wide settings. Every other file imports from here, so one change applies everywhere."""

from pathlib import Path

# Reproducibility: the same seed gives the same split and the same training run.
SEED = 42

# Folders (all relative to the repo root, so it works on Colab and locally).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                  # downloaded files go here (git-ignored)
SPLITS_CSV = DATA_DIR / "splits.csv"      # our 70/15/15 split (committed to git)

# PlantVillage on the Hugging Face Hub (official copy by the dataset author).
HF_DATASET_REPO = "mohanty/PlantVillage"

# Split ratios. Train gets whatever is left over (0.70).
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------- Training (Phase 2) ----------------
RUNS_DIR = ROOT / "runs"          # checkpoints + logs; on Colab we pass a Google Drive folder instead
REPORTS_DIR = ROOT / "reports"             # Phase 3: comparison table, charts, confusion matrices
IMG_SIZE = 224                    # same input size for every model, so the comparison is fair
BATCH_SIZE = 64
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1                 # LR rises linearly for 1 epoch, then follows a cosine curve down
LABEL_SMOOTHING = 0.1             # softens one-hot targets, a cheap regulariser
SMOKE_IMAGES = {"train": 64, "val": 32, "test": 32}  # smoke test: tiny fixed random sample, 1 epoch

# Per-model settings, in training order.
#   timm_id: pretrained weights from the timm library (None = our own CNN, trained from scratch)
#   lr: from-scratch training tolerates a higher LR; fine-tuning pretrained weights needs a small one
#   patience: early stopping - stop after this many epochs without a better validation macro-F1
MODELS = {
    "custom_cnn":      {"timm_id": None,                                          "lr": 1e-3, "epochs": 30, "patience": 5},
    "efficientnetv2_s": {"timm_id": "tf_efficientnetv2_s.in21k_ft_in1k",          "lr": 3e-4, "epochs": 10, "patience": 3},
    "convnext_tiny":   {"timm_id": "convnext_tiny.fb_in22k_ft_in1k",              "lr": 2e-4, "epochs": 10, "patience": 3},
    "vit_small":       {"timm_id": "vit_small_patch16_224.augreg_in21k_ft_in1k",  "lr": 1e-4, "epochs": 10, "patience": 3},
}
