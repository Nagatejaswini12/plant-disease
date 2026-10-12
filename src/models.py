"""The four models we compare: our own small CNN (baseline) and three pretrained timm models."""

import timm
import torch
import torch.nn as nn

from src.config import MODELS

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions (each followed by BatchNorm + ReLU), then 2x2 max-pool halves the size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Baseline trained from scratch: 4 conv blocks (224 -> 14 px, 3 -> 256 channels), then a linear classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling: one number per channel
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a model by its short name from config.MODELS."""
    timm_id = MODELS[name]["timm_id"]
    if timm_id is None:
        return CustomCNN(num_classes)
    # timm downloads ImageNet weights and swaps the 1000-class head for a new num_classes head.
    return timm.create_model(timm_id, pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a best.pt saved by train.py. Returns (model in eval mode, class names)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)  # weights come from the file
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


def normalization(model: nn.Module):
    """Mean/std the model expects. Pretrained models must see inputs normalised like their training data."""
    cfg = getattr(model, "pretrained_cfg", None)
    return (cfg["mean"], cfg["std"]) if cfg else (IMAGENET_MEAN, IMAGENET_STD)
