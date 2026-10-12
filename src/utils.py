"""Small helpers shared by the whole project."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every source of randomness so experiments can be repeated exactly."""
    random.seed(seed)                     # Python's built-in random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch on CPU
    torch.cuda.manual_seed_all(seed)      # PyTorch on every GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuDNN picks the fastest conv algorithm at random by default; force the deterministic one.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Self-check: re-seeding must reproduce exactly the same random numbers.
    set_seed(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b, f"seeding is not reproducible: {a} != {b}"
    print("set_seed OK:", a)
