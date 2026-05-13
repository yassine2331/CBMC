"""
Pendulum dataset loader.
Images are 96x96 RGBA PNGs. Labels are encoded in the filename:
    a_<v0>_<v1>_<v2>_<v3>.png  →  4 continuous values

Returns (image, concepts, target) where:
    image    : (4, img_size, img_size) RGBA tensor
    concepts : (4,) float tensor — the 4 physical values (same as target)
    target   : (4,) float tensor — same as concepts (physical values ARE the task)

label_mean and label_std are returned for normalizing targets during training.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PendulumDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root      = root
        self.transform = transform
        self.files     = sorted(f for f in os.listdir(root) if f.endswith(".png"))

        # Parse labels from filenames: a_v0_v1_v2_v3.png
        labels = []
        for f in self.files:
            parts = f.replace(".png", "").split("_")
            # parts[0] is 'a', parts[1:] are the values
            # handle negative numbers: split on '_' after prefix
            vals = [float(v) for v in parts[1:] if v != ""]
            labels.append(vals[:4])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        concepts = self.labels[idx]
        target   = self.labels[idx]   # concepts == target for pendulum
        return img, concepts, target


def get_pendulum(
    data_dir: str = "data/generated/pendulum",
    batch_size: int = 64,
    num_workers: int = 2,
    img_size: int = 64,             # resize to 64x64 to reduce memory
):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),       # (4, H, W), values in [0,1]
    ])

    train_set = PendulumDataset(os.path.join(data_dir, "train"), transform=transform)
    test_set  = PendulumDataset(os.path.join(data_dir, "test"),  transform=transform)

    # Compute per-dimension mean/std from training labels for normalization
    label_mean = train_set.labels.mean(0)
    label_std  = train_set.labels.std(0).clamp(min=1e-6)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, label_mean, label_std
