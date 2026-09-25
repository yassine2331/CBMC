"""
Arithmetic MNIST dataset loader.
Source: generate_data_numbers.py

Each sample is a 3-panel image: [digit_a | operator | digit_b]
Returns (image, concepts, target) where:
    image    : (3, img_size, img_size) tensor — grayscale→RGB, normalized
    concepts : (2,) float tensor — [digit_a, digit_b], values in [1,9]
    target   : scalar float — result of the operation (a+b, a-b, a*b, a/b)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont


class ArithmeticMNISTDataset(Dataset):
    def __init__(
        self,
        mnist_root: str = "data/generated/MNIST",
        train: bool = True,
        num_samples: int = 10000,
        img_size: int = 64,
        operators: tuple = ('+', '-', 'x', '/'),
        digits=None,            # None = all 1-9; e.g. (1, 2, 3) restricts to those digits
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.img_size    = img_size
        self.operators   = operators

        rng = random.Random(seed)
        self.operator_list = [rng.choice(operators) for _ in range(num_samples)]

        self.mnist = datasets.MNIST(
            root=mnist_root, train=train, download=True, transform=None
        )

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # Pre-compute per-digit index lists for fast, targeted sampling.
        allowed = sorted(set(digits) if digits is not None else range(1, 10))
        self._allowed_digits = allowed
        self._digit_indices: dict = {d: [] for d in allowed}
        for i in range(len(self.mnist)):
            label = self.mnist.targets[i].item()
            if label in self._digit_indices:
                self._digit_indices[label].append(i)

        # Don't store font object here — PIL FreeTypeFont can't be pickled
        # across DataLoader worker processes. Load it lazily in __getitem__.

    def __len__(self):
        return self.num_samples

    def _get_font(self):
        try:
            return ImageFont.truetype("arial.ttf", 20)
        except Exception:
            return ImageFont.load_default()

    def __getitem__(self, idx):
        rng = random.Random(idx)

        # Sample two digits from the allowed set using pre-computed indices
        d_a   = rng.choice(self._allowed_digits)
        img1, a = self.mnist[rng.choice(self._digit_indices[d_a])]
        d_b   = rng.choice(self._allowed_digits)
        img2, b = self.mnist[rng.choice(self._digit_indices[d_b])]

        op = self.operator_list[idx]

        if   op == '+': result = float(a + b)
        elif op == '-': result = float(a - b)
        elif op == 'x': result = float(a * b)
        elif op == '/': result = float(a) / float(b)

        # Build 84×28 canvas: [digit | operator | digit]
        canvas = Image.new("L", (84, 28), color=255)
        canvas.paste(img1, (0, 0))

        font = self._get_font()
        op_canvas = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(op_canvas)
        try:
            bbox = draw.textbbox((0, 0), op, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(op, font=font)
        draw.text(((28 - tw) // 2, (28 - th) // 2), op, fill=255, font=font)
        canvas.paste(op_canvas, (28, 0))
        canvas.paste(img2, (56, 0))

        x = self.transform(canvas)
        concepts = torch.tensor([float(a), float(b)], dtype=torch.float32)
        target   = torch.tensor(result, dtype=torch.float32)
        return x, concepts, target


def get_arithmetic_mnist(
    mnist_root: str  = "data/generated/MNIST",
    img_size: int    = 64,
    operators: tuple = ('+','x'),
    digits=None,
    num_train: int   = 10000,
    num_test: int    = 2000,
    batch_size: int  = 64,
    num_workers: int = 2,
    seed: int        = 42,
):
    train_set = ArithmeticMNISTDataset(
        mnist_root=mnist_root, train=True,
        num_samples=num_train, img_size=img_size,
        operators=operators, digits=digits, seed=seed,
    )
    test_set = ArithmeticMNISTDataset(
        mnist_root=mnist_root, train=False,
        num_samples=num_test, img_size=img_size,
        operators=operators, digits=digits, seed=seed + 1,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
