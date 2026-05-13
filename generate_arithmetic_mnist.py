import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json
from itertools import product as iterproduct

# ── paths ──────────────────────────────────────────────────────────────────────


SAVE_DIR    = "data/generated/arithmetic_mnist"   # where the dataset will be written

CONCEPT_NAMES = ["first_digit", "second_digit", "operator"]

# ── helpers ────────────────────────────────────────────────────────────────────

def build_mnist_index(mnist_dataset):
    """Return a dict  digit -> [list of indices]  for fast sampling."""
    index = {d: [] for d in range(1, 10)}   # digits 1-9 (0 excluded)
    for i, (_, label) in enumerate(mnist_dataset):
        if label in index:
            index[label].append(i)
    return index


def make_canvas(img1, img2, op, font):
    """Paste  [digit | operator | digit]  on an 84×28 grayscale canvas."""
    canvas = Image.new("L", (84, 28), color=255)
    canvas.paste(img1, (0, 0))

    op_canvas = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(op_canvas)
    try:
        bbox = draw.textbbox((0, 0), op, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(op, font=font)
    draw.text(((28 - tw) // 2, (28 - th) // 2), op, fill=255, font=font)

    canvas.paste(op_canvas, (28, 0))
    canvas.paste(img2, (56, 0))
    return canvas


def compute_result(a, b, op):
    if op == '+': return a + b
    if op == '-': return a - b
    if op == 'x': return a * b
    if op == '/': return a / b


# ── main generator ─────────────────────────────────────────────────────────────

def generate_and_save_dataset(
        mnist_root  = "./data",
        train       = True,
        num_samples = 10_000,
        img_size    = 224,
        operators   = ('+', '-', 'x', '/'),
        save_dir    = SAVE_DIR,
):
    """
    Generate `num_samples` examples with **uniform** distribution over
        • left digit   (1-9)
        • right digit  (1-9)
        • operator
    and save them to disk.

    Each row is saved as:
        images/  <idx>.pt          – float32 tensor  C×H×W
        labels.pt                  – int64 tensor  (N, 4)
                                     columns: [left_digit, right_digit, op_idx, result_int]
        metadata.json              – operator list, split info, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ── load MNIST ──────────────────────────────────────────────────────────
    mnist = datasets.MNIST(root=mnist_root, train=train, download=True, transform=None)
    digit_index = build_mnist_index(mnist)   # digit -> list[idx]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    try:
        font = ImageFont.truetype("arial.ttf", 200)
    except:
        font = ImageFont.load_default()

    digits    = list(range(1, 10))          # 1 … 9
    op_list   = list(operators)
    op2idx    = {op: i for i, op in enumerate(op_list)}

    labels = np.zeros((num_samples, 4), dtype=np.float32)
    # columns: left_digit | right_digit | op_idx | result

    print(f"Generating {num_samples} samples …")
    for idx in range(num_samples):
        # ── uniform sampling ────────────────────────────────────────────────
        a  = random.choice(digits)
        b  = random.choice(digits)
        op = random.choice(op_list)

        # pick a random MNIST image for each digit
        i1 = random.choice(digit_index[a])
        i2 = random.choice(digit_index[b])
        img1, _ = mnist[i1]
        img2, _ = mnist[i2]

        canvas = make_canvas(img1, img2, op, font)
        x = transform(canvas)               # float32  C×H×W  in [0, 1]

        result = compute_result(a, b, op)

        # save image tensor
        torch.save(x, os.path.join(img_dir, f"{idx}.pt"))

        # store label row
        labels[idx] = [a, b, op2idx[op], result]

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{num_samples}")

    # ── save labels & metadata ───────────────────────────────────────────────
    torch.save(torch.tensor(labels, dtype=torch.float32),
               os.path.join(save_dir, "labels.pt"))

    meta = {
        "num_samples" : num_samples,
        "img_size"    : img_size,
        "operators"   : op_list,
        "op2idx"      : op2idx,
        "digits"      : digits,
        "split"       : "train" if train else "test",
        "label_cols"  : ["left_digit", "right_digit", "op_idx", "result"],
        "concept_names": CONCEPT_NAMES,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to  '{save_dir}'")
    print(f"  images/          {num_samples} × .pt files")
    print(f"  labels.pt        shape {labels.shape}   cols: {meta['label_cols']}")
    print(f"  metadata.json")
    return save_dir


# ── dataset class that reads from disk ─────────────────────────────────────────

class SavedArithmeticMNIST(Dataset):
    """
    Loads the pre-generated dataset from disk.

    __getitem__ returns:
        x  – image tensor  (C, H, W)
        c  – concept tensor [left_digit, right_digit, op_idx]   float32
        y  – result  (scalar float32)
    """
    def __init__(self, save_dir=SAVE_DIR):
        self.img_dir = os.path.join(save_dir, "images")
        self.labels  = torch.load(os.path.join(save_dir, "labels.pt"))  # (N, 4)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.load(os.path.join(self.img_dir, f"{idx}.pt"))
        row = self.labels[idx]                       # [left, right, op_idx, result]
        c = row[:3]                                  # concepts: left, right, op_idx
        y = row[3]                                   # result
        return x, c, y


# ── quick verification / notebook demo ────────────────────────────────────────

def verify_distribution(save_dir=SAVE_DIR, plot=True):
    import matplotlib.pyplot as plt

    labels = torch.load(os.path.join(save_dir, "labels.pt")).numpy()
    with open(os.path.join(save_dir, "metadata.json")) as f:
        meta = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Left digit", "Right digit", "Operator"]

    for col, (ax, title) in enumerate(zip(axes, titles)):
        vals = labels[:, col]
        if col < 2:                          # digit columns
            unique, counts = np.unique(vals, return_counts=True)
            ax.bar(unique.astype(int), counts)
            ax.set_xticks(unique.astype(int))
        else:                                # operator column
            unique, counts = np.unique(vals, return_counts=True)
            op_labels = [meta["operators"][int(i)] for i in unique]
            ax.bar(op_labels, counts)
        ax.set_title(title)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distribution_check.pdf"))
    plt.show()
    print("Distribution plot saved.")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. generate & save
    save_dir = generate_and_save_dataset(
        mnist_root  = "./data",
        train       = True,
        num_samples = 100_000,
        img_size    = 224,
        operators   = ('+', '-', 'x', '/'),
        save_dir    = SAVE_DIR,
    )

    # 2. verify distributions
    verify_distribution(save_dir)

    # 3. test the dataset class
    ds = SavedArithmeticMNIST(save_dir)
    x, c, y = ds[0]
    print(f"\nSample 0:")
    print(f"  image shape : {x.shape}")
    print(f"  concepts    : left={int(c[0])}, right={int(c[1])}, op_idx={int(c[2])}")
    print(f"  result      : {y.item()}")