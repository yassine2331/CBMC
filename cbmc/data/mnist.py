import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(data_dir: str = "data/raw", batch_size: int = 128, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_set = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
