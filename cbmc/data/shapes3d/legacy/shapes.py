import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import re 

# ==========================================
# 1. Folder Driver Class (Replaces H5)
# ==========================================
class FolderDataSource:
    """
    Handles loading images from a folder and labels from a CSV.
    """
    def __init__(self, root_dir: str, metadata_file: str = "metadata.csv"):
        self.root_dir = Path(root_dir)
        self.imgs_dir = self.root_dir / "images"
        
        # Load metadata (CelebA style)
        metadata_path = self.root_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        else: 
            print(f"Loading metadata from {metadata_path}") 
            
        self.df = pd.read_csv(metadata_path)
        self.total_len = len(self.df)

    def get_image(self, idx: int) -> Image.Image:
        # Get filename from CSV and open
        img_name = self.df.iloc[idx, 0] 
        img_path = self.imgs_dir / img_name
        # Keeping it as PIL Image so transforms.ToTensor() works correctly
        return Image.open(img_path).convert("RGB")

    def get_label(self, idx: int):
        # labels are in columns 1 onwards

        target_raw_str = self.df.iloc[idx, 1:].values[0]

        # 1. Remove the brackets
        clean_str = target_raw_str.replace('[', '').replace(']', '')

        # 2. .split() without arguments automatically handles multiple spaces and newlines
        target_list = clean_str.split()

        # 3. (Optional) Convert strings to actual numbers
        target_numbers = [float(x) for x in target_list]

        return target_numbers

    def __len__(self):
        return self.total_len


# ==========================================
# 2. PyTorch Dataset Wrapper
# ==========================================
class Shapes3DDataset(Dataset):
    def __init__(
        self,
        source: FolderDataSource,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.source = source
        self.transform = transform
        self.indices = np.arange(len(source)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        
        # 1. Get Image
        img = self.source.get_image(real_idx)
        
        # 2. Apply Transform
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        # 3. Get Target - FIXED LINE BELOW
        target_raw = self.source.get_label(real_idx)
   
        try:
            # Convert to float first to handle string-encoded numbers
            target = torch.tensor(target_raw, dtype=torch.float32)
            #apply a transformation to the targets 
            #first 4 elements are position (continuous) between 0 and 1
            #fifth element is shape (categorical: 0,1,2)
            #sixth is continuous scale [-30 to 30]
            #mapping the continuses values to -1 to 1
            #target[0:4] = (target[0:4] * 2.0 - 1.0)*4
            target[5] = (target[5] / 30.0)

            target[0:3] = ((target[0:3]*10/9) * 2.0 - 1.0)
            target[3] = (((target[3]-0.75)*2) * 2.0 - 1.0)
            target[4] = (target[4] - 1.5)/1.5 
            

            
            
        except (ValueError, TypeError):
            # Fallback if the data is missing or truly non-numeric
            target = torch.tensor(0.0, dtype=torch.float32)
            raise ValueError(f"Non-numeric label encountered at index {real_idx}: {target_raw}")
          
        return img_t, target


# ==========================================
# 3. Lightning DataModule
# ==========================================
class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str, # Path to the EXTRACTED folder
        train_batch_size: int = 128, # Increased for better GPU utilization
        val_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True, # Set to True for GPU
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Initialize the single source
        full_source = FolderDataSource(self.data_dir)
        total_len = len(full_source)
        
        # Reproducible splitting
        rng = np.random.RandomState(self.seed)
        all_indices = np.arange(total_len)
        rng.shuffle(all_indices)

        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)

        # Use the same source object across datasets; Folder reading is thread-safe
        self.train_dataset = Shapes3DDataset(full_source, all_indices[:train_end], default_transform)
        self.val_dataset = Shapes3DDataset(full_source, all_indices[train_end:val_end], default_transform)
        self.test_dataset = Shapes3DDataset(full_source, all_indices[val_end:], default_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )