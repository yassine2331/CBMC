import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, List, Tuple, Union, Callable, Sequence

# ==========================================
# 1. Default Transformation
# ==========================================
def get_default_transform() -> transforms.Compose:
    """
    Returns a standard transformation:
    1. Converts HWC [0,255] numpy array to CHW [0.0, 1.0] FloatTensor.
    2. Normalizes to [-1, 1] (mean=0.5, std=0.5).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# ==========================================
# 2. File Reader Class (The "Driver")
# ==========================================
class H5DataSource:
    """
    Handles low-level HDF5 file operations.
    - Manages file handles (lazy loading).
    - Handles in-memory caching.
    - Safe for multiprocessing if instantiated per process (handled by Dataset).
    """
    def __init__(self, h5_path: str, image_key: str = "images", label_key: Optional[str] = None, in_memory: bool = False):
        self.h5_path = h5_path
        self.image_key = image_key
        self._user_label_key = label_key
        self.in_memory = in_memory
        
        self._h5_file = None
        self._images_cache = None
        self._labels_cache = None
        
        # Open briefly to validate keys and get length
        with h5py.File(self.h5_path, "r") as f:
            if self.image_key not in f:
                raise KeyError(f"Image key '{self.image_key}' not found in {h5_path}.")
            
            self.total_len = f[self.image_key].shape[0]

            # Auto-infer label key if missing
            if self._user_label_key is None:
                for candidate in ("labels", "label", "latents_values", "latents_classes"):
                    if candidate in f:
                        self._user_label_key = candidate
                        break
            
            # Preload if requested
            if self.in_memory:
                self._images_cache = f[self.image_key][:]
                if self._user_label_key:
                    self._labels_cache = f[self._user_label_key][:]

    def _ensure_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")

    def get_image(self, idx: int) -> np.ndarray:
        if self._images_cache is not None:
            return self._images_cache[idx]
        self._ensure_open()
        return self._h5_file[self.image_key][idx]

    def get_label(self, idx: int):
        if not self._user_label_key:
            return None
        if self._labels_cache is not None:
            return self._labels_cache[idx]
        self._ensure_open()
        return self._h5_file[self._user_label_key][idx]

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __len__(self):
        return self.total_len

# ==========================================
# 3. Dataset Class (The "Transformer")
# ==========================================
class Shapes3DDataset(Dataset):
    """
    PyTorch interface. 
    - Wraps H5DataSource.
    - Applies Transforms.
    - Handles Subsetting (via indices).
    """
    def __init__(
        self,
        source: H5DataSource,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.source = source
        self.transform = transform
        self.target_transform = target_transform
        
        # If no indices provided, use the whole dataset
        if indices is None:
            self.indices = np.arange(len(source))
        else:
            self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int, float]]:
        # Map logical index (0..subset_len) to physical index (0..total_file_len)
        real_idx = int(self.indices[idx])
        
        # 1. Get Raw Data
        img_data = self.source.get_image(real_idx) # Returns HWC numpy
        img_data = np.asarray(img_data)
        
        # 2. Apply Transform
        if self.transform:
            img_t = self.transform(img_data)
        else:
            # Fallback if no transform provided
            if img_data.ndim == 2: img_data = np.expand_dims(img_data, -1)
            img_t = torch.from_numpy(img_data).permute(2,0,1).float() / 255.0

        # 3. Get Target
        target = self.source.get_label(real_idx)
        if self.target_transform and target is not None:
            target = self.target_transform(target)
            
        # 4. Standardize Target format
        if target is not None and not isinstance(target, torch.Tensor):
            try:
                target = torch.tensor(target)
            except:
                pass

        return img_t, target

    def __getstate__(self):
        # Pickling safety for DataLoaders
        state = self.__dict__.copy()
        # Ensure the source's internal file handle is closed before pickling
        self.source.close() 
        return state

# ==========================================
# 4. Data Module (The "Manager")
# ==========================================
class Shapes3DDataModule:
    """
    High-level manager.
    - Handles Train/Val/Test Splitting with FIXED SEED.
    - Creates DataLoaders.
    - Assigns default transforms if none provided.
    """
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        # Ratios for Train and Val (Test is remainder)
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        transform: Optional[Callable] = None,
        num_workers: int = 2,
        in_memory: bool = False
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_workers = num_workers
        self.in_memory = in_memory
        
        # Use default transform if None provided
        self.transform = transform if transform is not None else get_default_transform()

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Returns (train_loader, val_loader, test_loader)"""
        
        # 1. Initialize Source (Lightweight check)
        temp_source = H5DataSource(self.data_path)
        total_len = len(temp_source)
        temp_source.close()

        # 2. Create Reproducible Split Indices
        # Using numpy RandomState ensures this is isolated from global random seed
        rng = np.random.RandomState(self.seed)
        all_indices = np.arange(total_len)
        rng.shuffle(all_indices)

        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))

        train_indices = all_indices[:train_end]
        val_indices = all_indices[train_end:val_end]
        test_indices = all_indices[val_end:]

        # 3. Create Datasets Factory
        def create_loader(indices, shuffle):
            source = H5DataSource(self.data_path, in_memory=self.in_memory)
            ds = Shapes3DDataset(
                source=source, 
                indices=indices, 
                transform=self.transform
            )
            return DataLoader(
                ds, 
                batch_size=self.batch_size, 
                shuffle=shuffle, 
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

        train_loader = create_loader(train_indices, shuffle=True)
        val_loader = create_loader(val_indices, shuffle=False)
        test_loader = create_loader(test_indices, shuffle=False)
        
        print(f"Dataset Split (Seed={self.seed}): {len(train_indices)} Train, {len(val_indices)} Val, {len(test_indices)} Test")
        return train_loader, val_loader, test_loader

# ==========================================
# Example Usage & Visualization
# ==========================================
if __name__ == "__main__":
    # Ensure file exists for the test
    dataset_path = '3dshapes.h5'
    
    if os.path.exists(dataset_path):
        # 1. Initialize the Module with 3-way split
        dm = Shapes3DDataModule(
            data_path=dataset_path,
            batch_size=4,       # Small batch for visualization
            train_ratio=0.8,    # 80% Train
            val_ratio=0.1,      # 10% Val
            # Remaining 10% is Test
            seed=42
        )

        # 2. Get Loaders
        train_loader, val_loader, test_loader = dm.get_loaders()

        # 3. Visualize 4 examples from the Test Set
        print("\n--- Visualizing Test Examples ---")
        
        # Get one batch
        imgs, labels = next(iter(test_loader))
        
        # Prepare plot
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        # We need to de-normalize for display because our default transform creates [-1, 1]
        # (val * 0.5) + 0.5 maps [-1, 1] back to [0, 1]
        
        for i in range(4):
            ax = axes[i]
            
            # De-normalize: C,H,W -> H,W,C
            img_tensor = imgs[i].permute(1, 2, 0) # to HWC
            img_display = (img_tensor * 0.5) + 0.5
            img_display = img_display.clamp(0, 1).numpy()
            
            ax.imshow(img_display)
            ax.axis('off')
            
            # Formatting label string
            # 3DShapes labels usually correspond to [floor_hue, wall_hue, object_hue, scale, shape, orientation]
            label_vals = labels[i].numpy()
            label_str = "\n".join([f"{v:.2f}" for v in label_vals])
            ax.set_title(f"Target:\n{label_str}", fontsize=9)

        plt.suptitle(f"4 Random Examples from Test Set (Seed={dm.seed})")
        plt.tight_layout()
        plt.show()

    else:
        print(f"File {dataset_path} not found. Please provide a valid h5 file.")