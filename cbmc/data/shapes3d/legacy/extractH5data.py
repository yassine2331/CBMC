import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_h5_to_folder(h5_path, extract_path, batch_size=1000):
    os.makedirs(extract_path, exist_ok=True)
    imgs_dir = os.path.join(extract_path, "images")
    os.makedirs(imgs_dir, exist_ok=True)
    
    with h5py.File(h5_path, "r", swmr=True) as f:
        images_ds = f['images']  # HDF5 dataset object
        labels_ds = f['labels'] 
        total_imgs = len(images_ds)
        
        metadata = []
        
        # Process in chunks (batches)
        for start_idx in tqdm(range(0, total_imgs, batch_size), desc="Extracting Batches"):
            end_idx = min(start_idx + batch_size, total_imgs)
            
            # 1. Batch Read: This is the speed booster
            # Loading a block of images into RAM at once
            batch_images = images_ds[start_idx:end_idx]
            batch_labels = labels_ds[start_idx:end_idx]
            
            # 2. Process the loaded batch
            for i, img_np in enumerate(batch_images):
                global_idx = start_idx + i
                img_name = f"{global_idx:06d}.jpg"
                img_path = os.path.join(imgs_dir, img_name)
                
                # Save to disk
                Image.fromarray(img_np).save(img_path, quality=95)
                
                # Store metadata in list
                metadata.append({
                    "image_id": img_name, 
                    "label": batch_labels[i]
                })
            
        # Save metadata CSV
        print("Saving metadata CSV...")
        pd.DataFrame(metadata).to_csv(os.path.join(extract_path, "metadata.csv"), index=False)

if __name__ == "__main__":  
    # 1000 is a good balance for 12GB RAM
    extract_h5_to_folder("3dshapes.h5", "./shapes3d_extracted", batch_size=100)