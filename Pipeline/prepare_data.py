import os
import sys

# --- FIX: ROBUST GDAL PATHS FOR WINDOWS (COPY OF WORKING CONFIG) ---
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"

# 1. Define paths
gdal_data = os.path.join(conda_env_path, r"Library\share\gdal")
gdal_plugins = os.path.join(conda_env_path, r"Library\lib\gdalplugins")
gdal_bin = os.path.join(conda_env_path, r"Library\bin") 

# 2. Set Environment Variables
os.environ['GDAL_DATA'] = gdal_data
os.environ['GDAL_DRIVER_PATH'] = gdal_plugins

# 3. Add 'bin' to the System PATH (Fixes Error 126)
os.environ['PATH'] = gdal_bin + ";" + os.environ['PATH']

# 4. Explicitly add DLL directories (Required for Python 3.8+)
try:
    os.add_dll_directory(gdal_bin)
    os.add_dll_directory(gdal_plugins)
except AttributeError:
    pass 

print("Environment paths set. Starting Tiling...")
# ------------------------------------------------------------------

import cv2
import numpy as np
import rasterio
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = "data/dop/"
MASK_DIR = "data/masks/"
OUTPUT_DIR = "dataset/"
TILE_SIZE = 512
STRIDE = 256  # Overlap of 50%

# Create output directories
for subset in ['train', 'val']:
    os.makedirs(os.path.join(OUTPUT_DIR, subset, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, subset, 'masks'), exist_ok=True)

def tile_image_and_mask(img_path, mask_path, filename_prefix):
    # Read Image (JP2)
    with rasterio.open(img_path) as src_img:
        # Read RGB channels (1, 2, 3)
        image = src_img.read([1, 2, 3]) 
        image = np.moveaxis(image, 0, -1) # Convert to (H, W, C)

    # Read Mask (TIF)
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1) # Read single channel

    h, w, _ = image.shape

    # Sliding Window Slicing
    count = 0
    for y in range(0, h - TILE_SIZE + 1, STRIDE):
        for x in range(0, w - TILE_SIZE + 1, STRIDE):
            
            # Crop
            img_tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Filter: Skip mostly empty tiles (less than 1% building)
            if np.sum(mask_tile) < (TILE_SIZE * TILE_SIZE * 0.01):
                continue
                
            # Random Split
            subset = 'val' if np.random.rand() < 0.2 else 'train'
            
            # Save Files
            save_name = f"{filename_prefix}_{y}_{x}.png"
            
            # Save Image (RGB -> BGR for OpenCV)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, subset, 'images', save_name),
                cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
            )
            
            # Save Mask
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, subset, 'masks', save_name),
                mask_tile
            )
            count += 1
            
    return count

# --- EXECUTION ---
jp2_files = glob(os.path.join(IMAGE_DIR, "*.jp2"))
print(f"Found {len(jp2_files)} source images.")

total_tiles = 0
for img_path in tqdm(jp2_files):
    filename = os.path.basename(img_path).replace(".jp2", "")
    mask_path = os.path.join(MASK_DIR, filename + "_mask.tif")
    
    if os.path.exists(mask_path):
        try:
            num = tile_image_and_mask(img_path, mask_path, filename)
            total_tiles += num
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Warning: Mask not found for {filename}")

print(f"Done! Generated {total_tiles} tiles in '{OUTPUT_DIR}'")