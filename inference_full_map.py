import os
import sys

# --- SAFETY FIX: WINDOWS GDAL PATHS ---
# (Standard fix for your environment)
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"
os.environ['GDAL_DATA'] = os.path.join(conda_env_path, r"Library\share\gdal")
os.environ['GDAL_DRIVER_PATH'] = os.path.join(conda_env_path, r"Library\lib\gdalplugins")
os.environ['PATH'] = os.path.join(conda_env_path, r"Library\bin") + ";" + os.environ['PATH']
try:
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\bin"))
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\lib\gdalplugins"))
except: pass

import torch
import numpy as np
import cv2
import rasterio
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = './best_model.pth'
INPUT_DIR = 'data/dop/'          # Where your big .jp2 files are
OUTPUT_DIR = 'final_maps/'       # Where the results will go
TILE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_large_image(model, img_path, save_path):
    """
    Scans a large geospatial image and saves the prediction as a GeoTIFF.
    """
    print(f"Reading: {os.path.basename(img_path)}...")
    
    with rasterio.open(img_path) as src:
        # Get metadata (Coordinate System, Transform, etc.)
        profile = src.profile
        w, h = src.width, src.height
        
        # Update profile for output (We are writing a 1-channel mask, not 3-channel RGB)
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            driver='GTiff'
        )
        
        # Read the full image into memory (RAM is cheap, VRAM is expensive)
        # We read first 3 bands (RGB)
        full_image = src.read([1, 2, 3]) 
        # Move axis to (H, W, C) for OpenCV/PyTorch
        full_image = np.moveaxis(full_image, 0, -1)
    
    # Create an empty canvas for the result
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    
    # --- SLIDING WINDOW INFERENCE ---
    # We iterate through the image in chunks
    print(f"   Scanning {w}x{h} pixels...")
    
    model.eval()
    with torch.no_grad():
        for y in tqdm(range(0, h, TILE_SIZE), desc="   Processing Rows"):
            for x in range(0, w, TILE_SIZE):
                
                # 1. Cut the tile
                # Handle edge cases where the tile goes off the map
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                
                tile = full_image[y:y_end, x:x_end]
                
                # Pad if tile is smaller than 512 (at the edges)
                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
                # 2. Pre-process for AI
                # RGB -> Tensor (CHW) -> Normalize
                input_tensor = tile.transpose(2, 0, 1).astype('float32')
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
                
                # 3. Predict
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                
                # 4. Post-process
                # Remove batch dim, move to CPU, convert to binary
                mask = probs.squeeze().cpu().numpy()
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # 5. Place on Canvas (Crop padding if necessary)
                real_h = y_end - y
                real_w = x_end - x
                prediction_map[y:y_end, x:x_end] = mask_binary[:real_h, :real_w]

    # --- SAVE AS GEOTIFF ---
    print(f"   Saving GeoTIFF to {save_path}...")
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(prediction_map, 1) # Write to band 1
        
    print("   Done.")

# --- EXECUTION ---
print("--- FULL MAP INFERENCE ---")
print(f"Loading Model: {MODEL_PATH}")

# Fix for PyTorch 2.6 security warning
model = torch.load(MODEL_PATH, weights_only=False)
model = model.to(DEVICE)

# Find images
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jp2')]

if not image_files:
    # Fallback to TIF if you converted them earlier
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')]

print(f"Found {len(image_files)} maps to process.")

for img_file in image_files:
    in_path = os.path.join(INPUT_DIR, img_file)
    out_path = os.path.join(OUTPUT_DIR, img_file.replace('.jp2', '_prediction.tif').replace('.tif', '_prediction.tif'))
    
    predict_large_image(model, in_path, out_path)

print("\nAll maps processed! Project Complete.")