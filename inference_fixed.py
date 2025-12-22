import os
import sys

# --- SAFETY FIX: WINDOWS GDAL PATHS ---
# This block is critical for finding the .jp2 driver on Windows
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"
os.environ['GDAL_DATA'] = os.path.join(conda_env_path, r"Library\share\gdal")
# vvv THIS WAS MISSING vvv
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
INPUT_DIR = 'data/dop/'
OUTPUT_DIR = 'final_maps_fixed/' 
TILE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_large_image(model, img_path, save_path):
    print(f"Reading: {os.path.basename(img_path)}...")
    
    with rasterio.open(img_path) as src:
        profile = src.profile
        w, h = src.width, src.height
        
        # Read the full image
        full_image = src.read([1, 2, 3]) 
        # (C, H, W) -> (H, W, C)
        full_image = np.moveaxis(full_image, 0, -1)
        
        # --- CRITICAL FIX: 16-BIT to 8-BIT CONVERSION ---
        # If the image uses 16-bit values (0-65535), scale them down to 8-bit (0-255)
        if full_image.max() > 255:
            print("   ⚠️ Detected 16-bit image. Scaling down to 8-bit...")
            # Scale 0-65535 to 0-255
            full_image = (full_image / 256).astype(np.uint8)
        else:
            full_image = full_image.astype(np.uint8)
        
        # Update profile for output
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', driver='GTiff')
    
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    
    print(f"   Scanning {w}x{h} pixels...")
    model.eval()
    
    with torch.no_grad():
        for y in tqdm(range(0, h, TILE_SIZE), desc="   Processing"):
            for x in range(0, w, TILE_SIZE):
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                
                tile = full_image[y:y_end, x:x_end]
                
                # Pad edges
                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
                # Preprocess
                input_tensor = tile.transpose(2, 0, 1).astype('float32')
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
                
                # Predict
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                mask = probs.squeeze().cpu().numpy()
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Save to canvas
                real_h = y_end - y
                real_w = x_end - x
                prediction_map[y:y_end, x:x_end] = mask_binary[:real_h, :real_w]

    print(f"   Saving Fixed GeoTIFF...")
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(prediction_map, 1)

# --- EXECUTION ---
print("--- FULL MAP INFERENCE (16-BIT FIX) ---")

# Fix for PyTorch 2.6 security warning
model = torch.load(MODEL_PATH, weights_only=False)
model = model.to(DEVICE)

image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jp2')]

for img_file in image_files:
    in_path = os.path.join(INPUT_DIR, img_file)
    # Save to the new FIXED folder
    out_path = os.path.join(OUTPUT_DIR, img_file.replace('.jp2', '_prediction.tif'))
    
    predict_large_image(model, in_path, out_path)

print("\nAll maps fixed.")