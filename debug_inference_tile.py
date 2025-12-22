import os
import rasterio
import cv2
import torch
import numpy as np

# --- SAFETY FIX ---
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"
os.environ['GDAL_DATA'] = os.path.join(conda_env_path, r"Library\share\gdal")
os.environ['GDAL_DRIVER_PATH'] = os.path.join(conda_env_path, r"Library\lib\gdalplugins") 
os.environ['PATH'] = os.path.join(conda_env_path, r"Library\bin") + ";" + os.environ['PATH']
try:
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\bin"))
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\lib\gdalplugins"))
except: pass

# --- CONFIGURATION ---
INPUT_FILE = 'data/dop/dop10rgbi_32_364_5620_1_nw_2025.jp2' # Pick the first file
MODEL_PATH = './best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_debug_test():
    print(f"--- DEBUGGING TILE INFERENCE ---")
    
    # 1. Read a specific chunk (e.g., from the middle of the map)
    # We read a 512x512 window at offset x=2000, y=2000
    window = rasterio.windows.Window(2000, 2000, 512, 512)
    
    print(f"1. Reading tile from {INPUT_FILE}...")
    with rasterio.open(INPUT_FILE) as src:
        # Read RGB bands (1, 2, 3)
        tile = src.read([1, 2, 3], window=window)
        
        # Convert (3, 512, 512) -> (512, 512, 3)
        tile = np.moveaxis(tile, 0, -1)
        
        print(f"   Tile Shape: {tile.shape}")
        print(f"   Tile Min: {tile.min()}, Max: {tile.max()}, Mean: {tile.mean():.2f}")
        
        # Save INPUT to verify colors
        # OpenCV expects BGR, so we swap RGB -> BGR for saving
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_input.png", tile_bgr)
        print("   Saved 'debug_input.png'. CHECK THIS IMAGE!")

    # 2. Load Model
    print("2. Loading Model...")
    model = torch.load(MODEL_PATH, weights_only=False)
    model.to(DEVICE)
    model.eval()

    # 3. Predict
    print("3. Running AI...")
    with torch.no_grad():
        # Preprocess: (512, 512, 3) -> (1, 3, 512, 512)
        input_tensor = tile.transpose(2, 0, 1).astype('float32')
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
        
        print(f"   Input Tensor Stats -> Min: {input_tensor.min().item()}, Max: {input_tensor.max().item()}")
        
        # Forward Pass
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        
        # Postprocess
        mask = probs.squeeze().cpu().numpy()
        
        print(f"   Prediction Stats -> Min: {mask.min():.4f}, Max: {mask.max():.4f}, Mean: {mask.mean():.4f}")
        
        # Save OUTPUT
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite("debug_prediction.png", mask_vis)
        print("   Saved 'debug_prediction.png'. CHECK THIS IMAGE!")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        run_debug_test()
    else:
        print(f"Error: Could not find {INPUT_FILE}")