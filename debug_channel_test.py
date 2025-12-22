import os
import rasterio
import torch
import numpy as np
import cv2

# --- SAFETY FIX ---
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"
os.environ['GDAL_DATA'] = os.path.join(conda_env_path, r"Library\share\gdal")
os.environ['GDAL_DRIVER_PATH'] = os.path.join(conda_env_path, r"Library\lib\gdalplugins") 
os.environ['PATH'] = os.path.join(conda_env_path, r"Library\bin") + ";" + os.environ['PATH']
try:
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\bin"))
except: pass

# --- CONFIGURATION ---
INPUT_FILE = 'data/dop/dop10rgbi_32_364_5620_1_nw_2025.jp2' 
MODEL_PATH = './best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_inference(name, tensor_data, model):
    print(f"\n--- TESTING: {name} ---")
    
    # Prepare Tensor
    tensor = torch.from_numpy(tensor_data).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = probs.squeeze().cpu().numpy()
        
    min_v, max_v, mean_v = mask.min(), mask.max(), mask.mean()
    print(f"   Probability Stats -> Min: {min_v:.4f}, Max: {max_v:.4f}")
    
    # Score: We want 'Spread' (Difference between Max and Min)
    score = max_v - min_v
    print(f"   Confidence Score: {score:.4f} (Higher is Better)")
    
    # Save image for visual check
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(f"debug_test_{name}.png", mask_vis)

def run_channel_tests():
    # Load Model
    print("Loading Model...")
    model = torch.load(MODEL_PATH, weights_only=False)
    model.to(DEVICE)
    model.eval()
    
    # Read Tile
    window = rasterio.windows.Window(2000, 2000, 512, 512)
    with rasterio.open(INPUT_FILE) as src:
        # Read as RGB
        base_tile = src.read([1, 2, 3], window=window)
        # Convert to CHW float
        base_tile = base_tile.astype('float32')
    
    # TEST 1: Standard RGB (0-255)
    test_inference("RGB_Raw", base_tile, model)

    # TEST 2: Swapped BGR (0-255)
    # Flip the channel dimension (0)
    bgr_tile = base_tile[::-1, :, :].copy() 
    test_inference("BGR_Swapped", bgr_tile, model)
    
    # TEST 3: Higher Threshold Simulation
    # If standard RGB is the best we can do, we might just need a higher cutoff
    print("\n--- ANALYSIS ---")
    print("Check the generated PNG files.")
    print("If 'RGB_Raw' looks good but just gray, we will use a threshold of 0.6.")

if __name__ == "__main__":
    run_channel_tests()