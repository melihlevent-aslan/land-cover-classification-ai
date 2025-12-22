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
except: pass

# --- CONFIGURATION ---
# Pick the same file
INPUT_FILE = 'data/dop/dop10rgbi_32_364_5620_1_nw_2025.jp2' 
MODEL_PATH = './best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_test():
    print(f"--- TESTING NORMALIZATION FIX ---")
    
    # Read the same window
    window = rasterio.windows.Window(2000, 2000, 512, 512)
    
    with rasterio.open(INPUT_FILE) as src:
        tile = src.read([1, 2, 3], window=window)
        tile = np.moveaxis(tile, 0, -1) # (512, 512, 3)

    model = torch.load(MODEL_PATH, weights_only=False)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Preprocess
        input_tensor = tile.transpose(2, 0, 1).astype('float32')
        
        # --- THE FIX IS HERE ---
        # Divide by 255.0 to squeeze values between 0.0 and 1.0
        input_tensor = input_tensor / 255.0
        # -----------------------

        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)
        
        print(f"   Input Tensor Stats -> Min: {input_tensor.min():.4f}, Max: {input_tensor.max():.4f}")
        print("   (Should now be between 0.0 and 1.0)")
        
        # Predict
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = probs.squeeze().cpu().numpy()
        
        print(f"   Prediction Stats -> Min: {mask.min():.4f}, Max: {mask.max():.4f}")
        
        # Save Output
        # Now we expect solid Black (0) and Solid White (255)
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite("debug_prediction_FIXED.png", mask_vis)
        print("   Saved 'debug_prediction_FIXED.png'. Look at it!")

if __name__ == "__main__":
    run_test()