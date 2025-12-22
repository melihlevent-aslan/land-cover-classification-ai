import os
import rasterio
import numpy as np

# --- SAFETY FIX: WINDOWS GDAL PATHS ---
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"
os.environ['GDAL_DATA'] = os.path.join(conda_env_path, r"Library\share\gdal")
os.environ['GDAL_DRIVER_PATH'] = os.path.join(conda_env_path, r"Library\lib\gdalplugins")
os.environ['PATH'] = os.path.join(conda_env_path, r"Library\bin") + ";" + os.environ['PATH']
try:
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\bin"))
    os.add_dll_directory(os.path.join(conda_env_path, r"Library\lib\gdalplugins"))
except: pass

# --- CONFIGURATION ---
INPUT_DIR = 'data/dop/'

def inspect_file(path):
    print(f"\nInspecting: {os.path.basename(path)}")
    try:
        with rasterio.open(path) as src:
            print(f"   Driver: {src.driver}")
            print(f"   Size: {src.width} x {src.height}")
            print(f"   Bands (Channels): {src.count}")
            print(f"   Data Type: {src.dtypes}")
            
            # Read all bands to check values
            data = src.read()
            
            print(f"   Min Value: {data.min()}")
            print(f"   Max Value: {data.max()}")
            print(f"   Mean Value: {data.mean():.2f}")
            
            # Check just the first small corner to see pixel examples
            sample = data[:, 0:5, 0:5]
            print(f"   Sample Pixels (Top-Left corner):\n{sample}")
            
    except Exception as e:
        print(f"   ERROR reading file: {e}")

# --- RUN ---
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jp2')]
if files:
    inspect_file(os.path.join(INPUT_DIR, files[0]))
else:
    print("No .jp2 files found.")