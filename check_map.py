import rasterio
import numpy as np
import cv2
import os

# Pick the first file found in final_maps
map_dir = "final_maps"
files = [f for f in os.listdir(map_dir) if f.endswith(".tif")]

if not files:
    print("No maps found!")
else:
    target_file = os.path.join(map_dir, files[0])
    print(f"Inspecting: {target_file}...")

    with rasterio.open(target_file) as src:
        data = src.read(1) # Read the first channel
        
        # Count values
        unique, counts = np.unique(data, return_counts=True)
        stats = dict(zip(unique, counts))
        
        print("\n--- PIXEL STATISTICS ---")
        if 1 in stats:
            print(f"✅ FOUND BUILDINGS! Pixel count: {stats[1]:,}")
        else:
            print("❌ No buildings found (Pure Black).")
            
        print(f"   Background pixels: {stats.get(0, 0):,}")
        
        # Create a VISIBLE version (multiply by 255)
        visible_img = (data * 255).astype(np.uint8)
        cv2.imwrite("proof_of_success.png", visible_img)
        print("\nSaved 'proof_of_success.png'. Open it to see the white buildings!")