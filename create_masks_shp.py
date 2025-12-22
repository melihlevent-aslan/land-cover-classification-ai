import os
import sys

# --- FIX: ROBUST GDAL PATHS FOR WINDOWS ---
conda_env_path = r"C:\Users\NewFo\anaconda3\envs\spatial_ai"

# 1. Define paths
gdal_data = os.path.join(conda_env_path, r"Library\share\gdal")
gdal_plugins = os.path.join(conda_env_path, r"Library\lib\gdalplugins")
gdal_bin = os.path.join(conda_env_path, r"Library\bin") # <--- CRITICAL MISSING LINK

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
    pass # For older Python versions

print("Environment paths set. Attempting to load rasterio...")

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

# --- CONFIGURATION ---
IMAGE_DIR = "data/dop/"      # Folder with your .jp2 images
MASK_DIR = "data/masks/"     # Output folder
SHAPEFILE_PATH = "data/shapefiles/2022_hu_shp.shp"  

os.makedirs(MASK_DIR, exist_ok=True)

def generate_mask_from_shapefile(jp2_path, gdf_buildings):
    print(f"Processing: {jp2_path}...")
    
    try:
        with rasterio.open(jp2_path) as src:
            out_shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs
            
            # 1. Spatial Filter: Get only buildings that intersect with this image
            from shapely.geometry import box
            bounds = src.bounds
            bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Reproject bounding box to match Shapefile CRS if needed
            if gdf_buildings.crs != crs:
                # Temporary reproject vector to image CRS for filtering
                pass 

            # Filter the buildings (Spatial Indexing makes this fast)
            gdf_local = gdf_buildings.to_crs(crs).clip(bbox)
            
            if gdf_local.empty:
                print("  No buildings found in this image area.")
                mask = np.zeros(out_shape, dtype='uint8')
            else:
                # 2. Rasterize
                print(f"  Found {len(gdf_local)} buildings. Rasterizing...")
                shapes = ((geom, 1) for geom in gdf_local.geometry)
                
                mask = rasterize(
                    shapes=shapes,
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype='uint8'
                )

            # 3. Save Mask
            filename = os.path.basename(jp2_path).replace(".jp2", "_mask.tif")
            out_path = os.path.join(MASK_DIR, filename)
            
            with rasterio.open(
                out_path, 'w',
                driver='GTiff',
                height=out_shape[0],
                width=out_shape[1],
                count=1,
                dtype='uint8',
                crs=crs,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(mask, 1)
                
            print(f"  Saved: {out_path}")

    except Exception as e:
        print(f"Error processing {jp2_path}: {e}")

# --- EXECUTION ---
print("Loading Shapefile...")
if not os.path.exists(SHAPEFILE_PATH):
    print(f"ERROR: Shapefile not found at {SHAPEFILE_PATH}")
    sys.exit()

gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"Loaded {len(gdf)} polygons.")

# 2. Process all images
jp2_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jp2')]

for jp2 in jp2_files:
    generate_mask_from_shapefile(os.path.join(IMAGE_DIR, jp2), gdf)