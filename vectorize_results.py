import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = 'final_maps_threshold/'
OUTPUT_DIR = 'final_vectors_threshold/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def vectorize_map(tif_path, output_path):
    print(f"Vectorizing: {os.path.basename(tif_path)}...")
    
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # --- FIX: STRICT MASKING ---
        # Only create shapes where the image value is EXACTLY 1 (Building)
        # We explicitly exclude 0 (Background)
        mask = (image == 1)
        
        # If the image is all black, skip it
        if not mask.any():
            print("   Skipping: No buildings detected.")
            return

        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=transform)
            )
        )
        
        # 2. Convert to Geometries
        geoms = list(results)
        
        if not geoms:
            print("   Warning: Generator empty.")
            return

        # 3. Create GeoDataFrame
        polygons = [shape(g['geometry']) for g in geoms]
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
        
        # --- FIX: FILTER TINY NOISE ---
        # Real houses are rarely smaller than 10 square meters. 
        # Removing "pixel noise" (< 5m²) cleans up those 0.005 values.
        gdf['area'] = gdf.geometry.area
        initial_count = len(gdf)
        gdf = gdf[gdf['area'] > 5.0]  # Filter out noise < 5m²
        filtered_count = len(gdf)
        
        print(f"   Removed {initial_count - filtered_count} tiny noise artifacts.")
        
        # 4. Save
        # Drop the temporary 'area' column if you want, or keep it
        print(f"   Saving {len(gdf)} valid building polygons to {output_path}...")
        gdf.to_file(output_path, driver="GPKG")

# --- EXECUTION ---
print("--- RASTER TO VECTOR CONVERSION ---")

tif_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')]

if not tif_files:
    print("No TIFF maps found! Run inference_full_map.py first.")
else:
    for tif in tif_files:
        in_path = os.path.join(INPUT_DIR, tif)
        out_path = os.path.join(OUTPUT_DIR, tif.replace('.tif', '.gpkg'))
        
        vectorize_map(in_path, out_path)

print("\nDone! Drag and drop the files in 'final_vectors' into QGIS.")