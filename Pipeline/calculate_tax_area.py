import os
import geopandas as gpd
import pandas as pd

# --- CONFIGURATION ---
VECTOR_DIR = 'final_vectors_threshold/'
OUTPUT_CSV = 'final_tax_report.csv'

# Standard "Rainwater Tax" in NRW (Approximate example)
# Price per square meter of sealed surface
TAX_RATE_PER_SQM = 0.85  # Euros per m² (Example value)

def calculate_areas(gpkg_path):
    print(f"\nAnalyzing: {os.path.basename(gpkg_path)}...")
    
    # 1. Load the Vector Data
    gdf = gpd.read_file(gpkg_path)
    
    # 2. Check Coordinate System (CRS)
    # We need a metric system (meters), not Degrees (Lat/Lon)
    print(f"   Coordinate System: {gdf.crs}")
    
    if gdf.crs.is_geographic:
        print("   WARNING: Map is in Degrees (Lat/Lon). Reprojecting to UTM 32N for accurate area...")
        gdf = gdf.to_crs(epsg=25832) # Standard for Germany
    
    # 3. Calculate Area
    # In UTM, .area returns square meters directly
    gdf['area_m2'] = gdf.geometry.area
    
    # 4. Filter Noise (Optional)
    # Remove tiny specks (e.g., < 5m²) that might be artifacts
    original_count = len(gdf)
    gdf = gdf[gdf['area_m2'] > 5.0]
    filtered_count = len(gdf)
    print(f"   Filtered {original_count - filtered_count} tiny artifacts (<5m²).")
    
    # 5. Calculate Tax
    gdf['estimated_tax_euro'] = gdf['area_m2'] * TAX_RATE_PER_SQM
    
    # 6. Summary Stats
    total_area = gdf['area_m2'].sum()
    total_tax = gdf['estimated_tax_euro'].sum()
    
    print(f"   ✅ Total Sealed Surface: {total_area:,.2f} m²")
    print(f"   💰 Estimated Annual Tax Revenue: €{total_tax:,.2f}")
    
    return gdf[['area_m2', 'estimated_tax_euro']]

# --- EXECUTION ---
all_data = []

files = [f for f in os.listdir(VECTOR_DIR) if f.endswith('.gpkg')]

if not files:
    print("No vector files found! Run 'vectorize_results.py' first.")
else:
    for f in files:
        path = os.path.join(VECTOR_DIR, f)
        df = calculate_areas(path)
        df['source_file'] = f
        all_data.append(df)

    # Save complete report
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index_label="Building_ID")
        print(f"\n📄 Final Report saved to: {OUTPUT_CSV}")
        print("   (Open this in Excel to show your 'Client')")