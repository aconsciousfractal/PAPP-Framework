import sys
import os
import numpy as np
import pandas as pd
import scipy.spatial
from scipy.spatial import ConvexHull, distance

# Add path to import PhiGapMechanism
sys.path.append(os.path.dirname(__file__))

# Try importing the generator class
try:
    from test_phi_gap_mechanism import PhiGapMechanism
except ImportError:
    # Fallback or strict error
    print("Error: Could not import PhiGapMechanism from test_phi_gap_mechanism.py")
    sys.exit(1)

def calculate_shape_metrics(points):
    if len(points) < 4:
        return 0, 0, 0 # Not a 3D volume
    
    try:
        hull = ConvexHull(points)
        volume = hull.volume
        area = hull.area
        
        # Sphericity: (36 * pi * V^2) / A^3
        # 1.0 = Perfect Sphere, < 1.0 = Irregular
        isoperimetric_quotient = (36 * np.pi * (volume**2)) / (area**3)
        return volume, area, isoperimetric_quotient
    except:
        return 0, 0, 0

def compute_crystallinity(points, sample_size=1000):
    # Radial Distribution Function roughness
    if len(points) < 10:
        return 0.0
        
    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        pts = points[indices]
    else:
        pts = points
        
    try:
        dists = distance.pdist(pts)
        hist, bin_edges = np.histogram(dists, bins=50, density=True)
        
        # Crystallinity Index = StdDev / Mean (Peak Sharpness)
        mean_prob = np.mean(hist)
        if mean_prob > 0:
            return np.std(hist) / mean_prob
        return 0.0
    except:
        return 0.0

def run_census(input_csv):
    print(f"Reading Seeds from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Prepare Output
    results = []
    
    # Check column names (handling the previous issue)
    seed_col = 'Seed' if 'Seed' in df.columns else 'Seed_String'
    
    total = len(df)
    print(f"Starting Physical Analysis on {total} species...")
    
    for i, row in df.iterrows():
        seed_str = row[seed_col]
        # Clean string "[1 2 3 4]" or "[1, 2, 3, 4]"
        s_clean = str(seed_str).replace('[', '').replace(']', '').replace(',', ' ').strip()
        seed = [int(x) for x in s_clean.split()]
        
        # Generate Geometry
        pg = PhiGapMechanism(seed)
        # We need to run the generation steps manually or does init do it? 
        # Checking previous usage: needs steps.
        # Actually usually test_phi_gap_mechanism has a run() or individual steps.
        # I'll replicate the standard pipeline from generate_colossus.py
        
        try:
            # Minimal pipeline to get vertices
            pg.step_1_beta_helix()
            pg.step_3_rg_flow()
            pg.step_4_5_triple()
            pg.step_6_vertex_count()
            pg.step_coordinate_generation()
            
            points = pg.vertices
            
            # Metrics
            v_total = len(points)
            vol, area, sphericity = calculate_shape_metrics(points)
            c_index = compute_crystallinity(points)
            
            density = 0
            if vol > 0:
                density = v_total / vol
            
            # Classify State
            if c_index > 0.8:
                state = "Solid Crystal"
            elif c_index > 0.4:
                state = "Liquid Crystal"
            else:
                state = "Amorphous Gas"
            
            # Append result
            results.append({
                "Seed": seed_str,
                "V_Total": v_total,
                "Volume": vol,
                "Area": area,
                "Sphericity": sphericity,
                "Density": density,
                "Crystallinity_Index": c_index,
                "Phase_State": state
            })
            
        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            continue
            
        if i % 50 == 0:
            print(f"  Processed {i}/{total}...")

    # Export
    out_csv = "METATRON_PHYSICAL_CENSUS.csv"
    print(f"Exporting to {out_csv}...")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default
        csv_path = r"p:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds\METATRON_PHYLOGENY_CENSUS.csv"
        
    run_census(csv_path)
