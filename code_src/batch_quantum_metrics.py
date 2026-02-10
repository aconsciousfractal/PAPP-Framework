import sys
import os
import numpy as np
import pandas as pd
import scipy.spatial

# Add path to import PhiGapMechanism
sys.path.append(os.path.dirname(__file__))

try:
    from test_phi_gap_mechanism import PhiGapMechanism
except ImportError:
    print("Error: Could not import PhiGapMechanism")
    sys.exit(1)

# --- QUANTUM METRIC LOGIC (Adapted from quantum_metric_simulator.py) ---

def calculate_angle(v1, v2, v3):
    a = v2 - v1
    b = v3 - v1
    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    if la == 0 or lb == 0: return 0.0
    a /= la
    b /= lb
    dot = np.dot(a, b)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def compute_gaussian_curvature(vertices, faces):
    num_verts = len(vertices)
    angle_sums = np.zeros(num_verts)
    vertex_is_on_hull = np.zeros(num_verts, dtype=bool)
    
    for face in faces:
        n = len(face)
        for i in range(n):
            idx_curr = face[i]
            vertex_is_on_hull[idx_curr] = True
            
            idx_prev = face[i-1]
            idx_next = face[(i+1)%n]
            
            v_curr = vertices[idx_curr]
            v_prev = vertices[idx_prev]
            v_next = vertices[idx_next]
            
            angle = calculate_angle(v_curr, v_prev, v_next)
            angle_sums[idx_curr] += angle
            
    curvature = np.zeros(num_verts)
    # K = 2pi - sum(angles) on hull
    curvature[vertex_is_on_hull] = 2 * np.pi - angle_sums[vertex_is_on_hull]
    return curvature

def value_to_color(val, min_val, max_val):
    if max_val == min_val:
        norm = 0.5
    else:
        norm = (val - min_val) / (max_val - min_val)
    
    if norm < 0.5:
        local_t = norm * 2
        return 0, local_t, 1 - local_t # Blue -> Green
    else:
        local_t = (norm - 0.5) * 2
        return local_t, 1 - local_t, 0 # Green -> Red

def save_colored_obj(vertices, faces, curvatures, filename):
    min_k = np.min(curvatures)
    max_k = np.max(curvatures)
    
    with open(filename, 'w') as f:
        f.write("# Metatron Quantum Metric Analysis\n")
        f.write(f"g QuantumMetricObject\n")
        
        for i, v in enumerate(vertices):
            k = curvatures[i]
            r, g, b = value_to_color(k, min_k, max_k)
            # OBJ format with vertex colors: v x y z r g b
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            
        for face in faces:
            # OBJ is 1-indexed
            f_str = " ".join([str(idx+1) for idx in face])
            f.write(f"f {f_str}\n")

# --- BATCH PROCESS ---

def run_batch_quantum_metrics(csv_path, output_dir):
    print(f"Reading Seeds from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Target Output Directory: {output_dir}")
    
    seed_col = 'Seed' if 'Seed' in df.columns else 'Seed_String'
    
    total = len(df)
    print(f"Starting Batch Quantum Metric Generation for {total} objects...")
    
    processed = 0
    errors = 0
    
    for i, row in df.iterrows():
        try:
            seed_str = row[seed_col]
            s_clean = str(seed_str).replace('[', '').replace(']', '').replace(',', ' ').strip()
            seed = [int(x) for x in s_clean.split()]
            
            # Generate Geometry
            pg = PhiGapMechanism(seed)
            pg.step_1_beta_helix()
            pg.step_3_rg_flow()
            pg.step_4_5_triple()
            pg.step_6_vertex_count()
            pg.step_coordinate_generation()
            
            points = pg.vertices
            if len(points) < 4:
                # Cannot make hull
                continue
                
            # Compute Hull (Topology)
            hull = scipy.spatial.ConvexHull(points)
            faces = hull.simplices # Array of indices
            
            # Compute Quantum Metric (Curvature)
            curvatures = compute_gaussian_curvature(points, faces)
            
            # Export
            v_val = len(points)
            seed_fname = f"{seed[0]}_{seed[1]}_{seed[2]}_{seed[3]}"
            filename = f"Element_V{v_val}_phi_gap_{seed_fname}_QUANTUM_METRIC.obj"
            out_path = os.path.join(output_dir, filename)
            
            save_colored_obj(points, faces, curvatures, out_path)
            
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed}/{total}...")
                
        except Exception as e:
            # print(f"Error on {seed}: {e}")
            errors += 1
            continue
            
    print(f"Batch Complete. Generated {processed} files. Errors: {errors}")

if __name__ == "__main__":
    # Default Paths
    csv_input = r"p:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds\Unique_Elements\Deep Analysis\CSV\METATRON_PHYLOGENY_CENSUS.csv"
    output_folder = r"p:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds\Unique_Elements\1111 solids\Quantum Metrics"
    
    if len(sys.argv) > 2:
        csv_input = sys.argv[1]
        output_folder = sys.argv[2]
        
    run_batch_quantum_metrics(csv_input, output_folder)
