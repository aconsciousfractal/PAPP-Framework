import numpy as np
import itertools
import os

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def value_to_color(val, min_val, max_val):
    if max_val == min_val: norm = 0.5
    else: norm = (val - min_val) / (max_val - min_val)
    if norm < 0.25: r, g, b = 0, norm*4, 1
    elif norm < 0.5: r, g, b = 0, 1, 1-(norm-0.25)*4
    elif norm < 0.75: r, g, b = (norm-0.5)*4, 1, 0
    else: r, g, b = 1, 1-(norm-0.75)*4, 0
    return r, g, b

def stereographic_project(points_4d, R=2.5):
    w = points_4d[:, 3]
    scale = 1.0 / (R - w)
    points_3d = points_4d[:, :3] * scale[:, np.newaxis]
    return points_3d

# --- 1. GENERATE E8 ROOTS (240 Vertices in 8D) ---
def generate_e8_roots():
    print("Generating E8 Roots (8D)...")
    roots = []
    
    # Type A: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    # Positions: Choose 2 from 8 -> 28 pairs.
    # Signs: 4 combinations per pair. 28 * 4 = 112.
    for indices in itertools.combinations(range(8), 2):
        for s1, s2 in itertools.product([-1, 1], repeat=2):
            v = np.zeros(8)
            v[indices[0]] = s1
            v[indices[1]] = s2
            roots.append(v)
            
    print(f"Type A Roots: {len(roots)}")
    
    # Type B: Signs of (±0.5, ..., ±0.5) with Even number of minus signs.
    # Total signs = 2^8 = 256. Half have even minus signs -> 128.
    # Note: E8 usually scaled so these are integers E8 = D8 + (1/2, ..., 1/2) + D8?
    # Standard normalization: Roots have squared length 2.
    # Type A: 1^2 + 1^2 = 2. Correct.
    # Type B: 8 * (0.5^2) = 8 * 0.25 = 2. Correct.
    
    type_b_count = 0
    base = np.ones(8) * 0.5
    for i in range(256):
        # Generate signs from binary representation
        signs = []
        temp = i
        minus_count = 0
        for b in range(8):
            if (temp >> b) & 1:
                signs.append(-1)
                minus_count += 1
            else:
                signs.append(1)
        
        if minus_count % 2 == 0:
            v = base * np.array(signs)
            roots.append(v)
            type_b_count += 1
            
    print(f"Type B Roots: {type_b_count}")
    
    roots = np.array(roots)
    print(f"Total E8 Roots: {len(roots)}")
    return roots

# --- 2. THE GOLDEN FOLD (8D -> 4D) ---
def project_e8_to_h4(roots_8d):
    """
    Project 8D roots to 4D using the Golden Fold.
    u = v_left + phi * v_right
    This splits the 240 roots into two subsets of 120 in 4D (scaled).
    """
    print("Folding 8D -> 4D...")
    v_left = roots_8d[:, :4]
    v_right = roots_8d[:, 4:]
    
    # Simple fold? Or do we need specific basis alignment?
    # The standard projection from E8 to H4 uses the fact that E8 roots 
    # project to two copies of H4 roots (multiplied by c and s).
    # H4 roots (600-cell) have varying lengths? No, 600-cell is uniform.
    # Let's try the simple fold first.
    
    # u = v_left + PHI * v_right
    # Let's verify lengths.
    
    u = v_left + PHI * v_right
    return u

def generate_e8_gosset():
    # 1. Roots
    roots_8d = generate_e8_roots()
    
    # 2. Edges (8D)
    # In E8, edges exist between roots with squared distance = 2 (minimal distance).
    # Since all roots have length^2 = 2, distance^2 = |u-v|^2 = 2+2 - 2u.v = 4 - 2u.v
    # Min dist sq = 2 implies 4 - 2u.v = 2 => 2u.v = 2 => u.v = 1.
    # Roots form edges if their dot product is 1. (Angle 60 degrees).
    print("Computing Edges (Dot Product = 1)...")
    edges = []
    # This is O(N^2) but N=240, so 240*240 = 57600. Fast.
    
    # Vectorized dot product
    dots = np.dot(roots_8d, roots_8d.T)
    # Indices where dot is close to 1
    pairs = np.argwhere(np.isclose(dots, 1.0))
    
    # This gives (i, j) and (j, i). Filter unique.
    unique_edges = set()
    for i, j in pairs:
        if i < j:
            unique_edges.add((i, j))
            
    print(f"E8 4_21 Edges: {len(unique_edges)} (Expected 6720)")
    
    # 3. Projection to 3D
    verts_4d = project_e8_to_h4(roots_8d)
    
    # Normalize 4D for visualization?
    # The fold produces vectors of different lengths?
    norms_4d = np.linalg.norm(verts_4d, axis=1)
    
    # Analysis of Shells in 4D
    print("Analyzing 4D Shells (Radii)...")
    unique_radii = np.unique(np.round(norms_4d, 5))
    print(f"Radii Found (Folded): {unique_radii}")
    
    # If we see two distinct radii, we have the "Two 600-cells".
    
    # Project to 3D for OBJ
    # We keep the scale for the "Metric", but maybe normalize for "Semantic" view?
    # Let's keep scale to see the nesting.
    
    verts_3d = stereographic_project(verts_4d, R=np.max(norms_4d) + 0.5)
    
    # 4. Export
    out_dir = r"P:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds"
    
    # Files
    fname_metric = os.path.join(out_dir, "E8_Gosset_QUANTUM_METRIC.obj")
    
    # Use 8D W-coordinate (let's say 8th dim) or 4D W for color?
    # Let's use the 4D Radius (Shell) as the Metric Color to highlight the two 600-cells.
    metric_values = norms_4d 
    min_m, max_m = np.min(metric_values), np.max(metric_values)
    
    with open(fname_metric, 'w') as f:
        f.write("# E8 Gosset Polytope (4_21) - 8D Roots -> 4D Fold\n")
        f.write(f"# Vertices: {len(roots_8d)}\n")
        f.write(f"# Edges: {len(unique_edges)}\n")
        f.write(f"# 4D Radii: {unique_radii}\n")
        
        for i, v in enumerate(verts_3d):
            # Color by 4D Radius (Blue=Inner, Red=Outer)
            val = metric_values[i]
            r, g, b = value_to_color(val, min_m, max_m)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            
        for i, j in unique_edges:
            f.write(f"l {i+1} {j+1}\n")
            
    print(f"Exported: {fname_metric}")

if __name__ == "__main__":
    generate_e8_gosset()
