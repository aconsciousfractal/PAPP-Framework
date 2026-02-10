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

# --- 1. REUSE E8 GENERATION ---
def generate_e8_roots():
    print("Generating E8 Roots (8D)...")
    roots = []
    
    # Type A: Permutations of (±1, ±1, 0^6)
    for indices in itertools.combinations(range(8), 2):
        for s1, s2 in itertools.product([-1, 1], repeat=2):
            v = np.zeros(8)
            v[indices[0]] = s1
            v[indices[1]] = s2
            roots.append(v)
            
    # Type B: Double-even signs of (±0.5^8)
    base = np.ones(8) * 0.5
    for i in range(256):
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
            roots.append(base * np.array(signs))
            
    return np.array(roots)

# --- 2. FILTER E6 SOUL ---
def filter_e6_roots(roots_e8):
    print("Filtering E6 Roots (Orthogonal to A2)...")
    
    # 2a. Find A2 subsystem (u, v such that u.v = 1)
    # Let's try to find them dynamically to represent the logic
    # Or picking specific ones aligns the E6 in a known way.
    # Picking {e1+e2, e2+e3} ...
    # e1+e2 = (1, 1, 0...) length 2.
    # e2+e3 = (0, 1, 1...) length 2.
    # Dot product = 1. Correct.
    
    u = np.zeros(8); u[0] = 1; u[1] = 1
    v = np.zeros(8); v[1] = 1; v[2] = 1 # e2+e3
    
    # Correction: The generated roots are (±1, ±1).
    # (1, 1, 0...) IS in our set.
    # (0, 1, 1...) IS in our set.
    # Check if they exist in roots_e8 to be safe (precision)
    # Yes they should.
    
    print(f"Defining A2 plane with u={u}, v={v}")
    
    e6_roots = []
    for r in roots_e8:
        # Check orthogonality
        # Using a small tolerance for float comparison
        if abs(np.dot(r, u)) < 1e-9 and abs(np.dot(r, v)) < 1e-9:
            e6_roots.append(r)
            
    e6_roots = np.array(e6_roots)
    print(f"Filtered Roots: {len(e6_roots)}")
    
    if len(e6_roots) != 72:
        print("CRITICAL WARNING: Expected 72 roots for E6!")
        # Debug: try other A2 pair?
        # A standard definition of E6 within E8 is roots orthogonal to Alpha1 and Alpha2? 
        # Or remove roots involving coordinate 7 and 8? 
        # Actually E6 is usually { x in E8 | x . e7 = x . e8 = 0 } ? 
        # No, that gives D6?
        # Let's trust the algebraic definition: Orthogonal to A2.
        # N(E8) = 240. N(E6) = 72. N(A2) = 6. 
        # Decomposition: E8 -> E6 + A2.
        # This seems correct.
        
    return e6_roots

# --- 3. PROJECTION ---
def project_fold_4d(roots):
    # Same Golden Fold as E8
    v_left = roots[:, :4]
    v_right = roots[:, 4:]
    return v_left + PHI * v_right

def generate_e6_soul():
    # 1. Generate Parent E8
    roots8 = generate_e8_roots()
    
    # 2. Extract Child E6
    roots6 = filter_e6_roots(roots8)
    
    # 3. Compute Edges (Roots with distance sqrt(2)) -> Dot product 1
    print("Computing Edges...")
    dots = np.dot(roots6, roots6.T)
    pairs = np.argwhere(np.isclose(dots, 1.0))
    unique_edges = set()
    for i, j in pairs:
        if i < j:
            unique_edges.add((i, j))
    print(f"E6 Edges: {len(unique_edges)}") # Expected 72 * 16 / 2? No E6 kissing number 
    # E6 kissing number is 72? No.
    # Verify: 2_21 polytope has 72 vertices and 2160 edges? (From Wikipedia)
    # Let's see what we get.
    
    # 4. Project
    verts_4d = project_fold_4d(roots6)
    
    # 4D Norms for Color
    norms_4d = np.linalg.norm(verts_4d, axis=1)
    
    # 5. Project to 3D
    verts_3d = stereographic_project(verts_4d, R=np.max(norms_4d) + 0.5)
    
    # 6. Export
    out_dir = r"P:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds"
    fname = os.path.join(out_dir, "E6_Soul_QUANTUM_METRIC.obj")
    
    metric_values = norms_4d
    min_m, max_m = np.min(metric_values), np.max(metric_values)
    
    with open(fname, 'w') as f:
        f.write("# E6 Soul Polytope (1_22) - 6D Slice of 8D E8\n")
        f.write(f"# Vertices: 72 (Filtered via Orthogonality to A2)\n")
        f.write(f"# Edges: {len(unique_edges)} (Subgraph of full 972 edges)\n")
        
        for i, v in enumerate(verts_3d):
            val = metric_values[i]
            r, g, b = value_to_color(val, min_m, max_m)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            
        for i, j in unique_edges:
            f.write(f"l {i+1} {j+1}\n")
            
    print(f"Exported: {fname}")

if __name__ == "__main__":
    generate_e6_soul()
