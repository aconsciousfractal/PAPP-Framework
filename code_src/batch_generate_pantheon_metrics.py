import numpy as np
import itertools
from scipy.spatial import ConvexHull, Delaunay
import sys
import os

PHI = (1.0 + np.sqrt(5.0)) / 2.0

# --- COLOR UTILS ---
def value_to_color(val, min_val, max_val):
    if max_val == min_val: norm = 0.5
    else: norm = (val - min_val) / (max_val - min_val)
    if norm < 0.25: r, g, b = 0, norm*4, 1
    elif norm < 0.5: r, g, b = 0, 1, 1-(norm-0.25)*4
    elif norm < 0.75: r, g, b = (norm-0.5)*4, 1, 0
    else: r, g, b = 1, 1-(norm-0.75)*4, 0
    return r, g, b

def stereographic_project(points_4d, radius_offset=0.2):
    w = points_4d[:, 3]
    R = np.max(w) + radius_offset
    scale = 1.0 / (R - w)
    points_3d = points_4d[:, :3] * scale[:, np.newaxis]
    return points_3d

def export_metric_obj(verts_4d, faces, filename, name):
    print(f"Exporting {name} to {filename}...")
    metric_values = verts_4d[:, 3]
    min_m, max_m = np.min(metric_values), np.max(metric_values)
    verts_3d = stereographic_project(verts_4d)
    
    with open(filename, 'w') as f:
        f.write(f"# {name} - Quantum Metric (W-Depth)\n")
        for i, v in enumerate(verts_3d):
            val = metric_values[i]
            r, g, b = value_to_color(val, min_m, max_m)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print("Done.")

def get_hull_faces(verts_4d):
    hull = ConvexHull(verts_4d)
    all_faces = set()
    for simplex in hull.simplices:
        for i in range(4):
            face = [simplex[j] for j in range(4) if j != i]
            face.sort()
            all_faces.add(tuple(face))
    return list(all_faces)

# --- GENERATORS ---

def gen_5_cell():
    a = 1.0 / np.sqrt(5)
    verts = [
        [1, 1, 1, -a], [1, -1, -1, -a], [-1, 1, -1, -a], [-1, -1, 1, -a], [0, 0, 0, 4*a]
    ]
    return np.array(verts)

def gen_8_cell():
    return np.array(list(itertools.product([-1, 1], repeat=4)))

def gen_16_cell():
    verts = []
    for i in range(4):
        v = [0]*4; v[i]=1; verts.append(v)
        v = [0]*4; v[i]=-1; verts.append(v)
    return np.array(verts)

def gen_24_cell():
    vertices = []
    # Permutations of (±1, ±1, 0, 0)
    for p in itertools.combinations(range(4), 2):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0]*4; v[p[0]]=s1; v[p[1]]=s2
                vertices.append(v)
    return np.array(vertices)

def gen_600_cell_verts():
    # Helper for 120-cell
    vertices = []
    # 1. (±2, 0, 0, 0)
    for i in range(4):
        v = [0.0]*4; v[i] = 2.0; vertices.append(v); v=[0.0]*4; v[i]=-2.0; vertices.append(v)
    # 2. (±1, ±1, ±1, ±1)
    for p in itertools.product([-1.0, 1.0], repeat=4): vertices.append(list(p))
    # 3. Even perms of (±phi, ±1, ±1/phi, 0)
    phi = PHI; inv_phi = 1.0/phi
    base = [phi, 1.0, inv_phi, 0.0]
    even_perms = []
    for p in itertools.permutations(range(4)):
        inv = 0
        for i in range(4):
            for j in range(i+1, 4):
                if p[i]>p[j]: inv+=1
        if inv%2==0: even_perms.append(p)
    
    for p_idx in even_perms:
        v_base = [base[p_idx[i]] for i in range(4)]
        nz = [i for i,x in enumerate(v_base) if x!=0]
        for signs in itertools.product([-1,1], repeat=3):
            v=list(v_base)
            for k,s in enumerate(signs): v[nz[k]]*=s
            vertices.append(v)
    return np.array(vertices)

def gen_120_cell():
    print("Generating 120-Cell via 600-Cell Dual...")
    # 1. Get 600-cell vertices
    v600 = gen_600_cell_verts()
    # 2. Compute Delaunay Triangulation (Cells)
    # Since 600-cell vertices are on S3, Delaunay of them might give the 600 tetrahedra.
    # Note: hull.simplices gives the SURFACE cells of the 600-cell (Tetrahedra).
    # Wait, 600-cell IS the surface of the 4D hull of these points.
    # The facets of the ConvexHull(v600) ARE the cells of the 600-cell.
    hull = ConvexHull(v600)
    print(f"600-Cell Facets (Tetrahedra): {len(hull.simplices)}") # Should be 600
    
    # 3. The dual vertices correspond to the centers of these cells.
    # Center of a tetrahedron = average of its 4 vertices.
    centers = []
    for simplex in hull.simplices:
        # Simplex indices
        cell_verts = v600[simplex]
        center = np.mean(cell_verts, axis=0)
        centers.append(center)
        
    v120 = np.array(centers)
    # Project to sphere (normalize) standard 120-cell
    # Note: 120-cell vertices are usually at norm squared = 8 or something.
    norms = np.linalg.norm(v120, axis=1)
    avg_norm = np.mean(norms)
    # Normalize to constant radius to be clean
    v120 = v120 / norms[:, np.newaxis] * 2.0 # Scale to 2.0 like others
    
    # 4. Filter for unique vertices? (Should be 600)
    # Check simple uniqueness
    unique_v = []
    seen = set()
    for v in v120:
        t = tuple(np.round(v, 4))
        if t not in seen:
            seen.add(t)
            unique_v.append(v)
    
    final_v120 = np.array(unique_v)
    print(f"120-Cell Vertices: {len(final_v120)} (Expected 600)")
    return final_v120

def main():
    out_dir = r"P:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds"
    
    # 1. 5-Cell
    v5 = gen_5_cell()
    f5 = get_hull_faces(v5)
    export_metric_obj(v5, f5, os.path.join(out_dir, "5_cell_semantic_QUANTUM_METRIC.obj"), "5-Cell")
    
    # 2. 8-Cell
    v8 = gen_8_cell()
    f8 = get_hull_faces(v8)
    export_metric_obj(v8, f8, os.path.join(out_dir, "8_cell_semantic_QUANTUM_METRIC.obj"), "8-Cell")
    
    # 3. 16-Cell
    v16 = gen_16_cell()
    f16 = get_hull_faces(v16)
    export_metric_obj(v16, f16, os.path.join(out_dir, "16_cell_semantic_QUANTUM_METRIC.obj"), "16-Cell")
    
    # 4. 24-Cell
    v24 = gen_24_cell()
    # 24-cell edge filtering is needed? 
    # Let's use hull. If hull gives extra faces, usually they are internal if points not on boundary.
    # All 24 vertices are on sphere. Hull is the boundary.
    # Hull gives tetrahedra (facets). Boundary of facets -> Triangles.
    f24 = get_hull_faces(v24)
    export_metric_obj(v24, f24, os.path.join(out_dir, "24_cell_semantic_QUANTUM_METRIC.obj"), "24-Cell")
    
    # 5. 120-Cell (SKIPPED: Generated separately with correct duality logic)
    # v120 = gen_120_cell()
    # f120 = get_hull_faces(v120)
    # export_metric_obj(v120, f120, os.path.join(out_dir, "120_cell_semantic_QUANTUM_METRIC.obj"), "120-Cell")

if __name__ == "__main__":
    main()
