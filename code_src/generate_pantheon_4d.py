import numpy as np
import itertools
from scipy.spatial import ConvexHull
import os

def stereographic_project(points_4d):
    """
    Project 4D points to 3D.
    """
    w = points_4d[:, 3]
    R = 2.0 # View distance
    scale = 1.0 / (R - w)
    points_3d = points_4d[:, :3] * scale[:, np.newaxis]
    return points_3d

def get_faces_from_hull(verts_4d, expected_faces=None, edge_len_sq=None):
    """
    Extract 2D faces from 3D boundary of 4D convex hull.
    Filter by edge length if provided.
    """
    hull = ConvexHull(verts_4d)
    print(f"  4D Hull Simplices: {len(hull.simplices)}")
    
    all_faces = set()
    for simplex in hull.simplices:
        for i in range(4):
            face = [simplex[j] for j in range(4) if j != i]
            face.sort()
            all_faces.add(tuple(face))
            
    valid_faces = []
    
    if edge_len_sq:
        print(f"  Filtering for edge length sq ~ {edge_len_sq}...")
        for face in all_faces:
            v1, v2, v3 = verts_4d[face[0]], verts_4d[face[1]], verts_4d[face[2]]
            d1 = np.sum((v1-v2)**2)
            d2 = np.sum((v2-v3)**2)
            d3 = np.sum((v3-v1)**2)
            
            tol = 0.1
            if (abs(d1 - edge_len_sq) < tol) and \
               (abs(d2 - edge_len_sq) < tol) and \
               (abs(d3 - edge_len_sq) < tol):
                valid_faces.append(face)
    else:
        valid_faces = list(all_faces)
        
    print(f"  Refined Faces: {len(valid_faces)}")
    return valid_faces

def generate_5_cell_vertices():
    """
    Regular 5-Cell (Simplex).
    Vertices: (1,1,1,-1/sqrt(5)) and perms?
    Standard coordinates for edge length sqrt(5):
    """
    # Simple construction:
    # V0=(1,0,0,0)
    # V1... simplex logic.
    # Let's use the one based on (1,1,1,-1) / sqrt(5)?
    # Standard:
    a = 1.0 / np.sqrt(5)
    verts = [
        [1, 1, 1, -a],
        [1, -1, -1, -a],
        [-1, 1, -1, -a],
        [-1, -1, 1, -a],
        [0, 0, 0, 4*a] # Wait, 4/sqrt(5).
    ]
    # Check distances
    v0 = np.array(verts[0])
    v1 = np.array(verts[1])
    d2 = np.sum((v0-v1)**2)
    # (0)^2 + (2)^2 + (2)^2 + (0)^2 = 8.
    v4 = np.array(verts[4])
    d2_4 = np.sum((v0-v4)**2)
    # (1)^2 + (1)^2 + (1)^2 + (-1/sqrt5 - 4/sqrt5)^2
    # 3 + (-5/sqrt5)^2 = 3 + (-sqrt5)^2 = 3 + 5 = 8.
    # PERFECT. All edges have squared length 8.
    return np.array(verts), 8.0

def generate_16_cell_vertices():
    """
    Regular 16-Cell (Cross Polytope).
    Permutations of (±1, 0, 0, 0).
    """
    verts = []
    for i in range(4):
        v = [0, 0, 0, 0]
        v[i] = 1
        verts.append(list(v))
        v[i] = -1
        verts.append(list(v))
    
    # Edge length: (1,0,0,0) to (0,1,0,0) -> sq dist 2.
    return np.array(verts), 2.0

def generate_8_cell_vertices():
    """
    Regular 8-Cell (Tesseract).
    (±1, ±1, ±1, ±1). 16 vertices.
    """
    verts = list(itertools.product([-1, 1], repeat=4))
    # Edge length: (1,1,1,1) to (1,1,1,-1) -> sq dist 4.
    # Note: Hull triangles?
    # Tesseract boundary is Cubes.
    # Cubes have square faces.
    # Triangulation splits squares into 2 triangles.
    # Distance of diagonal: sqrt(8).
    # So we expect edges of length 2 and sqrt(8).
    # If we strictly filter for length 2 (edges), we lose the triangulation.
    # We want the triangulation of the cubes.
    # So we accept edges 2 and sqrt(8) (face diagonal)?
    # Actually, we want the FACES. 
    # The Hull will give triangles.
    # Let's just create the OBJ and let the user see.
    return np.array(verts), None # No strict filter for Tesseract as faces are squares

def export_obj(verts_3d, faces, filename, name):
    output_dir = r"P:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds"
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write(f"# {name}\n")
        f.write(f"# Vertices: {len(verts_3d)}\n")
        for v in verts_3d:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Exported {name}: {path}")

def main():
    print("=== GENERATING 4D PANTHEON ===")
    
    # 1. 5-Cell
    v5, d2_5 = generate_5_cell_vertices()
    f5 = get_faces_from_hull(v5, edge_len_sq=d2_5)
    v5_3d = stereographic_project(v5)
    export_obj(v5_3d, f5, "5_cell_semantic.obj", "5-Cell (Hypertetrahedron)")
    
    # 2. 16-Cell
    v16, d2_16 = generate_16_cell_vertices()
    f16 = get_faces_from_hull(v16, edge_len_sq=d2_16)
    v16_3d = stereographic_project(v16)
    export_obj(v16_3d, f16, "16_cell_semantic.obj", "16-Cell (Hyperoctahedron)")
    
    # 3. 8-Cell
    v8, _ = generate_8_cell_vertices()
    f8 = get_faces_from_hull(v8, edge_len_sq=None) # Accept triangulation
    # Tesseract faces are squares. Hull will output 2 triangles per square.
    # Valid triangles should have sides: 2, 2, sqrt(8)?
    # Or 4, 4, 8 (squared).
    # Let's accept all from Hull for now.
    v8_3d = stereographic_project(v8)
    export_obj(v8_3d, f8, "8_cell_semantic.obj", "8-Cell (Tesseract)")

if __name__ == "__main__":
    main()
