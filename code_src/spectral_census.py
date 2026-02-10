import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

# Add path to import PhiGapMechanism
sys.path.append(os.path.dirname(__file__))

try:
    from test_phi_gap_mechanism import PhiGapMechanism
except ImportError:
    print("Error: Could not import PhiGapMechanism")
    sys.exit(1)

def build_graph_laplacian(vertices, faces):
    n = len(vertices)
    adj = scipy.sparse.lil_matrix((n, n))
    
    # Faces are list of lists of indices
    for face in faces:
        k_len = len(face)
        for i in range(k_len):
            v1 = face[i]
            v2 = face[(i+1)%k_len]
            # Undirected graph
            adj[v1, v2] = 1
            adj[v2, v1] = 1
            
    adj = adj.tocsr()
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = scipy.sparse.diags(degrees)
    L = D - adj
    return L

def get_eigenvalues(L, k=10):
    # Calculate k smallest eigenvalues
    # L is real symmetric (or close enough for Graph Laplacian)
    # Using 'SM' (Smallest Magnitude) with shift-invert mode to find near-zero
    try:
        n = L.shape[0]
        if n <= k:
            # Full dense solver for tiny matrices
            eigenvalues = np.linalg.eigvalsh(L.toarray())
            return sorted(eigenvalues)[:k]
        
        # Sparse solver
        # shift -0.01 to handle singular matrix (0 eigenvalues)
        sigma = -0.01
        vals, vecs = scipy.sparse.linalg.eigsh(L, k=k, sigma=sigma, which='LM')
        return sorted(vals)
    except Exception as e:
        print(f"  Eigen Solver Error: {e}")
        return [0]*k

def run_spectral_census(input_csv):
    print(f"Reading Seeds from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except:
        print("Failed to read CSV")
        return

    seed_col = 'Seed' if 'Seed' in df.columns else 'Seed_String'
    
    results = []
    total = len(df)
    print(f"Starting Spectral Analysis on {total} species...")
    
    for i, row in df.iterrows():
        seed_str = row[seed_col]
        # Clean string
        s_clean = str(seed_str).replace('[', '').replace(']', '').replace(',', ' ').strip()
        seed = [int(x) for x in s_clean.split()]
        
        try:
            # Generate Geometry
            pg = PhiGapMechanism(seed)
            pg.step_1_beta_helix()
            pg.step_3_rg_flow()
            pg.step_4_5_triple()
            pg.step_6_vertex_count()
            pg.step_8_topology()
            pg.step_coordinate_generation()
            
            # Need Faces for Laplacian
            # PhiGapMechanism stores .faces
            if not hasattr(pg, 'faces') or not pg.faces:
                # If no faces (e.g. V < 4), skip
                # Actually step_coordinate_generation calls ConvexHull usually? 
                # Oh wait, pg.step_coordinate_generation only generates points?
                # We need to manually generate hull faces if not provided.
                # Let's check test_phi_gap_mechanism. 
                # It usually doesn't store faces in 'pg' instance unless export is called.
                # We'll regenerate Hull here.
                if len(pg.vertices) >= 4:
                    hull = scipy.spatial.ConvexHull(pg.vertices)
                    # Hull simplices are faces (triangles)
                    faces = hull.simplices
                else:
                    faces = []
            else:
                faces = pg.faces

            if len(pg.vertices) < 2:
                results.append({"Seed": seed_str, "V": len(pg.vertices), "Error": "Too small"})
                continue
                
            # Laplacian
            L = build_graph_laplacian(pg.vertices, faces)
            
            # Eigenvalues
            evals = get_eigenvalues(L, k=10)
            
            # Count Zero Modes (approx < 1e-5)
            zeros = sum(1 for e in evals if abs(e) < 1e-4)
            
            # Spectral Gap (First non-zero)
            # Find first eval > 1e-4
            gap = 0.0
            for e in evals:
                if e > 1e-4:
                    gap = e
                    break
                    
            entry = {
                "Seed": seed_str,
                "V": len(pg.vertices),
                "Zero_Modes": zeros,
                "Fundamental_Freq": gap,
                "Eigenvalues": str([round(e, 4) for e in evals])
            }
            results.append(entry)
            
        except Exception as e:
            # print(f"Error on {seed}: {e}")
            results.append({"Seed": seed_str, "Error": str(e)})

        if i % 25 == 0:
            print(f"  Processed {i}/{total} species...")

    # Export
    out_csv = "METATRON_SPECTRAL_CENSUS.csv"
    print(f"Exporting to {out_csv}...")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r"p:\GitHub_puba\HAN\METATRON_PHYLOGENY_CENSUS.csv"
        
    run_spectral_census(csv_path)
