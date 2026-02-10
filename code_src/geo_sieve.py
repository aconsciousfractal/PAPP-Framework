import numpy as np
from Bio.PDB import PDBParser
import argparse
import sys
import os
import glob
import warnings

# Suppress BioPython warnings
warnings.filterwarnings("ignore")

def get_ca_coords(pdb_file):
    """Extracts CA atom coordinates from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('S', pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    return np.array(coords)

def load_obj_vertices(obj_file):
    """Parses OBJ file and returns vertices as numpy array."""
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def kabsch_rmsd(P, Q):
    """
    Calculates RMSD between two point sets P and Q using Kabsch algorithm.
    P and Q must be (N, 3) arrays.
    """
    if len(P) != len(Q):
        return float('inf')
        
    # Center the points
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    # Compute covariance matrix
    H = np.dot(P_centered.T, Q_centered)
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation
    R = np.dot(Vt.T, U.T)
    
    # Check reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # Rotate P
    P_rotated = np.dot(P_centered, R)
    
    # RMSD
    return np.sqrt(np.mean(np.sum((P_rotated - Q_centered)**2, axis=1)))

def normalize_scale(P):
    """Normalizes point cloud to unit sphere scaling for shape comparison."""
    centroid = np.mean(P, axis=0)
    centered = P - centroid
    dist = np.sqrt(np.sum(centered**2, axis=1))
    max_dist = np.max(dist)
    if max_dist == 0: return P
    return centered / max_dist

def resample_curve(points, n_target):
    """
    Resamples a 3D curve defined by 'points' to have exactly 'n_target' points
    while preserving the shape/arc-length distribution.
    """
    if len(points) < 2: return points
    
    # Calculate cumulative arc length
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dist[-1]
    
    # Create target distances
    target_dists = np.linspace(0, total_length, n_target)
    
    # Interpolate for each dimension
    new_points = np.zeros((n_target, 3))
    for i in range(3):
        new_points[:, i] = np.interp(target_dists, cum_dist, points[:, i])
        
    return new_points

if __name__ == "__main__":
    print("--- PAPP Geometric Sieve Starting ---", flush=True)
    parser = argparse.ArgumentParser(description="Screen PDB files against PAPP Geometry")
    parser.add_argument("--target", help="Path to PAPP OBJ file", required=True)
    parser.add_argument("--db", help="Directory containing PDB files to screen", required=True)
    parser.add_argument("--threshold", help="RMSD Threshold", type=float, default=2.0)
    parser.add_argument("--tolerance", help="Vertex Count Tolerance (default: 0)", type=int, default=0)
    
    args = parser.parse_args()
    
    target_path = os.path.abspath(args.target)
    db_path = os.path.abspath(args.db)
    
    if not os.path.exists(target_path):
        print(f"Error: Target {target_path} not found")
        sys.exit(1)
        
    papp_verts = load_obj_vertices(target_path)
    papp_norm = normalize_scale(papp_verts)
    n_papp = len(papp_verts)
    
    print(f"Target: {os.path.basename(target_path)} (Vertices: {n_papp})")
    print(f"Scanning {db_path} (By Chain)...")
    
    pdbs = glob.glob(os.path.join(db_path, "*.pdb"))
    if not pdbs:
        print("No PDB files found.")
        sys.exit(0)
        
    matches = []
    
    parser_pdb = PDBParser(QUIET=True)

    for pdb in pdbs:
        try:
            structure = parser_pdb.get_structure('S', pdb)
            
            # Iterate over all chains in the first model (usually sufficient)
            try:
                model = structure[0]
            except:
                continue

            for chain in model:
                # Extract CA for this chain
                chain_coords = []
                for residue in chain:
                    if 'CA' in residue:
                        chain_coords.append(residue['CA'].get_coord())
                
                chain_coords = np.array(chain_coords)
                n_prot = len(chain_coords)
                
                # Filter by Vertex Count
                if abs(n_prot - n_papp) > args.tolerance:
                    continue
                
                # Resample if needed to match dimensions for Kabsch
                if n_prot != n_papp:
                    chain_for_rms = resample_curve(chain_coords, n_papp)
                else:
                    chain_for_rms = chain_coords
                
                prot_norm = normalize_scale(chain_for_rms)
                
                # Check for NaNs or degeneracy
                if len(prot_norm) < 3: continue
                
                try:
                    rmsd = kabsch_rmsd(papp_norm, prot_norm)
                except:
                    continue # SVD fail?

                if rmsd < args.threshold:
                    print(f"[MATCH!] {os.path.basename(pdb)} Chain {chain.id} | N={n_prot} | RMSD: {rmsd:.4f}")
                    matches.append((f"{os.path.basename(pdb)}_{chain.id}", rmsd))
                elif rmsd < args.threshold * 1.5:
                     print(f"[CANDIDATE] {os.path.basename(pdb)} Chain {chain.id} | N={n_prot} | RMSD: {rmsd:.4f}")

        except Exception as e:
            print(f"Error reading {pdb}: {str(e)}")
            
    print(f"--- Scan Complete. Found {len(matches)} matches. ---")
