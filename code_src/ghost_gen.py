import numpy as np
from Bio.PDB import StructureBuilder, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
import argparse
import sys
import os

def load_obj(filename):
    """Parses OBJ file and returns vertices as numpy array."""
    vertices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def scale_to_protein(vertices, target_edge_length=3.8):
    """Scales the point cloud so average nearest-neighbor distance is ~3.8A."""
    # 1. Calculate current average edge length (approximate via MST or NN)
    # Simple heuristic: average distance to nearest neighbor
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    distances, _ = tree.query(vertices, k=2) # k=2 because k=1 is self (dist=0)
    avg_dist = np.mean(distances[:, 1])
    
    scale_factor = target_edge_length / avg_dist
    print(f"[Scaling] Original Avg Dist: {avg_dist:.3f}. Scaling by {scale_factor:.3f}")
    return vertices * scale_factor

def linearize_vertices(vertices):
    """
    Converts a cloud of 3D points into a linear path (Polymer Chain).
    Strategy: Greedy Nearest Neighbor.
    """
    unvisited = set(range(len(vertices)))
    path_indices = []
    
    # Start with vertex closest to origin (or arbitrary)
    current = 0
    path_indices.append(current)
    unvisited.remove(current)
    
    while unvisited:
        # Find nearest unvisited neighbor
        best_dist = float('inf')
        best_idx = -1
        
        curr_pos = vertices[current]
        
        for idx in unvisited:
            dist = np.linalg.norm(vertices[idx] - curr_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        if best_idx != -1:
            path_indices.append(best_idx)
            unvisited.remove(best_idx)
            current = best_idx
        else:
            break # Should not happen unless set is empty
            
    return vertices[path_indices]

def create_pdb(vertices, output_filename, residue_name="GLY"):
    """
    Creates a PDB file from a list of vertices, treating them as CA atoms.
    """
    structure = Structure("Ghost")
    model = Model(0)
    structure.add(model)
    chain = Chain("A")
    model.add(chain)
    
    for i, coord in enumerate(vertices):
        # Create Residue
        res_id = (' ', i + 1, ' ') # hetero flag, sequence identifier, insertion code
        res = Residue(res_id, residue_name, ' ')
        
        # Create CA Atom
        atom = Atom("CA", coord, 1.0, 0.0, " ", "CA", i + 1, "C")
        res.add(atom)
        
        chain.add(res)
        
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filename)
    print(f"[Success] Written {len(vertices)} residues to {output_filename}")

if __name__ == "__main__":
    print("--- PAPP Ghost Generator Starting ---", flush=True)
    parser = argparse.ArgumentParser(description="Convert OBJ geometry to PDB Ghost Template")
    parser.add_argument("input_obj", help="Path to source OBJ file")
    parser.add_argument("--out", help="Output PDB filename", default=None)
    parser.add_argument("--res", help="Residue Name (default: GLY)", default="GLY")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_path = os.path.abspath(args.input_obj)
    print(f"Input Target: {input_path}", flush=True)
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}", flush=True)
        sys.exit(1)
        
    verts = load_obj(input_path)
    print(f"Loaded {len(verts)} vertices.", flush=True)
    
    scaled_verts = scale_to_protein(verts)
    ordered_verts = linearize_vertices(scaled_verts)
    
    # Resolve output path
    if args.out:
        out_name = os.path.abspath(args.out)
    else:
        out_name = input_path.replace(".obj", "_ghost.pdb")
        
    # Ensure output directory exists
    out_dir = os.path.dirname(out_name)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}", flush=True)
        
    create_pdb(ordered_verts, out_name, args.res)
    print("--- Done ---", flush=True)
