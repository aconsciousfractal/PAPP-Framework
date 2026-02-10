"""
Topological Component Count Validation for PAPP Paper
======================================================

Validates the 9-component claim for V=18 structure using spectral graph theory.

Method:
- Load 3D mesh from OBJ file (vertices + faces)
- Build adjacency graph from face connectivity (combinatorial topology)
- Count components via Laplacian zero eigenvalue multiplicity
- Generate figure showing spectral evidence

Mathematical Principle:
For a graph with k connected components, the graph Laplacian L = D - A
has exactly k eigenvalues equal to 0 (zero modes).

This component count is an INTRINSIC topological property, independent
of any arbitrary threshold parameters.

Author: HAN Framework
Date: February 9, 2026
Reference: Section 6.3 - Component Counting via Spectral Analysis
"""

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import os
import sys

def load_obj(filepath):
    """
    Load 3D mesh from OBJ file
    
    Returns:
        vertices: (N, 3) array of 3D vertex coordinates
        faces: list of face vertex indices
        seed: 4D seed from comment (if present)
    """
    vertices = []
    faces = []
    seed = None
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            if line.startswith('# Seed:'):
                seed_str = line.split('Seed:')[1].strip()
                try:
                    seed = eval(seed_str)
                except:
                    seed = seed_str
                    
            elif parts[0] == 'v':
                # Take only first 3 coordinates (3D projection)
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
            elif parts[0] == 'f':
                # Parse face indices (handle v/vt/vn format)
                face_idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face_idxs)
    
    return np.array(vertices), faces, seed

def build_graph_laplacian(vertices, faces):
    """
    Build combinatorial graph Laplacian L = D - A from mesh topology
    
    The adjacency matrix A is constructed from face connectivity:
    - If vertices i and j share an edge in any face, then A[i,j] = 1
    
    Args:
        vertices: (N, 3) vertex array
        faces: list of face vertex indices
    
    Returns:
        L: sparse Laplacian matrix (N, N)
    """
    n = len(vertices)
    
    # Build adjacency matrix from face connectivity
    adj = scipy.sparse.lil_matrix((n, n))
    
    for face in faces:
        k = len(face)
        for i in range(k):
            v1 = face[i]
            v2 = face[(i+1) % k]  # Next vertex in face (cyclic)
            adj[v1, v2] = 1
            adj[v2, v1] = 1
    
    # Convert to CSR for efficiency
    adj = adj.tocsr()
    
    # Degree matrix
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = scipy.sparse.diags(degrees)
    
    # Laplacian L = D - A
    L = D - adj
    
    return L

def count_components_spectral(L, k_eigs=20):
    """
    Count connected components via Laplacian spectral analysis
    
    Args:
        L: sparse Laplacian matrix
        k_eigs: number of eigenvalues to compute
    
    Returns:
        num_components: number of disconnected components
        eigenvalues: array of smallest eigenvalues
    """
    N = L.shape[0]
    k_eigs = min(k_eigs, N - 2)
    
    try:
        # Compute smallest eigenvalues using shift-invert
        # sigma=-0.01 allows targeting zero eigenvalues without singularity
        eigenvalues, _ = eigsh(L, k=k_eigs, which='LM', sigma=-0.01)
        eigenvalues = np.sort(eigenvalues)
        
        # Count eigenvalues within numerical tolerance of zero
        ZERO_TOL = 1e-8
        num_components = np.sum(np.abs(eigenvalues) < ZERO_TOL)
        
        return num_components, eigenvalues
        
    except Exception as e:
        print(f"⚠️  Eigenvalue computation failed: {e}")
        print(f"   Attempting dense fallback...")
        
        if N < 2000:
            from scipy.linalg import eigh
            L_dense = L.toarray()
            eigenvalues, _ = eigh(L_dense)
            eigenvalues = eigenvalues[:k_eigs]
            
            num_components = np.sum(np.abs(eigenvalues) < ZERO_TOL)
            return num_components, eigenvalues
        else:
            return None, None

def generate_spectral_figure(eigenvalues, num_components, seed, output_path):
    """
    Generate publication-quality figure showing spectral evidence
    
    Args:
        eigenvalues: array of Laplacian eigenvalues
        num_components: number of components (zero modes)
        seed: 4D seed vector
        output_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot eigenvalues
    indices = np.arange(len(eigenvalues))
    
    # Color zero modes differently
    ZERO_TOL = 1e-8
    zero_mask = np.abs(eigenvalues) < ZERO_TOL
    nonzero_mask = ~zero_mask
    
    # Plot zero modes (red circles)
    ax.scatter(indices[zero_mask], eigenvalues[zero_mask], 
               s=120, c='red', marker='o', edgecolors='darkred', linewidth=2,
               label=f'Zero Modes (k={num_components})', zorder=3)
    
    # Plot non-zero eigenvalues (blue circles)
    ax.scatter(indices[nonzero_mask], eigenvalues[nonzero_mask], 
               s=80, c='steelblue', marker='o', alpha=0.7,
               label='Non-zero Eigenvalues', zorder=2)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3, zorder=1)
    
    # Add spectral gap annotation
    if np.any(nonzero_mask):
        first_nonzero = eigenvalues[nonzero_mask][0]
        ax.annotate(f'Spectral Gap\nλ = {first_nonzero:.4f}',
                   xy=(np.where(nonzero_mask)[0][0], first_nonzero),
                   xytext=(np.where(nonzero_mask)[0][0] + 2, first_nonzero + 0.5),
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                   fontsize=11, color='darkgreen', weight='bold')
    
    # Styling
    ax.set_xlabel('Eigenvalue Index', fontsize=13, weight='bold')
    ax.set_ylabel('Eigenvalue (λ)', fontsize=13, weight='bold')
    ax.set_title(f'Laplacian Spectrum: V=18 Structure (Seed {seed})\n'
                 f'Component Count k={num_components} (Zero Eigenvalue Multiplicity)',
                 fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Set y-axis to start from small negative value to show zero clearly
    ymin = min(-0.5, eigenvalues[0] - 0.5)
    ymax = max(5.0, eigenvalues[-1] + 1.0)
    ax.set_ylim(ymin, ymax)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved: {output_path}")
    plt.close()

def validate_component_count(obj_file, output_dir):
    """
    Validate component count and generate figure
    
    Args:
        obj_file: path to OBJ file
        output_dir: directory to save output figure
    """
    print("="*70)
    print("TOPOLOGICAL COMPONENT COUNT VALIDATION")
    print("="*70)
    
    # Load mesh
    vertices, faces, seed = load_obj(obj_file)
    
    filename = os.path.basename(obj_file)
    print(f"\nFile: {filename}")
    print(f"Seed: {seed}")
    print(f"3D Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    
    # Build Laplacian from mesh topology
    print(f"\nBuilding Laplacian from face connectivity...")
    L = build_graph_laplacian(vertices, faces)
    
    # Count edges
    num_edges = (L.nnz - len(vertices)) // 2  # Exclude diagonal, divide by 2 for symmetry
    print(f"Edges: {num_edges}")
    
    # Count components via spectral method
    print(f"\nComputing Laplacian spectrum (targeting zero modes)...")
    num_components, eigenvalues = count_components_spectral(L, k_eigs=20)
    
    if num_components is not None:
        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"\n  Number of Disconnected Components: {num_components}")
        
        print(f"\n  Spectral Evidence (first 10 eigenvalues):")
        for i in range(min(10, len(eigenvalues))):
            marker = "  ← ZERO MODE" if abs(eigenvalues[i]) < 1e-8 else ""
            print(f"    λ_{i} = {eigenvalues[i]:.8f}{marker}")
        
        print(f"\n{'='*70}")
        
        # Generate figure
        output_path = os.path.join(output_dir, "component_count_validation.pdf")
        generate_spectral_figure(eigenvalues, num_components, seed, output_path)
        
        # Also save PNG version
        output_path_png = os.path.join(output_dir, "component_count_validation.png")
        generate_spectral_figure(eigenvalues, num_components, seed, output_path_png)
        
        return num_components
    else:
        print("\n❌ FAILED: Could not compute component count.")
        return None

if __name__ == "__main__":
    # Paths relative to paper directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.dirname(script_dir)
    
    # Input: V=18 structure from assets
    obj_file = os.path.join(paper_dir, "assets", "models_obj", 
                            "PAPP Polychora 4D_Reconstructed", 
                            "Element_V18_RECONSTRUCTED_4D.obj")
    
    # Output: figures directory in paper_build
    output_dir = os.path.join(paper_dir, "paper_build", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run validation
    num_components = validate_component_count(obj_file, output_dir)
    
    print(f"\n{'='*70}")
    if num_components == 9:
        print("✅ VALIDATION SUCCESS: 9-component count confirmed for V=18")
    else:
        print(f"⚠️  VALIDATION ISSUE: Found {num_components} components (expected 9)")
    print(f"{'='*70}")
