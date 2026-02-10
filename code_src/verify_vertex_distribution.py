"""
Verify Vertex Distribution in Tetrahedral Cascade
==================================================

Analyze the actual vertex distribution in generated 4D polychora
to verify if coefficients (1, 2, 2, 1) emerge naturally.

Goal: Extract 4D coordinates and analyze their distribution across
the four orthogonal regions (A, B, C, D).
"""

import sys
import numpy as np
from pathlib import Path
import re

# Setup paths
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir / "code_src"))
sys.path.insert(0, str(scripts_dir.parent.parent.parent / "scripts"))

from grant_4d_constructor import Grant4DParameters, Grant4DPolychoron
from test_phi_gap_mechanism import PhiGapMechanism

def analyze_seed(seed, verbose=True):
    """
    Analyze vertex distribution for a given seed.
    
    Returns distribution counts across 4 regions.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"ANALYZING SEED: {seed}")
        print(f"{'='*70}")
    
    # Step 1: Generate Grant parameters
    pg = PhiGapMechanism(seed)
    
    # Suppress output
    import os
    import contextlib
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            pg.step_1_beta_helix()
            pg.step_3_rg_flow()
            pg.step_4_5_triple()
    
    a, b, c = pg.triple
    d = np.sqrt(a**2 + b**2 + c**2)
    
    # Grant parameters
    params = Grant4DParameters(a=float(a), b=float(b), c=float(c), d=float(d))
    
    if verbose:
        print(f"Grant Parameters: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
        print(f"Predicted V_4D = a + 2b + 2c + d = {params.V_4D:.2f}")
    
    # Step 2: Generate 4D vertices
    poly = Grant4DPolychoron(params, name=f"Test_{seed}")
    vertices_4d = poly.generate_vertices_4d(method='tetrahedral_cascade')
    
    V_actual = len(vertices_4d)
    
    if verbose:
        print(f"Actual V_4D = {V_actual}")
        print(f"Match: {abs(V_actual - params.V_4D) < 1.5}")
    
    # Step 3: Analyze distribution across regions
    # Classify each vertex by dominant coordinate
    
    region_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for vertex in vertices_4d:
        x, y, z, w = vertex
        
        # Find dominant coordinate (largest absolute value)
        coords = {'A': abs(x), 'B': abs(y), 'C': abs(z), 'D': abs(w)}
        dominant = max(coords, key=coords.get)
        region_counts[dominant] += 1
    
    if verbose:
        print(f"\nVertex Distribution by Dominant Coordinate:")
        print(f"  Region A (x-axis): {region_counts['A']} vertices")
        print(f"  Region B (y-axis): {region_counts['B']} vertices")
        print(f"  Region C (z-axis): {region_counts['C']} vertices")
        print(f"  Region D (w-axis): {region_counts['D']} vertices")
        print(f"  Total: {sum(region_counts.values())}")
        
        # Compare with predicted distribution
        predicted_total = a + 2*b + 2*c + d
        print(f"\nPredicted Distribution (from formula):")
        print(f"  Region A: {a:.1f}  (coefficient: 1)")
        print(f"  Region B: {2*b:.1f}  (coefficient: 2)")
        print(f"  Region C: {2*c:.1f}  (coefficient: 2)")
        print(f"  Region D: {d:.1f}  (coefficient: 1)")
        print(f"  Total: {predicted_total:.1f}")
        
        # Ratios
        print(f"\nRatio Analysis (Actual / Predicted):")
        if a > 0:
            print(f"  A: {region_counts['A'] / a:.3f}")
        if b > 0:
            print(f"  B: {region_counts['B'] / (2*b):.3f}")
        if c > 0:
            print(f"  C: {region_counts['C'] / (2*c):.3f}")
        if d > 0:
            print(f"  D: {region_counts['D'] / d:.3f}")
    
    return {
        'seed': seed,
        'params': (a, b, c, d),
        'V_predicted': params.V_4D,
        'V_actual': V_actual,
        'distribution': region_counts,
        'vertices_4d': vertices_4d
    }

def verify_euler_characteristic(vertices_4d):
    """
    Compute 4D convex hull and verify Euler characteristic.
    """
    from scipy.spatial import ConvexHull
    from itertools import combinations
    
    try:
        # Compute 4D convex hull
        hull = ConvexHull(vertices_4d)
        
        # In 4D, hull.simplices are 4-simplices (tetrahedra in 4D)
        # Each simplex has 5 vertices (not 4!)
        V = len(hull.vertices)  # Use hull vertices, not all vertices
        C = len(hull.simplices)  # Number of 4-cells
        
        # Compute edges and faces (more complex)
        # For now, just compute from topology
        edges = set()
        faces = set()
        
        for simplex in hull.simplices:
            # Each 4-simplex has 5 vertices, 10 edges, 10 faces
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
            
            # Faces are 3-vertex combinations (triangular faces)
            for face in combinations(simplex, 3):
                faces.add(tuple(sorted(face)))
        
        E = len(edges)
        F = len(faces)
        
        chi = V - E + F - C
        
        return {
            'V': V,
            'E': E,
            'F': F,
            'C': C,
            'chi_4': chi,
            'valid': abs(chi) < 1e-6
        }
    except Exception as e:
        print(f"ERROR computing hull: {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("TETRAHEDRAL CASCADE VERTEX DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Test cases with known parameters
    test_seeds = [
        [1, 1, 1, 1],      # V=5: Simplest case (a=b=c=1)
        [1, 3, 3, 3],      # V=6
        [3, 64, 64, 68],   # V=28: From paper
        [13, 44, 44, 44],  # V=60: From paper
    ]
    
    results = []
    
    for seed in test_seeds:
        result = analyze_seed(seed, verbose=True)
        results.append(result)
        
        # Verify Euler characteristic
        print(f"\nComputing 4D Convex Hull...")
        euler_data = verify_euler_characteristic(result['vertices_4d'])
        
        if euler_data:
            print(f"  V={euler_data['V']}, E={euler_data['E']}, F={euler_data['F']}, C={euler_data['C']}")
            print(f"  χ₄ = {euler_data['chi_4']}")
            print(f"  Status: {'✓ VALID' if euler_data['valid'] else '✗ INVALID'}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for result in results:
        seed = result['seed']
        a, b, c, d = result['params']
        dist = result['distribution']
        
        print(f"\nSeed {seed}:")
        print(f"  (a, b, c, d) = ({a:.1f}, {b:.1f}, {c:.1f}, {d:.1f})")
        print(f"  Predicted: 1×{a:.1f} + 2×{b:.1f} + 2×{c:.1f} + 1×{d:.1f} = {result['V_predicted']:.1f}")
        print(f"  Actual: {dist['A']}×A + {dist['B']}×B + {dist['C']}×C + {dist['D']}×D = {result['V_actual']}")
        
        # Check if distribution matches (1, 2, 2, 1) pattern
        ratio_B_to_A = dist['B'] / dist['A'] if dist['A'] > 0 else 0
        ratio_C_to_A = dist['C'] / dist['A'] if dist['A'] > 0 else 0
        
        print(f"  Ratios: B/A={ratio_B_to_A:.2f}, C/A={ratio_C_to_A:.2f}")
        
        if 1.8 < ratio_B_to_A < 2.2 and 1.8 < ratio_C_to_A < 2.2:
            print(f"  Pattern (1, 2, 2, 1): ✓ CONFIRMED")
        else:
            print(f"  Pattern (1, 2, 2, 1): ✗ NOT OBSERVED")
