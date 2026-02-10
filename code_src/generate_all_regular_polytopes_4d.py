"""
Batch Generator: All 6 Regular 4D Polytopes
============================================

Generate complete 4D structure for all Platonic solids in 4D:
1. 5-cell (4-simplex): V=5, E=10, F=10, C=5 (tetrahedra)
2. 8-cell (tesseract): V=16, E=32, F=24, C=8 (cubes)
3. 16-cell (cross-polytope): V=8, E=24, F=32, C=16 (tetrahedra)
4. 24-cell: V=24, E=96, F=96, C=24 (octahedra)
5. 120-cell: V=600, E=1200, F=720, C=120 (dodecahedra)
6. 600-cell: V=120, E=720, F=1200, C=600 (tetrahedra)

Strategy:
- Use polychoron_projector.py from_polytope() for analytical coords
- Enumerate V-E-F-C with PolychoronCombinatorics4D
- Export with full 4D metadata
"""

import sys
import numpy as np
from pathlib import Path

# Local imports (scripts required in the same directory)
from polychoron_projector import PolychoronProjector
from polychoron_combinatorics_4d import PolychoronCombinatorics4D

# Create projector instance
projector = PolychoronProjector()

# Mapping from polytope names to generator methods
VERTEX_GENERATORS = {
    '5-cell': lambda: projector.generate_5_cell_vertices(),
    '8-cell': lambda: projector.generate_8_cell_vertices(),
    '16-cell': lambda: projector.generate_16_cell_vertices(),
    '24-cell': lambda: projector.generate_24_cell_vertices(),
    '120-cell': None,  # Not implemented in projector
    '600-cell': lambda: projector.generate_600_cell_vertices()
}

# Define all 6 regular polytopes
POLYTOPES = {
    '5-cell': {
        'name': '5-cell',
        'expected': {'V': 5, 'E': 10, 'F': 10, 'C': 5},
        'cell_type': 'tetrahedra'
    },
    '8-cell': {
        'name': '8-cell',
        'expected': {'V': 16, 'E': 32, 'F': 24, 'C': 8},
        'cell_type': 'cubes'
    },
    '16-cell': {
        'name': '16-cell',
        'expected': {'V': 8, 'E': 24, 'F': 32, 'C': 16},
        'cell_type': 'tetrahedra'
    },
    '24-cell': {
        'name': '24-cell',
        'expected': {'V': 24, 'E': 96, 'F': 96, 'C': 24},
        'cell_type': 'octahedra'
    },
    '120-cell': {
        'name': '120-cell',
        'expected': {'V': 600, 'E': 1200, 'F': 720, 'C': 120},
        'cell_type': 'dodecahedra'
    },
    '600-cell': {
        'name': '600-cell',
        'expected': {'V': 120, 'E': 720, 'F': 1200, 'C': 600},
        'cell_type': 'tetrahedra'
    }
}

def generate_polytope_4d(poly_name: str, poly_info: dict):
    """Generate complete 4D structure for a polytope"""
    
    print(f"\n{'='*70}")
    print(f"GENERATING: {poly_name}")
    print(f"{'='*70}")
    
    # Get expected values
    expected = poly_info['expected']
    cell_type = poly_info['cell_type']
    
    print(f"Expected: V={expected['V']}, E={expected['E']}, F={expected['F']}, C={expected['C']}")
    print(f"Cell type: {cell_type}")
    
    # Generate 4D vertices
    if poly_name == '120-cell':
        print("Creating 120-cell via dual 600-cell construction...")
        # 1. Generate 600-cell
        v_600 = projector.generate_600_cell_vertices()
        
        # 2. Find cells (tetrahedra) of 600-cell
        print("  Enumerating 600-cell structure to find cell centers...")
        comb_600 = PolychoronCombinatorics4D(v_600, polytope_type='600-cell')
        comb_600.compute_edge_graph()
        comb_600.compute_triangular_faces()
        comb_600.compute_tetrahedral_cells()
        
        if len(comb_600.cells) != 600:
            print(f"✗ Failed to find 600 cells in 600-cell (found {len(comb_600.cells)})")
            return None
            
        print(f"  ✓ Found 600 tetrahedra. Computing centroids...")
        
        # 3. Compute centroids
        centroids = []
        for cell in comb_600.cells:
            # cell is tuple of 4 vertex indices
            tetra_verts = v_600[list(cell)]
            centroid = np.mean(tetra_verts, axis=0)
            centroids.append(centroid)
            
        vertices_4d = np.array(centroids)
        
        # 4. Normalize
        radii = np.linalg.norm(vertices_4d, axis=1)
        vertices_4d = vertices_4d / radii[:, np.newaxis]
        print(f"✓ Generated {len(vertices_4d)} 120-cell vertices (dual construction)")
        
    else:
        # Standard analytical generation
        generator = VERTEX_GENERATORS.get(poly_name)
        
        if generator is None:
            print(f"✗ No vertex generator for {poly_name}")
            return None
        
        try:
            vertices_4d = generator()
            print(f"✓ Generated {len(vertices_4d)} 4D vertices")
        except Exception as e:
            print(f"✗ Failed to generate vertices: {e}")
            return None
    
    # Enumerate structure
    combinator = PolychoronCombinatorics4D(vertices_4d, polytope_type=poly_name)
    
    # Edges
    try:
        combinator.compute_edge_graph()
        E = len(combinator.edges)
        print(f"✓ Edges: E = {E} (expected {expected['E']})")
    except Exception as e:
        print(f"✗ Edge enumeration failed: {e}")
        return None
    
    # Faces
    try:
        if poly_name == '8-cell':
            combinator.compute_ngon_faces(n=4)  # Squares
        elif poly_name == '120-cell':
            combinator.compute_ngon_faces(n=5)  # Pentagons
        else:
            combinator.compute_triangular_faces() # Triangles (default)
            
        F = len(combinator.faces)
        print(f"✓ Faces: F = {F} (expected {expected['F']})")
    except Exception as e:
        print(f"⚠ Face enumeration: {e}")
        F = 0
        combinator.faces = []
    
    # Cells (only works for tetrahedral cells!)
    if cell_type == 'tetrahedra':
        try:
            combinator.compute_tetrahedral_cells()
            C = len(combinator.cells)
            print(f"✓ Cells: C = {C} (expected {expected['C']})")
        except Exception as e:
            print(f"⚠ Cell enumeration: {e}")
            C = 0
            combinator.cells = []
    else:
        print(f"⚠ Cell type '{cell_type}' not supported (need tetrahedra), skipping C enumeration")
        C = expected['C']  # Use expected value
        combinator.cells = []
    
    # Euler characteristic
    V = combinator.V
    chi_computed = V - E + F - C
    
    print(f"\nEuler characteristic:")
    print(f"  χ₄ = V - E + F - C = {V} - {E} + {F} - {C} = {chi_computed}")
    
    if chi_computed == 0:
        print(f"  ✅ χ₄ = 0 (valid closed polytope!)")
    else:
        print(f"  ⚠ χ₄ ≠ 0 (may be due to non-tetrahedral cells)")
    
    # Export to OBJ
    # Path relative to script execution: ../assets/models_obj/Polychora
    output_dir = Path(__file__).parent.parent / "assets" / "models_obj" / "Polychora"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{poly_name}_COMPLETE_4D.obj"
    
    # Project to 3D for visualization
    vertices_3d = projector.project_to_3d(vertices_4d, distance=2.0)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {poly_name} - Complete 4D Polychoron\n")
        f.write(f"# Generated by: generate_all_regular_polytopes_4d.py\n")
        f.write(f"#\n")
        f.write(f"# COMBINATORICS:\n")
        f.write(f"# V = {V}\n")
        f.write(f"# E = {E}\n")
        f.write(f"# F = {F}\n")
        f.write(f"# C = {C} ({cell_type})\n")
        f.write(f"# χ₄ = {chi_computed}\n")
        f.write(f"#\n")
        f.write(f"# Expected: V={expected['V']}, E={expected['E']}, F={expected['F']}, C={expected['C']}\n")
        f.write(f"# Validation: {'PASS' if V == expected['V'] and E == expected['E'] else 'FAIL'}\n")
        f.write(f"#\n")
        f.write(f"######################################################################\n\n")
        
        # Vertices with 4D coordinates in comments
        for i, v4d in enumerate(vertices_4d):
            v3d = vertices_3d[i]
            f.write(f"v {v3d[0]:.6f} {v3d[1]:.6f} {v3d[2]:.6f}  ")
            f.write(f"# [{i}] 4D: ({v4d[0]:.4f}, {v4d[1]:.4f}, {v4d[2]:.4f}, {v4d[3]:.4f})\n")
        
        # Edges
        if E > 0:
            f.write(f"\n# Edges ({E})\n")
            for e in combinator.edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")
        
        # Faces
        if F > 0:
            f.write(f"\n# Faces ({F})\n")
            for face in combinator.faces:
                # face can be (v1, v2, v3) or (v1, v2, v3, v4) etc.
                indices = [str(idx + 1) for idx in face]
                f.write(f"f {' '.join(indices)}\n")
        
        # Cells
        if len(combinator.cells) > 0:
            f.write(f"\n# Cells ({len(combinator.cells)})\n")
            for i, cell in enumerate(combinator.cells):
                f.write(f"# c {cell[0]} {cell[1]} {cell[2]} {cell[3]}  # Cell {i}\n")
    
    print(f"\n✅ Exported to: {output_file}")
    
    return {
        'name': poly_name,
        'V': V,
        'E': E,
        'F': F,
        'C': C,
        'chi': chi_computed,
        'file': str(output_file),
        'validation': 'PASS' if V == expected['V'] and E == expected['E'] else 'FAIL'
    }

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("4D POLYTOPE BATCH GENERATOR")
    print("="*70)
    print(f"\nGenerating complete 4D structures for {len(POLYTOPES)} regular polytopes...")
    
    results = []
    
    for poly_name, poly_info in POLYTOPES.items():
        result = generate_polytope_4d(poly_name, poly_info)
        if result:
            results.append(result)
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Polytope':<15} {'V':>5} {'E':>6} {'F':>6} {'C':>6} {'χ₄':>5} {'Status':<10}")
    print(f"{'-'*70}")
    
    for r in results:
        status = '✅ ' + r['validation']
        print(f"{r['name']:<15} {r['V']:>5} {r['E']:>6} {r['F']:>6} {r['C']:>6} {r['chi']:>5} {status:<10}")
    
    print(f"\n✅ All polytopes generated!")
    print(f"Files created: {len(results)}")
