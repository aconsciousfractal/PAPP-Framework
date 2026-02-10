"""
Reconstruct PAPP Irregular Polychora (4D)
==========================================

Reconstructs the complete 4D topological structure of PAPP-generated irregular polychora
from quantum metric seed data.

Input:  Element_V*_phi_gap_*_QUANTUM_METRIC.obj (3D projections)
Output: Element_V*_RECONSTRUCTED_4D.obj (full 4D V-E-F-C structure)

The reconstruction process:
1. Extracts phi-gap seed [k1, k2, k3, k4] from filename
2. Computes Grant 4D parameters via Phi-Gap mechanism
3. Generates 4D vertex cloud using tetrahedral cascade
4. Computes 4D convex hull topology (Qhull)
5. Projects to 3D via Hopf fibration
6. Validates 4D Euler characteristic: χ₄ = V - E + F - C = 0
7. Exports OBJ with complete combinatorial data

Usage:
    python reconstruct_papp_polychora.py [--start INDEX] [--end INDEX]

Arguments:
    --start   Starting index in sorted array (default: 0)
    --end     Ending index in sorted array (default: 500)

Example:
    python reconstruct_papp_polychora.py --start 0 --end 500

Author: PAPP Protocol Development Team
License: See LICENSE file
"""

import sys
import os
import re
import argparse
import numpy as np
from pathlib import Path

# Setup paths for PAPP repository structure
repo_root = Path(__file__).parent.parent
code_src_dir = repo_root / "code_src"
assets_dir = repo_root / "assets" / "models_obj"

# For Grant modules, check if they exist in code_src, otherwise use Metatron scripts
grant_modules_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "scripts"

sys.path.insert(0, str(code_src_dir))
if grant_modules_dir.exists():
    sys.path.insert(0, str(grant_modules_dir))

try:
    from grant_4d_constructor import Grant4DParameters, Grant4DPolychoron
    from test_phi_gap_mechanism import PhiGapMechanism
    from polychoron_combinatorics_4d import PolychoronCombinatorics4D
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print(f"Required modules:")
    print(f"  - grant_4d_constructor.py")
    print(f"  - test_phi_gap_mechanism.py")
    print(f"  - polychoron_combinatorics_4d.py")
    print(f"\nSearched in:")
    print(f"  - {code_src_dir}")
    print(f"  - {grant_modules_dir}")
    sys.exit(1)


def parse_seed_from_filename(filename):
    """
    Extract phi-gap seed [k1, k2, k3, k4] from PAPP filename.
    
    Args:
        filename: Element_V*_phi_gap_k1_k2_k3_k4_QUANTUM_METRIC.obj
    
    Returns:
        List[int] or None
    """
    match = re.search(r"phi_gap_(\d+)_(\d+)_(\d+)_(\d+)", filename)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return None


def get_grant_params_from_seed(seed):
    """
    Compute Grant 4D parameters (a,b,c,d) from phi-gap seed via RG flow.
    
    Args:
        seed: [k1, k2, k3, k4] integer array
    
    Returns:
        Grant4DParameters instance
    """
    pg = PhiGapMechanism(seed)
    
    # Suppress verbose output during computation
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        pg.step_1_beta_helix()
        pg.step_3_rg_flow()
        pg.step_4_5_triple()
    finally:
        sys.stdout = original_stdout
    
    a, b, c = pg.triple
    d = np.sqrt(a**2 + b**2 + c**2)
    
    return Grant4DParameters(a=float(a), b=float(b), c=float(c), d=float(d))


def reconstruct_element(filename, target_name, output_dir):
    """
    Reconstruct full 4D topology of a single PAPP element.
    
    Args:
        filename: Source OBJ filename
        target_name: Output identifier (e.g., "Element_V5")
        output_dir: Target directory for reconstructed OBJ
    """
    # 1. Extract seed from filename
    seed = parse_seed_from_filename(str(filename))
    if not seed:
        print(f"  WARNING: Could not parse seed from {filename}, skipping")
        return
    
    print(f"\nRECONSTRUCTING {target_name} (Seed: {seed})")
    
    # 2. Compute Grant 4D parameters
    params = get_grant_params_from_seed(seed)
    
    # 3. Generate 4D polychoron structure
    poly = Grant4DPolychoron(params, name=target_name)
    vertices_4d = poly.generate_vertices_4d(method='tetrahedral_cascade')
    vertices_3d = poly.project_to_3d(method='hopf')
    
    # 4. Compute 4D convex hull topology
    try:
        combinatorics = PolychoronCombinatorics4D.from_point_cloud(vertices_4d)
    except Exception as e:
        print(f"  ERROR: Topology computation failed: {e}")
        return
    
    # Extract combinatorial data
    V = len(combinatorics.vertices)
    E = len(combinatorics.edges)
    F = len(combinatorics.faces)
    C = len(combinatorics.cells)
    
    # 5. Validate 4D Euler characteristic
    chi = V - E + F - C
    status = "VALID" if chi == 0 else "INVALID"
    
    print(f"  Hull: V={V}, E={E}, F={F}, C={C}")
    print(f"  Chi_4 = {chi} [{status}]")
    
    # 6. Export to OBJ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{target_name}_RECONSTRUCTED_4D.obj"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header with metadata
        f.write(f"# PAPP Irregular Polychoron - 4D Reconstruction\n")
        f.write(f"# Element: {target_name}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"#\n")
        f.write(f"# 4D COMBINATORICS:\n")
        f.write(f"#   Vertices (V): {V}\n")
        f.write(f"#   Edges    (E): {E}\n")
        f.write(f"#   Faces    (F): {F}\n")
        f.write(f"#   Cells    (C): {C}\n")
        f.write(f"#\n")
        f.write(f"# EULER CHARACTERISTIC: Chi_4 = V - E + F - C = {chi}\n")
        f.write(f"# STATUS: {status}\n")
        f.write(f"#\n")
        f.write(f"# Projection: Hopf fibration S^3 -> S^2\n")
        f.write(f"# Coordinates: 3D stereographic projection of 4D boundary\n")
        f.write(f"#\n")
        f.write(f"{'#' * 70}\n\n")
        
        # Vertices (3D projection)
        f.write(f"# Vertices ({len(vertices_3d)} projected from 4D)\n")
        for v3 in vertices_3d:
            f.write(f"v {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
        
        # Edges
        f.write(f"\n# Edges ({E})\n")
        for i, j in combinatorics.edges:
            f.write(f"l {i+1} {j+1}\n")
        
        # Faces (triangular)
        f.write(f"\n# Faces - Triangular ({F})\n")
        for face in combinatorics.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Cells (as comments - OBJ format limitation)
        f.write(f"\n# Cells - Tetrahedral ({C})\n")
        f.write(f"# Note: OBJ format does not support 4D cells natively\n")
        f.write(f"# Cell data preserved in comments for reference\n")
        for idx, cell in enumerate(combinatorics.cells):
            f.write(f"# c {cell[0]+1} {cell[1]+1} {cell[2]+1} {cell[3]+1}  # Cell {idx+1}\n")
    
    print(f"  Exported: {output_file.name}")


def main():
    """Main reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="Reconstruct PAPP irregular polychora 4D topology"
    )
    parser.add_argument(
        '--start', type=int, default=0,
        help='Starting index in sorted element array (default: 0)'
    )
    parser.add_argument(
        '--end', type=int, default=500,
        help='Ending index in sorted element array (default: 500)'
    )
    args = parser.parse_args()
    
    # Locate source files
    quantum_metrics_dir = assets_dir / "1111 obj" / "1111 obj Quantum Metrics"
    output_dir = assets_dir / "PAPP Polychora 4D_Reconstructed"
    
    if not quantum_metrics_dir.exists():
        print(f"ERROR: Source directory not found: {quantum_metrics_dir}")
        sys.exit(1)
    
    # Scan for quantum metric files
    all_files = list(quantum_metrics_dir.glob("Element_V*_QUANTUM_METRIC.obj"))
    print(f"\n{'='*70}")
    print(f"PAPP IRREGULAR POLYCHORA - 4D RECONSTRUCTION")
    print(f"{'='*70}")
    print(f"Source: {quantum_metrics_dir.name}/")
    print(f"Total files: {len(all_files)}")
    
    # Extract and sort by V-number
    def extract_v_number(filepath):
        """Extract numeric V value from Element_V123_*.obj"""
        stem = filepath.stem
        parts = stem.split('_')
        if len(parts) >= 2 and parts[1].startswith('V'):
            try:
                return int(parts[1][1:])
            except ValueError:
                return 999999
        return 999999
    
    sorted_files = sorted(all_files, key=extract_v_number)
    
    # Select processing range
    targets = sorted_files[args.start:args.end]
    
    if not targets:
        print(f"ERROR: No files in range [{args.start}:{args.end}]")
        sys.exit(1)
    
    v_start = extract_v_number(targets[0])
    v_end = extract_v_number(targets[-1])
    
    print(f"\nProcessing range: [{args.start}:{args.end}]")
    print(f"Elements: V{v_start} to V{v_end} ({len(targets)} files)")
    print(f"Output: {output_dir.name}/")
    print(f"\n{'='*70}\n")
    
    # Process each element
    success_count = 0
    fail_count = 0
    
    for idx, filepath in enumerate(targets, 1):
        v_num = extract_v_number(filepath)
        target_name = f"Element_V{v_num}"
        
        print(f"[{idx}/{len(targets)}] {target_name}...", end=" ")
        
        try:
            reconstruct_element(filepath.name, target_name, output_dir)
            success_count += 1
        except Exception as e:
            print(f"\n  [FAILED] {e}")
            fail_count += 1
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {success_count}/{len(targets)}")
    print(f"Failed: {fail_count}/{len(targets)}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
