"""
Reverse-Engineer the Tetrahedral Cascade Algorithm
===================================================

GOAL: Understand why V = a + 2b + 2c + d works.

Strategy: Analyze the ACTUAL code logic without presuppositions.
"""

import numpy as np
from pathlib import Path
import sys

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir / "code_src"))
sys.path.insert(0, str(scripts_dir.parent.parent.parent / "scripts"))

from grant_4d_constructor import Grant4DParameters

def analyze_allocation_logic(a, b, c, d, n_points):
    """
    Reproduce the exact allocation logic from tetrahedral_cascade_4d
    """
    total = a + 2*b + 2*c + d
    
    # This is the ACTUAL code
    n_a = max(1, int(round(n_points * a / total)))
    n_b = max(1, int(round(n_points * 2*b / total)))
    n_c = max(1, int(round(n_points * 2*c / total)))
    n_d = n_points - n_a - n_b - n_c
    
    print(f"\n{'='*70}")
    print(f"ALLOCATION LOGIC ANALYSIS")
    print(f"{'='*70}")
    print(f"Parameters: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
    print(f"Target vertices: {n_points}")
    print(f"Formula total: a + 2b + 2c + d = {total:.2f}")
    print(f"\nAllocation fractions:")
    print(f"  Region A: {n_points} × ({a:.2f} / {total:.2f}) = {n_points * a / total:.2f} → {n_a}")
    print(f"  Region B: {n_points} × (2×{b:.2f} / {total:.2f}) = {n_points * 2*b / total:.2f} → {n_b}")
    print(f"  Region C: {n_points} × (2×{c:.2f} / {total:.2f}) = {n_points * 2*c / total:.2f} → {n_c}")
    print(f"  Region D: {n_points} - {n_a} - {n_b} - {n_c} = {n_d}")
    print(f"\nActual distribution: {n_a} + {n_b} + {n_c} + {n_d} = {n_a + n_b + n_c + n_d}")
    
    # KEY INSIGHT: Check if allocation preserves (1, 2, 2, 1) RATIOS
    if n_a > 0:
        ratio_b = n_b / n_a
        ratio_c = n_c / n_a
        ratio_d = n_d / n_a
        
        print(f"\nRatio Analysis (relative to Region A):")
        print(f"  B/A = {ratio_b:.3f}  (target: {2*b/a:.3f})")
        print(f"  C/A = {ratio_c:.3f}  (target: {2*c/a:.3f})")
        print(f"  D/A = {ratio_d:.3f}  (target: {d/a:.3f})")
    
    return n_a, n_b, n_c, n_d

def theorem_statement_from_code():
    """
    Derive the theorem from ACTUAL code behavior.
    """
    print(f"\n{'='*70}")
    print("THEOREM DERIVATION FROM CODE")
    print(f"{'='*70}")
    
    print("""
The tetrahedral_cascade_4d algorithm works as follows:

1. INPUT: Grant parameters (a, b, c, d) and target vertex count n
2. FORMULA: total = a + 2b + 2c + d
3. ALLOCATION:
   - Region A: n_a = round(n × a / total)
   - Region B: n_b = round(n × 2b / total)  [DOUBLED!]
   - Region C: n_c = round(n × 2c / total)  [DOUBLED!]
   - Region D: n_d = n - n_a - n_b - n_c
4. GUARANTEE: n_a + n_b + n_c + n_d = n (by construction)

CRITICAL OBSERVATION:
The formula V = a + 2b + 2c + d is NOT derived.
It is ASSUMED in the allocation step (line: total = a + 2*b + 2*c + d).

CIRCULAR REASONING:
- Code uses formula to allocate vertices
- Allocation produces vertex count matching formula
- But formula is not proven, just encoded!

REAL QUESTION:
Why does THIS allocation scheme produce valid 4D convex polytopes
with χ₄ = 0?
    """)

def search_for_geometric_interpretation():
    """
    Try to find geometric meaning of (1, 2, 2, 1) pattern.
    """
    print(f"\n{'='*70}")
    print("GEOMETRIC INTERPRETATION SEARCH")
    print(f"{'='*70}")
    
    print("""
HYPOTHESIS 1: Orbit multiplicity under B_4 symmetry
- B_4 has orbit sizes: 8, 48, 16, 16 for different vertex types
- Doesn't match (a, 2b, 2c, d) directly unless specific scaling

HYPOTHESIS 2: Hopf fibration fiber structure
- S³ → S² has fibers S¹ (circles)
- Each base point has 1D fiber
- Could (1, 2, 2, 1) reflect base + fiber decomposition?

HYPOTHESIS 3: Dimensional cascade
- Start with a (1D boundary points)
- Add 2b (2D layer, doubled for +/- symmetry)
- Add 2c (3D layer, doubled for +/- symmetry)  
- Add d (4D interior/apex)
- Total: a + 2b + 2c + d

HYPOTHESIS 4: Quaternion basis decomposition
- Quaternions: q = a₀·1 + a₁·i + a₂·j + a₃·k
- Could (1, 2, 2, 1) reflect:
  * 1 real part (a)
  * 2 imaginary pairs: (i,j) → 2b, (k,w) → 2c
  * 1 norm (d)

HYPOTHESIS 5 (MOST PROMISING): Layer-by-layer stacking
- 3D Grant: V₃D = a + 2b + c
- Extend to 4D by adding layers:
  * Keep a (1D boundary)
  * Keep 2b (2D layers)
  * DOUBLE c → 2c (3D layers need +/- w direction!)
  * Add d (4D radius/apex)
- Formula: V₄D = a + 2b + 2c + d ✓

This explains the +c term naturally!
    """)

def verify_hypothesis_5():
    """
    Test Hypothesis 5: 3D → 4D layer extension
    """
    print(f"\n{'='*70}")
    print("TESTING HYPOTHESIS 5: Layer Extension")
    print(f"{'='*70}")
    
    print("""
Grant 3D Formula: V₃D = a + 2b + c
Interpretation: 
- a: Boundary points (1D edges)
- 2b: Layer 1 (2D face structure)
- c: Layer 2 (3D volume apex)

Extension to 4D:
- a: Boundary (unchanged)
- 2b: Layer 1 (2D, unchanged)
- c → 2c: Layer 2 (3D, DOUBLED for ±w direction)
- d: Layer 3 (4D apex)

Result: V₄D = a + 2b + 2c + d

EVIDENCE NEEDED:
1. Show that 3D Grant structure exists in 4D cross-section
2. Prove doubling of c is geometrically necessary
3. Explain role of d as 4D "depth"
    """)

if __name__ == "__main__":
    # Test with simple case
    print("="*70)
    print("RIGOROUS ANALYSIS: TETRAHEDRAL CASCADE ALGORITHM")
    print("="*70)
    
    # Test Case 1: Simplest (a=b=c=1)
    print("\n### TEST CASE 1: Symmetric (1, 1, 1, √3)")
    a, b, c = 1, 1, 1.414
    d = np.sqrt(a**2 + b**2 + c**2)
    V_pred = a + 2*b + 2*c + d
    analyze_allocation_logic(a, b, c, d, int(round(V_pred)))
    
    # Test Case 2: Scaled
    print("\n### TEST CASE 2: Scaled (3, 3, 4.24, 6)")
    a, b, c = 3, 3, 4.24
    d = np.sqrt(a**2 + b**2 + c**2)
    V_pred = a + 2*b + 2*c + d
    analyze_allocation_logic(a, b, c, d, int(round(V_pred)))
    
    # Derive theorem
    theorem_statement_from_code()
    
    # Search for geometric meaning
    search_for_geometric_interpretation()
    
    # Test hypothesis
    verify_hypothesis_5()
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
STATUS: Formula V = a + 2b + 2c + d is ALGORITHMICALLY VERIFIED
        but NOT GEOMETRICALLY DERIVED.

RECOMMENDATION: 
1. Rename from "Theorem" to "Construction Algorithm"
2. Focus on proving χ₄ = 0 for this construction
3. Find geometric interpretation of (1, 2, 2, 1) pattern

STRONGEST LEAD: Hypothesis 5 (Layer Extension)
- Explains +c naturally as dimensional doubling
- Connects to Grant 3D formula
- Testable via cross-section analysis
    """)
