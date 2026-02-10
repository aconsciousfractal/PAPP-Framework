"""
METATRON UNIVERSAL POLYTOPE GENERATOR
======================================

Phase 1.1 Implementation - Week 1
Generates 3D polytopes from Metatron's Cube seed using Grant's Projection Theorem

Author: HAN Framework
Date: February 2026
Status: ACTIVE DEVELOPMENT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


@dataclass
class MetatronSeed:
    """
    Metatron's Cube base structure
    13 circles (F7 Fibonacci) in 2D, 64 spheres in 3D
    """
    circles_2d: int = 13  # F7 Fibonacci
    spheres_3d: int = 64  # 2^6 Binary
    layers: int = 4       # 4D depth
    
    # Golden Lattice correction (Hentsch)
    golden_s: float = 0.874  # PHI**(-1.5) approx
    golden_r_deg: float = 22.24  # Rotation angle
    
    # Volume constants
    volume_quantum: int = 5904
    lucas_L2_cubed: int = 27  # L2^3
    universal_bridge: int = 159408  # F12 * L10 * L2^2


class GrantTriangleSolver:
    """
    Solves Grant's formula: V = a + 2b + c
    Where (a,b,c) is a Pythagorean triangle: a^2 + b^2 = c^2
    
    Face type k = 6 - integer_count(a, b, c)
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def is_integer(self, value: float, tolerance: float = 0.01) -> bool:
        """Check if value is close to an integer"""
        return abs(value - round(value)) < tolerance
    
    def verify_pythagorean(self, a: float, b: float, c: float) -> Tuple[bool, float]:
        """
        Verify Pythagorean relation: a^2 + b^2 = c^2
        
        Returns:
            (is_valid, error)
        """
        lhs = a**2 + b**2
        rhs = c**2
        error = abs(lhs - rhs)
        is_valid = error < self.tolerance
        
        return is_valid, error
    
    def count_integers(self, a: float, b: float, c: float) -> int:
        """Count how many of (a,b,c) are integers"""
        return sum([
            self.is_integer(a),
            self.is_integer(b),
            self.is_integer(c)
        ])
    
    def compute_face_type(self, a: float, b: float, c: float) -> int:
        """
        Compute k (face polygon type)
        k = 6 - integer_count
        
        Returns:
            k: 3=triangle, 4=quad, 5=pentagon, 6=hexagon
        """
        int_count = self.count_integers(a, b, c)
        k = 6 - int_count
        return k
    
    def solve_for_V_exact(self, V_target: int, k_target: int) -> Optional[Tuple[float, float, float]]:
        """
        Solve for triangle (a,b,c) given V and k
        
        Method: For k=5 (pentagons), we need 1 integer
        Try integer values of 'a' and solve for b
        
        System:
            V = a + 2b + c
            a^2 + b^2 = c^2
            integer_count(a,b,c) = 1
        
        Returns:
            (a, b, c) or None if no solution found
        """
        integer_count_needed = 6 - k_target
        
        if integer_count_needed == 1:  # k=5 (pentagons)
            # Try integer values of 'a'
            for a_val in range(1, int(V_target/2)):
                # From V = a + 2b + c and c^2 = a^2 + b^2
                # We get: V = a + 2b + sqrt(a^2 + b^2)
                # Solve for b using Newton-Raphson
                
                b_val = self._solve_b_for_given_a(a_val, V_target)
                
                if b_val is not None and b_val > 0:
                    c_val = np.sqrt(a_val**2 + b_val**2)
                    
                    # Verify constraints
                    is_pyth, error = self.verify_pythagorean(a_val, b_val, c_val)
                    V_check = a_val + 2*b_val + c_val
                    
                    if is_pyth and abs(V_check - V_target) < 0.01:
                        # Check integer count
                        if not self.is_integer(b_val) and not self.is_integer(c_val):
                            return (float(a_val), b_val, c_val)
        
        # If no solution found, return None
        return None
    
    def _solve_b_for_given_a(self, a: float, V: float, max_iter: int = 100) -> Optional[float]:
        """
        Solve for b given a and V using Newton-Raphson
        
        Equation: f(b) = a + 2b + sqrt(a^2 + b^2) - V = 0
        Derivative: f'(b) = 2 + b/sqrt(a^2 + b^2)
        """
        # Initial guess
        b = V / 3.0
        
        for _ in range(max_iter):
            # Function value
            f = a + 2*b + np.sqrt(a**2 + b**2) - V
            
            # Check convergence
            if abs(f) < 1e-10:
                return b
            
            # Derivative
            df = 2 + b / np.sqrt(a**2 + b**2)
            
            # Newton step
            b = b - f / df
            
            # Ensure b stays positive
            if b <= 0:
                return None
        
        return b if b > 0 else None


class PolytopeTopology:
    """
    Compute topological properties of polytope
    using Grant's formulas
    """
    
    @staticmethod
    def compute_topology(V: float, k: int) -> Dict[str, float]:
        """
        Compute F (faces) and E (edges) from V and k
        
        Formulas:
            F = 2(V - 2) / (k - 2)
            E = k * F / 2
            chi = V - E + F (should be 2)
        
        Returns:
            Dictionary with V, F, E, k, chi
        """
        if k <= 2:
            raise ValueError(f"k must be > 2, got {k}")
        
        F = 2 * (V - 2) / (k - 2)
        E = k * F / 2
        chi = V - E + F
        
        return {
            'V': V,
            'F': F,
            'E': E,
            'k': k,
            'chi': chi
        }
    
    @staticmethod
    def is_stable(V: int, k: int) -> bool:
        """
        Check if polytope is stable (F must be integer)
        
        Condition: 2(V-2) must be divisible by (k-2)
        """
        return (2 * (V - 2)) % (k - 2) == 0


class MetatronUniversalProjector:
    """
    Main class for generating polytopes from Metatron's Cube seed
    
    Usage:
        projector = MetatronUniversalProjector()
        polytope = projector.generate_polytope(V_target=26, k_target=5)
    """
    
    def __init__(self, seed: Optional[MetatronSeed] = None):
        self.seed = seed or MetatronSeed()
        self.solver = GrantTriangleSolver()
        self.topology = PolytopeTopology()
    
    def generate_polytope(self, V_target: int, k_target: int, name: Optional[str] = None) -> Dict:
        """
        Generate a polytope with V vertices and k-gon faces
        
        Args:
            V_target: Number of vertices
            k_target: Face type (3=tri, 4=quad, 5=penta, 6=hexa)
            name: Optional name for the polytope
        
        Returns:
            Dictionary containing all polytope properties
        """
        # Check stability
        if not self.topology.is_stable(V_target, k_target):
            print(f"WARNING: Polytope V={V_target}, k={k_target} is unstable (F non-integer)")
        
        # Solve for triangle
        triangle = self.solver.solve_for_V_exact(V_target, k_target)
        
        if triangle is None:
            raise ValueError(f"Could not find triangle for V={V_target}, k={k_target}")
        
        a, b, c = triangle
        
        # Verify Pythagorean
        is_pyth, pyth_error = self.solver.verify_pythagorean(a, b, c)
        
        # Compute topology
        topo = self.topology.compute_topology(V_target, k_target)
        
        # Compute Metatron ratio
        metatron_ratio = V_target / self.seed.circles_2d
        
        # Assemble result
        result = {
            'name': name or f"Polytope_V{V_target}_k{k_target}",
            'triangle': {
                'a': a,
                'b': b,
                'c': c
            },
            'topology': topo,
            'validation': {
                'pythagorean': is_pyth,
                'pythagorean_error': pyth_error,
                'integer_count': self.solver.count_integers(a, b, c),
                'V_check': a + 2*b + c,
                'chi_check': abs(topo['chi'] - 2) < 0.01
            },
            'metatron': {
                'ratio': metatron_ratio,
                'base_V': self.seed.circles_2d
            }
        }
        
        return result
    
        return family

    def generate_metatron_family_extended(self, max_m: int = 30, k: int = 6) -> List[Dict]:
        """
        V = 13(2 + 3m) for m = 0,1,2,...
        Targeting V=806 for m=20.
        """
        results = []
        for m in range(max_m + 1):
            V = 13 * (2 + 3*m)
            # Check stability for k=6 (hexagon faces)
            if not self.topology.is_stable(V, k):
                continue
            try:
                polytope = self.generate_polytope(V, k, name=f"Metatron_Family_m{m}")
                results.append(polytope)
                print(f"Generated: {polytope['name']} (V={V}, F={polytope['topology']['F']:.0f})")
            except Exception as e:
                print(f"Failed V={V}, m={m}: {e}")
        return results

    def find_fibonacci_lucas_triangles(self, max_c: int = 1000) -> List[Dict]:
        """
        Find Pythagorean triples (a,b,c) where at least one is Fibonacci or Lucas.
        V = a + 2b + c
        """
        def get_fib(n):
            F = [1, 1]
            for i in range(2, n): F.append(F[-1] + F[-2])
            return F
        def get_luc(n):
            L = [1, 3]
            for i in range(2, n): L.append(L[-1] + L[-2])
            return L
        
        special = set(get_fib(20) + get_luc(20))
        results = []
        for c in range(5, max_c + 1):
            for a in range(3, c):
                b_sq = c**2 - a**2
                b = int(np.sqrt(b_sq))
                if b**2 == b_sq and a < b:
                    if a in special or b in special or c in special:
                        V = a + 2*b + c
                        # Valid for any integer triple
                        results.append({
                            'a': a, 'b': b, 'c': c, 'V': V,
                            'fib_luc_match': [x for x in [a,b,c] if x in special]
                        })
        return results

    def test_enkehedron_terminal(self) -> bool:
        """
        Check if V=42 has any integer triangle solutions.
        c must be integer (Grant conjecture).
        """
        c = 42
        c_sq = c**2
        found = []
        for a in range(1, c):
            b_sq = c_sq - a**2
            b = int(np.sqrt(b_sq))
            if b**2 == b_sq and a < b < c:
                found.append((a, b, c))
        return len(found) == 0
    
    def export_to_json(self, polytope: Dict, filename: str):
        """Export polytope data to JSON file"""
        # Convert numpy types to Python native types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        polytope_clean = convert_types(polytope)
        
        with open(filename, 'w') as f:
            json.dump(polytope_clean, f, indent=2)
        print(f"Exported to {filename}")
    
    def print_polytope_summary(self, polytope: Dict):
        """Print human-readable summary of polytope"""
        print("\n" + "="*80)
        print(f"POLYTOPE: {polytope['name']}")
        print("="*80)
        
        tri = polytope['triangle']
        print(f"\nGenerating Triangle:")
        print(f"  a = {tri['a']:.6f}")
        print(f"  b = {tri['b']:.6f}")
        print(f"  c = {tri['c']:.6f}")
        
        topo = polytope['topology']
        print(f"\nTopology:")
        print(f"  V (vertices) = {topo['V']:.0f}")
        print(f"  F (faces)    = {topo['F']:.0f} {self._face_name(topo['k'])}s")
        print(f"  E (edges)    = {topo['E']:.0f}")
        print(f"  k (face type) = {topo['k']}")
        print(f"  chi (Euler)   = {topo['chi']:.2f}")
        
        val = polytope['validation']
        print(f"\nValidation:")
        print(f"  Pythagorean: {'PASS' if val['pythagorean'] else 'FAIL'} (error: {val['pythagorean_error']:.2e})")
        print(f"  Integer count: {val['integer_count']}")
        print(f"  V = a+2b+c: {val['V_check']:.6f}")
        print(f"  chi = 2: {'PASS' if val['chi_check'] else 'FAIL'}")
        
        met = polytope['metatron']
        print(f"\nMetatron Connection:")
        print(f"  Base V: {met['base_V']}")
        print(f"  Ratio: {met['ratio']:.2f}x")
        
        print("="*80 + "\n")
    
    def _face_name(self, k: int) -> str:
        """Get name of k-gon"""
        names = {3: 'triangle', 4: 'quadrilateral', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon'}
        return names.get(k, f'{k}-gon')


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("METATRON UNIVERSAL POLYTOPE GENERATOR - TEST RUN")
    print("="*80 + "\n")
    
    # Initialize projector
    projector = MetatronUniversalProjector()
    
    # Test 1: Generate Granthahedron (V=26, k=5)
    print("\n[TEST 1] Generating Granthahedron...")
    granthahedron = projector.generate_polytope(
        V_target=26,
        k_target=5,
        name="Granthahedron (Robert Grant, 2021)"
    )
    projector.print_polytope_summary(granthahedron)
    
    # Test 2: Generate Enkehedron equivalent (we know it's V=42, but k=7)
    # For now, test V=42 with k=5 to see if stable
    print("\n[TEST 2] Testing V=42 with k=5...")
    try:
        test_42 = projector.generate_polytope(V_target=42, k_target=5, name="V42_k5_test")
        projector.print_polytope_summary(test_42)
    except ValueError as e:
        print(f"V=42, k=5 failed: {e}")
    
    # Test 3: Generate Metatron family (V=13n, k=5)
    print("\n[TEST 3] Generating Metatron family (V=13n, k=5)...")
    family = projector.generate_metatron_family(max_multiplier=10, k=5)
    
    print(f"\nSuccessfully generated {len(family)} polytopes in family:")
    for poly in family:
        V = poly['topology']['V']
        F = poly['topology']['F']
        print(f"  {poly['name']:15s}: V={V:.0f}, F={F:.0f}")
    
    # Test 4: Export to JSON
    print("\n[TEST 4] Exporting Granthahedron to JSON...")
    projector.export_to_json(granthahedron, "granthahedron.json")
    
    print("\n" + "="*80)
    print("TEST RUN COMPLETE")
    print("="*80 + "\n")
