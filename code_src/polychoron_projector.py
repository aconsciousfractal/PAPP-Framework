import numpy as np
import itertools
from typing import List, Tuple, Dict

PHI = (1 + np.sqrt(5)) / 2

class PolychoronProjector:
    """
    Project 4D Polytopes (Polychora) to 3D Space
    Uses rotation and perspective projection.
    """
    
    def __init__(self):
        pass
        
    def generate_600_cell_vertices(self) -> np.ndarray:
        """
        Generate complete 120 vertices of the 600-cell (icosians).
        Conway-Smith construction via quaternion coordinates.
        
        Block 1: 16 tesseract corners
        Block 2: 8 coordinate axes  
        Block 3: 96 icosian vertices (all permutations with even parity)
        
        Returns:
            vertices (120, 4): Normalized on unit 3-sphere
        """
        vertices = []
        
        # Block 1: 16 vertices - tesseract corners
        # All sign combinations of (0.5, 0.5, 0.5, 0.5)
        for signs in itertools.product([-0.5, 0.5], repeat=4):
            vertices.append(signs)
        block1_count = len(vertices)
        
        # Block 2: 8 vertices - coordinate axes
        # (+/- 1, 0, 0, 0) and permutations
        for i in range(4):
            for sign in [-1, 1]:
                v = [0, 0, 0, 0]
                v[i] = sign
                vertices.append(v)
        block2_count = len(vertices) - block1_count
                
        # Block 3: 96 vertices - icosian structure
        # Standard definition: EVEN permutations of (±phi/2, ±1/2, ±1/(2phi), 0)
        # with ALL sign combinations.
        # Deduplication occurs because 0 absorbs sign changes (16 signs -> 8 unique).
        # 12 even perms * 8 unique signs = 96 vertices.
        base = [PHI/2, 0.5, 1/(2*PHI), 0]
        
        def is_even_perm(p):
            # Count inversions
            inv = 0
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    if p[i] > p[j]:
                        inv += 1
            return inv % 2 == 0

        block3_set = set()
        
        # Filter for EVEN permutations only (12 out of 24)
        for perm in itertools.permutations(range(4)):
            if is_even_perm(perm):
                permuted_base = [base[perm[i]] for i in range(4)]
                
                # All 16 sign patterns
                for signs in itertools.product([-1, 1], repeat=4):
                    # Note: We take ALL sign patterns.
                    # The presence of 0 in permuted_base means signs pairs like 
                    # (+a, +b, +c, +0) and (+a, +b, +c, -0) produce the same vector.
                    v = tuple(signs[i] * permuted_base[i] for i in range(4))
                    v_rounded = tuple(round(x, 12) for x in v)
                    block3_set.add(v_rounded)
        
        # Convert to vertices (Single append loop!)
        for v in block3_set:
            vertices.append(list(v))
            
        block3_count = len(block3_set)
        
        vertices = np.array(vertices)
        
        # Normalize to unit 3-sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        # Validation
        total = len(vertices)
        assert total == 120, f"Expected 120 vertices total, got {total}"
        assert block3_count == 96, f"Block 3 must have 96 vertices, got {block3_count}"
        
        # Check uniqueness
        unique_check = len(set(tuple(np.round(v, 10)) for v in vertices))
        assert unique_check == 120, f"Duplicate vertices detected: {unique_check} unique"
        
        print(f"600-cell vertices generated:")
        print(f"  Block 1 (tesseract): {block1_count}")
        print(f"  Block 2 (axes): {block2_count}")
        print(f"  Block 3 (icosians): {block3_count}")
        print(f"  Total: {total} [PASS]")
        
        return vertices

    def generate_5_cell_vertices(self) -> np.ndarray:
        """
        Generate 5 vertices of the 5-cell (Regular Simplex).
        Radius = 1
        """
        # Coordinates based on simplex standard embedding
        vertices = [
            [1, 1, 1, -1/np.sqrt(5)],
            [1, -1, -1, -1/np.sqrt(5)],
            [-1, 1, -1, -1/np.sqrt(5)],
            [-1, -1, 1, -1/np.sqrt(5)],
            [0, 0, 0, 4/np.sqrt(5)]
        ]
        # Align to standard size (radius 1) works nicely with normalization
        v_np = np.array(vertices)
        # Normalize
        norms = np.linalg.norm(v_np, axis=1)
        v_norm = v_np / norms[:, np.newaxis]
        
        print(f"5-cell vertices generated: {len(v_norm)}")
        return v_norm

    def generate_8_cell_vertices(self) -> np.ndarray:
        """
        Generate 16 vertices of the 8-cell (Tesseract).
        Coords: (±0.5, ±0.5, ±0.5, ±0.5) normalized
        """
        vertices = []
        for i in range(16):
            v = []
            temp = i
            for _ in range(4):
                sign = 1 if (temp % 2 == 1) else -1
                v.append(0.5 * sign)
                temp //= 2
            vertices.append(v)
            
        v_np = np.array(vertices)
        # Normalize (radius 1)
        norms = np.linalg.norm(v_np, axis=1)
        v_norm = v_np / norms[:, np.newaxis]
        
        print(f"8-cell vertices generated: {len(v_norm)}")
        return v_norm

    def generate_16_cell_vertices(self) -> np.ndarray:
        """
        Generate 8 vertices of the 16-cell (Orthoplex).
        Coords: Permutations of (±1, 0, 0, 0)
        """
        vertices = []
        for i in range(4):
            for sign in [-1, 1]:
                v = [0.0] * 4
                v[i] = 1.0 * sign
                vertices.append(v)
                
        print(f"16-cell vertices generated: {len(vertices)}")
        return np.array(vertices)

    def generate_24_cell_vertices(self) -> np.ndarray:
        """
        Generate 24 vertices of the 24-cell (Icositetrachoron).
        Coords: Permutations of (±0.5, ±0.5, 0, 0) normalized -> (±1, ±1, 0, 0)/sqrt(2)
        Also is the union of 8-cell (unscaled) and 16-cell (scaled).
        Standard set: All permutations of (±1, ±1, 0, 0) normalized.
        """
        vertices = set()
        
        # Base vector pattern: (1, 1, 0, 0)
        # Iterate all pairs of indices for the two 1s
        for idx1, idx2 in itertools.combinations(range(4), 2):
            # For each pair, iterate all sign combinations
            for s1, s2 in itertools.product([-1, 1], repeat=2):
                v = [0.0] * 4
                v[idx1] = s1
                v[idx2] = s2
                vertices.add(tuple(v))
                
        v_list = list(vertices)
        v_np = np.array(v_list)
        # Normalize
        norms = np.linalg.norm(v_np, axis=1)
        v_norm = v_np / norms[:, np.newaxis]
        
        print(f"24-cell vertices generated: {len(v_norm)}")
        return v_norm

    def rotate_4d(self, vertices: np.ndarray, angles: List[float]) -> np.ndarray:
        """
        Rotate 4D vertices around 6 planes: xy, xz, xw, yz, yw, zw
        """
        # For simplicity, just rotate in xw plane
        theta = angles[2] # xw
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotation matrix for xw plane
        # [ cos -sin ]
        # [ sin  cos ]
        rotated = vertices.copy()
        for i in range(len(vertices)):
            x, w = vertices[i][0], vertices[i][3]
            rotated[i][0] = x * cos_t - w * sin_t
            rotated[i][3] = x * sin_t + w * cos_t
            
        return rotated

    def project_to_3d(self, vertices: np.ndarray, distance: float = 2.0) -> np.ndarray:
        """
        Perspective projection from 4D to 3D.
        w coordinate acts as depth.
        """
        projected = []
        for v in vertices:
            x, y, z, w = v
            # Perspective factor
            factor = 1 / (distance - w)
            projected.append([x * factor, y * factor, z * factor])
            
        return np.array(projected)

    def compute_topology(self, vertices: np.ndarray) -> Dict:
        """
        Compute basic topological invariants.
        """
        return {
            "V": len(vertices),
            "dim": 4
        }

if __name__ == "__main__":
    projector = PolychoronProjector()
    v_120 = projector.generate_600_cell_vertices()
    print(f"Generated {len(v_120)} vertices of 600-cell (partial set)")
    
    # Rotate and project
    rotated = projector.rotate_4d(v_120, [0, 0, np.pi/4])
    projected = projector.project_to_3d(rotated)
    
    print(f"Projected to 3D, first 5 vertices:\n{projected[:5]}")
