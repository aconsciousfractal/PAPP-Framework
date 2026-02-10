import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from collections import defaultdict
import sys
import os

# Usage: python phylogenetic_tree_generator.py <atlas_path>

def parse_atlas_lazy(filename):
    print(f"Scanning Atlas {filename} for species (Fast Mode)...")
    
    species_map = {} # Key: (V, Triple_String) -> Seed
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Quick check for table row
            if not line.startswith("| `["): continue
            
            # Fast Split
            # Example: | `[4, 4, 4, 7]` | 63 | 2 | (8.0, 15.0, 17.0) | Stable |
            parts = line.split('|')
            if len(parts) < 5: continue
            
            try:
                # Seed is in parts[1] -> `[4, 4, 4, 7]`
                seed_str = parts[1].strip().strip('`')
                
                # V is in parts[2] -> 63
                v_str = parts[2].strip()
                
                # Triple is in parts[4] -> (8.0, 15.0, 17.0)
                triple_str = parts[4].strip()
                
                # Validation
                if not seed_str.startswith('[') or not triple_str.startswith('('):
                    continue
                    
                phenotype_key = (int(v_str), triple_str)
                
                if phenotype_key not in species_map:
                    # Parse seed only when needed
                    seed = eval(seed_str)
                    species_map[phenotype_key] = seed
            except:
                continue
    
    print(f"Found {len(species_map)} unique species.")
    return species_map

def generate_phylo_tree(species_map, output_prefix):
    print("Calculating Genetic Distances...")
    
    # Prepare data for clustering
    labels = []
    data_points = []
    
    # Sort by V for initial ordering logic (optional)
    sorted_species = sorted(species_map.items(), key=lambda x: x[0][0])
    
    for (v, triple), seed in sorted_species:
        # Label: "V=63 [4,4,4,7]"
        label = f"V={v} {seed}"
        labels.append(label)
        data_points.append(seed)
    
    X = np.array(data_points)
    
    # Hierarchical Clustering
    # Metric: Euclidean distance in Seed Space (4D integer lattice)
    # Method: Ward's method minimizes variance within clusters
    print("Clustering...")
    Z = sch.linkage(X, method='ward', metric='euclidean')
    
    # Plot Dendrogram
    print("Generating Dendrogram...")
    plt.figure(figsize=(20, 12))
    plt.title("Metatron Phylogenetic Tree (The Tree of Life)\nClustering of Geometric Species by 4D Seed Proximity")
    plt.xlabel("Species (V and Seed)")
    plt.ylabel("Genetic Distance (Euclidean)")
    
    # Dendrogram
    dend = sch.dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=6.,
        labels=labels,
        show_contracted=True
    )
    
    plt.tight_layout()
    out_png = f"{output_prefix}_DENDROGRAM.png"
    plt.savefig(out_png, dpi=300)
    print(f"Tree saved to {out_png}")
    
    # Analyze Clusters to find Ancestors
    # We can look at the roots of the main branches.
    # Identifying "Clades".
    
    return Z, labels, data_points

if __name__ == "__main__":
    if len(sys.argv) > 1:
        atlas_path = sys.argv[1]
    else:
        # Default
        atlas_path = r"p:\GitHub_puba\HAN\FRAMEWORK\04-SOFTWARE\Metatron's Cube\Seeds\ATLAS_INDEX_DEEP.md"
        
    base_dir = os.path.dirname(atlas_path)
    output_prefix = os.path.join(base_dir, "METATRON_TREE_OF_LIFE")
    
    species = parse_atlas_lazy(atlas_path)
    generate_phylo_tree(species, output_prefix)
