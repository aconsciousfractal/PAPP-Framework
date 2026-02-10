#!/usr/bin/env python3
"""PAPP Quick Start Example

This Python script demonstrates basic PAPP functionality with 4 examples:
1. Census data analysis
2. 3D model visualization
3. Crystallinity distribution
4. Phylogenetic family statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "assets" / "models_obj" / "1111 obj" / "1111 obj Quantum Metrics"

print("="*60)
print("PAPP Quick Start Example")
print("="*60)

# Example 1: Load and analyze census data
print("\nExample 1: Census Data Analysis")
print("-" * 60)

# Load physical census
df_phys = pd.read_csv(DATA_DIR / "PHYSICAL_CENSUS.csv")
print(f"Loaded {len(df_phys)} configurations")

# Find ground state
ground_state = df_phys.loc[df_phys['V_Total'].idxmin()]
print(f"\nGround State Properties:")
print(f"  Vertices: {ground_state['V_Total']}")
print(f"  Crystallinity: {ground_state['Crystallinity_Index']:.3f}")
print(f"  Phase: {ground_state['Phase_State']}")

# Phase distribution
phase_counts = df_phys['Phase_State'].value_counts()
print(f"\nPhase Distribution:")
for phase, count in phase_counts.items():
    print(f"  {phase}: {count} ({count/len(df_phys)*100:.1f}%)")

# Example 2: Load and visualize a 3D model
print("\nExample 2: 3D Model Visualization")
print("-" * 60)

def load_obj_vertices(filepath):
    """Load vertices from OBJ file."""
    vertices = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('v '):
                coords = [float(x) for x in line.split()[1:4]]
                vertices.append(coords)
    return np.array(vertices)

# Load V=18 ground state
model_file = MODELS_DIR / "Element_V18_phi_gap_5_5_5_5_QUANTUM_METRIC.obj"
vertices = load_obj_vertices(model_file)
print(f"Loaded {len(vertices)} vertices from {model_file.name}")

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
           c=np.arange(len(vertices)), cmap='viridis', s=100, alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('V=18 Ground State Structure', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("example_v18_visualization.png", dpi=150)
print("Saved visualization to: example_v18_visualization.png")

# Example 3: Crystallinity distribution
print("\nExample 3: Crystallinity Distribution")
print("-" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_phys['Crystallinity_Index'], bins=50, 
        color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(ground_state['Crystallinity_Index'], color='red', 
           linestyle='--', linewidth=2, label='Ground State (V=18)')
ax.set_xlabel('Crystallinity Index', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Crystallinity Indices', fontsize=14, weight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("example_crystallinity_distribution.png", dpi=150)
print("Saved distribution to: example_crystallinity_distribution.png")

# Example 4: Family statistics
print("\nExample 4: Phylogenetic Family Statistics")
print("-" * 60)

df_phylo = pd.read_csv(DATA_DIR / "PHYLOGENY_CENSUS.csv")
distance_col = None
for candidate in ("Distance_to_Centroid", "Distance_To_Centroid", "DistanceToCentroid"):
    if candidate in df_phylo.columns:
        distance_col = candidate
        break

if distance_col is None:
    raise KeyError(
        "No distance-to-centroid column found in PHYLOGENY_CENSUS.csv. "
        f"Available columns: {list(df_phylo.columns)}"
    )

family_stats = df_phylo.groupby('Family_ID')[distance_col].agg(['count', 'mean', 'std'])
family_stats.columns = ['Members', 'Avg Distance', 'Std Distance']
print(family_stats.to_string())

print("\n" + "="*60)
print("[OK] Quick start examples completed!")
print("="*60)
print("\nGenerated files:")
print("  - example_v18_visualization.png")
print("  - example_crystallinity_distribution.png")
