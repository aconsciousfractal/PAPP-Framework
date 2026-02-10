import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import os

# Publication-Quality Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0
})

# Color Palette (Academic-grade)
COLORS = {
    'primary': '#2E3440',      # Dark Blue-Grey
    'secondary': '#5E81AC',    # Steel Blue
    'accent': '#BF616A',       # Muted Red
    'highlight': '#EBCB8B',    # Gold
    'success': '#A3BE8C',      # Green
    'grid': '#D8DEE9',         # Light Grey
    'bg': '#ECEFF4'            # Off-white
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_build", "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "assets", "models_obj")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_obj_vertices(filepath):
    """Load vertices from OBJ file."""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def load_obj_faces(filepath):
    """Load faces from OBJ file."""
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.split()
                # OBJ is 1-indexed
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    return faces

def figure_1_ground_state_v18():
    """
    Figure 1: The Ground State (V=18)
    3D visualization of the archetypal 18-vertex structure showing:
    - Vertex positions
    - Connectivity (edges from faces)
    - Radial symmetry
    """
    # Load authentic V=18 ground state model
    obj_candidates = [
        os.path.join(MODELS_DIR, "1111 obj", "1111 obj Quantum Metrics", "Element_V18_phi_gap_5_5_5_5_QUANTUM_METRIC.obj"),
        os.path.join(MODELS_DIR, "1111 obj Quantum Metrics", "Element_V18_phi_gap_5_5_5_5_QUANTUM_METRIC.obj"),
        os.path.join(MODELS_DIR, "ground_state_v18.obj")  # Fallback if file moved
    ]
    
    obj_file = None
    for candidate in obj_candidates:
        if os.path.exists(candidate):
            obj_file = candidate
            break
    
    if not obj_file:
        print("WARNING: No V=18 model found, skipping Figure 1")
        return
    
    vertices = load_obj_vertices(obj_file)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main 3D view
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                c=COLORS['accent'], s=80, alpha=0.8, edgecolors=COLORS['primary'], linewidths=1.5)
    
    # Try to draw edges if faces exist
    try:
        faces = load_obj_faces(obj_file)
        for face in faces[:30]:  # Limit for clarity
            if len(face) >= 2:
                for i in range(len(face)):
                    v1 = vertices[face[i]]
                    v2 = vertices[face[(i+1) % len(face)]]
                    ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                            color=COLORS['secondary'], alpha=0.3, linewidth=0.8)
    except:
        pass
    
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title(f'Ground State Structure (V={len(vertices)})', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.2)
    
    # Projection XY
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(vertices[:, 0], vertices[:, 1], c=COLORS['primary'], s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Radial Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    radii = np.linalg.norm(vertices - np.mean(vertices, axis=0), axis=1)
    ax3.hist(radii, bins=15, color=COLORS['secondary'], alpha=0.7, edgecolor=COLORS['primary'])
    ax3.set_xlabel('Radial Distance')
    ax3.set_ylabel('Vertex Count')
    ax3.set_title('Radial Symmetry Profile')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_GroundState_V18.png"))
    plt.close()
    print(f"✓ Generated Fig1_GroundState_V18.png")

def figure_2_saturation_curve():
    """
    Figure 2: Discovery Saturation Curve
    Shows the growth of unique topological families with increasing N_max.
    Uses real data from SATURATION_DATA.csv
    """
    csv_path = os.path.join(DATA_DIR, "SATURATION_DATA.csv")
    
    if not os.path.exists(csv_path):
        print("WARNING: SATURATION_DATA.csv not found, skipping Figure 2")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Raw Growth Curve
    ax1.plot(df['N_Max_Element'], df['Total_Unique_Families'], 
             color=COLORS['secondary'], linewidth=2.5, label='Unique Families')
    ax1.fill_between(df['N_Max_Element'], 0, df['Total_Unique_Families'], 
                      color=COLORS['secondary'], alpha=0.15)
    
    # Mark key milestones
    milestones = [18, 60, 120]
    for m in milestones:
        if m in df['N_Max_Element'].values:
            idx = df[df['N_Max_Element'] == m].index[0]
            count = df.loc[idx, 'Total_Unique_Families']
            ax1.scatter([m], [count], color=COLORS['accent'], s=100, zorder=5, edgecolors='white', linewidths=2)
            ax1.annotate(f'N={m}\n({count})', xy=(m, count), xytext=(m+5, count+50),
                        fontsize=9, ha='left', color=COLORS['accent'])
    
    ax1.set_xlabel('Maximum Element (N)')
    ax1.set_ylabel('Total Unique Families')
    ax1.set_title('Topological Family Discovery Curve')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Right: Growth Rate (Derivative)
    growth_rate = np.diff(df['Total_Unique_Families'].values)
    ax2.plot(df['N_Max_Element'].values[1:], growth_rate, 
             color=COLORS['primary'], linewidth=2, label='Discovery Rate (dF/dN)')
    ax2.axhline(y=np.mean(growth_rate), color=COLORS['accent'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Mean Rate: {np.mean(growth_rate):.1f}')
    
    ax2.set_xlabel('Maximum Element (N)')
    ax2.set_ylabel('New Families per Step')
    ax2.set_title('Discovery Rate Analysis')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Saturation_Curve.png"))
    plt.close()
    print(f"✓ Generated Fig2_Saturation_Curve.png")

def figure_3_pantheon_spectrum():
    """
    Figure 3: The Pantheon Spectrum
    Visualization of the 4D polytope hierarchy with vertex counts and symmetry properties.
    """
    # Data from the paper's Pantheon analysis
    pantheon_data = {
        'Name': ['5-Cell', '8-Cell', '16-Cell', '24-Cell', '120-Cell', '600-Cell', 
                 'Grand\n600-Cell', 'E6 Soul', 'E8 Gosset'],
        'Vertices': [5, 16, 8, 24, 600, 120, 720, 72, 240],
        'Symmetry': ['A₄', 'C₄', 'D₄', 'F₄', 'H₄', 'H₄', 'H₄*', 'E₆', 'E₈'],
        'Type': ['Regular', 'Regular', 'Regular', 'Regular', 'Regular', 'Regular', 
                 'Star', 'Lie', 'Lie']
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Vertex Count Spectrum
    colors_map = {'Regular': COLORS['secondary'], 'Star': COLORS['accent'], 'Lie': COLORS['highlight']}
    colors = [colors_map[t] for t in pantheon_data['Type']]
    
    bars = ax1.bar(range(len(pantheon_data['Name'])), pantheon_data['Vertices'], 
                   color=colors, alpha=0.8, edgecolor=COLORS['primary'], linewidth=1.5)
    
    ax1.set_xticks(range(len(pantheon_data['Name'])))
    ax1.set_xticklabels(pantheon_data['Name'], rotation=0, ha='center')
    ax1.set_ylabel('Vertex Count (V)')
    ax1.set_title('4D Polytope Pantheon: Vertex Spectrum', fontsize=13, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add symmetry labels
    for i, (v, sym) in enumerate(zip(pantheon_data['Vertices'], pantheon_data['Symmetry'])):
        ax1.text(i, v * 1.2, f'{v}\n{sym}', ha='center', va='bottom', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[k], edgecolor=COLORS['primary'], label=k, alpha=0.8) 
                      for k in colors_map.keys()]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Bottom: Connectivity Matrix Heatmap (Conceptual)
    # Show which polytopes embed in higher dimensions
    connectivity = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],  # 5-cell
        [0, 1, 1, 1, 0, 0, 0, 0, 0],  # 8-cell
        [0, 0, 1, 1, 0, 0, 0, 0, 0],  # 16-cell
        [0, 0, 0, 1, 1, 1, 0, 1, 0],  # 24-cell
        [0, 0, 0, 0, 1, 1, 1, 0, 0],  # 120-cell
        [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 600-cell
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Grand
        [0, 0, 0, 0, 0, 0, 0, 1, 1],  # E6
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # E8
    ])
    
    im = ax2.imshow(connectivity, cmap='Blues', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(len(pantheon_data['Name'])))
    ax2.set_yticks(range(len(pantheon_data['Name'])))
    ax2.set_xticklabels(pantheon_data['Name'], rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(pantheon_data['Name'], fontsize=9)
    ax2.set_title('Structural Containment Hierarchy', fontsize=12, weight='bold')
    
    plt.colorbar(im, ax=ax2, label='Contains Structure')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Pantheon_Spectrum.png"))
    plt.close()
    print(f"✓ Generated Fig3_Pantheon_Spectrum.png")

def figure_4_crystallinity_evolution(filename="Fig4_Crystallinity_Evolution.png"):
    """
    Figure 4: Crystallinity Index Evolution
    Shows how the crystallinity index evolves with vertex count V,
    highlighting phase transitions from Amorphous → Liquid → Solid Crystal.
    """
    # Load physical census data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "PHYSICAL_CENSUS.csv")
    df = pd.read_csv(data_path)
    
    # Phase colors matching census figures
    PHASE_COLORS = {
        'Amorphous Gas': '#FF6B6B',
        'Liquid Crystal': '#FFA500',
        'Solid Crystal': '#4169E1'
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Main plot: Crystallinity vs V, colored by phase
    for phase in df['Phase_State'].unique():
        df_phase = df[df['Phase_State'] == phase]
        ax1.scatter(df_phase['V_Total'], df_phase['Crystallinity_Index'],
                    c=PHASE_COLORS.get(phase, '#888888'),
                    label=phase,
                    s=60, alpha=0.7, edgecolors='black', linewidths=0.8)
    
    # Add trend line
    valid_data = df[df['Crystallinity_Index'] > 0]
    if len(valid_data) > 10:
        z = np.polyfit(valid_data['V_Total'], valid_data['Crystallinity_Index'], 3)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['V_Total'].min(), valid_data['V_Total'].max(), 200)
        ax1.plot(x_trend, p(x_trend), 'k--', linewidth=2.5, alpha=0.6, label='Polynomial Trend')
    
    ax1.set_xlabel('Number of Vertices (V)', fontsize=12, weight='bold')
    ax1.set_ylabel('Crystallinity Index', fontsize=12, weight='bold')
    ax1.set_title('Crystallinity Evolution Across Configuration Space', fontsize=14, weight='bold', pad=15)
    ax1.legend(loc='best', framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=-0.1)
    
    # Bottom plot: Phase distribution histogram
    phase_counts = df.groupby(['V_Total', 'Phase_State']).size().unstack(fill_value=0)
    
    # Create stacked bar chart
    bottom = np.zeros(len(phase_counts))
    x_positions = phase_counts.index
    
    for phase in ['Amorphous Gas', 'Liquid Crystal', 'Solid Crystal']:
        if phase in phase_counts.columns:
            ax2.bar(x_positions, phase_counts[phase], 
                    bottom=bottom,
                    color=PHASE_COLORS.get(phase, '#888888'),
                    edgecolor='black',
                    linewidth=0.5,
                    label=phase,
                    width=1.0)
            bottom += phase_counts[phase].values
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=11, weight='bold')
    ax2.set_ylabel('Count', fontsize=11, weight='bold')
    ax2.set_title('Phase Distribution', fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"✓ Generated {filename}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PAPP Figure Generation Suite - Professional Edition")
    print("="*60 + "\n")
    
    try:
        figure_1_ground_state_v18()
        figure_2_saturation_curve()
        figure_3_pantheon_spectrum()
        # Figure 4 removed - see generate_new_figures.py for updated figures
        
        print(f"\n{'='*60}")
        print(f"✓ Figures 1-3 generated successfully!")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Note: For additional figures (4-7), run generate_new_figures.py")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
