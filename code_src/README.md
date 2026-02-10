# Model Generation Scripts

This directory contains all scripts used to generate the 1111+ OBJ models referenced in the PAPP paper.

## Scripts Overview

### 📦 Batch Generation Scripts

#### `batch_quantum_metrics.py`
**Purpose**: Generates the 1111 quantum metric OBJ files  
**Output**: `assets/models_obj/1111 obj Quantum Metrics/Element_V*.obj`  
**Parameters**:
- Phi-gap ranges: [1,1,1,1] through [n,n,n,n]
- Total configurations: 1111
- Metrics: Volume, sphericity, crystallinity, density

**Usage**:
```powershell
python batch_quantum_metrics.py
```

**Expected Runtime**: ~45 minutes (1111 models)

---

#### `reconstruct_papp_polychora.py`
**Purpose**: Reconstructs full 4D topology of PAPP irregular polychora from quantum metrics  
**Input**: `assets/models_obj/1111 obj/1111 obj Quantum Metrics/Element_V*_QUANTUM_METRIC.obj`  
**Output**: `assets/models_obj/PAPP Polychora 4D_Reconstructed/Element_V*_RECONSTRUCTED_4D.obj`

**Key Features**:
- Extracts phi-gap seed [k1, k2, k3, k4] from filename
- Computes Grant 4D parameters (a, b, c, d) via Phi-Gap mechanism
- Generates 4D vertex cloud using tetrahedral cascade method
- Computes 4D convex hull topology using Qhull
- Projects to 3D via Hopf fibration (S³ → S²)
- Validates 4D Euler characteristic: χ₄ = V - E + F - C = 0
- Exports complete combinatorial data (vertices, edges, faces, cells)

**Formula Used**: `V_4D = a + 2b + 2c + d` (PAPP 4D vertex formula)

**Usage**:
```powershell
# Reconstruct first 500 elements
python reconstruct_papp_polychora.py --start 0 --end 500

# Reconstruct specific range
python reconstruct_papp_polychora.py --start 500 --end 900

# Reconstruct all 1111 elements
python reconstruct_papp_polychora.py --start 0 --end 1111
```

**Expected Runtime**: 
- First 500: ~30 minutes
- All 1111: ~1 hour

**Output Structure**:
- V (vertices): 3D projected coordinates
- E (edges): Line segments in 4D
- F (faces): Triangular faces
- C (cells): Tetrahedral 4D cells (in comments)

---

#### `batch_generate_pantheon_metrics.py`
**Purpose**: Generates semantic quantum metrics for the 4D Pantheon  
**Output**: `assets/models_obj/*_semantic_QUANTUM_METRIC.obj`  
**Polytopes Generated**:
- 5-cell (Pentatope)
- 8-cell (Tesseract)
- 16-cell (Hexadecachoron)
- 24-cell (Icositetrachoron)
- 120-cell (Hecatonicosachoron)
- 600-cell (Hexacosichoron)

**Usage**:
```powershell
python batch_generate_pantheon_metrics.py
```

**Expected Runtime**: ~10 minutes (6 models)

---

### 🔧 Individual Polytope Generators

#### `generate_pantheon_4d.py`
**Purpose**: Main 4D polytope constructor  
**Output**: Individual Pantheon polytopes with full topology

#### `generate_e6_soul.py`
**Purpose**: E6 Gosset polytope (72 vertices)  
**Output**: `assets/models_obj/E6_72V_GOSSET_SOUL.obj`  
**Dimension**: 6D projected to 3D

#### `generate_e8_gosset.py`
**Purpose**: E8 Gosset polytope (240 vertices)  
**Output**: `assets/models_obj/E8_240V_GOSSET_LATTICE.obj`  
**Dimension**: 8D projected to 3D

#### `generate_all_regular_polytopes_4d.py`
**Purpose**: Generates complete 4D-validated regular polytopes (validation set)
**Output**: `assets/models_obj/Polychora/*.obj`
**Polytopes**:
- 5-cell, 8-cell, 16-cell, 24-cell, 120-cell, 600-cell
**Features**:
- Analytical 4D vertex generation
- Full combinatorial checking (V, E, F, C)
- 4D Euler characteristic verification (χ₄=0)

---

### 🌌 Universal Projector (n=101-200)

#### `metatron_universal_projector.py`
**Purpose**: Extended parameter search beyond n=100  
**Output**: `assets/models_obj/Post_n100_332 New_Species/*.obj`  
**Discoveries**: 332 unique configurations (n=101-200)

**Usage**:
```powershell
python metatron_universal_projector.py --n_min 101 --n_max 200
```

**Expected Runtime**: ~2 hours (332 models)

---

## Data Pipeline

```
┌─────────────────────────────┐
│  Phi-Gap Parameters         │
│  [k1, k2, k3, k4]          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4D Polytope Construction   │
│  - Vertices in 4D           │
│  - Hopf fibration           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3D Projection (PAPP)       │
│  - Grant's projection       │
│  - Quantum Metric overlay   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  OBJ Export                 │
│  - Vertices (v x y z)       │
│  - Faces (f v1 v2 v3 ...)   │
└─────────────────────────────┘
```

---

## Census Generation Scripts

### `physical_state_census.py`
**Purpose**: Analyze physical properties of all 1111 models  
**Output**: `data/PHYSICAL_CENSUS.csv`  
**Metrics**:
- Volume (4D → 3D compression)
- Sphericity index
- Density (vertices/volume)
- Crystallinity index
- Phase state classification

**Usage**:
```powershell
python physical_state_census.py
```

---

### `spectral_census.py`
**Purpose**: Laplacian spectral analysis  
**Output**: `data/SPECTRAL_CENSUS.csv`  
**Metrics**:
- Zero modes count
- Fundamental frequency
- Eigenvalue spectrum
- Spectral gap

**Usage**:
```powershell
python spectral_census.py
```

---

### `phylogenetic_tree_generator.py`
**Purpose**: Classify configurations into phylogenetic families  
**Output**: `data/PHYLOGENY_CENSUS.csv`  
**Features**:
- Family ID assignment (5 families)
- Distance to family centroid
- Structural similarity metrics

**Usage**:
```powershell
python phylogenetic_tree_generator.py
```

---

## Figure Generation Scripts

### `generate_paper_figures.py`
**Purpose**: Generate main paper figures (Fig1-4)  
**Output**: `paper_build/figures/Fig1-4.png`  
**Figures**:
1. Ground State V=18
2. Saturation Curve
3. Pantheon Spectrum
4. Crystallinity Evolution

**Usage**:
```powershell
python generate_paper_figures.py
```

---

### `generate_census_figures.py`
**Purpose**: Generate census analysis figures (Fig5-13)  
**Output**: `paper_build/figures/Fig5-13.png`  
**Figures**:
5. Family Evolution
6. Centroid Distance Heatmap
7. Phase Diagram
8. Sphericity-Volume
9. Density Distribution
10. Zero Modes Evolution
11. Fundamental Frequency Decay
12. Eigenvalue Spectrum
13. Summary Dashboard

**Usage**:
```powershell
python generate_census_figures.py
```

---

## Reproduction Instructions

### Full Pipeline (from scratch)

```powershell
# Step 1: Generate 1111 quantum metrics (~45 min)
python batch_quantum_metrics.py

# Step 2: Generate Pantheon polytopes (~10 min)
python batch_generate_pantheon_metrics.py

# Step 3: Generate E6/E8 (~5 min)
python generate_e6_soul.py
python generate_e8_gosset.py

# Step 4: Extended search n=101-200 (~2 hours)
python metatron_universal_projector.py --n_min 101 --n_max 200

# Step 5: Generate census data (~30 min)
python physical_state_census.py
python spectral_census.py
python phylogenetic_tree_generator.py

# Step 6: Generate all figures (~5 min)
python generate_paper_figures.py
python generate_census_figures.py
```

**Total Time**: ~4 hours

---

## Dependencies

All scripts require:

```
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
scipy>=1.7.0
seaborn>=0.11.0
```

Install via:
```powershell
pip install numpy matplotlib pandas scipy seaborn
```

---

## Output Structure

```
PAPP Repository/
├── assets/
│   └── models_obj/
│       ├── 5_cell_semantic_QUANTUM_METRIC.obj          [Pantheon]
│       ├── 8_cell_semantic_QUANTUM_METRIC.obj
│       ├── ...
│       ├── E6_72V_GOSSET_SOUL.obj                      [E6/E8]
│       ├── E8_240V_GOSSET_LATTICE.obj
│       ├── 1111 obj Quantum Metrics/                   [1111 models]
│       │   ├── Element_V5_phi_gap_1_1_1_1.obj
│       │   ├── Element_V18_phi_gap_5_5_5_5.obj
│       │   └── ...
│       └── Post_n100_332 New_Species/                  [n=101-200]
│           ├── Element_V*.obj (332 files)
│           └── ...
├── data/
│   ├── PHYLOGENY_CENSUS.csv                            [1111 rows]
│   ├── PHYSICAL_CENSUS.csv                             [1111 rows]
│   ├── SPECTRAL_CENSUS.csv                             [1111 rows]
│   └── SATURATION_DATA.csv                             [saturation curve]
└── paper_build/
    └── figures/
        ├── Fig1_GroundState_V18.png
        ├── ...
        └── Fig13_Summary_Dashboard.png                 [13 figures]
```

---

## Validation

To verify complete reproduction:

```powershell
# Check OBJ count
Get-ChildItem "assets/models_obj/1111 obj Quantum Metrics/*.obj" | Measure-Object # Should be 1111
Get-ChildItem "assets/models_obj/Post_n100_332 New_Species/*.obj" | Measure-Object # Should be 332

# Check CSV rows
(Get-Content "data/PHYSICAL_CENSUS.csv" | Measure-Object -Line).Lines - 1 # Should be 1111
(Get-Content "data/PHYLOGENY_CENSUS.csv" | Measure-Object -Line).Lines - 1 # Should be 1111
(Get-Content "data/SPECTRAL_CENSUS.csv" | Measure-Object -Line).Lines - 1 # Should be 1111

# Check figures
Get-ChildItem "paper_build/figures/*.png" | Measure-Object # Should be 13
```

---

## Citation

If using these scripts, please cite:

```bibtex
@article{Babanskyy2026PAPP,
  author  = {Babanskyy, Oleksiy},
  title   = {Polytopic Archetypal Projection Protocol (PAPP): A Computational Survey},
  year    = {2026},
  note    = {arXiv preprint arXiv:2602.xxxxx}
}
```

---

## License

Code: GNU GPL v3  
Data: CC BY 4.0  
Paper: CC BY 4.0

---

## Contact

For questions or issues:
- **Email**: aconsciousfractal@gmail.com
- **GitHub**: https://github.com/aconsciousfractal/PAPP-Framework
- **ORCID**: https://orcid.org/0009-0001-6176-6208

---

**Last Updated**: 2026-02-06
