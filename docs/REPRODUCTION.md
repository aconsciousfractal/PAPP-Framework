# Full Reproduction Guide

This guide provides step-by-step instructions to reproduce **all** results from the PAPP paper, starting from zero assets.

**Total Time**: ~4 hours  
**Disk Space Required**: 2+ GB

---

## Prerequisites

1. ✅ Completed [Installation](INSTALLATION.md)
2. ✅ Virtual environment activated
3. ✅ At least 4 hours of uninterrupted time

---

## Phase 1: Model Generation (2 hours 30 minutes)

### Step 1.1: Generate 1111 Quantum Metrics (45 minutes)

```bash
cd code_src
python batch_quantum_metrics.py
```

**Output**: 1111 OBJ files in `assets/models_obj/1111 obj Quantum Metrics/`

**Progress Indicators**:
- Every 100 models: Progress bar update
- Expected: `Element_V5_phi_gap_1_1_1_1.obj` through `Element_V59591_phi_gap_...obj`

**Verification**:
```powershell
# Windows: Count files
(Get-ChildItem "../assets/models_obj/1111 obj Quantum Metrics/*.obj" | Measure-Object).Count
# Expected: 1111

# Linux/macOS:
ls -1 ../assets/models_obj/1111\ obj\ Quantum\ Metrics/*.obj | wc -l
# Expected: 1111
```

---

### Step 1.2: Generate Pantheon Polytopes (10 minutes)

```bash
python batch_generate_pantheon_metrics.py
```

**Output**: 6 semantic OBJ files in `assets/models_obj/`
- `5_cell_semantic_QUANTUM_METRIC.obj` (5 vertices)
- `8_cell_semantic_QUANTUM_METRIC.obj` (16 vertices)
- `16_cell_semantic_QUANTUM_METRIC.obj` (8 vertices)
- `24_cell_semantic_QUANTUM_METRIC.obj` (24 vertices)
- `120_cell_semantic_QUANTUM_METRIC.obj` (600 vertices)
- `600_cell_semantic_QUANTUM_METRIC.obj` (120 vertices)

**Verification**:
```bash
ls -lh ../assets/models_obj/*_semantic_QUANTUM_METRIC.obj
# Should show 6 files
```

---

### Step 1.3: Generate E6/E8 Lattices (5 minutes)

```bash
python generate_e6_soul.py  # ~2 minutes
python generate_e8_gosset.py  # ~3 minutes
```

**Output**:
- `assets/models_obj/E6_72V_GOSSET_SOUL.obj` (72 vertices)
- `assets/models_obj/E8_240V_GOSSET_LATTICE.obj` (240 vertices)

**Verification**:
```bash
python -c "
with open('../assets/models_obj/E6_72V_GOSSET_SOUL.obj') as f:
    vertices = [l for l in f if l.startswith('v ')]; print(f'E6: {len(vertices)} vertices')
with open('../assets/models_obj/E8_240V_GOSSET_LATTICE.obj') as f:
    vertices = [l for l in f if l.startswith('v ')]; print(f'E8: {len(vertices)} vertices')
"
# Expected:
# E6: 72 vertices
# E8: 240 vertices
```

---

### Step 1.4: Extended Search n=101-200 (90 minutes)

**NOTE**: This is computationally intensive. Skip if time-limited.

```bash
python metatron_universal_projector.py --n_min 101 --n_max 200
```

**Output**: 332 OBJ files in `assets/models_obj/Post_n100_332 New_Species/`

**Verification**:
```bash
# Count files
ls -1 ../assets/models_obj/Post_n100_332\ New_Species/*.obj | wc -l
# Expected: 332
```

---

## Phase 2: Census Data Generation (30 minutes)

### Step 2.1: Physical State Census (15 minutes)

```bash
python physical_state_census.py
```

**Output**: `data/PHYSICAL_CENSUS.csv` (1111 rows)

**Columns**:
- V_Total, Volume_4D, Volume_3D, Sphericity, Density, Crystallinity_Index, Phase_State

**Verification**:
```bash
# Check row count
python -c "import pandas as pd; df = pd.read_csv('../data/PHYSICAL_CENSUS.csv'); print(f'Rows: {len(df)}')"
# Expected: Rows: 1111

# Preview data
head -n 5 ../data/PHYSICAL_CENSUS.csv
```

---

### Step 2.2: Spectral Census (10 minutes)

```bash
python spectral_census.py
```

**Output**: `data/SPECTRAL_CENSUS.csv` (1111 rows)

**Columns**:
- Zero_Modes, Fundamental_Freq, Spectral_Gap, Eigenvalue_Spectrum

---

### Step 2.3: Phylogenetic Tree (5 minutes)

```bash
python phylogenetic_tree_generator.py
```

**Output**: `data/PHYLOGENY_CENSUS.csv` (1111 rows)

**Columns**:
- Family_ID, Distance_To_Centroid, Similarity_Score

**Verification (All CSVs)**:
```bash
python -c "
import pandas as pd
phy = pd.read_csv('../data/PHYLOGENY_CENSUS.csv')
phys = pd.read_csv('../data/PHYSICAL_CENSUS.csv')
spec = pd.read_csv('../data/SPECTRAL_CENSUS.csv')
print(f'PHYLOGENY: {len(phy)} rows')
print(f'PHYSICAL: {len(phys)} rows')
print(f'SPECTRAL: {len(spec)} rows')
assert len(phy) == len(phys) == len(spec) == 1111, 'Row count mismatch!'
print('✓ All census files verified!')
"
```

---

## Phase 3: Figure Generation (5 minutes)

### Step 3.1: Main Paper Figures (3 minutes)

```bash
python generate_paper_figures.py
```

**Output**: `paper_build/figures/Fig1-4.png`
- Fig1_GroundState_V18.png
- Fig2_Saturation_Curve.png
- Fig3_Pantheon_Spectrum.png
- Fig4_Crystallinity_Evolution.png

---

### Step 3.2: Census Analysis Figures (2 minutes)

```bash
python generate_census_figures.py
```

**Output**: `paper_build/figures/Fig5-13.png`
- Fig5 through Fig13 (9 figures)

**Verification (All Figures)**:
```bash
ls -1 ../paper_build/figures/*.png | wc -l
# Expected: 13
```

---

## Final Verification Checklist

Run this comprehensive check:

```bash
cd code_src
python -c "
import os
from pathlib import Path

# Count OBJ files
obj_1111 = len(list(Path('../assets/models_obj/1111 obj Quantum Metrics').glob('*.obj')))
obj_pantheon = len(list(Path('../assets/models_obj').glob('*_semantic_QUANTUM_METRIC.obj')))
obj_e6e8 = len(list(Path('../assets/models_obj').glob('E*_*.obj')))

# Count CSV files
import pandas as pd
csv_files = ['PHYLOGENY_CENSUS.csv', 'PHYSICAL_CENSUS.csv', 'SPECTRAL_CENSUS.csv']
csv_rows = [len(pd.read_csv(f'../data/{f}')) for f in csv_files]

# Count figures
figs = len(list(Path('../paper_build/figures').glob('Fig*.png')))

# Report
print('='*60)
print('PAPP REPRODUCTION VERIFICATION')
print('='*60)
print(f'✓ 1111 Quantum Metrics: {obj_1111}/1111')
print(f'✓ Pantheon Polytopes: {obj_pantheon}/6')
print(f'✓ E6/E8 Lattices: {obj_e6e8}/2')
print(f'✓ Census CSVs: {len([r for r in csv_rows if r == 1111])}/3 (1111 rows each)')
print(f'✓ Figures: {figs}/13')
print('='*60)

# Check success
all_good = (obj_1111 == 1111 and obj_pantheon == 6 and obj_e6e8 == 2 
            and all(r == 1111 for r in csv_rows) and figs == 13)
if all_good:
    print('🎉 FULL REPRODUCTION SUCCESSFUL!')
else:
    print('⚠️  Some files missing - review steps above')
print('='*60)
"
```

**Expected Output**:
```
============================================================
PAPP REPRODUCTION VERIFICATION
============================================================
✓ 1111 Quantum Metrics: 1111/1111
✓ Pantheon Polytopes: 6/6
✓ E6/E8 Lattices: 2/2
✓ Census CSVs: 3/3 (1111 rows each)
✓ Figures: 13/13
============================================================
🎉 FULL REPRODUCTION SUCCESSFUL!
============================================================
```

---

## Expected Runtime Summary

| Phase | Step | Time | Output |
|-------|------|------|--------|
| 1 | 1111 Quantum Metrics | 45 min | 1111 OBJ files |
| 1 | Pantheon | 10 min | 6 OBJ files |
| 1 | E6/E8 | 5 min | 2 OBJ files |
| 1 | n=101-200 | 90 min | 332 OBJ files (optional) |
| 2 | Physical Census | 15 min | PHYSICAL_CENSUS.csv |
| 2 | Spectral Census | 10 min | SPECTRAL_CENSUS.csv |
| 2 | Phylogenetic | 5 min | PHYLOGENY_CENSUS.csv |
| 3 | Paper Figures | 3 min | Fig1-4.png |
| 3 | Census Figures | 2 min | Fig5-13.png |
| **TOTAL** | | **4 hours** | **1,451 OBJ + 3 CSV + 13 PNG** |

---

## Troubleshooting

### Issue: Script crashes midway

**Solution**: Check RAM usage. Close other applications.
```bash
# Resume from checkpoint (if implemented)
python batch_quantum_metrics.py --resume
```

### Issue: Figures look different from paper

**Solution**: Ensure exact package versions
```bash
pip freeze | grep -E "(numpy|matplotlib|pandas|scipy|seaborn)"
```

### Issue: File count mismatch

**Solution**: Re-run specific step
```bash
# Example: Re-generate Pantheon
python batch_generate_pantheon_metrics.py --force
```

---

## Optional: Compare with Published Data

Download official dataset from Zenodo:
```bash
wget https://zenodo.org/record/xxxxxx/files/PAPP_official_data.zip
unzip PAPP_official_data.zip

# Compare CSVs
diff -q data/PHYSICAL_CENSUS.csv PAPP_official_data/PHYSICAL_CENSUS.csv
```

---

## Next Steps

✅ **Reproduction Complete!**

Now you can:
1. Explore data with [examples/quickstart.ipynb](../examples/quickstart.ipynb)
2. Run custom analyses (see [API.md](API.md))
3. Submit modifications via Pull Request

---

**Questions?** Open an issue: https://github.com/aconsciousfractal/PAPP-Framework/issues
