# PAPP API Reference

PAPP is currently organized as a **script-based research repository**, not as a versioned Python package with a stable import API.

This document therefore focuses on the **supported entrypoints (scripts)**, their **inputs/outputs**, and the **data/asset formats** used throughout the repo.

If/when the code is packaged (e.g. `papp` module), this document can be expanded with stable function/class references.

---

## Entry Points (Scripts)

All scripts live in `code_src/` and are typically executed from that directory.

### Model generation

- `code_src/batch_quantum_metrics.py`
    - Generates the 1111 quantum metric OBJ files.
    - Output directory: `assets/models_obj/1111 obj/1111 obj Quantum Metrics/`

- `code_src/batch_generate_pantheon_metrics.py`
    - Generates the 4D “Pantheon” semantic quantum metric OBJ files.
    - Output directory: `assets/models_obj/`

- `code_src/generate_e6_soul.py`, `code_src/generate_e8_gosset.py`
    - Generates E6/E8 lattice-derived OBJ models.
    - Output directory: `assets/models_obj/`

- `code_src/metatron_universal_projector.py`
    - Extended search (n-range driven) producing “New Species” OBJs.
    - Output directory: `assets/models_obj/Post_n100_332 New_Species/`

### Census generation

- `code_src/physical_state_census.py`
    - Produces: `data/PHYSICAL_CENSUS.csv`

- `code_src/spectral_census.py`
    - Produces: `data/SPECTRAL_CENSUS.csv`

- `code_src/phylogenetic_tree_generator.py`
    - Produces: `data/PHYLOGENY_CENSUS.csv`

### Figure generation

- `code_src/generate_paper_figures.py`
    - Generates the core paper figures into `paper_build/figures/`.

Note: If additional figure scripts are added/renamed, update both this file and the main README so the docs stay consistent.

---

## File Formats

### OBJ models

OBJ files are stored under `assets/models_obj/`.

Typical patterns:

- 1111 quantum metrics:
    - `assets/models_obj/1111 obj/1111 obj Quantum Metrics/Element_V*_phi_gap_*_QUANTUM_METRIC.obj`

- Pantheon semantic quantum metrics:
    - `assets/models_obj/*_semantic_QUANTUM_METRIC.obj`

### Census CSVs

- `data/PHYSICAL_CENSUS.csv`
    - Includes columns such as: `V_Total`, `Volume_4D`, `Volume_3D`, `Sphericity`, `Density`, `Crystallinity_Index`, `Phase_State`.

- `data/SPECTRAL_CENSUS.csv`
    - Includes columns such as: `Zero_Modes`, `Fundamental_Freq`, `Spectral_Gap`, `Eigenvalue_Spectrum`.

- `data/PHYLOGENY_CENSUS.csv`
    - Includes columns such as: `Family_ID`, `Distance_To_Centroid`, `Similarity_Score`.

---

## See Also

- [Installation Guide](INSTALLATION.md)
- [Reproduction Guide](REPRODUCTION.md)
- [Code Documentation](../code_src/README.md)
- [Main README](../README.md)
