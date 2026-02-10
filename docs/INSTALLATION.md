# Installation Guide

This guide provides detailed installation instructions for the PAPP Framework.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4 GB
- **Disk Space**: 2 GB (for complete dataset)
- **Internet**: Required for package installation

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 8 GB
- **Disk Space**: 5 GB
- **Processor**: Multi-core (for faster generation)

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/aconsciousfractal/PAPP-Framework.git
cd PAPP-Framework

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import numpy, matplotlib, pandas, scipy, seaborn; print('✓ All dependencies installed!')"
```

### Method 2: Conda Installation

```bash
# 1. Clone repository
git clone https://github.com/aconsciousfractal/PAPP-Framework.git
cd PAPP-Framework

# 2. Create conda environment
conda create -n papp python=3.10

# 3. Activate environment
conda activate papp

# 4. Install dependencies
conda install numpy matplotlib pandas scipy seaborn

# 5. Verify installation
python -c "import numpy, matplotlib, pandas, scipy, seaborn; print('✓ All dependencies installed!')"
```

### Method 3: Development Installation

For contributors who want to modify the code:

```bash
# 1. Clone repository
git clone https://github.com/aconsciousfractal/PAPP-Framework.git
cd PAPP-Framework

# 2. Install in editable mode
pip install -e .

# 3. Install development dependencies
pip install pytest black flake8 mypy

# 4. Run tests
pytest tests/
```

---

## Dependency Details

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Numerical computations, array operations |
| `matplotlib` | ≥3.4.0 | Figure generation, 3D plotting |
| `pandas` | ≥1.3.0 | CSV data handling, analysis |
| `scipy` | ≥1.7.0 | Spectral analysis, Laplacian eigenvalues |
| `seaborn` | ≥0.11.0 | Statistical visualizations |

---

## Platform-Specific Notes

### Windows

If you encounter `Microsoft Visual C++ 14.0 or greater is required` error:
1. Install Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
2. Or use pre-built binary wheels: `pip install --only-binary :all: scipy`

### macOS

If you encounter SSL certificate errors:
```bash
# Install certificates
/Applications/Python\ 3.10/Install\ Certificates.command
```

### Linux

Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv
```

---

## Verification

Test your installation:

```bash
cd code_src

# Quick test: Generate single figure (~30 seconds)
python -c "
from generate_paper_figures import load_obj_vertices
import os
vertices = load_obj_vertices('../assets/models_obj/5_cell_semantic_QUANTUM_METRIC.obj')
print(f'✓ Loaded {len(vertices)} vertices')
"
```

Expected output:
```
✓ Loaded 5 vertices
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`

**Solution**: Ensure virtual environment is activated
```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Issue: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Ensure you're in the correct directory
```bash
cd code_src  # For running scripts
```

### Issue: Slow installation

**Solution**: Use faster mirror
```bash
pip install -r requirements.txt -i https://pypi.org/simple
```

---

## Next Steps

After successful installation:

1. **Quick Start**: See [README.md](../README.md#-quick-start)
2. **Generate Figures**: Run `generate_paper_figures.py`
3. **Full Reproduction**: See [REPRODUCTION.md](REPRODUCTION.md)

---

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv

# Remove repository
cd ..
rm -rf PAPP-Framework
```

---

**Need Help?** Open an issue on GitHub: https://github.com/aconsciousfractal/PAPP-Framework/issues
