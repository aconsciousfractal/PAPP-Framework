# PAPP Quick Start Guide

This document provides a 5-minute introduction to using the PAPP framework.

## 📦 Installation (1 minute)

```bash
git clone https://github.com/aconsciousfractal/PAPP-Framework.git
cd PAPP-Framework
pip install -r requirements.txt
```

## 🚀 Generate Your First Figure (2 minutes)

```bash
cd code_src
python generate_paper_figures.py
```

View output in `paper_build/figures/`

## 📊 Explore Census Data (1 minute)

```python
import pandas as pd

# Load data
df = pd.read_csv("../data/PHYSICAL_CENSUS.csv")

# Find ground state
ground_state = df.loc[df['V_Total'].idxmin()]
print(f"Ground State: V={ground_state['V_Total']}")
print(f"Crystallinity: {ground_state['Crystallinity_Index']:.3f}")

# Phase distribution
print(df['Phase_State'].value_counts())
```

## 🎨 Visualize a Model (1 minute)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load V=18 ground state
vertices = []
with open("../assets/models_obj/1111 obj Quantum Metrics/Element_V18_phi_gap_5_5_5_5.obj") as f:
    for line in f:
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:4]])

vertices = np.array(vertices)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
plt.show()
```

## ✅ Next Steps

- Full reproduction guide: [docs/REPRODUCTION.md](../docs/REPRODUCTION.md)
- API reference: [docs/API.md](../docs/API.md)
- Examples: [examples/quickstart.py](../examples/quickstart.py)

## 📖 Documentation

- [README.md](../README.md) - Main documentation
- [INSTALLATION.md](../docs/INSTALLATION.md) - Detailed setup
- [REPRODUCTION.md](../docs/REPRODUCTION.md) - Full pipeline (4 hours)

---

**Questions?** Open an issue: https://github.com/aconsciousfractal/PAPP-Framework/issues
