# Contributing to PAPP

Thank you for your interest in contributing to the Polytopic Archetypal Projection Protocol!

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

---

## Ways to Contribute

### 🐛 Report Bugs
- Check [existing issues](https://github.com/aconsciousfractal/PAPP-Framework/issues)
- Use bug report template
- Include: Python version, OS, error message, minimal reproducible example

### 💡 Suggest Features
- Open a feature request issue
- Describe use case and expected behavior
- Discuss before implementing large changes

### 📖 Improve Documentation
- Fix typos, clarify instructions
- Add examples or tutorials
- Translate documentation

### 🔧 Submit Code
- Fix bugs
- Implement new features
- Optimize performance
- Add tests

---

## Development Setup

```bash
# 1. Fork repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/PAPP-Framework.git
cd PAPP-Framework

# 3. Add upstream remote
git remote add upstream https://github.com/aconsciousfractal/PAPP-Framework.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 5. Install in editable mode with dev dependencies
pip install -e .
pip install pytest black flake8 mypy

# 6. Create feature branch
git checkout -b feature/your-feature-name
```

---

## ⚖️ Contributor License Agreement (CLA)

**Important**: Before contributing code, you must sign the Contributor License Agreement.

### Why We Require a CLA

The CLA:
- ✅ Protects the project from legal issues
- ✅ Ensures we can defend against patent trolls
- ✅ Enables dual licensing (GPL v3 + commercial)
- ✅ Allows future license updates if needed
- ✅ Maintains your rights to your own code

### How to Sign

1. **Read the CLA**: [CLA.md](CLA.md)
2. **Create signature file**: `CLA-signatures/CLA-signed-[YourGitHubUsername].md`
3. **Fill in template**:
   ```markdown
   # CLA Signature
   
   I have read and agree to the PAPP Contributor License Agreement (CLA v1.0).
   
   **Full Name**: Your Full Name
   **GitHub Username**: [@YourUsername](https://github.com/YourUsername)
   **Email**: your.email@example.com
   **Date**: 2026-MM-DD
   **Signature**: Your Full Name
   
   ## Contributions
   I confirm this CLA applies to all my past and future contributions to PAPP.
   ```
4. **Submit via PR**: Include CLA signature in your first Pull Request

### First-Time Contributors

Your first PR should include:
1. ✅ Your signed CLA in `CLA-signatures/`
2. ✅ Your actual code contribution

We'll review the CLA first, then the code.

### Questions About CLA?

- Read [CLA.md](CLA.md) for full terms
- Email: aconsciousfractal@gmail.com
- Open discussion: GitHub Discussions

---

## Code Style

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use `black` for auto-formatting: `black code_src/`
- Use `flake8` for linting: `flake8 code_src/`
- Type hints encouraged: `mypy code_src/`

### Example

```python
def calculate_crystallinity(vertices: np.ndarray, volume: float) -> float:
    """Calculate crystallinity index from vertices and volume.
    
    Args:
        vertices: Nx3 array of vertex coordinates
        volume: Total volume of polytope
        
    Returns:
        Crystallinity index between 0 (amorphous) and 1 (perfect crystal)
    """
    # Implementation
    ...
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactor
- `perf`: Performance improvement
- `test`: Add/update tests
- `chore`: Maintenance (dependencies, build)

**Examples**:
```
feat(generation): add support for n=201-300 range

fix(census): correct crystallinity calculation for V<10

docs(readme): add installation troubleshooting section
```

---

## Pull Request Process

### Before Submitting

1. ✅ Code follows style guide
2. ✅ Tests pass: `pytest tests/`
3. ✅ Linting passes: `flake8 code_src/`
4. ✅ Documentation updated (if needed)
5. ✅ Commit messages follow format

### Submitting PR

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub:
   - Clear title and description
   - Reference related issues (#123)
   - Include screenshots if UI changes
   - Request review from maintainers

4. **Address review feedback**:
   - Make requested changes
   - Push updates to same branch
   - Reply to comments

### PR Checklist Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Tests pass locally
- [ ] No new warnings
```

---

## Testing

### Run Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_generation.py

# With coverage
pytest --cov=code_src tests/
```

### Write Tests

```python
# tests/test_crystallinity.py
import pytest
import numpy as np
from code_src.physical_state_census import calculate_crystallinity

def test_crystallinity_perfect_cube():
    """Test crystallinity of perfect cube (should be ~1.0)"""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    volume = 1.0
    
    crystallinity = calculate_crystallinity(vertices, volume)
    assert 0.9 < crystallinity < 1.0, "Perfect cube should have high crystallinity"

def test_crystallinity_random_points():
    """Test crystallinity of random points (should be low)"""
    np.random.seed(42)
    vertices = np.random.rand(50, 3)
    volume = 1.0
    
    crystallinity = calculate_crystallinity(vertices, volume)
    assert 0.0 < crystallinity < 0.3, "Random points should have low crystallinity"
```

---

## Documentation

### Docstring Format (NumPy Style)

```python
def project_4d_to_3d(vertices_4d, method="hopf"):
    """Project 4D vertices to 3D using specified method.
    
    Parameters
    ----------
    vertices_4d : np.ndarray, shape (N, 4)
        4D vertex coordinates
    method : str, default="hopf"
        Projection method: "hopf" or "perspective"
        
    Returns
    -------
    vertices_3d : np.ndarray, shape (N, 3)
        Projected 3D coordinates
        
    Raises
    ------
    ValueError
        If vertices_4d is not Nx4 array
        If method is unknown
        
    Examples
    --------
    >>> vertices_4d = np.random.rand(10, 4)
    >>> vertices_3d = project_4d_to_3d(vertices_4d)
    >>> vertices_3d.shape
    (10, 3)
    
    Notes
    -----
    Hopf fibration uses S³ → S² mapping described in Grant (2024).
    """
    ...
```

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment.

### Standards

**Positive Behaviors**:
- ✅ Respectful communication
- ✅ Constructive feedback
- ✅ Gracefully accepting criticism
- ✅ Focusing on what's best for community

**Unacceptable Behaviors**:
- ❌ Harassment or discrimination
- ❌ Trolling or insulting comments
- ❌ Personal or political attacks
- ❌ Publishing others' private information

### Enforcement
Report violations to aconsciousfractal@gmail.com. Maintainers will review and take appropriate action.

---

## License

By contributing, you agree that your contributions will be licensed under the GNU GPL v3 License (code) and CC BY 4.0 (data/documentation).

---

## Questions?

- 💬 Discussions: https://github.com/aconsciousfractal/PAPP-Framework/discussions
- 📧 Email: aconsciousfractal@gmail.com
- 🐛 Issues: https://github.com/aconsciousfractal/PAPP-Framework/issues

---

**Thank you for contributing! 🎉**
