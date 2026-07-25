# AGENTS.md — Gurobi Machine Learning

This file provides guidance for AI agents (and developers) working in this repository.

## Repository Overview

**Gurobi Machine Learning** (`gurobi-machinelearning`) is an open-source Python package that formulates trained regression models as constraints inside a [gurobipy](https://pypi.org/project/gurobipy/) optimization model, enabling them to be solved with the Gurobi solver.

Supported ML frameworks:
- **scikit-learn** — linear models, decision trees, pipelines, gradient boosted regressors, etc.
- **Keras / TensorFlow** — neural networks with ReLU activation
- **PyTorch** — neural networks with ReLU activation
- **XGBoost** — gradient boosted trees
- **LightGBM** — gradient boosted trees
- **ONNX** — generic model interchange format

Source layout:
```
src/gurobi_ml/          # Main package
  add_predictor.py      # Public entry point: add_predictor_constr()
  register_user_predictor.py
  registered_predictors.py
  sklearn/              # scikit-learn predictor support
  keras/                # Keras predictor support
  torch/                # PyTorch predictor support
  xgboost/              # XGBoost predictor support
  lightgbm/             # LightGBM predictor support
  onnx/                 # ONNX predictor support
  modeling/             # Shared modeling utilities
tests/                  # pytest test suites, one directory per framework
docs/                   # Sphinx documentation
notebooks/              # Example Jupyter notebooks
```

## Setting Up a Development Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install tox tox-uv
```

A valid Gurobi license is required to run some tests. Set the `GRB_LICENSE_FILE` environment variable to point to your license file.

## Running Tests

Tests are managed by [tox](https://tox.readthedocs.io/). The test matrix covers Python 3.11–3.13 and Gurobi 11–13.

**Run a focused subset (recommended for local iteration):**

```bash
# scikit-learn tests against Gurobi 13
tox -e py311-sklearn-gurobi13

# PyTorch tests
tox -e py311-pytorch-gurobi13

# Keras tests
tox -e py311-keras-gurobi13

# XGBoost tests
tox -e py311-xgboost-gurobi13

# LightGBM tests
tox -e py311-lightgbm-gurobi13

# ONNX tests
tox -e py311-onnx-gurobi13

# All deps at once
tox -e py311-all_deps-gurobi13

# Tests that require no optional dependencies
tox -e py311-no_deps
```

**Run pytest directly** (after manually installing deps):

```bash
pytest tests/test_sklearn
pytest tests/test_pytorch
# etc.
```

**Run docs build:**

```bash
tox -e docs
```

## Linting and Code Style

Pre-commit hooks enforce code style. Install and run them with:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Or via tox:

```bash
tox -e pre-commit
```

The hooks run:
- **ruff** — linting and formatting (replaces flake8/black/isort)
- **autoflake** — removes unused imports
- **pyupgrade** — upgrades syntax to Python 3.11+
- **nbstripout** — strips notebook output before commit
- **gitlint** — validates commit message format
- Standard pre-commit checks: trailing whitespace, end-of-file fixer, YAML/TOML/JSON validation, merge conflict detection

All source files under `src/` and notebooks must include the Apache-2.0 license header (enforced by `.github/hooks/license_embedded.py`).

## Code Conventions

- **Python version target:** supported runtime `>=3.10` (see `pyproject.toml`), but CI/tooling targets Python 3.11–3.13 (`pyupgrade --py311-plus`).
- **Formatter/linter:** `ruff` — do not use `black` or `flake8` directly
- **Imports:** unused imports are removed automatically by `autoflake`; keep imports clean
- **License header:** every `.py` file in `src/` and every `.ipynb` must begin with the Apache-2.0 copyright header (see any existing source file for the exact format)
- **Docstrings:** use NumPy-style docstrings for public functions and classes (consistent with the Sphinx docs)
- **Commit messages:** follow [gitlint](https://jorisroovers.com/gitlint/) conventions — subject line ≤ 72 characters, imperative mood

## Key Public API

```python
from gurobi_ml import add_predictor_constr, register_predictor_constr

# Insert a trained predictor as constraints into a gurobipy model
pred_constr = add_predictor_constr(gp_model, predictor, input_vars, output_vars)

# Register a custom predictor class
register_predictor_constr(MyPredictorClass, MyConstraintClass)
```

## Adding Support for a New ML Framework

1. Create a new subdirectory under `src/gurobi_ml/<framework>/`.
2. Implement a class inheriting from the appropriate base in `src/gurobi_ml/modeling/`.
3. Register the new predictor in `src/gurobi_ml/registered_predictors.py`.
4. Add a `requirements.<framework>.txt` for the optional dependency.
5. Add a test directory `tests/test_<framework>/` with pytest tests.
6. Add a `tox.ini` section for the new framework (following the existing pattern).
7. Update the documentation in `docs/`.

## Pull Request Guidelines

- Link every PR to an existing GitHub issue.
- Include tests for all changed behavior.
- All pre-commit hooks must pass (`tox -e pre-commit`).
- All relevant test environments must pass.
- Respond to all review comments; resolve them after changes are applied.

## Useful Resources

- [Documentation](https://gurobi-optimization-gurobi-machine-learning.readthedocs-hosted.com/)
- [GitHub Issues](https://github.com/Gurobi/gurobi-machinelearning/issues)
