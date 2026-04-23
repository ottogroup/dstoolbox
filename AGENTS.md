# AGENTS.md: dstoolbox

## Project Overview

**dstoolbox** (v0.14.0) is a scikit-learn and pandas utility library, maintained by Otto Group BI in **life support mode** (compatibility updates only, no new major features).

- **Python**: 3.8, 3.9, 3.10 | **License**: Apache 2.0 | **Default branch**: `master`

## Tech Stack

- **Core**: numpy >=1.22.3, pandas >=1.4.2, scikit-learn >=1.1.0, scipy >=1.8.0
- **Dev**: pytest, pytest-cov, pylint, jupyter
- **Build**: setuptools, build, wheel

## Repository Structure

```
dstoolbox/
├── dstoolbox/
│   ├── pipeline.py             # SliceMixin, DictFeatureUnion, DataFrameFeatureUnion, TimedPipeline, PipelineY (deprecated)
│   ├── cluster.py              # HierarchicalClustering
│   ├── utils.py                # normalize_matrix, cosine_similarity (NO internal imports!)
│   ├── visualization.py        # NOT production-ready (untested, deps not in requirements)
│   ├── transformers/
│   │   ├── casting.py          # ToDataFrame
│   │   ├── preprocessing.py    # XLabelEncoder, ParallelFunctionTransformer
│   │   ├── slicing.py          # ItemSelector
│   │   ├── padding.py          # Padder2d, Padder3d
│   │   ├── text.py             # TextFeaturizer
│   │   └── tests/
│   ├── models/text.py          # W2VClassifier
│   └── tests/                  # test_pipeline.py, test_cluster.py, test_utils.py
├── notebooks/                  # Example Jupyter notebooks
├── setup.py / setup.cfg        # pytest config: testpaths, coverage, suffixes=Spec
├── requirements.txt / requirements-dev.txt
├── VERSION                     # Plain text version string (read by setup.py)
└── pylintrc                    # Strict: max line 80, docstrings >=10 chars required
```

## Local Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Commands

```bash
pytest dstoolbox                # All tests + coverage (from setup.cfg)
pytest dstoolbox -vv --pdb     # Verbose + debugger
pylint dstoolbox               # Must pass in CI
python -m build                # Wheel + sdist
```

## Key Classes

### `pipeline.py`
- **`PipelineY`** — DEPRECATED; use `sklearn.compose.TransformedTargetRegressor`
- **`SliceMixin`** — enables `pipeline[0:2]` slicing
- **`DictFeatureUnion`** — FeatureUnion returning dicts
- **`DataFrameFeatureUnion`** — FeatureUnion returning DataFrames with column names
- **`TimedPipeline`** — per-step timing stored in `timing_summary_`

### `transformers/`
All extend `BaseEstimator + TransformerMixin`:
- **`ToDataFrame`** — arrays/Series/dicts/lists → DataFrame
- **`XLabelEncoder`** — maps unknowns to `<UNKNOWN>` instead of raising
- **`ParallelFunctionTransformer`** — joblib.Parallel execution
- **`ItemSelector`** — select by key(s), callable, or slice
- **`Padder2d/3d`** — sequence padding

## Conventions

- **`utils.py`**: NEVER import other dstoolbox modules — prevents circular imports
- **Pylint**: max line 80, docstrings required (>=10 chars) for all non-test non-dunder
- **Test naming**: `test_*.py` (not `*_test.py`), class-based grouping
- **Coverage target**: 100%
- **Pylint good names**: i, j, k, X, y, Xt, Xs, yt, f

### Transformer Pattern
```python
class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param): self.param = param
    def fit(self, X, y=None): return self
    def transform(self, X): return X_transformed
```

## CI/CD

- **`build_test_python.yml`**: Push to `master` or `dev` → Python 3.8/3.9/3.10 → pylint → pytest
- **`deploy_to_pypi.yml`**: GitHub Release → `python -m build` → PyPI trusted publishing
- **Release**: Edit `VERSION` file → commit → tag `v{version}` → create GitHub Release

## Key Gotchas

1. **`visualization.py` is NOT production-ready** — deps not in requirements; don't use in prod
2. **`utils.py` must have no internal imports** — circular import risk
3. **`PipelineY` deprecated** — don't add features; logs DeprecationWarning
4. **Test discovery**: `python_files = test*py` in setup.cfg (not `*_test.py`)
5. **Default branch is `master`** (not `main`)\n