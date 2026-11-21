# ModelGuard — Production ML Reliability Control Plane

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)

ModelGuard is a local-first MLOps control plane for governing the operational lifecycle of tabular machine-learning models. It connects **data provenance, model lineage, deployment governance, and observable production behavior**.

---

## Current Status: Work Package A (Data Foundation & Validation) Completed

- [x] **Dataset Registry:** Logical dataset entity with task type, citation, and licensing.
- [x] **Immutable Versioning:** Streaming SHA-256 calculation, byte sizing, and schema hashing.
- [x] **Data Contract Engine:** Versioned Pandera contract for UCI Metro Interstate Traffic Volume.
- [x] **Validation Reporting:** Categorized blocking errors vs. warnings persisted as auditable records.
- [x] **FastAPI Control Plane:** `/api/v1/datasets`, `/api/v1/dataset-versions`, `/api/v1/validation-runs`, `/api/v1/health`.
- [x] **Test Suite:** 100% passing unit and API test coverage with deterministic fault-injection fixtures.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### 3. Run Tests and Linting
```bash
# Run test suite
pytest -v

# Run linter
ruff check .
```

### 4. Start Control Plane API
```bash
uvicorn app.main:app --reload --port 8000
```
Open interactive Swagger docs at `http://localhost:8000/docs`.

---

## Datasets & Attribution

ModelGuard uses official benchmark datasets with strict licensing attribution:
- **Metro Interstate Traffic Volume:** Hogue, J. (2019), UCI Machine Learning Repository, DOI: [10.24432/C5X60B](https://doi.org/10.24432/C5X60B), licensed under CC BY 4.0. See [DATASETS.md](docs/DATASETS.md).
