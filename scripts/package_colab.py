"""Build and verify ModelGuard_Colab_Package.zip for Google Colab execution.

Strict rules:
- Top-level directory: ModelGuard_Colab_Package/
- Includes required backend app modules, pyproject.toml, manifests, registered v1.0.1 CSV, notebook.
- Excludes .venv, .git, cache, secrets, and SQLite database.
- Verifies dataset SHA-256 and 48,204 row count before and after packaging.
- Verifies 18-cell notebook integrity.
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import nbformat as nbf
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.domain.hashing import calculate_sha256  # noqa: E402

EXPECTED_DATASET_NAME = "metro_interstate_traffic_volume"
EXPECTED_DATASET_ID = 492
EXPECTED_VERSION_TAG = "v1.0.1"
EXPECTED_VERSION_UUID = "8da56ac6-8614-4789-943c-5e612008e77b"
EXPECTED_SHA256 = "dad82c80c03d6cd881d60e1acc8e1237a75e241727fe367b997aa62f83c32201"
EXPECTED_ROW_COUNT = 48204

EXCLUDED_SUBSTRINGS = [
    ".venv",
    "venv",
    ".git",
    "node_modules",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "modelguard.db",
]


def pre_packaging_verification() -> tuple[Path, Path, Path, Path]:
    print("=== Step 1: Pre-Packaging Verification ===")

    # 1. Dataset file verification
    dataset_file = repo_root / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.1_dad82c80.csv"
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Registered dataset v1.0.1 file not found at: {dataset_file}")

    actual_hash = calculate_sha256(dataset_file)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Dataset SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_hash}")

    df = pd.read_csv(dataset_file)
    actual_rows = len(df)
    if actual_rows != EXPECTED_ROW_COUNT:
        raise ValueError(f"Dataset row count mismatch! Expected {EXPECTED_ROW_COUNT}, got {actual_rows}")
    print(f"[OK] Dataset file verified: {dataset_file.name} ({actual_rows:,} rows, SHA-256: {actual_hash})")

    # 2. Manifest verification
    manifest_file = repo_root / "datasets" / "manifests" / "metro_traffic_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found at: {manifest_file}")

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("name") != EXPECTED_DATASET_NAME:
        raise ValueError(f"Manifest name mismatch: {manifest.get('name')}")
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(f"Manifest dataset_id mismatch: {manifest.get('dataset_id')}")
    if manifest.get("version_tag") != EXPECTED_VERSION_TAG:
        raise ValueError(f"Manifest version_tag mismatch: {manifest.get('version_tag')}")
    if manifest.get("dataset_version_id") != EXPECTED_VERSION_UUID:
        raise ValueError(f"Manifest dataset_version_id mismatch: {manifest.get('dataset_version_id')}")
    if manifest.get("sha256") != EXPECTED_SHA256:
        raise ValueError(f"Manifest sha256 mismatch: {manifest.get('sha256')}")
    print(f"[OK] Manifest verified: version={EXPECTED_VERSION_TAG}, UUID={EXPECTED_VERSION_UUID}")

    # 3. Cleaning provenance JSON
    provenance_file = repo_root / "datasets" / "manifests" / "metro_traffic_cleaning_provenance_v1.0.1.json"
    if not provenance_file.is_file():
        raise FileNotFoundError(f"Cleaning provenance file not found at: {provenance_file}")
    print(f"[OK] Cleaning provenance JSON verified: {provenance_file.name}")

    # 4. Notebook verification
    notebook_file = repo_root / "experiments" / "notebooks" / "metro_traffic_training.ipynb"
    if not notebook_file.is_file():
        raise FileNotFoundError(f"Notebook file not found at: {notebook_file}")

    with notebook_file.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    if len(nb.cells) != 18:
        raise ValueError(f"Notebook cell count mismatch! Expected 18, found {len(nb.cells)}")
    print(f"[OK] Notebook verified: {notebook_file.name} (exactly 18 cells, nbformat {nb.nbformat}.{nb.nbformat_minor})")

    return dataset_file, manifest_file, provenance_file, notebook_file


def collect_package_files(
    dataset_file: Path,
    manifest_file: Path,
    provenance_file: Path,
    notebook_file: Path,
) -> list[tuple[Path, str]]:
    """Collect all files to include with their relative archive paths."""
    prefix = "ModelGuard_Colab_Package"
    files_to_pack: list[tuple[Path, str]] = []

    # 1. pyproject.toml
    pyproject_file = repo_root / "backend" / "pyproject.toml"
    files_to_pack.append((pyproject_file, f"{prefix}/backend/pyproject.toml"))

    # 2. backend/app modules
    app_dir = repo_root / "backend" / "app"
    app_subdirs = ["core", "db", "domain", "ml", "schemas"]

    # Root app __init__.py
    root_app_init = app_dir / "__init__.py"
    if root_app_init.exists():
        files_to_pack.append((root_app_init, f"{prefix}/backend/app/__init__.py"))

    for subdir in app_subdirs:
        sub_path = app_dir / subdir
        for f in sub_path.rglob("*.py"):
            # Exclude __pycache__
            if "__pycache__" in f.parts:
                continue
            rel_within_app = f.relative_to(app_dir)
            archive_path = f"{prefix}/backend/app/{rel_within_app.as_posix()}"
            files_to_pack.append((f, archive_path))

    # 3. Datasets & manifests
    files_to_pack.append((manifest_file, f"{prefix}/datasets/manifests/{manifest_file.name}"))
    files_to_pack.append((provenance_file, f"{prefix}/datasets/manifests/{provenance_file.name}"))

    # 4. Storage dataset snapshot
    files_to_pack.append((dataset_file, f"{prefix}/storage/datasets/{dataset_file.name}"))

    # 5. Notebook
    files_to_pack.append((notebook_file, f"{prefix}/experiments/notebooks/{notebook_file.name}"))

    return files_to_pack


def build_and_verify_zip() -> dict:
    dataset_file, manifest_file, provenance_file, notebook_file = pre_packaging_verification()
    files_to_pack = collect_package_files(dataset_file, manifest_file, provenance_file, notebook_file)

    print("\n=== Step 2: Security & Exclusions Audit ===")
    for src_path, arc_path in files_to_pack:
        for excl in EXCLUDED_SUBSTRINGS:
            if excl in arc_path or excl in src_path.name:
                raise ValueError(f"SECURITY BREACH: Excluded file pattern '{excl}' found in package: {arc_path}")
    print(f"[OK] Pre-packaging security check passed for all {len(files_to_pack)} candidate files.")

    # Create ZIP archive
    zip_path = repo_root / "ModelGuard_Colab_Package.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"\n=== Step 3: Creating Archive: {zip_path.name} ===")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_path in files_to_pack:
            zf.write(src_path, arc_path)

    zip_size_bytes = zip_path.stat().st_size
    print(f"Archive created: {zip_path} ({zip_size_bytes:,} bytes, {zip_size_bytes / (1024**2):.2f} MB)")

    # Step 4: Post-packaging verification
    print("\n=== Step 4: Post-Packaging Verification & Extraction Test ===")
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupted = zf.testzip()
        if corrupted is not None:
            raise ValueError(f"ZIP integrity test failed on file: {corrupted}")
        print("[OK] ZIP file CRC32 integrity test passed (testzip() is clean).")

        namelist = zf.namelist()
        print(f"[OK] Total entries in archive: {len(namelist)}")

        # Verify no forbidden files in namelist
        for name in namelist:
            for excl in EXCLUDED_SUBSTRINGS:
                if excl in name:
                    raise ValueError(f"SECURITY BREACH in archive namelist: '{name}' matches '{excl}'")

    # Extract to temporary directory and test artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_extract_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract_path)

        extracted_pkg = tmp_extract_path / "ModelGuard_Colab_Package"
        extracted_dataset = extracted_pkg / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.1_dad82c80.csv"
        extracted_manifest = extracted_pkg / "datasets" / "manifests" / "metro_traffic_manifest.json"
        extracted_provenance = extracted_pkg / "datasets" / "manifests" / "metro_traffic_cleaning_provenance_v1.0.1.json"
        extracted_notebook = extracted_pkg / "experiments" / "notebooks" / "metro_traffic_training.ipynb"
        extracted_pyproject = extracted_pkg / "backend" / "pyproject.toml"

        assert extracted_dataset.is_file(), "Extracted dataset missing"
        assert extracted_manifest.is_file(), "Extracted manifest missing"
        assert extracted_provenance.is_file(), "Extracted provenance missing"
        assert extracted_notebook.is_file(), "Extracted notebook missing"
        assert extracted_pyproject.is_file(), "Extracted pyproject.toml missing"

        # SHA-256 verification of extracted dataset
        extracted_hash = calculate_sha256(extracted_dataset)
        if extracted_hash != EXPECTED_SHA256:
            raise ValueError(f"Extracted dataset SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {extracted_hash}")
        print(f"[OK] Extracted dataset SHA-256 verified: {extracted_hash}")

        # Row count verification
        extracted_df = pd.read_csv(extracted_dataset)
        if len(extracted_df) != EXPECTED_ROW_COUNT:
            raise ValueError(f"Extracted dataset row count mismatch! Expected {EXPECTED_ROW_COUNT}, got {len(extracted_df)}")
        print(f"[OK] Extracted dataset row count verified: {len(extracted_df):,} rows")

        # Notebook cell count verification
        with extracted_notebook.open("r", encoding="utf-8") as f:
            nb_extracted = nbf.read(f, as_version=4)
        if len(nb_extracted.cells) != 18:
            raise ValueError(f"Extracted notebook cell count mismatch! Expected 18, found {len(nb_extracted.cells)}")
        print(f"[OK] Extracted notebook verified: exactly 18 cells, nbformat {nb_extracted.nbformat}.{nb_extracted.nbformat_minor}")

        # Python import verification from extracted backend
        sys.path.insert(0, str(extracted_pkg / "backend"))
        from app.domain.contracts.metro_traffic_v1 import MetroTrafficContractV1
        from app.ml.training import TrainingService
        from app.schemas.training import SplitConfig

        contract = MetroTrafficContractV1()
        assert contract.version_tag == "v1"
        splitter_cfg = SplitConfig()
        svc = TrainingService(split_config=splitter_cfg)
        assert svc is not None
        print("[OK] Extracted backend modules verified: clean import and initialization successful.")

    print("\n=== PACKAGE VERIFICATION COMPLETE: ALL CHECKS PASSED ===")
    return {
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_size_bytes,
        "zip_size_mb": round(zip_size_bytes / (1024**2), 2),
        "file_count": len(files_to_pack),
        "archive_entries": len(namelist),
        "dataset_sha256": extracted_hash,
        "dataset_rows": len(extracted_df),
        "notebook_cells": len(nb_extracted.cells),
        "status": "READY FOR GOOGLE COLAB UPLOAD",
    }


if __name__ == "__main__":
    result = build_and_verify_zip()
    print("\nRESULT_JSON:")
    print(json.dumps(result, indent=2))
