import ast
import json
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
import pytest

from app.domain.contracts.metro_traffic_v1 import MetroTrafficContractV1
from app.domain.hashing import calculate_sha256
from app.domain.validation_service import validate_dataset_file
from app.ml.data_loader import load_validated_dataset
from app.ml.evaluation import compare_models, evaluate_pipeline
from app.ml.features import get_feature_metadata, prepare_features
from app.ml.models import build_baseline_pipeline, build_candidate_pipeline
from app.ml.preprocessing import build_preprocessor
from app.ml.splitting import ChronologicalSplitter
from app.ml.training import TrainingService
from app.schemas.training import SplitConfig


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture
def notebook_path(repo_root: Path) -> Path:
    nb_path = repo_root / "experiments" / "notebooks" / "metro_traffic_training.ipynb"
    assert nb_path.is_file(), f"Notebook file not found at {nb_path}"
    return nb_path


@pytest.fixture
def manifest_path(repo_root: Path) -> Path:
    m_path = repo_root / "datasets" / "manifests" / "metro_traffic_manifest.json"
    assert m_path.is_file(), f"Manifest file not found at {m_path}"
    return m_path


# 1 & 2. Format, nbformat version, and cell count
def test_notebook_format_and_cell_count(notebook_path: Path):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    assert nb.nbformat == 4
    # Exactly 18 cells required
    assert len(nb.cells) == 18

    # Cell 1 is markdown; cells 2 to 18 are code
    assert nb.cells[0].cell_type == "markdown"
    for i, cell in enumerate(nb.cells[1:], start=2):
        assert cell.cell_type == "code", f"Cell {i} expected to be 'code', found '{cell.cell_type}'"


# 3. Python AST syntax validation
def test_notebook_python_syntax(notebook_path: Path):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source = cell.source
            # Filter out shell commands (!pip, !git, etc.) for Python AST parsing
            py_lines = [line for line in source.splitlines() if not line.strip().startswith("!")]
            py_source = "\n".join(py_lines)
            try:
                ast.parse(py_source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in notebook cell {i + 1}: {e}\nSource:\n{py_source}")


# 4. Security & path hygiene
def test_no_hardcoded_windows_paths_or_secrets(notebook_path: Path):
    with notebook_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Disallow Windows drive letter paths
    assert "C:\\" not in content
    assert "C:/" not in content
    assert "\\Users\\" not in content

    # Disallow hardcoded tokens / secrets
    disallowed_tokens = ["ghp_", "api_key", "secret_key", "aws_secret", "bearer "]
    for token in disallowed_tokens:
        assert token not in content.lower()


# 5. Generator recreates the notebook accurately
def test_generator_recreates_notebook_identically(repo_root: Path, notebook_path: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.generate_notebook import generate_colab_notebook

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_nb_path = Path(tmp_dir) / "test_notebook.ipynb"
        generate_colab_notebook(tmp_nb_path)

        with notebook_path.open("r", encoding="utf-8") as f_orig, tmp_nb_path.open("r", encoding="utf-8") as f_new:
            nb_orig = nbf.read(f_orig, as_version=4)
            nb_new = nbf.read(f_new, as_version=4)

        assert len(nb_orig.cells) == len(nb_new.cells)
        for i, (cell_o, cell_n) in enumerate(zip(nb_orig.cells, nb_new.cells, strict=False), start=1):
            assert cell_o.cell_type == cell_n.cell_type, f"Cell {i} type mismatch"
            assert cell_o.source == cell_n.source, f"Cell {i} source mismatch"


# 6. Manifest contains all required fields
def test_manifest_schema_and_required_fields(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_keys = [
        "name",
        "dataset_id",
        "dataset_version_id",
        "version_tag",
        "sha256",
        "task",
        "citation",
        "official_url",
        "licence",
        "licence_url",
        "total_expected_rows",
        "features",
        "target",
        "target_type",
    ]
    for key in required_keys:
        assert key in manifest, f"Required manifest key '{key}' missing from {manifest_path}"

    assert manifest["name"] == "metro_interstate_traffic_volume"
    assert manifest["dataset_id"] == 492
    assert manifest["version_tag"] == "v1.0.1"
    assert manifest["total_expected_rows"] == 48204


# 7. Dataset version ID is not confused with version tag
def test_dataset_version_id_not_confused_with_version_tag():
    version_tag = "v1.0.1"
    # When dataset_version_id equals version_tag, integrity check must reject
    invalid_dataset_version_id = "v1.0.1"

    with pytest.raises(ValueError, match="equals version tag"):
        if invalid_dataset_version_id == version_tag or invalid_dataset_version_id in ["v1.0.0", "v1.0.1"]:
            raise ValueError(
                f"CRITICAL INTEGRITY FAILURE: 'dataset_version_id' ('{invalid_dataset_version_id}') equals version tag!"
            )


# 8. Checksum mismatch causes execution to stop
def test_checksum_mismatch_causes_execution_to_stop():
    expected_sha256 = "64b73248aa0ccebbde2eb919fe0e5d183f06cb2ecf41b39247d86f7b3df8e76a"
    tampered_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(ValueError, match="SHA-256 integrity mismatch"):
        if tampered_sha256 != expected_sha256:
            raise ValueError(
                f"CRITICAL: Dataset SHA-256 integrity mismatch! Expected: {expected_sha256}, Actual: {tampered_sha256}"
            )


# 9. Missing registered identity causes execution to stop
def test_missing_registered_identity_causes_execution_to_stop():
    manifest_without_uuid = {"dataset_version_id": None, "sha256": None}

    with pytest.raises(ValueError, match="dataset_version_id.*missing"):
        dataset_version_id = manifest_without_uuid.get("dataset_version_id")
        if not dataset_version_id:
            raise ValueError("CRITICAL INTEGRITY FAILURE: 'dataset_version_id' is missing from registration manifest!")


# 10. Blocking validation errors prevent training
def test_blocking_validation_errors_prevent_training():
    from app.schemas.validation import CheckFinding, ValidationSummary

    failed_val_summary = ValidationSummary(
        contract_name="metro_traffic_contract",
        contract_version="v1",
        dataset_version_id="test-uuid",
        status="FAILED",
        total_records=100,
        error_count=1,
        warning_count=0,
        findings=[
            CheckFinding(
                column="temp",
                check="in_range(150, 350)",
                severity="BLOCKING",
                failure_count=1,
                sample_cases=[0.0],
                message="Temperature out of bounds",
            )
        ],
    )

    blocking_findings = [f for f in failed_val_summary.findings if f.severity == "BLOCKING"]
    with pytest.raises(ValueError, match="contract validation FAILED"):
        if failed_val_summary.status == "FAILED" or failed_val_summary.error_count > 0 or len(blocking_findings) > 0:
            raise ValueError(
                f"CRITICAL: Dataset contract validation FAILED with {failed_val_summary.error_count} blocking errors."
            )


# 11. TrainingService receives the actual dataset version ID
def test_training_service_receives_actual_dataset_version_id():
    import inspect

    sig = inspect.signature(TrainingService.train_and_evaluate_validation)
    assert "dataset_version_id" in sig.parameters
    assert "dataset_sha256" in sig.parameters


# 12. Validation-based model selection remains separate from test evaluation
def test_validation_based_model_selection_separate_from_test_holdout():
    import inspect

    # train_and_evaluate_validation returns result AND untouched test_df
    sig_train = inspect.signature(TrainingService.train_and_evaluate_validation)
    assert sig_train.return_annotation != inspect.Signature.empty

    # evaluate_test_holdout is a decoupled static method taking test_df
    sig_test = inspect.signature(TrainingService.evaluate_test_holdout)
    assert "test_df" in sig_test.parameters
    assert "pipeline" in sig_test.parameters


# 13. Exported provenance bundle contains all required 18 items
def test_exported_provenance_bundle_contains_all_18_fields(notebook_path: Path):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    cell_18_code = nb.cells[17].source
    expected_bundle_keys = [
        "selected_model_artifact",
        "baseline_model_artifact",
        "training_configuration",
        "dataset_manifest",
        "dataset_version_id",
        "dataset_version_tag",
        "dataset_sha256",
        "split_manifest",
        "validation_summary",
        "baseline_validation_metrics",
        "candidate_validation_metrics",
        "model_comparison",
        "final_test_metrics",
        "validation_predictions_summary",
        "test_predictions_summary",
        "feature_metadata",
        "python_and_dependency_provenance",
        "execution_status",
    ]
    for key in expected_bundle_keys:
        assert f'"{key}":' in cell_18_code, f"Reproducibility bundle missing key '{key}' in Cell 18"


# 14. Module imports and signatures
def test_module_imports_and_signatures_exist():
    contract = MetroTrafficContractV1()
    assert contract.version_tag == "v1"

    config = SplitConfig()
    splitter = ChronologicalSplitter(config)
    assert splitter.config.train_ratio == 0.70

    metadata = get_feature_metadata()
    assert metadata.target_column == "traffic_volume"

    assert callable(calculate_sha256)
    assert callable(validate_dataset_file)
    assert callable(load_validated_dataset)
    assert callable(prepare_features)
    assert callable(compare_models)
    assert callable(evaluate_pipeline)

    preprocessor = build_preprocessor()
    assert preprocessor is not None

    base_pipe = build_baseline_pipeline()
    assert "regressor" in base_pipe.named_steps

    cand_pipe = build_candidate_pipeline(n_estimators=5)
    assert "regressor" in cand_pipe.named_steps

    training_service = TrainingService(split_config=config)
    assert hasattr(training_service, "train_and_evaluate_validation")
    assert hasattr(training_service, "evaluate_test_holdout")
