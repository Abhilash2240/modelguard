from pathlib import Path

import pandas as pd
from app.core.errors import EntityNotFoundError, StorageError
from app.domain.contracts.base import BaseDataContract
from app.domain.contracts.metro_traffic_v1 import MetroTrafficContractV1
from app.schemas.validation import ValidationSummary

CONTRACT_REGISTRY: dict[str, type[BaseDataContract]] = {
    "metro_interstate_traffic_volume:v1": MetroTrafficContractV1,
}


def get_contract(dataset_name: str, version_tag: str = "v1") -> BaseDataContract:
    """Resolve a versioned contract implementation by dataset name and version."""
    key = f"{dataset_name}:{version_tag}"
    contract_cls = CONTRACT_REGISTRY.get(key)
    if not contract_cls:
        # Fallback to metro traffic contract if it's the primary dataset
        if "metro" in dataset_name.lower():
            return MetroTrafficContractV1()
        raise EntityNotFoundError(f"No contract registered for '{dataset_name}' with version '{version_tag}'.")
    return contract_cls()


def validate_dataset_file(
    file_path: Path,
    dataset_name: str,
    dataset_version_id: str,
    contract_version: str = "v1",
) -> ValidationSummary:
    """Load dataset file from disk and execute contract validation."""
    if not file_path.is_file():
        raise StorageError(f"Dataset snapshot file not found: {file_path}")

    try:
        # Read CSV snapshot
        df = pd.read_csv(file_path)
    except Exception as e:
        raise StorageError(f"Failed to parse CSV dataset: {e}")

    contract = get_contract(dataset_name, contract_version)
    return contract.validate(df, dataset_version_id)
