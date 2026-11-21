from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from app.schemas.validation import ValidationSummary


class BaseDataContract(ABC):
    """Abstract base class for all ModelGuard dataset contracts."""

    @property
    @abstractmethod
    def version_tag(self) -> str:
        """The version tag of this contract (e.g. 'v1')."""
        pass

    @property
    @abstractmethod
    def contract_schema_json(self) -> dict[str, Any]:
        """Declarative description of the contract schema."""
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame, dataset_version_id: str) -> ValidationSummary:
        """Validate a pandas DataFrame against this contract and return structured findings."""
        pass
