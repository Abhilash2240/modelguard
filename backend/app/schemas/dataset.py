from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    nullable: bool = False


class DatasetBase(BaseModel):
    name: str = Field(..., max_length=100, description="Unique name of dataset")
    task_type: str = Field(
        ..., max_length=50, description="ML task type: regression or classification"
    )
    source_uri: str = Field(..., max_length=255, description="Official source URL or reference")
    licence: str = Field(..., max_length=100, description="Licence tag (e.g. CC BY 4.0)")
    description: str | None = Field(None, description="Detailed dataset description")


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DatasetVersionCreate(BaseModel):
    version_tag: str = Field(
        ..., pattern=r"^v[0-9]+(\.[0-9]+)*$", description="Version string, e.g. v1.0.0"
    )


class DatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version_tag: str
    storage_uri: str
    sha256_hash: str
    byte_size: int
    row_count: int
    columns_json: list[dict[str, Any]]
    schema_hash: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DatasetDetailResponse(DatasetResponse):
    versions: list[DatasetVersionResponse] = []
