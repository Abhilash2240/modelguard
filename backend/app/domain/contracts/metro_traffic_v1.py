from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from app.domain.contracts.base import BaseDataContract
from app.schemas.validation import CheckFinding, ValidationSummary

# Allowable weather main categories from UCI documentation
VALID_WEATHER_MAIN = [
    "Clear",
    "Clouds",
    "Rain",
    "Drizzle",
    "Mist",
    "Haze",
    "Fog",
    "Thunderstorm",
    "Snow",
    "Squall",
    "Smoke",
]

# Pandera DataFrame Schema for Metro Interstate Traffic Volume
metro_traffic_schema_v1 = pa.DataFrameSchema(
    columns={
        "holiday": pa.Column(
            pa.String,
            nullable=True,
            required=True,
            description="US national or state holiday or None",
        ),
        "temp": pa.Column(
            pa.Float,
            checks=[
                pa.Check.gt(150.0, error="Temperature must be > 150.0 Kelvin (sensor error filter)"),
                pa.Check.lt(350.0, error="Temperature must be < 350.0 Kelvin (extreme physics bound)"),
            ],
            nullable=True,
            required=True,
            description="Average temperature in Kelvin",
        ),
        "rain_1h": pa.Column(
            pa.Float,
            checks=[
                pa.Check.ge(0.0, error="Rainfall must be non-negative"),
                pa.Check.lt(500.0, error="Hourly rainfall must be < 500 mm"),
            ],
            nullable=True,
            required=True,
            description="Amount in mm of rain that occurred in the hour",
        ),
        "snow_1h": pa.Column(
            pa.Float,
            checks=[
                pa.Check.ge(0.0, error="Snowfall must be non-negative"),
                pa.Check.lt(50.0, error="Hourly snowfall must be < 50 mm"),
            ],
            nullable=False,
            required=True,
            description="Amount in mm of snow that occurred in the hour",
        ),
        "clouds_all": pa.Column(
            pa.Int,
            checks=[
                pa.Check.ge(0, error="Cloud coverage percentage must be >= 0"),
                pa.Check.le(100, error="Cloud coverage percentage must be <= 100"),
            ],
            nullable=False,
            required=True,
            description="Percentage of cloud cover",
        ),
        "weather_main": pa.Column(
            pa.String,
            checks=[
                pa.Check.isin(
                    VALID_WEATHER_MAIN,
                    error=f"Weather category must be one of {VALID_WEATHER_MAIN}",
                ),
            ],
            nullable=False,
            required=True,
            description="Short textual description of the current weather",
        ),
        "weather_description": pa.Column(
            pa.String,
            nullable=False,
            required=True,
            description="Granular textual description of the current weather",
        ),
        "date_time": pa.Column(
            pa.String,
            nullable=False,
            required=True,
            description="Hour of the data collected in local CST (YYYY-MM-DD HH:MM:SS)",
        ),
        "traffic_volume": pa.Column(
            pa.Int,
            checks=[
                pa.Check.ge(0, error="Traffic volume must be >= 0"),
                pa.Check.le(12000, error="Traffic volume must be <= 12000 vehicles/hour"),
            ],
            nullable=False,
            required=True,
            description="Hourly I-94 westbound traffic volume target",
        ),
    },
    strict=True,
    coerce=True,
)


class MetroTrafficContractV1(BaseDataContract):
    """Version 1 Data Contract for UCI Metro Interstate Traffic Volume dataset."""

    @property
    def version_tag(self) -> str:
        return "v1"

    @property
    def contract_schema_json(self) -> dict[str, Any]:
        return {
            "name": "metro_interstate_traffic_volume",
            "version": "v1",
            "required_columns": list(metro_traffic_schema_v1.columns.keys()),
            "valid_weather_main": VALID_WEATHER_MAIN,
            "bounds": {
                "temp": {"min": 150.0, "max": 350.0, "unit": "Kelvin"},
                "rain_1h": {"min": 0.0, "max": 500.0, "unit": "mm"},
                "snow_1h": {"min": 0.0, "max": 50.0, "unit": "mm"},
                "clouds_all": {"min": 0, "max": 100, "unit": "%"},
                "traffic_volume": {"min": 0, "max": 12000, "unit": "vehicles"},
            },
        }

    def validate(self, df: pd.DataFrame, dataset_version_id: str) -> ValidationSummary:
        findings: list[CheckFinding] = []
        total_records = len(df)

        try:
            metro_traffic_schema_v1.validate(df, lazy=True)
        except SchemaErrors as err:
            failure_df = err.failure_cases
            # Pandas groupby drops NaN by default; pass dropna=False so column=None is preserved
            grouped = failure_df.groupby(["column", "check"], dropna=False)
            for (col, check_name), group in grouped:
                sample_cases = group["failure_case"].head(5).tolist()
                col_name = str(col) if pd.notna(col) else None
                # If column is None and check is column_in_dataframe, sample_cases holds missing col name
                if col_name is None and check_name == "column_in_dataframe" and sample_cases:
                    col_name = str(sample_cases[0])
                findings.append(
                    CheckFinding(
                        column=col_name,
                        check=str(check_name),
                        severity="BLOCKING",
                        failure_count=len(group),
                        sample_cases=[str(c) for c in sample_cases],
                        message=f"Check '{check_name}' failed on column '{col_name}' for {len(group)} records.",
                    )
                )
        except Exception as err:
            findings.append(
                CheckFinding(
                    column=None,
                    check="schema_validation",
                    severity="BLOCKING",
                    failure_count=1,
                    sample_cases=[],
                    message=f"Schema validation error: {err}",
                )
            )

        # Check for non-schema warnings (e.g. timestamp duplicate warnings)
        if "date_time" in df.columns:
            duplicates = df.duplicated(subset=["date_time"]).sum()
            if duplicates > 0:
                findings.append(
                    CheckFinding(
                        column="date_time",
                        check="unique_timestamps",
                        severity="WARNING",
                        failure_count=int(duplicates),
                        sample_cases=[],
                        message=f"Found {duplicates} duplicate timestamps in dataset.",
                    )
                )

        error_count = sum(1 for f in findings if f.severity == "BLOCKING")
        warning_count = sum(1 for f in findings if f.severity == "WARNING")

        if error_count > 0:
            final_status = "FAILED"
        elif warning_count > 0:
            final_status = "WARNING"
        else:
            final_status = "PASSED"

        return ValidationSummary(
            status=final_status,
            dataset_version_id=dataset_version_id,
            contract_version=self.version_tag,
            total_records=total_records,
            error_count=error_count,
            warning_count=warning_count,
            findings=findings,
        )
