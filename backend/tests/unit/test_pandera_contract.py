from pathlib import Path

import pandas as pd

from app.domain.contracts.metro_traffic_v1 import MetroTrafficContractV1


def test_contract_metadata():
    contract = MetroTrafficContractV1()
    assert contract.version_tag == "v1"
    assert "traffic_volume" in contract.contract_schema_json["required_columns"]
    assert contract.contract_schema_json["bounds"]["temp"]["min"] == 150.0


def test_contract_validates_clean_data(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    contract = MetroTrafficContractV1()
    summary = contract.validate(df, dataset_version_id="test-v1")

    assert summary.status == "PASSED"
    assert summary.error_count == 0
    assert summary.total_records == len(df)
    assert len(summary.findings) == 0


def test_contract_detects_sensor_and_category_failures(invalid_csv_path: Path):
    df = pd.read_csv(invalid_csv_path)
    contract = MetroTrafficContractV1()
    summary = contract.validate(df, dataset_version_id="test-v1")

    assert summary.status == "FAILED"
    assert summary.error_count > 0
    # Findings should include temp check and weather_main category failure
    failed_columns = {f.column for f in summary.findings}
    assert "temp" in failed_columns
    assert "weather_main" in failed_columns
    assert "traffic_volume" in failed_columns


def test_contract_missing_required_column():
    df = pd.DataFrame(
        {
            "temp": [290.0],
            "rain_1h": [0.0],
            # weather_main and traffic_volume are completely missing
        }
    )
    contract = MetroTrafficContractV1()
    summary = contract.validate(df, dataset_version_id="test-missing")

    assert summary.status == "FAILED"
    assert summary.error_count > 0


def test_contract_duplicate_timestamps_warning(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    # Duplicate first row timestamp
    df.loc[1, "date_time"] = df.loc[0, "date_time"]
    contract = MetroTrafficContractV1()
    summary = contract.validate(df, dataset_version_id="test-dup")

    # Should raise a WARNING for duplicate timestamps, not block if other checks pass
    assert summary.warning_count >= 1
    assert any(f.check == "unique_timestamps" for f in summary.findings)


def test_contract_permits_nullable_temp_and_rain(valid_csv_path: Path):
    df = pd.read_csv(valid_csv_path)
    # Set temp and rain_1h in first row to None/NaN
    df.loc[0, "temp"] = None
    df.loc[0, "rain_1h"] = None

    contract = MetroTrafficContractV1()
    summary = contract.validate(df, dataset_version_id="test-null-sensors")

    assert summary.status == "PASSED"
    assert summary.error_count == 0
