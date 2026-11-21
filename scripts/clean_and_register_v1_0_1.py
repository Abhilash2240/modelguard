"""Clean UCI Metro Interstate Traffic Volume dataset and register version v1.0.1.

Applies controlled remediation for 11 known sensor anomalies:
- 10 temperature values of 0.0 K converted to missing
- 1 rainfall value of 9831.3 mm converted to missing

Preserves original raw snapshot v1.0.0 unchanged.
Generates full cleaning provenance report.
Registers immutable version v1.0.1 in ModelGuard database and validates against MetroTrafficContractV1.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Ensure backend modules are on sys.path
repo_root = Path(__file__).resolve().parent.parent
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.db.session import SessionLocal  # noqa: E402
from app.domain.hashing import calculate_sha256  # noqa: E402
from app.services.dataset_service import DatasetService  # noqa: E402


def main() -> None:
    print("=== MODELGUARD CONTROLLED DATA CLEANING & v1.0.1 REGISTRATION ===")

    # 1. Source raw snapshot paths and verification
    raw_snapshot_path = repo_root / "storage" / "datasets" / "metro_interstate_traffic_volume_v1.0.0_749c90d7.csv"
    if not raw_snapshot_path.is_file():
        raise FileNotFoundError(f"Raw snapshot v1.0.0 not found at: {raw_snapshot_path}")

    raw_sha256 = calculate_sha256(raw_snapshot_path)
    expected_raw_sha256 = "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e"
    if raw_sha256 != expected_raw_sha256:
        raise ValueError(f"Raw snapshot SHA-256 mismatch! Expected {expected_raw_sha256}, got {raw_sha256}")

    raw_df = pd.read_csv(raw_snapshot_path)
    orig_row_count = len(raw_df)
    print(f"Verified raw snapshot: {raw_snapshot_path.name}")
    print(f"Raw rows: {orig_row_count:,}, SHA-256: {raw_sha256}")

    # 2. Inspect anomaly records
    temp_anomalies = raw_df[raw_df["temp"] <= 150.0]
    temp_anomaly_count = len(temp_anomalies)
    if temp_anomaly_count != 10:
        raise ValueError(f"Unexpected temperature anomaly count: expected 10, found {temp_anomaly_count}")

    non_zero_temp = temp_anomalies[temp_anomalies["temp"] != 0.0]
    if len(non_zero_temp) > 0:
        raise ValueError(f"Found temperature anomalies with temp != 0.0: {non_zero_temp}")

    rain_anomalies = raw_df[raw_df["rain_1h"] >= 500.0]
    rain_anomaly_count = len(rain_anomalies)
    if rain_anomaly_count != 1:
        raise ValueError(f"Unexpected rainfall anomaly count: expected 1, found {rain_anomaly_count}")

    non_9831_rain = rain_anomalies[rain_anomalies["rain_1h"] != 9831.3]
    if len(non_9831_rain) > 0:
        raise ValueError(f"Found rainfall anomalies with rain_1h != 9831.3: {non_9831_rain}")

    print(f"Confirmed exactly {temp_anomaly_count} temp == 0.0 K anomalies.")
    print(f"Confirmed exactly {rain_anomaly_count} rain_1h == 9831.3 mm anomaly.")

    # 3. Deterministic cleaning transformation
    cleaned_df = raw_df.copy()
    cleaned_df.loc[cleaned_df["temp"] <= 150.0, "temp"] = None
    cleaned_df.loc[cleaned_df["rain_1h"] >= 500.0, "rain_1h"] = None

    cleaned_row_count = len(cleaned_df)
    if cleaned_row_count != orig_row_count:
        raise ValueError(f"Row count altered during cleaning! Original: {orig_row_count}, Cleaned: {cleaned_row_count}")

    # Verify no remaining out-of-bound values in non-null entries
    remaining_bad_temp = cleaned_df[cleaned_df["temp"].notna() & ((cleaned_df["temp"] <= 150.0) | (cleaned_df["temp"] >= 350.0))]
    if len(remaining_bad_temp) > 0:
        raise ValueError(f"Remaining bad temperature values found: {len(remaining_bad_temp)}")

    remaining_bad_rain = cleaned_df[cleaned_df["rain_1h"].notna() & ((cleaned_df["rain_1h"] < 0.0) | (cleaned_df["rain_1h"] >= 500.0))]
    if len(remaining_bad_rain) > 0:
        raise ValueError(f"Remaining bad rainfall values found: {len(remaining_bad_rain)}")

    # 4. Save cleaned staging file
    staging_dir = repo_root / "data" / "cleaned"
    staging_dir.mkdir(parents=True, exist_ok=True)
    cleaned_staging_path = staging_dir / "Metro_Interstate_Traffic_Volume_v1.0.1.csv"
    cleaned_df.to_csv(cleaned_staging_path, index=False)
    cleaned_sha256 = calculate_sha256(cleaned_staging_path)
    print(f"Cleaned staging file written: {cleaned_staging_path}")
    print(f"Cleaned SHA-256: {cleaned_sha256}")

    # 5. Generate cleaning provenance report
    cleaning_report = {
        "cleaning_policy_version": "v1.0",
        "cleaning_timestamp_utc": datetime.now(UTC).isoformat(),
        "source_dataset_uuid": "5fe9641a-0283-45b5-bfbf-f8449fdfcf73",
        "source_dataset_version_uuid": "5781b57e-aa75-40f9-a5d5-592ad6160ba0",
        "source_version_tag": "v1.0.0",
        "source_sha256": raw_sha256,
        "original_row_count": orig_row_count,
        "cleaned_row_count": cleaned_row_count,
        "original_sha256": raw_sha256,
        "cleaned_sha256": cleaned_sha256,
        "columns_modified": ["temp", "rain_1h"],
        "modifications": [
            {
                "column": "temp",
                "condition": "temp <= 150.0 (specifically temp == 0.0)",
                "action": "convert_to_null",
                "count": temp_anomaly_count,
                "reason": "MN DOT weather sensor dropout recording unphysical absolute zero (0.0 K / -273.15 C)",
            },
            {
                "column": "rain_1h",
                "condition": "rain_1h >= 500.0 (specifically rain_1h == 9831.3)",
                "action": "convert_to_null",
                "count": rain_anomaly_count,
                "reason": "MN DOT rain gauge glitch recording impossible torrential rainfall (9,831.3 mm in 1 hour)",
            },
        ],
        "columns_preserved_unaltered": [
            "holiday",
            "snow_1h",
            "clouds_all",
            "weather_main",
            "weather_description",
            "date_time",
            "traffic_volume",
        ],
        "row_preservation_policy": "All 48,204 rows preserved. No rows dropped. Duplicate timestamps preserved as warnings.",
        "imputation_policy": "Downstream ColumnTransformer uses SimpleImputer(strategy='median') fitted exclusively on training splits.",
    }

    report_path = repo_root / "datasets" / "manifests" / "metro_traffic_cleaning_provenance_v1.0.1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)
    print(f"Cleaning provenance report written: {report_path}")

    # 6. Register version v1.0.1 in ModelGuard database
    db = SessionLocal()
    try:
        service = DatasetService(db)
        dataset = service.get_dataset("5fe9641a-0283-45b5-bfbf-f8449fdfcf73")
        print(f"Found logical dataset: {dataset.name} (UUID: {dataset.id})")

        # Check if v1.0.1 is already registered
        from app.db import models
        existing_v101 = (
            db.query(models.DatasetVersion)
            .filter(
                models.DatasetVersion.dataset_id == dataset.id,
                models.DatasetVersion.version_tag == "v1.0.1",
            )
            .first()
        )

        if existing_v101:
            print(f"Version v1.0.1 already registered: UUID={existing_v101.id}")
            v101_record = existing_v101
        else:
            v101_record = service.create_dataset_version(
                dataset_id=dataset.id,
                version_tag="v1.0.1",
                source_file_path=cleaned_staging_path,
            )
            print("Successfully created DatasetVersion v1.0.1:")
            print(f"  UUID       : {v101_record.id}")
            print(f"  Storage URI: {v101_record.storage_uri}")
            print(f"  SHA-256    : {v101_record.sha256_hash}")
            print(f"  Rows       : {v101_record.row_count}")

        # 7. Validate v1.0.1 with MetroTrafficContractV1
        print("\n=== Validating DatasetVersion v1.0.1 against MetroTrafficContractV1 ===")
        val_run = service.validate_version(v101_record.id, contract_version="v1")
        print(f"ValidationRun ID: {val_run.id}")
        print(f"Status          : {val_run.status}")
        print(f"Blocking Errors : {val_run.error_count}")
        print(f"Warnings        : {val_run.warning_count}")
        for finding in val_run.details_json.get("findings", []):
            print(f"  - [{finding.get('severity')}] {finding.get('column')}: {finding.get('check')} -> {finding.get('message')}")

        if val_run.error_count > 0 or val_run.status == "FAILED":
            print("\n[VALIDATION FAILED] Blocking errors remain!")
        else:
            print("\n[VALIDATION SUCCESSFUL] Zero blocking errors! Status: WARNING (duplicate timestamps only)")

        # Output summary payload for report
        summary_payload = {
            "dataset_id": dataset.id,
            "version_v101_id": v101_record.id,
            "version_v101_tag": v101_record.version_tag,
            "version_v101_sha256": v101_record.sha256_hash,
            "version_v101_storage_uri": v101_record.storage_uri,
            "val_run_id": val_run.id,
            "val_run_status": val_run.status,
            "val_run_errors": val_run.error_count,
            "val_run_warnings": val_run.warning_count,
        }
        print("\nRESULT_JSON:")
        print(json.dumps(summary_payload, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
