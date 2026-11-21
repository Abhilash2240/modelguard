from pathlib import Path

from fastapi.testclient import TestClient


def test_dataset_lifecycle_api(client: TestClient, valid_csv_path: Path, invalid_csv_path: Path):
    # 1. Register a new logical dataset
    reg_payload = {
        "name": "metro_interstate_traffic_volume",
        "task_type": "regression",
        "source_uri": "https://archive.ics.uci.edu/dataset/492/urban+traffic",
        "licence": "CC BY 4.0",
        "description": "Metro interstate traffic volume regression benchmark",
    }
    resp = client.post("/api/v1/datasets", json=reg_payload)
    assert resp.status_code == 201
    dataset = resp.json()
    dataset_id = dataset["id"]
    assert dataset["name"] == "metro_interstate_traffic_volume"

    # 2. Reject duplicate registration with 409 Conflict
    resp_dup = client.post("/api/v1/datasets", json=reg_payload)
    assert resp_dup.status_code == 409

    # 3. Upload an immutable dataset version snapshot (valid fixture)
    with open(valid_csv_path, "rb") as f:
        resp_upload = client.post(
            f"/api/v1/datasets/{dataset_id}/versions",
            data={"version_tag": "v1.0.0"},
            files={"file": ("sample_metro_valid.csv", f, "text/csv")},
        )
    assert resp_upload.status_code == 201
    version_data = resp_upload.json()
    version_id = version_data["id"]
    assert version_data["version_tag"] == "v1.0.0"
    assert version_data["row_count"] == 10
    assert len(version_data["sha256_hash"]) == 64

    # 4. Reject duplicate version tag for the same dataset
    with open(valid_csv_path, "rb") as f:
        resp_dup_ver = client.post(
            f"/api/v1/datasets/{dataset_id}/versions",
            data={"version_tag": "v1.0.0"},
            files={"file": ("sample_metro_valid.csv", f, "text/csv")},
        )
    assert resp_dup_ver.status_code == 409

    # 5. Validate the clean version against MetroTrafficContractV1
    resp_val = client.post(f"/api/v1/dataset-versions/{version_id}/validate?contract_version=v1")
    assert resp_val.status_code == 200
    val_data = resp_val.json()
    assert val_data["status"] == "PASSED"
    assert val_data["error_count"] == 0
    assert val_data["warning_count"] == 0

    # 6. Upload an invalid version snapshot (invalid fixture)
    with open(invalid_csv_path, "rb") as f:
        resp_upload_inv = client.post(
            f"/api/v1/datasets/{dataset_id}/versions",
            data={"version_tag": "v2.0.0"},
            files={"file": ("sample_metro_invalid.csv", f, "text/csv")},
        )
    assert resp_upload_inv.status_code == 201
    inv_version_id = resp_upload_inv.json()["id"]

    # 7. Validate the invalid version -> must fail and return structured findings
    resp_val_inv = client.post(
        f"/api/v1/dataset-versions/{inv_version_id}/validate?contract_version=v1"
    )
    assert resp_val_inv.status_code == 200
    inv_val_data = resp_val_inv.json()
    assert inv_val_data["status"] == "FAILED"
    assert inv_val_data["error_count"] > 0
    assert len(inv_val_data["details_json"]["findings"]) > 0

    # 8. Query validation run directly
    run_id = inv_val_data["id"]
    resp_run = client.get(f"/api/v1/validation-runs/{run_id}")
    assert resp_run.status_code == 200
    assert resp_run.json()["id"] == run_id
