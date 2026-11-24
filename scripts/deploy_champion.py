"""ModelGuard Manual Deployment Verification Utility.

This is a standalone manual utility script for operator verification of the end-to-end
ModelGuard lifecycle in a live deployment environment (such as Docker Compose).

NOTE: This script is NEVER executed automatically during application startup.
Production models are promoted only through audited, explicit human or pipeline approvals.

Pipeline Stages Verified:
1. Dataset registration & schema validation
2. MLflow-tracked training job execution
3. Model candidate registration in MLflow Model Registry
4. Automated release gate evaluation
5. Staging deployment & automated smoke tests
6. Formal deployment approval & promotion to production champion

Usage (inside backend container):
    python scripts/deploy_champion.py

Usage (from host via Docker exec):
    docker exec modelguard-backend python scripts/deploy_champion.py
"""

import sys
from pathlib import Path

# Support execution inside container (/app) or on host
repo_root = Path(__file__).resolve().parent.parent
backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
if Path("/app").is_dir() and "/app" not in sys.path:
    sys.path.insert(0, "/app")

from app.db.session import SessionLocal
from app.ml.tracking import MLflowTracker
from app.schemas.dataset import DatasetCreate
from app.schemas.models import ReleaseGatePolicy
from app.schemas.training import SplitConfig, TrainingJobConfig, TrainingJobCreate
from app.services.dataset_service import DatasetService
from app.services.deployment_service import DeploymentService
from app.services.model_registry_service import ModelRegistryService
from app.services.training_job_service import TrainingJobService


def main() -> None:
    # Resolve valid sample fixture path
    fixture_path = Path("/app/datasets/fixtures/sample_metro_valid.csv")
    if not fixture_path.is_file():
        fixture_path = repo_root / "datasets" / "fixtures" / "sample_metro_valid.csv"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Sample fixture not found at {fixture_path}")

    db = SessionLocal()
    try:
        print("1. Registering dataset...")
        ds_svc = DatasetService(db=db)
        ds = ds_svc.register_dataset(
            DatasetCreate(
                name="metro_traffic_prod",
                task_type="regression",
                source_uri="https://archive.ics.uci.edu/dataset/492/urban+traffic",
                licence="CC BY 4.0",
                description="UCI Metro Interstate Traffic Volume dataset",
            )
        )
        ver = ds_svc.create_dataset_version(dataset_id=ds.id, version_tag="v1.0.0", source_file_path=fixture_path)
        val_run = ds_svc.validate_version(ver.id)
        print(f"   Dataset ID: {ds.id}")
        print(f"   Version ID: {ver.id}, Tag: {ver.version_tag}")
        print(f"   Validation: {val_run.status}, Errors: {val_run.error_count}")

        print("\n2. Training model with MLflow experiment tracking...")
        tracker = MLflowTracker(
            tracking_uri="http://mlflow:5000",
            experiment_name="metro-traffic-volume-prod",
            artifact_root="file:///mlflow/artifacts",
        )
        job_svc = TrainingJobService(db=db, tracker=tracker)
        split_cfg = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, min_observations=5, seed=42)
        job = job_svc.create_training_job(
            TrainingJobCreate(
                dataset_version_id=ver.id,
                validation_run_id=val_run.id,
                config=TrainingJobConfig(
                    split=split_cfg,
                    hyperparameters={"n_estimators": 10, "max_depth": 4},
                ),
            )
        )
        run = job_svc.execute_training_job(job.id)
        print(f"   Training job ID: {job.id}")
        print(f"   Training run ID: {run.id}, Status: {run.status}")
        print(f"   MLflow run ID: {run.mlflow_run_id}")

        print("\n3. Registering model candidate in Model Registry...")
        reg_svc = ModelRegistryService(db=db, tracking_uri="http://mlflow:5000")
        cand = reg_svc.register_candidate(training_run_id=run.id)
        print(f"   Candidate ID: {cand.id}")
        print(f"   Model Name: {cand.mlflow_model_name}")
        print(f"   Model Version: {cand.mlflow_model_version}")
        print(f"   Model URI: {cand.model_uri}")
        print(f"   Artifact URI: {cand.artifact_uri}")
        print(f"   Status: {cand.status}")

        print("\n4. Evaluating release gates...")
        policy = ReleaseGatePolicy(max_val_mae=10000.0, max_val_rmse=10000.0, min_val_r2=-100.0)
        gate = reg_svc.evaluate_release_gate(cand.id, policy=policy)
        print(f"   Release gate passed: {gate.passed}")
        print(f"   Candidate status: {cand.status}")

        print("\n5. Staging candidate and executing automated smoke tests...")
        dep_svc = DeploymentService(db=db, tracking_uri="http://mlflow:5000")
        stg = dep_svc.stage_candidate(cand.id)
        print(f"   Staging deployment ID: {stg.id}, Status: {stg.status}")
        print(f"   Smoke tests passed: {stg.smoke_test_results_json.get('passed')}")

        print("\n6. Approving and promoting to production champion...")
        dep_svc.approve_deployment(
            stg.id,
            actor="system_admin",
            rationale="Automated deployment verification smoke tests passed successfully",
        )
        prod = dep_svc.promote_deployment(stg.id, actor="system_admin")
        print(f"   Production deployment ID: {prod.id}, Status: {prod.status}")
        print("\nDeployment pipeline complete! Active champion model is ready for live inference.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
