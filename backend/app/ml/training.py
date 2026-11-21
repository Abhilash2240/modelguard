from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.evaluation import compare_models, evaluate_pipeline
from app.ml.features import prepare_features
from app.ml.models import build_baseline_pipeline, build_candidate_pipeline
from app.ml.splitting import ChronologicalSplitter
from app.schemas.evaluation import FeatureMetadata, ModelComparison, PartitionEvaluation
from app.schemas.training import SplitConfig, SplitManifest


@dataclass
class TrainingExperimentResult:
    """Complete result bundle of a model training experiment."""

    split_manifest: SplitManifest
    baseline_pipeline: Pipeline
    candidate_pipeline: Pipeline
    baseline_val_eval: PartitionEvaluation
    candidate_val_eval: PartitionEvaluation
    comparison: ModelComparison
    val_predictions: pd.DataFrame
    feature_metadata: FeatureMetadata


class TrainingService:
    """Orchestrates reproducible model training and evaluation.

    Constraints:
    - Never fits preprocessing or estimator on validation or test data.
    - Preprocessing is fitted exclusively on training rows within scikit-learn Pipeline.
    - Test evaluation is decoupled from model selection.
    """

    def __init__(self, split_config: SplitConfig | None = None):
        self.split_config = split_config or SplitConfig()
        self.splitter = ChronologicalSplitter(self.split_config)

    def train_and_evaluate_validation(
        self,
        df: pd.DataFrame,
        dataset_version_id: str,
        dataset_sha256: str,
        candidate_hyperparameters: dict[str, Any] | None = None,
    ) -> tuple[TrainingExperimentResult, pd.DataFrame]:
        """Perform chronological splitting, feature preparation, model fitting, and validation evaluation.

        Returns TrainingExperimentResult and the untouched test_df for downstream final evaluation.
        """
        # 1. Deterministic chronological split
        train_df, val_df, test_df, split_manifest = self.splitter.split(
            df=df,
            dataset_version_id=dataset_version_id,
            dataset_sha256=dataset_sha256,
        )

        # 2. Prepare features (target isolated, timestamp converted to causal features)
        X_train, y_train = prepare_features(train_df, target_col=self.split_config.target_column)
        X_val, y_val = prepare_features(val_df, target_col=self.split_config.target_column)

        if y_train is None or y_val is None:
            raise ValueError(f"Target column '{self.split_config.target_column}' is missing from partitions.")

        # 3. Fit Baseline Pipeline exclusively on training partition
        baseline_pipe = build_baseline_pipeline()
        baseline_pipe.fit(X_train, y_train)

        # 4. Fit Candidate Pipeline exclusively on training partition
        hp = candidate_hyperparameters or {}
        candidate_pipe = build_candidate_pipeline(random_state=self.split_config.seed, **hp)
        candidate_pipe.fit(X_train, y_train)

        # 5. Evaluate Baseline and Candidate on untouched Validation partition
        b_val_eval, b_preds = evaluate_pipeline(
            baseline_pipe, X_val, y_val, model_name="baseline_dummy", partition_name="validation"
        )
        c_val_eval, c_preds = evaluate_pipeline(
            candidate_pipe, X_val, y_val, model_name="candidate_random_forest", partition_name="validation"
        )

        # 6. Compare candidate against baseline
        comparison = compare_models(b_val_eval, c_val_eval, partition_name="validation")

        # 7. Package validation predictions
        val_predictions_df = pd.DataFrame(
            {
                "actual": y_val.values,
                "baseline_predicted": b_preds,
                "candidate_predicted": c_preds,
                "candidate_residual": y_val.values - c_preds,
            }
        )

        from app.ml.features import get_feature_metadata

        result = TrainingExperimentResult(
            split_manifest=split_manifest,
            baseline_pipeline=baseline_pipe,
            candidate_pipeline=candidate_pipe,
            baseline_val_eval=b_val_eval,
            candidate_val_eval=c_val_eval,
            comparison=comparison,
            val_predictions=val_predictions_df,
            feature_metadata=get_feature_metadata(),
        )

        return result, test_df

    @staticmethod
    def evaluate_test_holdout(
        pipeline: Pipeline,
        test_df: pd.DataFrame,
        target_col: str = "traffic_volume",
        model_name: str = "candidate_random_forest",
    ) -> tuple[PartitionEvaluation, pd.DataFrame]:
        """Evaluate a selected model on the final temporal test holdout.

        This method is strictly decoupled to ensure the test set is evaluated
        only after model selection and hyperparameter tuning are finalized.
        """
        X_test, y_test = prepare_features(test_df, target_col=target_col)
        if y_test is None:
            raise ValueError(f"Target column '{target_col}' not found in test partition.")

        test_eval, test_preds = evaluate_pipeline(
            pipeline=pipeline,
            X=X_test,
            y=y_test,
            model_name=model_name,
            partition_name="test",
        )

        test_predictions_df = pd.DataFrame(
            {
                "actual": y_test.values,
                "predicted": test_preds,
                "residual": y_test.values - test_preds,
            }
        )

        return test_eval, test_predictions_df
