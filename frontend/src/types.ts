export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

export interface DatasetVersion {
  id: string
  version_tag: string
  sha256_hash: string
  byte_size: number
  row_count: number
  created_at: string
}

export interface ValidationRun {
  id: string
  dataset_version_id: string
  status: 'PASSED' | 'FAILED' | 'WARNING'
  error_count: number
  warning_count: number
  details_json: Record<string, any>
  created_at: string
}

export interface Dataset {
  id: string
  name: string
  task_type: string
  source_uri: string
  licence: string
  created_at: string
  versions?: DatasetVersion[]
}

export interface TrainingRun {
  id: string
  training_job_id: string
  mlflow_run_id?: string
  dataset_version_id: string
  python_version: string
  random_seed: number
  metrics_json?: {
    candidate_val_mae?: number
    candidate_val_rmse?: number
    candidate_val_r2?: number
    baseline_val_mae?: number
    baseline_val_rmse?: number
    baseline_val_r2?: number
    [key: string]: any
  }
  artifact_uri?: string
  status: string
  started_at?: string
  completed_at?: string
  created_at: string
}

export interface TrainingJob {
  id: string
  dataset_version_id: string
  validation_run_id?: string
  status: string
  created_at: string
  updated_at: string
  training_run?: TrainingRun
}

export interface ModelCandidate {
  id: string
  training_run_id: string
  dataset_version_id: string
  mlflow_model_name: string
  mlflow_model_version: string
  model_uri: string
  status: 'REGISTERED' | 'GATE_PASSED' | 'STAGED' | 'APPROVED' | 'PROMOTED' | 'REJECTED' | 'ARCHIVED' | 'FAILED'
  metrics_json?: Record<string, any>
  gate_evaluation_json?: {
    passed: boolean
    rejection_reasons?: string[]
    policy_version?: string
    metrics?: Record<string, any>
  }
  created_at: string
}

export interface Deployment {
  id: string
  environment_id: string
  candidate_id: string
  previous_candidate_id?: string
  status: 'STAGED' | 'AWAITING_APPROVAL' | 'APPROVED' | 'PROMOTED' | 'REJECTED' | 'SMOKE_FAILED' | 'REPLACED'
  smoke_test_results_json?: {
    passed?: boolean
    model_loaded?: boolean
    latency_ms?: number
    sample_prediction?: number
    error?: string
    [key: string]: any
  }
  approved_by?: string
  approved_at?: string
  approval_rationale?: string
  created_at: string
  completed_at?: string
  environment_name?: string
  candidate_version?: string
}

export interface PredictionRequest {
  inputs: Array<Record<string, any>>
  model_version?: string
}

export interface PredictionResponse {
  inference_id: string
  model_name: string
  model_version: string
  predictions: number[]
  latency_ms: number
  timestamp: string
}

export interface FeedbackRequest {
  inference_id: string
  actual_value: number
  submitted_by?: string
  notes?: string
}

export interface FeedbackResponse {
  id: string
  inference_id: string
  actual_value: number
  submitted_at: string
  submitted_by: string
  notes?: string
}

export interface QualityReport {
  id: string
  model_name: string
  model_version: string
  window_start?: string
  window_end?: string
  sample_count: number
  labeled_sample_count: number
  label_coverage: number
  mae?: number
  rmse?: number
  r2?: number
  status: 'SUFFICIENT_LABELS' | 'INSUFFICIENT_LABELS'
  details_json?: Record<string, any>
  created_at: string
}

export interface DriftReport {
  id: string
  report_type: 'DATA_DRIFT' | 'PREDICTION_DRIFT'
  model_name: string
  model_version: string
  reference_sample_count: number
  current_sample_count: number
  detection_method: string
  policy_version: string
  is_drift_detected: boolean
  summary_json: {
    total_features_evaluated?: number
    drifting_feature_count?: number
    drifting_features?: string[]
    ks_statistic?: number
    p_value?: number
    psi?: number
    reference_mean?: number
    current_mean?: number
    [key: string]: any
  }
  limitations?: string
  created_at: string
}

export interface Alert {
  id: string
  report_id?: string
  model_name: string
  model_version: string
  alert_type: string
  severity: 'INFO' | 'WARNING' | 'CRITICAL'
  message: string
  triggering_evidence_json: Record<string, any>
  status: 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED'
  acknowledged_by?: string
  acknowledged_at?: string
  resolved_by?: string
  resolved_at?: string
  resolution_notes?: string
  created_at: string
}
