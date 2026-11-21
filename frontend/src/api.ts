import type {
  Alert,
  Dataset,
  Deployment,
  DriftReport,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  ModelCandidate,
  PredictionRequest,
  PredictionResponse,
  QualityReport,
  TrainingJob,
} from './types'

const BASE_URL = '/api/v1'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`
    try {
      const data = await res.json()
      if (data.detail) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(errorDetail)
  }
  return res.json()
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    return handleResponse<HealthResponse>(await fetch(`${BASE_URL}/health`))
  },

  async getMetricsRaw(): Promise<string> {
    const res = await fetch('/metrics')
    if (!res.ok) throw new Error(`Failed to load metrics: ${res.statusText}`)
    return res.text()
  },

  async getDatasets(): Promise<Dataset[]> {
    return handleResponse<Dataset[]>(await fetch(`${BASE_URL}/datasets`))
  },

  async getTrainingJobs(): Promise<TrainingJob[]> {
    try {
      return handleResponse<TrainingJob[]>(await fetch(`${BASE_URL}/training/jobs`))
    } catch {
      return []
    }
  },

  async getModels(): Promise<ModelCandidate[]> {
    return handleResponse<ModelCandidate[]>(await fetch(`${BASE_URL}/models`))
  },

  async stageModel(modelId: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await fetch(`${BASE_URL}/models/${modelId}/stage`, {
        method: 'POST',
      })
    )
  },

  async getDeployments(environment?: string): Promise<Deployment[]> {
    const url = environment ? `${BASE_URL}/deployments?environment=${encodeURIComponent(environment)}` : `${BASE_URL}/deployments`
    return handleResponse<Deployment[]>(await fetch(url))
  },

  async approveDeployment(deploymentId: string, actor: string, rationale: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await fetch(`${BASE_URL}/deployments/${deploymentId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, rationale }),
      })
    )
  },

  async rejectDeployment(deploymentId: string, actor: string, reason: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await fetch(`${BASE_URL}/deployments/${deploymentId}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, reason }),
      })
    )
  },

  async promoteDeployment(deploymentId: string, actor: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await fetch(`${BASE_URL}/deployments/${deploymentId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, rationale: `Promoted by ${actor}` }),
      })
    )
  },

  async rollbackDeployment(deploymentId: string, actor: string, reason: string, targetCandidateId?: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await fetch(`${BASE_URL}/deployments/${deploymentId}/rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, reason, target_candidate_id: targetCandidateId || null }),
      })
    )
  },

  async predict(payload: PredictionRequest): Promise<PredictionResponse> {
    return handleResponse<PredictionResponse>(
      await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    )
  },

  async submitFeedback(payload: FeedbackRequest): Promise<FeedbackResponse> {
    return handleResponse<FeedbackResponse>(
      await fetch(`${BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    )
  },

  async getQuality(modelName = 'metro-traffic-volume'): Promise<QualityReport> {
    return handleResponse<QualityReport>(await fetch(`${BASE_URL}/quality?model_name=${encodeURIComponent(modelName)}&min_samples=10`))
  },

  async getReports(): Promise<DriftReport[]> {
    return handleResponse<DriftReport[]>(await fetch(`${BASE_URL}/reports?limit=25`))
  },

  async triggerInputDrift(): Promise<DriftReport> {
    return handleResponse<DriftReport>(
      await fetch(`${BASE_URL}/drift/input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    )
  },

  async triggerPredictionDrift(): Promise<DriftReport> {
    return handleResponse<DriftReport>(
      await fetch(`${BASE_URL}/drift/prediction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
    )
  },

  async getAlerts(status?: string): Promise<Alert[]> {
    const url = status ? `${BASE_URL}/alerts?status=${encodeURIComponent(status)}` : `${BASE_URL}/alerts`
    return handleResponse<Alert[]>(await fetch(url))
  },

  async acknowledgeAlert(alertId: string, actor: string): Promise<Alert> {
    return handleResponse<Alert>(
      await fetch(`${BASE_URL}/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor }),
      })
    )
  },

  async resolveAlert(alertId: string, actor: string, notes: string): Promise<Alert> {
    return handleResponse<Alert>(
      await fetch(`${BASE_URL}/alerts/${alertId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, notes }),
      })
    )
  },
}
