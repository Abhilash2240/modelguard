import type {
  Alert,
  Dataset,
  Deployment,
  DriftReport,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  LoginRequest,
  ModelCandidate,
  PredictionRequest,
  PredictionResponse,
  QualityReport,
  TokenResponse,
  TrainingJob,
  UserProfile,
} from './types'

const BASE_URL = '/api/v1'

let authToken: string | null = typeof window !== 'undefined' ? localStorage.getItem('mg_auth_token') : null

export function setAuthToken(token: string | null): void {
  authToken = token
  if (typeof window !== 'undefined') {
    if (token) {
      localStorage.setItem('mg_auth_token', token)
    } else {
      localStorage.removeItem('mg_auth_token')
    }
  }
}

export function getAuthToken(): string | null {
  return authToken
}

async function request(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers || {})
  if (authToken && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${authToken}`)
  }
  return fetch(path, {
    ...options,
    headers,
    credentials: 'same-origin',
  })
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    if (res.status === 401 && typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('mg:unauthorized'))
    }
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
  // Authentication & Session
  async login(credentials: LoginRequest): Promise<TokenResponse> {
    const res = await request(`${BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    })
    const data = await handleResponse<TokenResponse>(res)
    setAuthToken(data.access_token)
    return data
  },

  async logout(): Promise<void> {
    try {
      await request(`${BASE_URL}/auth/logout`, { method: 'POST' })
    } catch {
      // ignore logout failure on network error
    } finally {
      setAuthToken(null)
    }
  },

  async getMe(): Promise<UserProfile> {
    return handleResponse<UserProfile>(await request(`${BASE_URL}/auth/me`))
  },

  // Health and System
  async getHealth(): Promise<HealthResponse> {
    return handleResponse<HealthResponse>(await request(`${BASE_URL}/health`))
  },

  async getMetricsRaw(): Promise<string> {
    const res = await request('/metrics')
    if (!res.ok) throw new Error(`Failed to load metrics: ${res.statusText}`)
    return res.text()
  },

  // Datasets
  async getDatasets(): Promise<Dataset[]> {
    return handleResponse<Dataset[]>(await request(`${BASE_URL}/datasets`))
  },

  // Training
  async getTrainingJobs(): Promise<TrainingJob[]> {
    return handleResponse<TrainingJob[]>(await request(`${BASE_URL}/training/jobs`))
  },

  // Models
  async getModels(): Promise<ModelCandidate[]> {
    return handleResponse<ModelCandidate[]>(await request(`${BASE_URL}/models`))
  },

  async stageModel(modelId: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await request(`${BASE_URL}/models/${modelId}/stage`, {
        method: 'POST',
      })
    )
  },

  // Deployments & Lifecycle
  async getDeployments(environment?: string): Promise<Deployment[]> {
    const url = environment ? `${BASE_URL}/deployments?environment=${encodeURIComponent(environment)}` : `${BASE_URL}/deployments`
    return handleResponse<Deployment[]>(await request(url))
  },

  async approveDeployment(deploymentId: string, actor: string, rationale: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await request(`${BASE_URL}/deployments/${deploymentId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, rationale }),
      })
    )
  },

  async rejectDeployment(deploymentId: string, actor: string, reason: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await request(`${BASE_URL}/deployments/${deploymentId}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, reason }),
      })
    )
  },

  async promoteDeployment(deploymentId: string, actor: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await request(`${BASE_URL}/deployments/${deploymentId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, rationale: `Promoted by ${actor}` }),
      })
    )
  },

  async rollbackDeployment(deploymentId: string, actor: string, reason: string, targetCandidateId?: string): Promise<Deployment> {
    return handleResponse<Deployment>(
      await request(`${BASE_URL}/deployments/${deploymentId}/rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, reason, target_candidate_id: targetCandidateId || null }),
      })
    )
  },

  // Inference & Diagnostics
  async predict(payload: PredictionRequest): Promise<PredictionResponse> {
    return handleResponse<PredictionResponse>(
      await request(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    )
  },

  async submitFeedback(payload: FeedbackRequest): Promise<FeedbackResponse> {
    return handleResponse<FeedbackResponse>(
      await request(`${BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    )
  },

  // Quality & Drift
  async getQuality(modelName = 'metro-traffic-volume'): Promise<QualityReport> {
    return handleResponse<QualityReport>(await request(`${BASE_URL}/quality?model_name=${encodeURIComponent(modelName)}&min_samples=10`))
  },

  async getReports(): Promise<DriftReport[]> {
    return handleResponse<DriftReport[]>(await request(`${BASE_URL}/reports?limit=25`))
  },

  async triggerInputDrift(scenario?: string): Promise<DriftReport> {
    return handleResponse<DriftReport>(
      await request(`${BASE_URL}/drift/input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenario || 'default' }),
      })
    )
  },

  async triggerPredictionDrift(scenario?: string): Promise<DriftReport> {
    return handleResponse<DriftReport>(
      await request(`${BASE_URL}/drift/prediction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenario || 'default' }),
      })
    )
  },

  // Alerts
  async getAlerts(status?: string): Promise<Alert[]> {
    const url = status ? `${BASE_URL}/alerts?status=${encodeURIComponent(status)}` : `${BASE_URL}/alerts`
    return handleResponse<Alert[]>(await request(url))
  },

  async acknowledgeAlert(alertId: string, actor: string): Promise<Alert> {
    return handleResponse<Alert>(
      await request(`${BASE_URL}/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor }),
      })
    )
  },

  async resolveAlert(alertId: string, actor: string, notes: string): Promise<Alert> {
    return handleResponse<Alert>(
      await request(`${BASE_URL}/alerts/${alertId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor, notes }),
      })
    )
  },
}
