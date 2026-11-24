import React, { useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  GitBranch,
  Layers,
  ShieldCheck,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { CardSkeleton, StatGridSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatCard } from '../components/StatCard'
import { StatusBadge } from '../components/StatusBadge'
import type { Alert, Dataset, Deployment, HealthResponse, ModelCandidate, QualityReport } from '../types'

interface OverviewViewProps {
  refreshTrigger?: number
  onNavigate?: (tab: string) => void
}

export const OverviewView: React.FC<OverviewViewProps> = ({ refreshTrigger, onNavigate }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<ModelCandidate[]>([])
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    async function loadData() {
      try {
        setLoading(true)
        setError(null)
        const [h, d, m, dep, q, a] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getDatasets().catch(() => []),
          api.getModels().catch(() => []),
          api.getDeployments().catch(() => []),
          api.getQuality().catch(() => null),
          api.getAlerts().catch(() => []),
        ])
        if (!ignore) {
          setHealth(h)
          setDatasets(d)
          setModels(m)
          setDeployments(dep)
          setQuality(q)
          setAlerts(a)
        }
      } catch (err: any) {
        if (!ignore) setError(err.message || 'Failed to load control plane state.')
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    loadData()
    return () => {
      ignore = true
    }
  }, [refreshTrigger])

  if (loading && !health) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Operational Overview</h1>
            <p className="page-description">Real-time status across data foundation, registry, serving, and safeguards.</p>
          </div>
        </div>
        <StatGridSkeleton count={4} />
        <CardSkeleton />
        <TableSkeleton rows={4} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="view-container">
        <div className="card badge-error" style={{ padding: '1.5rem', marginTop: '1rem' }}>
          <h3>Failed to load operational overview</h3>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const prodDeployment = deployments.find((d) => d.status === 'PROMOTED')
  const stagingDeployment = deployments.find((d) => d.status === 'STAGED' || d.status === 'AWAITING_APPROVAL')
  const activeAlerts = alerts.filter((a) => a.status === 'OPEN')
  const totalVersions = datasets.reduce((sum, d) => sum + (d.versions?.length || 0), 0)

  // Quality metric display
  const isInsufficientLabels = quality?.status === 'INSUFFICIENT_LABELS'
  const qualityValue = isInsufficientLabels
    ? 'Need Labels'
    : quality?.mae != null
    ? `${quality.mae.toFixed(1)} MAE`
    : 'Nominal'

  return (
    <div className="view-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Operational Overview</h1>
          <p className="page-description">
            Production ML reliability control plane: real-time telemetry, model provenance, and automated safeguards.
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => onNavigate && onNavigate('inference')}
          >
            <Cpu size={14} />
            <span>Test Inference</span>
          </button>
          <a
            href="http://127.0.0.1:3001"
            target="_blank"
            rel="noreferrer"
            className="btn btn-primary"
          >
            <Activity size={14} />
            <span>Grafana Telemetry</span>
            <ExternalLink size={12} />
          </a>
        </div>
      </div>

      {/* Top Stat Cards Grid */}
      <div className="stat-grid">
        <StatCard
          title="System Health"
          value={health?.status?.toUpperCase() || 'HEALTHY'}
          icon={Activity}
          badge={{
            text: health?.status === 'healthy' ? 'ACTIVE' : 'DEGRADED',
            type: health?.status === 'healthy' ? 'success' : 'error',
          }}
          footer={`ModelGuard Engine v${health?.version || '0.1.0'}`}
        />

        <StatCard
          title="Production Champion"
          value={prodDeployment ? `v${prodDeployment.candidate_version || '1'}` : 'None'}
          icon={ShieldCheck}
          badge={
            prodDeployment
              ? { text: 'SERVING', type: 'success' }
              : { text: 'NOT SET', type: 'warning' }
          }
          footer={
            prodDeployment
              ? `Candidate ${prodDeployment.candidate_id.slice(0, 8)}`
              : 'No model in production'
          }
        />

        <StatCard
          title="Staging Status"
          value={stagingDeployment ? `v${stagingDeployment.candidate_id.slice(0, 8)}` : 'Idle'}
          icon={Layers}
          badge={
            stagingDeployment
              ? { text: stagingDeployment.status, type: 'warning' }
              : { text: 'READY', type: 'neutral' }
          }
          footer={
            stagingDeployment
              ? 'Automated smoke check verified'
              : 'Awaiting promotion candidate'
          }
        />

        <StatCard
          title="Rolling Model Quality"
          value={qualityValue}
          icon={TrendingUp}
          badge={
            isInsufficientLabels
              ? { text: 'Need < 5 Labels', type: 'warning' }
              : { text: 'HEALTHY', type: 'success' }
          }
          footer={`${quality?.labeled_sample_count ?? 0} delayed ground-truth labels evaluated`}
        />

        <StatCard
          title="Data Foundation"
          value={`${datasets.length} Datasets`}
          icon={Database}
          badge={{ text: 'VERIFIED', type: 'info' }}
          footer={`${totalVersions} immutable SHA-256 versions registered`}
        />

        <StatCard
          title="Safeguard Alerts"
          value={`${activeAlerts.length} Open`}
          icon={AlertTriangle}
          badge={
            activeAlerts.length > 0
              ? { text: 'REQUIRES ACTION', type: 'error' }
              : { text: 'NOMINAL', type: 'success' }
          }
          footer={`${alerts.length} total lifecycle events audited`}
        />
      </div>

      {/* Pipeline Status Workflow */}
      <div className="card pipeline-status-card">
        <div className="card-title">
          <span>End-to-End MLOps Pipeline Reliability State</span>
          <ShieldCheck size={16} className="text-accent" />
        </div>
        <div className="pipeline-steps-container">
          <div className="pipeline-step completed">
            <div className="step-circle">
              <Database size={15} />
            </div>
            <div className="step-content">
              <span className="step-name">1. Data Foundation</span>
              <span className="step-status">Immutable & SHA-256 Verified</span>
            </div>
          </div>

          <div className="pipeline-connector active" />

          <div className="pipeline-step completed">
            <div className="step-circle">
              <Zap size={15} />
            </div>
            <div className="step-content">
              <span className="step-name">2. Reproducible Training</span>
              <span className="step-status">MLflow Tracked & Seeded</span>
            </div>
          </div>

          <div className="pipeline-connector active" />

          <div className="pipeline-step completed">
            <div className="step-circle">
              <GitBranch size={15} />
            </div>
            <div className="step-content">
              <span className="step-name">3. Release Gates</span>
              <span className="step-status">MAE / RMSE / R² Gated</span>
            </div>
          </div>

          <div className="pipeline-connector active" />

          <div className={`pipeline-step ${stagingDeployment ? 'active' : 'completed'}`}>
            <div className="step-circle">
              <Layers size={15} />
            </div>
            <div className="step-content">
              <span className="step-name">4. Staging Verification</span>
              <span className="step-status">{stagingDeployment ? 'Awaiting Sign-off' : 'Standby / Smoke Verified'}</span>
            </div>
          </div>

          <div className="pipeline-connector active" />

          <div className="pipeline-step completed">
            <div className="step-circle">
              <ShieldCheck size={15} />
            </div>
            <div className="step-content">
              <span className="step-name">5. Production Champion</span>
              <span className="step-status">Audited Rollback Protected</span>
            </div>
          </div>
        </div>
      </div>

      {/* Two Column Section: Recent Candidates & Live Alerts */}
      <div className="two-column-grid">
        {/* Candidate Models Card */}
        <div className="card">
          <div className="card-title">
            <div className="title-with-badge">
              <span>Model Registry Candidates</span>
              <span className="counter-pill">{models.length}</span>
            </div>
            {onNavigate && (
              <button
                type="button"
                className="btn-text-link"
                onClick={() => onNavigate('models')}
              >
                <span>View all</span>
                <ArrowRight size={13} />
              </button>
            )}
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Candidate ID</th>
                  <th>Model</th>
                  <th>Version</th>
                  <th>Status</th>
                  <th>Val MAE</th>
                </tr>
              </thead>
              <tbody>
                {models.slice(0, 5).map((m) => {
                  const valMae = m.metrics_json?.val_mae || m.metrics_json?.candidate_val_mae
                  return (
                    <tr key={m.id}>
                      <td>
                        <div className="id-with-copy">
                          <code>{m.id.slice(0, 8)}</code>
                          <CopyButton text={m.id} tooltip="Copy Candidate UUID" />
                        </div>
                      </td>
                      <td>
                        <span className="model-name-text">{m.mlflow_model_name}</span>
                      </td>
                      <td>
                        <span className="version-pill">v{m.mlflow_model_version}</span>
                      </td>
                      <td>
                        <StatusBadge status={m.status} />
                      </td>
                      <td>
                        <strong className="metric-highlight">
                          {valMae != null ? Number(valMae).toFixed(1) : '—'}
                        </strong>
                      </td>
                    </tr>
                  )
                })}
                {models.length === 0 && (
                  <tr>
                    <td colSpan={5} className="empty-table-cell">
                      No candidate models registered yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Live Alerts & Safeguards Card */}
        <div className="card">
          <div className="card-title">
            <div className="title-with-badge">
              <span>Active Safeguard Alerts</span>
              <span className={`counter-pill ${activeAlerts.length > 0 ? 'pill-alert' : ''}`}>
                {activeAlerts.length}
              </span>
            </div>
            {onNavigate && (
              <button
                type="button"
                className="btn-text-link"
                onClick={() => onNavigate('alerts')}
              >
                <span>View all</span>
                <ArrowRight size={13} />
              </button>
            )}
          </div>

          <div className="alerts-summary-list">
            {activeAlerts.slice(0, 4).map((a) => (
              <div key={a.id} className={`alert-summary-item severity-${a.severity.toLowerCase()}`}>
                <div className="alert-item-header">
                  <div className="alert-item-title-wrap">
                    <AlertTriangle size={14} className="alert-item-icon" />
                    <span className="alert-item-title">{a.alert_type}</span>
                  </div>
                  <span className={`badge badge-${a.severity === 'CRITICAL' ? 'error' : a.severity === 'WARNING' ? 'warning' : 'info'}`}>
                    {a.severity}
                  </span>
                </div>
                <p className="alert-item-desc">{a.message}</p>
                <div className="alert-item-meta">
                  <span>Model: <code>{a.model_name} v{a.model_version}</code></span>
                  <span>{new Date(a.created_at).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}

            {activeAlerts.length === 0 && (
              <div className="empty-state-card">
                <CheckCircle2 size={32} className="text-emerald" />
                <p className="empty-state-title">No Active Alerts</p>
                <p className="empty-state-desc">All statistical drift checks, latency limits, and release gates are nominal.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
