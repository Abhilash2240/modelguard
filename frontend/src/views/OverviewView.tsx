import React, { useEffect, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle, Database, GitBranch, Layers, ShieldCheck } from 'lucide-react'
import { api } from '../api'
import type { Alert, Dataset, Deployment, HealthResponse, ModelCandidate, QualityReport } from '../types'

export const OverviewView: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<ModelCandidate[]>([])
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)
        const [h, d, m, dep, q, a] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getDatasets().catch(() => []),
          api.getModels().catch(() => []),
          api.getDeployments().catch(() => []),
          api.getQuality().catch(() => null),
          api.getAlerts().catch(() => []),
        ])
        setHealth(h)
        setDatasets(d)
        setModels(m)
        setDeployments(dep)
        setQuality(q)
        setAlerts(a)
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  if (loading) {
    return <div className="card">Loading control plane metrics...</div>
  }

  if (error) {
    return <div className="card badge-error">Failed to load overview: {error}</div>
  }

  const prodDeployment = deployments.find((d) => d.status === 'PROMOTED')
  const stagingDeployment = deployments.find((d) => d.status === 'STAGED' || d.status === 'AWAITING_APPROVAL')
  const activeAlerts = alerts.filter((a) => a.status === 'OPEN')

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Operational Overview</h1>
          <p className="page-description">Real-time status across data foundation, registry, serving, and safeguards.</p>
        </div>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-header">
            <span>System Health</span>
            <Activity size={18} className="text-accent" />
          </div>
          <div className="stat-value" style={{ color: health?.status === 'healthy' ? '#34d399' : '#f87171' }}>
            {health?.status ? health.status.toUpperCase() : 'UNKNOWN'}
          </div>
          <div className="stat-footer">ModelGuard Control Plane v{health?.version || '0.1.0'}</div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>Production Champion</span>
            <ShieldCheck size={18} color="#818cf8" />
          </div>
          <div className="stat-value">
            {prodDeployment ? `v${prodDeployment.candidate_version || prodDeployment.candidate_id.slice(0, 8)}` : 'None'}
          </div>
          <div className="stat-footer">
            {prodDeployment ? `Promoted at ${new Date(prodDeployment.created_at).toLocaleTimeString()}` : 'No model in production'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>Staging Deployment</span>
            <Layers size={18} color="#fbbf24" />
          </div>
          <div className="stat-value">{stagingDeployment ? `v${stagingDeployment.candidate_id.slice(0, 8)}` : 'Idle'}</div>
          <div className="stat-footer">
            {stagingDeployment ? `Status: ${stagingDeployment.status}` : 'No candidate in staging'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>Active Alerts</span>
            <AlertTriangle size={18} color={activeAlerts.length > 0 ? '#f87171' : '#34d399'} />
          </div>
          <div className="stat-value" style={{ color: activeAlerts.length > 0 ? '#f87171' : '#34d399' }}>
            {activeAlerts.length}
          </div>
          <div className="stat-footer">{activeAlerts.length > 0 ? 'Requires operator review' : 'All systems nominal'}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        <div className="card">
          <div className="card-title">
            <span>Data Foundation & Verification</span>
            <Database size={16} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem' }}>
            <div>
              <strong>Registered Datasets:</strong> {datasets.length}
            </div>
            {datasets.map((d) => (
              <div key={d.id} style={{ padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
                <div style={{ fontWeight: 600 }}>{d.name}</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                  Task: {d.task_type} • Licence: {d.licence}
                </div>
              </div>
            ))}
            {datasets.length === 0 && <div style={{ color: 'var(--text-muted)' }}>No datasets registered yet.</div>}
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <span>Model Quality vs Delayed Feedback</span>
            <CheckCircle size={16} />
          </div>
          {quality ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem' }}>
              <div>
                <strong>Evaluation Status:</strong>{' '}
                <span className={quality.status === 'SUFFICIENT_LABELS' ? 'badge badge-success' : 'badge badge-warning'}>
                  {quality.status}
                </span>
              </div>
              <div>
                <strong>Sample Count:</strong> {quality.sample_count} (Labeled: {quality.labeled_sample_count})
              </div>
              <div>
                <strong>Coverage:</strong> {quality.label_coverage.toFixed(1)}%
              </div>
              {quality.mae !== null && quality.mae !== undefined ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>MAE</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{quality.mae.toFixed(2)}</div>
                  </div>
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>RMSE</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{quality.rmse?.toFixed(2)}</div>
                  </div>
                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '4px' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>R² SCORE</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{quality.r2?.toFixed(3)}</div>
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  Insufficient labeled feedback samples to report rolling quality metrics without statistical distortion.
                </div>
              )}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)' }}>No feedback quality reports generated yet.</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Recent Candidate Models</span>
          <GitBranch size={16} />
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Version</th>
                <th>Status</th>
                <th>Gate Verdict</th>
                <th>Val MAE</th>
                <th>Registered At</th>
              </tr>
            </thead>
            <tbody>
              {models.slice(0, 5).map((m) => (
                <tr key={m.id}>
                  <td><strong>{m.mlflow_model_name}</strong></td>
                  <td>v{m.mlflow_model_version}</td>
                  <td>
                    <span className={`badge ${
                      m.status === 'PROMOTED' ? 'badge-success' :
                      m.status === 'GATE_PASSED' ? 'badge-info' :
                      m.status === 'FAILED' || m.status === 'REJECTED' ? 'badge-error' : 'badge-neutral'
                    }`}>
                      {m.status}
                    </span>
                  </td>
                  <td>
                    {m.gate_evaluation_json?.passed ? (
                      <span className="badge badge-success">PASSED</span>
                    ) : m.gate_evaluation_json ? (
                      <span className="badge badge-error">FAILED</span>
                    ) : (
                      <span className="badge badge-neutral">UNTESTED</span>
                    )}
                  </td>
                  <td>{m.metrics_json?.candidate_val_mae?.toFixed(1) || '—'}</td>
                  <td>{new Date(m.created_at).toLocaleString()}</td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '1.5rem' }}>
                    No model candidates registered yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
