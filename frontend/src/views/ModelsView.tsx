import React, { useEffect, useState } from 'react'
import { CheckCircle, GitBranch, Layers, XCircle } from 'lucide-react'
import { api } from '../api'
import type { ModelCandidate } from '../types'

export const ModelsView: React.FC = () => {
  const [models, setModels] = useState<ModelCandidate[]>([])
  const [loading, setLoading] = useState(true)
  const [stagingId, setStagingId] = useState<string | null>(null)
  const [actionMsg, setActionMsg] = useState<{ text: string; error?: boolean } | null>(null)

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        const data = await api.getModels()
        if (!ignore) setModels(data)
      } catch (err: any) {
        if (!ignore) setActionMsg({ text: `Failed to load models: ${err.message}`, error: true })
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [])

  async function handleStage(modelId: string) {
    try {
      setStagingId(modelId)
      setActionMsg(null)
      await api.stageModel(modelId)
      setActionMsg({ text: `Successfully staged candidate ${modelId.slice(0, 8)} to Staging! Smoke checks passed.` })
      const data = await api.getModels()
      setModels(data)
    } catch (err: any) {
      setActionMsg({ text: `Staging failed: ${err.message}`, error: true })
    } finally {
      setStagingId(null)
    }
  }

  if (loading) return <div className="card">Loading registered models...</div>

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Registry & Release Gates</h1>
          <p className="page-description">Registered candidates, exact MLflow versions, release-gate policy enforcement, and promotion eligibility.</p>
        </div>
      </div>

      {actionMsg && (
        <div className={`card ${actionMsg.error ? 'badge-error' : 'badge-success'}`} style={{ marginBottom: '1rem', padding: '0.75rem 1rem' }}>
          {actionMsg.text}
        </div>
      )}

      <div className="card">
        <div className="card-title">
          <span>Registered Candidates</span>
          <GitBranch size={16} />
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Candidate ID</th>
                <th>Model Name</th>
                <th>Exact Version</th>
                <th>Status</th>
                <th>Release Gate</th>
                <th>Val MAE</th>
                <th>Val RMSE</th>
                <th>Val R²</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const gate = m.gate_evaluation_json
                const canStage = m.status === 'GATE_PASSED' || m.status === 'REGISTERED'

                return (
                  <tr key={m.id}>
                    <td><code>{m.id.slice(0, 8)}</code></td>
                    <td><strong>{m.mlflow_model_name}</strong></td>
                    <td><span className="badge badge-neutral">v{m.mlflow_model_version}</span></td>
                    <td>
                      <span className={`badge ${
                        m.status === 'PROMOTED' ? 'badge-success' :
                        m.status === 'STAGED' ? 'badge-warning' :
                        m.status === 'GATE_PASSED' ? 'badge-info' :
                        m.status === 'FAILED' || m.status === 'REJECTED' ? 'badge-error' : 'badge-neutral'
                      }`}>
                        {m.status}
                      </span>
                    </td>
                    <td>
                      {gate ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          {gate.passed ? (
                            <span className="badge badge-success"><CheckCircle size={12} style={{ marginRight: '4px' }} /> PASSED</span>
                          ) : (
                            <span className="badge badge-error" title={gate.rejection_reasons?.join('; ')}>
                              <XCircle size={12} style={{ marginRight: '4px' }} /> FAILED
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="badge badge-neutral">EVALUATED AT STAGE</span>
                      )}
                    </td>
                    <td>{m.metrics_json?.candidate_val_mae ? m.metrics_json.candidate_val_mae.toFixed(1) : '—'}</td>
                    <td>{m.metrics_json?.candidate_val_rmse ? m.metrics_json.candidate_val_rmse.toFixed(1) : '—'}</td>
                    <td>{m.metrics_json?.candidate_val_r2 ? m.metrics_json.candidate_val_r2.toFixed(3) : '—'}</td>
                    <td>
                      {canStage && (
                        <button
                          className="btn btn-primary"
                          style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                          onClick={() => handleStage(m.id)}
                          disabled={stagingId === m.id}
                        >
                          <Layers size={12} /> {stagingId === m.id ? 'Staging...' : 'Deploy to Staging'}
                        </button>
                      )}
                      {m.status === 'STAGED' && (
                        <span style={{ fontSize: '0.75rem', color: '#fbbf24' }}>In Staging</span>
                      )}
                      {m.status === 'PROMOTED' && (
                        <span style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 600 }}>Active Champion</span>
                      )}
                    </td>
                  </tr>
                )
              })}
              {models.length === 0 && (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No candidates registered yet.
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
