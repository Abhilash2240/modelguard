import React, { useEffect, useState } from 'react'
import {
  CheckCircle2,
  GitBranch,
  Layers,
  ShieldCheck,
  XCircle,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { Modal } from '../components/Modal'
import { CardSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useToast } from '../components/Toast'
import type { ModelCandidate } from '../types'

interface ModelsViewProps {
  refreshTrigger?: number
}

export const ModelsView: React.FC<ModelsViewProps> = ({ refreshTrigger }) => {
  const [models, setModels] = useState<ModelCandidate[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [stageCandidateModal, setStageCandidateModal] = useState<ModelCandidate | null>(null)
  const [isStaging, setIsStaging] = useState<boolean>(false)
  const { addToast } = useToast()

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getModels()
        if (!ignore) setModels(data)
      } catch (err: any) {
        if (!ignore) setError(err.message || 'Failed to load model registry.')
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [refreshTrigger])

  const handleStageConfirm = async () => {
    if (!stageCandidateModal) return
    setIsStaging(true)
    try {
      await api.stageModel(stageCandidateModal.id)
      addToast({
        type: 'success',
        title: 'Model Staged Successfully',
        message: `Candidate ${stageCandidateModal.id.slice(0, 8)} (v${stageCandidateModal.mlflow_model_version}) staged to Staging environment. Smoke checks passed!`,
      })
      setStageCandidateModal(null)
      const data = await api.getModels()
      setModels(data)
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Staging Failed',
        message: err.message || 'Failed to stage model candidate.',
      })
    } finally {
      setIsStaging(false)
    }
  }

  if (loading && models.length === 0) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Model Registry & Release Gates</h1>
            <p className="page-description">Registered candidates, exact MLflow versions, release-gate policy enforcement, and promotion eligibility.</p>
          </div>
        </div>
        <CardSkeleton />
        <TableSkeleton rows={5} />
      </div>
    )
  }

  return (
    <div className="view-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Registry & Release Gates</h1>
          <p className="page-description">
            Audited model candidate repository, automated release-gate policy verification, and controlled promotion to staging.
          </p>
        </div>
        <div className="header-meta-pill">
          <GitBranch size={14} />
          <span>{models.length} Registered Candidates</span>
        </div>
      </div>

      {error && (
        <div className="card badge-error" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
          <strong>Error loading model registry:</strong> {error}
        </div>
      )}

      {/* Main Candidates Card */}
      <div className="card">
        <div className="card-title">
          <div className="title-with-badge">
            <GitBranch size={16} className="text-accent" />
            <span>Registered Model Candidates</span>
            <span className="counter-pill">{models.length}</span>
          </div>
          <div className="gate-policy-indicator">
            <ShieldCheck size={14} className="text-emerald" />
            <span>Gate Policy v1: MAE &le; 1200 | RMSE &le; 1600 | R&sup2; &ge; 0.50</span>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Candidate ID</th>
                <th>Model Name</th>
                <th>Exact Version</th>
                <th>Lifecycle Status</th>
                <th>Release Gate Verification</th>
                <th>Val MAE</th>
                <th>Val RMSE</th>
                <th>Val R&sup2;</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const gate = m.gate_evaluation_json
                const gatePassed = gate?.passed ?? (m.status === 'GATE_PASSED' || m.status === 'PROMOTED' || m.status === 'STAGED')
                const canStage = m.status === 'GATE_PASSED' || m.status === 'REGISTERED'

                const mae = m.metrics_json?.val_mae || m.metrics_json?.candidate_val_mae
                const rmse = m.metrics_json?.val_rmse || m.metrics_json?.candidate_val_rmse
                const r2 = m.metrics_json?.val_r2 || m.metrics_json?.candidate_val_r2

                return (
                  <tr key={m.id}>
                    <td>
                      <div className="id-with-copy">
                        <code>{m.id.slice(0, 8)}</code>
                        <CopyButton text={m.id} tooltip="Copy Candidate UUID" />
                      </div>
                    </td>
                    <td>
                      <strong className="model-name-text">{m.mlflow_model_name}</strong>
                    </td>
                    <td>
                      <span className="version-pill">v{m.mlflow_model_version}</span>
                    </td>
                    <td>
                      <StatusBadge status={m.status} />
                    </td>
                    <td>
                      <div className="gate-checklist-cell">
                        {gatePassed ? (
                          <span className="badge badge-success">
                            <CheckCircle2 size={11} />
                            <span>Passed All Gates</span>
                          </span>
                        ) : (
                          <span className="badge badge-error">
                            <XCircle size={11} />
                            <span>Gate Failed</span>
                          </span>
                        )}
                        <span className="gate-detail-hint">
                          {gate?.rejection_reasons?.length
                            ? gate.rejection_reasons.join(', ')
                            : 'All thresholds satisfied'}
                        </span>
                      </div>
                    </td>
                    <td>
                      <strong className="metric-highlight">
                        {mae != null ? Number(mae).toFixed(1) : '—'}
                      </strong>
                    </td>
                    <td>
                      <span className="text-muted">
                        {rmse != null ? Number(rmse).toFixed(1) : '—'}
                      </span>
                    </td>
                    <td>
                      <span className="text-muted">
                        {r2 != null ? Number(r2).toFixed(3) : '—'}
                      </span>
                    </td>
                    <td>
                      {canStage ? (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => setStageCandidateModal(m)}
                        >
                          <Layers size={13} />
                          <span>Stage</span>
                        </button>
                      ) : (
                        <span className="text-muted" style={{ fontSize: '0.8rem' }}>
                          {m.status === 'STAGED' ? 'In Staging' : m.status === 'PROMOTED' ? 'Champion' : 'Locked'}
                        </span>
                      )}
                    </td>
                  </tr>
                )
              })}

              {models.length === 0 && (
                <tr>
                  <td colSpan={9} className="empty-table-cell">
                    No models registered in the control plane registry.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confirmation Modal for Staging */}
      {stageCandidateModal && (
        <Modal
          isOpen={true}
          onClose={() => !isStaging && setStageCandidateModal(null)}
          title={`Stage Candidate ${stageCandidateModal.id.slice(0, 8)}`}
          footer={
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setStageCandidateModal(null)}
                disabled={isStaging}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleStageConfirm}
                disabled={isStaging}
              >
                {isStaging ? 'Staging Candidate...' : 'Confirm & Run Smoke Tests'}
              </button>
            </>
          }
        >
          <div className="modal-body-content">
            <p>
              You are about to deploy candidate <strong>{stageCandidateModal.id}</strong> (
              {stageCandidateModal.mlflow_model_name} v{stageCandidateModal.mlflow_model_version}) to the{' '}
              <strong>Staging</strong> environment.
            </p>
            <div className="modal-info-box">
              <div className="info-row">
                <span>Validation MAE:</span>
                <strong>{stageCandidateModal.metrics_json?.candidate_val_mae?.toFixed(1) ?? '—'}</strong>
              </div>
              <div className="info-row">
                <span>Release Gate Policy:</span>
                <span className="text-emerald">PASSED (v1 Standard)</span>
              </div>
              <div className="info-row">
                <span>Automated Smoke Checks:</span>
                <span>Will execute input contract and latency verification upon staging.</span>
              </div>
            </div>
            <p className="text-muted" style={{ fontSize: '0.825rem', marginTop: '0.75rem' }}>
              Note: This action does not touch the active Production champion.
            </p>
          </div>
        </Modal>
      )}
    </div>
  )
}
