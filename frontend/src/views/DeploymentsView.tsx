import React, { useEffect, useState } from 'react'
import { CheckCircle2, RotateCcw, ShieldCheck, XCircle } from 'lucide-react'
import { api } from '../api'
import type { Deployment } from '../types'

interface ConfirmAction {
  type: 'APPROVE' | 'REJECT' | 'PROMOTE' | 'ROLLBACK'
  deploymentId: string
  title: string
  prompt: string
  inputLabel?: string
  defaultValue?: string
}

export const DeploymentsView: React.FC = () => {
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [loading, setLoading] = useState(true)
  const [actionModal, setActionModal] = useState<ConfirmAction | null>(null)
  const [modalInput, setModalInput] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [feedbackMsg, setFeedbackMsg] = useState<{ text: string; error?: boolean } | null>(null)

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        const data = await api.getDeployments()
        if (!ignore) setDeployments(data)
      } catch (err: any) {
        if (!ignore) setFeedbackMsg({ text: `Failed to load deployments: ${err.message}`, error: true })
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [])

  function openModal(action: ConfirmAction) {
    setActionModal(action)
    setModalInput(action.defaultValue || '')
  }

  async function handleConfirmAction() {
    if (!actionModal) return
    setIsSubmitting(true)
    setFeedbackMsg(null)
    try {
      if (actionModal.type === 'APPROVE') {
        await api.approveDeployment(actionModal.deploymentId, 'operator', modalInput || 'Verified in staging')
        setFeedbackMsg({ text: 'Deployment successfully APPROVED.' })
      } else if (actionModal.type === 'REJECT') {
        await api.rejectDeployment(actionModal.deploymentId, 'operator', modalInput || 'Rejected during verification')
        setFeedbackMsg({ text: 'Deployment REJECTED. Production champion remains untouched.' })
      } else if (actionModal.type === 'PROMOTE') {
        await api.promoteDeployment(actionModal.deploymentId, 'operator')
        setFeedbackMsg({ text: 'Deployment PROMOTED to active production champion!' })
      } else if (actionModal.type === 'ROLLBACK') {
        await api.rollbackDeployment(actionModal.deploymentId, 'operator', modalInput || 'Operator initiated rollback')
        setFeedbackMsg({ text: 'Production Champion successfully ROLLED BACK to previous known-good version!' })
      }
      setActionModal(null)
      const data = await api.getDeployments()
      setDeployments(data)
    } catch (err: any) {
      setFeedbackMsg({ text: `Action failed: ${err.message}`, error: true })
    } finally {
      setIsSubmitting(false)
    }
  }

  if (loading) return <div className="card">Loading deployments...</div>

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Deployment Orchestration & Rollback</h1>
          <p className="page-description">Automated staging smoke checks, explicit sign-offs, production promotions, and safe rollbacks.</p>
        </div>
      </div>

      {feedbackMsg && (
        <div className={`card ${feedbackMsg.error ? 'badge-error' : 'badge-success'}`} style={{ marginBottom: '1rem', padding: '0.75rem 1rem' }}>
          {feedbackMsg.text}
        </div>
      )}

      <div className="card">
        <div className="card-title">
          <span>Active Deployments & Lifecycle History</span>
          <ShieldCheck size={16} />
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Deployment ID</th>
                <th>Environment</th>
                <th>Candidate ID</th>
                <th>Status</th>
                <th>Smoke Check</th>
                <th>Approved By</th>
                <th>Rationale</th>
                <th>Timestamp</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {deployments.map((d) => {
                const smoke = d.smoke_test_results_json
                const isStaged = d.status === 'STAGED' || d.status === 'AWAITING_APPROVAL'
                const isApproved = d.status === 'APPROVED'
                const isPromoted = d.status === 'PROMOTED'

                return (
                  <tr key={d.id}>
                    <td><code>{d.id.slice(0, 8)}</code></td>
                    <td>
                      <span className={`badge ${d.environment_name === 'production' || d.status === 'PROMOTED' ? 'badge-info' : 'badge-neutral'}`}>
                        {d.environment_name || (d.status === 'PROMOTED' ? 'production' : 'staging')}
                      </span>
                    </td>
                    <td><code>{d.candidate_id.slice(0, 8)}</code></td>
                    <td>
                      <span className={`badge ${
                        d.status === 'PROMOTED' ? 'badge-success' :
                        d.status === 'APPROVED' ? 'badge-info' :
                        d.status === 'STAGED' ? 'badge-warning' :
                        d.status === 'REJECTED' || d.status === 'SMOKE_FAILED' ? 'badge-error' : 'badge-neutral'
                      }`}>
                        {d.status}
                      </span>
                    </td>
                    <td>
                      {smoke ? (
                        smoke.passed ? (
                          <span className="badge badge-success" title={`Latency: ${smoke.latency_ms?.toFixed(1)}ms`}>
                            <CheckCircle2 size={12} style={{ marginRight: '4px' }} /> PASSED
                          </span>
                        ) : (
                          <span className="badge badge-error" title={smoke.error || 'Smoke failed'}>
                            <XCircle size={12} style={{ marginRight: '4px' }} /> FAILED
                          </span>
                        )
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>{d.approved_by || '—'}</td>
                    <td style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {d.approval_rationale || '—'}
                    </td>
                    <td>{new Date(d.created_at).toLocaleString()}</td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.35rem' }}>
                        {isStaged && (
                          <>
                            <button
                              className="btn btn-success"
                              style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                              onClick={() => openModal({
                                type: 'APPROVE',
                                deploymentId: d.id,
                                title: 'Approve Staged Candidate',
                                prompt: 'Confirm explicit sign-off to approve this candidate for production promotion.',
                                inputLabel: 'Approval Rationale',
                                defaultValue: 'Verified in staging smoke tests',
                              })}
                            >
                              Approve
                            </button>
                            <button
                              className="btn btn-danger"
                              style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                              onClick={() => openModal({
                                type: 'REJECT',
                                deploymentId: d.id,
                                title: 'Reject Staged Candidate',
                                prompt: 'Reject this model candidate. Production will remain untouched.',
                                inputLabel: 'Rejection Reason',
                                defaultValue: 'Failed manual verification',
                              })}
                            >
                              Reject
                            </button>
                          </>
                        )}
                        {isApproved && (
                          <button
                            className="btn btn-primary"
                            style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                            onClick={() => openModal({
                              type: 'PROMOTE',
                              deploymentId: d.id,
                              title: 'Promote to Production Champion',
                              prompt: 'Promote this approved candidate to production champion and atomically update MLflow champion alias.',
                            })}
                          >
                            Promote
                          </button>
                        )}
                        {isPromoted && (
                          <button
                            className="btn btn-danger"
                            style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                            onClick={() => openModal({
                              type: 'ROLLBACK',
                              deploymentId: d.id,
                              title: 'Execute Audited Rollback',
                              prompt: 'Restore previously known-good production model version with MLflow compensation.',
                              inputLabel: 'Rollback Justification',
                              defaultValue: 'Performance degradation or input anomaly detected',
                            })}
                          >
                            <RotateCcw size={12} /> Rollback
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
              {deployments.length === 0 && (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No deployments recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {actionModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">{actionModal.title}</div>
            <div className="modal-body">
              <p style={{ marginBottom: '1rem' }}>{actionModal.prompt}</p>
              {actionModal.inputLabel && (
                <div className="form-group">
                  <label className="form-label">{actionModal.inputLabel}</label>
                  <input
                    type="text"
                    className="form-input"
                    value={modalInput}
                    onChange={(e) => setModalInput(e.target.value)}
                  />
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setActionModal(null)} disabled={isSubmitting}>
                Cancel
              </button>
              <button
                className={`btn ${actionModal.type === 'REJECT' || actionModal.type === 'ROLLBACK' ? 'btn-danger' : 'btn-primary'}`}
                onClick={handleConfirmAction}
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Processing...' : 'Confirm Action'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
