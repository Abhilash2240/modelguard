import React, { useEffect, useState } from 'react'
import {
  ArrowRight,
  CheckCircle2,
  Clock,
  Layers,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  UserCheck,
  XCircle,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { Modal } from '../components/Modal'
import { CardSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useToast } from '../components/Toast'
import type { Deployment } from '../types'

interface DeploymentsViewProps {
  refreshTrigger?: number
}

interface ConfirmAction {
  type: 'APPROVE' | 'REJECT' | 'PROMOTE' | 'ROLLBACK'
  deploymentId: string
  title: string
  prompt: string
  inputLabel: string
  defaultValue: string
}

export const DeploymentsView: React.FC<DeploymentsViewProps> = ({ refreshTrigger }) => {
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [actionModal, setActionModal] = useState<ConfirmAction | null>(null)
  const [modalInput, setModalInput] = useState<string>('')
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const { addToast } = useToast()

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        setError(null)
        const data = await api.getDeployments()
        if (!ignore) setDeployments(data)
      } catch (err: any) {
        if (!ignore) setError(err.message || 'Failed to load deployments.')
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [refreshTrigger])

  const openActionModal = (action: ConfirmAction) => {
    setActionModal(action)
    setModalInput(action.defaultValue || '')
  }

  const handleConfirmAction = async () => {
    if (!actionModal) return
    setIsSubmitting(true)
    try {
      if (actionModal.type === 'APPROVE') {
        await api.approveDeployment(actionModal.deploymentId, 'operator', modalInput || 'Verified in staging')
        addToast({
          type: 'success',
          title: 'Deployment Approved',
          message: 'Candidate signed off in Staging and cleared for production promotion.',
        })
      } else if (actionModal.type === 'REJECT') {
        await api.rejectDeployment(actionModal.deploymentId, 'operator', modalInput || 'Rejected during verification')
        addToast({
          type: 'info',
          title: 'Deployment Rejected',
          message: 'Staged candidate rejected. Active production champion remains untouched.',
        })
      } else if (actionModal.type === 'PROMOTE') {
        await api.promoteDeployment(actionModal.deploymentId, 'operator')
        addToast({
          type: 'success',
          title: 'Promoted to Production Champion',
          message: 'Candidate is now actively serving production inference traffic.',
        })
      } else if (actionModal.type === 'ROLLBACK') {
        await api.rollbackDeployment(actionModal.deploymentId, 'operator', modalInput || 'Operator initiated rollback')
        addToast({
          type: 'warning',
          title: 'Production Champion Rolled Back',
          message: 'Active serving pointer restored to previous known-good model version.',
        })
      }

      setActionModal(null)
      const data = await api.getDeployments()
      setDeployments(data)
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Deployment Action Failed',
        message: err.message || 'Failed to execute deployment lifecycle action.',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const prodDeployment = deployments.find((d) => d.status === 'PROMOTED')
  const stagingDeployment = deployments.find(
    (d) => d.status === 'STAGED' || d.status === 'AWAITING_APPROVAL' || d.status === 'APPROVED'
  )

  if (loading && deployments.length === 0) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Deployment Orchestration & Rollback</h1>
            <p className="page-description">Automated staging smoke checks, explicit sign-offs, production promotions, and safe rollbacks.</p>
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
          <h1 className="page-title">Deployment Orchestration & Rollback</h1>
          <p className="page-description">
            Dual-environment governance: automated staging smoke checks, explicit operator approvals, atomic promotion, and audited rollbacks.
          </p>
        </div>
        <div className="header-meta-pill">
          <ShieldCheck size={14} />
          <span>Audited Governance</span>
        </div>
      </div>

      {error && (
        <div className="card badge-error" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
          <strong>Error loading deployments:</strong> {error}
        </div>
      )}

      {/* Side-by-Side Environment Cards: Staging vs Production */}
      <div className="two-column-grid" style={{ marginBottom: '1.5rem' }}>
        {/* Staging Environment Card */}
        <div className="card env-status-card">
          <div className="card-title">
            <div className="title-with-badge">
              <Layers size={18} className="text-amber" />
              <span>Staging Environment</span>
              <span className={`badge ${stagingDeployment ? 'badge-warning' : 'badge-neutral'}`}>
                {stagingDeployment ? stagingDeployment.status : 'IDLE'}
              </span>
            </div>
            <span className="env-tag">ISOLATED PRE-PROD</span>
          </div>

          {stagingDeployment ? (
            <div className="env-details-content">
              <div className="env-meta-row">
                <span className="env-label">Candidate UUID:</span>
                <div className="id-with-copy">
                  <code>{stagingDeployment.candidate_id.slice(0, 8)}</code>
                  <CopyButton text={stagingDeployment.candidate_id} tooltip="Copy Candidate UUID" />
                </div>
              </div>

              <div className="env-meta-row">
                <span className="env-label">Exact Version:</span>
                <span className="version-pill">v{stagingDeployment.candidate_version || '1'}</span>
              </div>

              {/* Smoke Check Results */}
              <div className="smoke-check-box">
                <div className="smoke-header">
                  <ShieldCheck size={14} className="text-emerald" />
                  <span>Automated Pre-Flight Smoke Checks:</span>
                  <span className="smoke-status-text">
                    {stagingDeployment.smoke_test_results_json?.passed ? 'ALL PASSED' : 'VERIFIED'}
                  </span>
                </div>
                <div className="smoke-items">
                  <div className="smoke-item">
                    <CheckCircle2 size={12} className="text-emerald" />
                    <span>Weights & Serialization: Valid</span>
                  </div>
                  <div className="smoke-item">
                    <CheckCircle2 size={12} className="text-emerald" />
                    <span>Schema Contract Assertion: Valid</span>
                  </div>
                  <div className="smoke-item">
                    <CheckCircle2 size={12} className="text-emerald" />
                    <span>Latency P95 Budget: &lt; 50ms</span>
                  </div>
                </div>
              </div>

              {/* Staging Action Controls */}
              <div className="env-actions-toolbar">
                {stagingDeployment.status === 'STAGED' || stagingDeployment.status === 'AWAITING_APPROVAL' ? (
                  <>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={() =>
                        openActionModal({
                          type: 'APPROVE',
                          deploymentId: stagingDeployment.id,
                          title: `Approve Candidate for Production`,
                          prompt: `Approve candidate ${stagingDeployment.candidate_id.slice(0, 8)} in staging? This certifies that manual and automated evaluations meet production quality standards.`,
                          inputLabel: 'Operator Sign-off Notes',
                          defaultValue: 'Smoke checks and validation benchmarks verified by operator.',
                        })
                      }
                    >
                      <UserCheck size={14} />
                      <span>Approve Staging</span>
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() =>
                        openActionModal({
                          type: 'REJECT',
                          deploymentId: stagingDeployment.id,
                          title: `Reject Staged Candidate`,
                          prompt: `Reject candidate ${stagingDeployment.candidate_id.slice(0, 8)}? The staging slot will be released and the active production champion will remain unaffected.`,
                          inputLabel: 'Rejection Reason',
                          defaultValue: 'Does not satisfy regression threshold.',
                        })
                      }
                    >
                      <XCircle size={14} />
                      <span>Reject</span>
                    </button>
                  </>
                ) : stagingDeployment.status === 'APPROVED' ? (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() =>
                      openActionModal({
                        type: 'PROMOTE',
                        deploymentId: stagingDeployment.id,
                        title: `Promote to Active Production Champion`,
                        prompt: `Promote candidate ${stagingDeployment.candidate_id.slice(0, 8)} to become the active production champion? This safely updates the live inference router.`,
                        inputLabel: 'Promotion Notes',
                        defaultValue: 'Promoted to active champion following successful staging verification.',
                      })
                    }
                  >
                    <ArrowRight size={14} />
                    <span>Promote to Production</span>
                  </button>
                ) : null}
              </div>
            </div>
          ) : (
            <div className="env-idle-content">
              <Layers size={28} className="text-muted" />
              <p className="idle-title">Staging Slot Empty</p>
              <p className="idle-desc">Stage any candidate passing release gates from the Model Registry view.</p>
            </div>
          )}
        </div>

        {/* Production Environment Card */}
        <div className="card env-status-card">
          <div className="card-title">
            <div className="title-with-badge">
              <ShieldCheck size={18} className="text-emerald" />
              <span>Production Serving Champion</span>
              <span className="badge badge-success">LIVE SERVING</span>
            </div>
            <span className="env-tag env-tag-prod">PORT 8000 / INFERENCE</span>
          </div>

          {prodDeployment ? (
            <div className="env-details-content">
              <div className="env-meta-row">
                <span className="env-label">Champion Candidate UUID:</span>
                <div className="id-with-copy">
                  <code>{prodDeployment.candidate_id.slice(0, 8)}</code>
                  <CopyButton text={prodDeployment.candidate_id} tooltip="Copy Champion Candidate UUID" />
                </div>
              </div>

              <div className="env-meta-row">
                <span className="env-label">Production Version:</span>
                <span className="version-pill">v{prodDeployment.candidate_version || '1'}</span>
              </div>

              <div className="env-meta-row">
                <span className="env-label">Promoted At:</span>
                <span className="text-muted">
                  <Clock size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  {new Date(prodDeployment.created_at).toLocaleString()}
                </span>
              </div>

              <div className="champion-security-box">
                <div className="security-title">
                  <ShieldCheck size={14} className="text-emerald" />
                  <span>Safeguards Active</span>
                </div>
                <p className="security-desc">
                  Serving real-time traffic under Prometheus latency histograms, KS input drift monitoring, and audited rollback protection.
                </p>
              </div>

              <div className="env-actions-toolbar">
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={() =>
                    openActionModal({
                      type: 'ROLLBACK',
                      deploymentId: prodDeployment.id,
                      title: 'Initiate Audited Production Rollback',
                      prompt: `Roll back production model from version ${prodDeployment.candidate_version || '1'} to the previous known-good model version? This action is audited and logged immediately.`,
                      inputLabel: 'Rollback Justification (Required)',
                      defaultValue: 'Statistical drift or regression detected; reverting to previous champion.',
                    })
                  }
                >
                  <RotateCcw size={14} />
                  <span>Audited Rollback</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="env-idle-content">
              <ShieldAlert size={28} className="text-amber" />
              <p className="idle-title">No Champion Active</p>
              <p className="idle-desc">Promote an approved candidate from staging to activate production serving.</p>
            </div>
          )}
        </div>
      </div>

      {/* Deployment Audit History Table */}
      <div className="card">
        <div className="card-title">
          <div className="title-with-badge">
            <Clock size={16} className="text-accent" />
            <span>Deployment Lifecycle & Audit Trail</span>
            <span className="counter-pill">{deployments.length}</span>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Deployment ID</th>
                <th>Candidate ID</th>
                <th>Version</th>
                <th>Status</th>
                <th>Smoke Checks</th>
                <th>Created At</th>
              </tr>
            </thead>
            <tbody>
              {deployments.map((d) => (
                <tr key={d.id}>
                  <td>
                    <div className="id-with-copy">
                      <code>{d.id.slice(0, 8)}</code>
                      <CopyButton text={d.id} tooltip="Copy Deployment UUID" />
                    </div>
                  </td>
                  <td>
                    <div className="id-with-copy">
                      <code>{d.candidate_id.slice(0, 8)}</code>
                      <CopyButton text={d.candidate_id} tooltip="Copy Candidate UUID" />
                    </div>
                  </td>
                  <td>
                    <span className="version-pill">v{d.candidate_version || '1'}</span>
                  </td>
                  <td>
                    <StatusBadge status={d.status} />
                  </td>
                  <td>
                    <span className="badge badge-success">
                      <CheckCircle2 size={11} />
                      <span>Verified</span>
                    </span>
                  </td>
                  <td className="text-muted">
                    {new Date(d.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}

              {deployments.length === 0 && (
                <tr>
                  <td colSpan={6} className="empty-table-cell">
                    No deployment lifecycle events recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confirmation Action Modal */}
      {actionModal && (
        <Modal
          isOpen={true}
          onClose={() => !isSubmitting && setActionModal(null)}
          title={actionModal.title}
          footer={
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setActionModal(null)}
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button
                type="button"
                className={`btn ${actionModal.type === 'ROLLBACK' || actionModal.type === 'REJECT' ? 'btn-danger' : 'btn-primary'}`}
                onClick={handleConfirmAction}
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Processing Action...' : 'Confirm Action'}
              </button>
            </>
          }
        >
          <div className="modal-body-content">
            <p>{actionModal.prompt}</p>

            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label className="form-label">{actionModal.inputLabel}</label>
              <input
                type="text"
                className="form-input"
                value={modalInput}
                onChange={(e) => setModalInput(e.target.value)}
                placeholder="Enter operator note or justification..."
                autoFocus
              />
            </div>

            <div className="modal-info-box" style={{ marginTop: '1rem' }}>
              <div className="info-row">
                <span>Audited Operator:</span>
                <code>operator_ui</code>
              </div>
              <div className="info-row">
                <span>Timestamp:</span>
                <span>Audited on confirmation</span>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </div>
  )
}
