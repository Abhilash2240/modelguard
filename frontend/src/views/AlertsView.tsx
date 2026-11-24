import React, { useEffect, useMemo, useState } from 'react'
import {
  CheckCircle2,
  Search,
  ShieldAlert,
  UserCheck,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { Modal } from '../components/Modal'
import { CardSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useToast } from '../components/Toast'
import type { Alert } from '../types'

interface AlertsViewProps {
  refreshTrigger?: number
}

export const AlertsView: React.FC<AlertsViewProps> = ({ refreshTrigger }) => {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filterStatus, setFilterStatus] = useState<string>('ALL')
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL')
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [resolveModal, setResolveModal] = useState<Alert | null>(null)
  const [resolveNotes, setResolveNotes] = useState<string>('')
  const [isResolving, setIsResolving] = useState<boolean>(false)
  const { addToast } = useToast()

  const loadAlerts = async () => {
    try {
      setLoading(true)
      const data = await api.getAlerts()
      setAlerts(data)
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Error Loading Alerts',
        message: err.message || 'Failed to fetch alerts.',
      })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAlerts()
  }, [refreshTrigger])

  const handleAcknowledge = async (alertId: string) => {
    try {
      await api.acknowledgeAlert(alertId, 'operator')
      addToast({
        type: 'info',
        title: 'Alert Acknowledged',
        message: `Alert ${alertId.slice(0, 8)} flagged as acknowledged by operator.`,
      })
      await loadAlerts()
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Acknowledge Failed',
        message: err.message || 'Failed to acknowledge alert.',
      })
    }
  }

  const handleResolveSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!resolveModal) return
    setIsResolving(true)
    try {
      await api.resolveAlert(resolveModal.id, 'operator', resolveNotes || 'Remediated by operator')
      addToast({
        type: 'success',
        title: 'Alert Resolved',
        message: `Alert ${resolveModal.id.slice(0, 8)} successfully resolved.`,
      })
      setResolveModal(null)
      setResolveNotes('')
      await loadAlerts()
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Resolution Failed',
        message: err.message || 'Failed to resolve alert.',
      })
    } finally {
      setIsResolving(false)
    }
  }

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    return alerts.filter((a) => {
      const matchStatus = filterStatus === 'ALL' || a.status === filterStatus
      const matchSeverity = filterSeverity === 'ALL' || a.severity === filterSeverity
      const matchQuery =
        !searchQuery ||
        a.alert_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        a.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
        a.model_name.toLowerCase().includes(searchQuery.toLowerCase())

      return matchStatus && matchSeverity && matchQuery
    })
  }, [alerts, filterStatus, filterSeverity, searchQuery])

  const openCount = alerts.filter((a) => a.status === 'OPEN').length

  if (loading && alerts.length === 0) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Alert Management & Audit Log</h1>
            <p className="page-description">Actionable threshold violations, statistical drift alarms, and human operator resolutions.</p>
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
          <h1 className="page-title">Alert Management & Audit Log</h1>
          <p className="page-description">
            Safeguard event tracking, automated threshold violations, drift alarms, and human operator audit resolution trails.
          </p>
        </div>
        <div className="header-meta-pill">
          <ShieldAlert size={14} className={openCount > 0 ? 'text-rose' : 'text-emerald'} />
          <span>{openCount} Open Alerts</span>
        </div>
      </div>

      {/* Main Alerts Card */}
      <div className="card">
        <div className="card-title">
          <div className="title-with-badge">
            <ShieldAlert size={16} className="text-rose" />
            <span>Operational Safeguard Alerts</span>
            <span className="counter-pill">{filteredAlerts.length}</span>
          </div>

          {/* Search & Filters */}
          <div className="alerts-controls-toolbar">
            <div className="search-input-wrap">
              <Search size={14} className="search-icon" />
              <input
                type="text"
                className="search-input"
                placeholder="Search alerts by title or component..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="filter-select-group">
              <select
                className="form-select form-select-sm"
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
              >
                <option value="ALL">All Statuses</option>
                <option value="OPEN">Open</option>
                <option value="ACKNOWLEDGED">Acknowledged</option>
                <option value="RESOLVED">Resolved</option>
              </select>

              <select
                className="form-select form-select-sm"
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
              >
                <option value="ALL">All Severities</option>
                <option value="CRITICAL">Critical</option>
                <option value="WARNING">Warning</option>
                <option value="INFO">Info</option>
              </select>
            </div>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Alert ID</th>
                <th>Severity</th>
                <th>Model</th>
                <th>Alert Title & Description</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredAlerts.map((a) => (
                <tr key={a.id}>
                  <td>
                    <div className="id-with-copy">
                      <code>{a.id.slice(0, 8)}</code>
                      <CopyButton text={a.id} tooltip="Copy Alert UUID" />
                    </div>
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        a.severity === 'CRITICAL'
                          ? 'badge-error'
                          : a.severity === 'WARNING'
                          ? 'badge-warning'
                          : 'badge-info'
                      }`}
                    >
                      {a.severity}
                    </span>
                  </td>
                  <td>
                    <code>{a.model_name} v{a.model_version}</code>
                  </td>
                  <td>
                    <div className="alert-text-block">
                      <strong className="alert-title-text">{a.alert_type}</strong>
                      <span className="alert-desc-text">{a.message}</span>
                    </div>
                  </td>
                  <td>
                    <StatusBadge status={a.status} />
                  </td>
                  <td className="text-muted">
                    {new Date(a.created_at).toLocaleString()}
                  </td>
                  <td>
                    <div className="alert-actions-cell">
                      {a.status === 'OPEN' && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleAcknowledge(a.id)}
                          title="Acknowledge alert"
                        >
                          <UserCheck size={12} />
                          <span>Acknowledge</span>
                        </button>
                      )}
                      {a.status !== 'RESOLVED' && (
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          onClick={() => {
                            setResolveModal(a)
                            setResolveNotes('')
                          }}
                          title="Resolve alert"
                        >
                          <CheckCircle2 size={12} />
                          <span>Resolve</span>
                        </button>
                      )}
                      {a.status === 'RESOLVED' && (
                        <span className="text-muted" style={{ fontSize: '0.8rem' }}>
                          Resolved
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}

              {filteredAlerts.length === 0 && (
                <tr>
                  <td colSpan={7} className="empty-table-cell">
                    No alerts match the selected status, severity, or search query.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Resolve Alert Modal */}
      {resolveModal && (
        <Modal
          isOpen={true}
          onClose={() => !isResolving && setResolveModal(null)}
          title={`Resolve Alert: ${resolveModal.alert_type}`}
          footer={
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setResolveModal(null)}
                disabled={isResolving}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleResolveSubmit}
                disabled={isResolving}
              >
                {isResolving ? 'Resolving...' : 'Confirm Resolution'}
              </button>
            </>
          }
        >
          <form onSubmit={handleResolveSubmit} className="modal-body-content">
            <div className="modal-info-box">
              <div className="info-row">
                <span>Alert ID:</span>
                <code>{resolveModal.id}</code>
              </div>
              <div className="info-row">
                <span>Model:</span>
                <code>{resolveModal.model_name} v{resolveModal.model_version}</code>
              </div>
              <div className="info-row">
                <span>Severity:</span>
                <span className="text-rose">{resolveModal.severity}</span>
              </div>
            </div>

            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label className="form-label">Remediation Notes & Root Cause</label>
              <textarea
                className="form-input"
                rows={3}
                placeholder="Document remedial steps taken (e.g., retrained model, adjusted drift threshold)..."
                value={resolveNotes}
                onChange={(e) => setResolveNotes(e.target.value)}
                autoFocus
              />
            </div>
          </form>
        </Modal>
      )}
    </div>
  )
}
