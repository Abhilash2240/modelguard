import React, { useEffect, useState } from 'react'
import { Filter, ShieldAlert } from 'lucide-react'
import { api } from '../api'
import type { Alert } from '../types'

export const AlertsView: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [resolveModal, setResolveModal] = useState<Alert | null>(null)
  const [resolveNotes, setResolveNotes] = useState('')
  const [actionMsg, setActionMsg] = useState<{ text: string; error?: boolean } | null>(null)

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        const data = await api.getAlerts(filterStatus || undefined)
        if (!ignore) setAlerts(data)
      } catch (err: any) {
        if (!ignore) setActionMsg({ text: `Failed to load alerts: ${err.message}`, error: true })
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [filterStatus])

  async function handleAcknowledge(alertId: string) {
    try {
      setActionMsg(null)
      await api.acknowledgeAlert(alertId, 'operator')
      setActionMsg({ text: 'Alert ACKNOWLEDGED.' })
      const data = await api.getAlerts(filterStatus || undefined)
      setAlerts(data)
    } catch (err: any) {
      setActionMsg({ text: `Failed to acknowledge: ${err.message}`, error: true })
    }
  }

  async function handleResolveSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!resolveModal) return
    try {
      setActionMsg(null)
      await api.resolveAlert(resolveModal.id, 'operator', resolveNotes || 'Remediated by operator')
      setActionMsg({ text: 'Alert RESOLVED.' })
      setResolveModal(null)
      setResolveNotes('')
      const data = await api.getAlerts(filterStatus || undefined)
      setAlerts(data)
    } catch (err: any) {
      setActionMsg({ text: `Failed to resolve: ${err.message}`, error: true })
    }
  }

  if (loading && alerts.length === 0) return <div className="card">Loading alerts...</div>

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Alert Management & Audit Log</h1>
          <p className="page-description">Actionable threshold violations, statistical drift alarms, and human operator resolutions.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Filter size={14} color="var(--text-muted)" />
          <select
            className="form-select"
            style={{ width: 'auto' }}
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="">All Statuses</option>
            <option value="OPEN">Open</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="RESOLVED">Resolved</option>
          </select>
        </div>
      </div>

      {actionMsg && (
        <div className={`card ${actionMsg.error ? 'badge-error' : 'badge-success'}`} style={{ marginBottom: '1rem', padding: '0.75rem 1rem' }}>
          {actionMsg.text}
        </div>
      )}

      <div className="card">
        <div className="card-title">
          <span>Active & Historical Alerts</span>
          <ShieldAlert size={16} />
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Severity</th>
                <th>Type</th>
                <th>Model</th>
                <th>Message</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr key={a.id}>
                  <td>
                    <span className={`badge ${
                      a.severity === 'CRITICAL' ? 'badge-error' :
                      a.severity === 'WARNING' ? 'badge-warning' : 'badge-info'
                    }`}>
                      {a.severity}
                    </span>
                  </td>
                  <td><code>{a.alert_type}</code></td>
                  <td>{a.model_name}:v{a.model_version}</td>
                  <td style={{ fontSize: '0.8125rem', maxWidth: '300px' }}>
                    <div>{a.message}</div>
                    {a.resolution_notes && (
                      <div style={{ fontSize: '0.7rem', color: '#34d399', marginTop: '2px' }}>
                        Note: {a.resolution_notes}
                      </div>
                    )}
                  </td>
                  <td>
                    <span className={`badge ${
                      a.status === 'OPEN' ? 'badge-error' :
                      a.status === 'ACKNOWLEDGED' ? 'badge-warning' : 'badge-success'
                    }`}>
                      {a.status}
                    </span>
                  </td>
                  <td>{new Date(a.created_at).toLocaleString()}</td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.35rem' }}>
                      {a.status === 'OPEN' && (
                        <button
                          className="btn btn-secondary"
                          style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                          onClick={() => handleAcknowledge(a.id)}
                        >
                          Acknowledge
                        </button>
                      )}
                      {a.status !== 'RESOLVED' && (
                        <button
                          className="btn btn-success"
                          style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem' }}
                          onClick={() => {
                            setResolveModal(a)
                            setResolveNotes('')
                          }}
                        >
                          Resolve
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              {alerts.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No alerts found for selected filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {resolveModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">Resolve Alert ({resolveModal.alert_type})</div>
            <form onSubmit={handleResolveSubmit}>
              <div className="modal-body">
                <p style={{ marginBottom: '1rem' }}>{resolveModal.message}</p>
                <div className="form-group">
                  <label className="form-label">Remediation Notes</label>
                  <textarea
                    className="form-textarea"
                    placeholder="Describe corrective action taken (e.g. sensor recalibrated, model rolled back)..."
                    value={resolveNotes}
                    onChange={(e) => setResolveNotes(e.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setResolveModal(null)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-success">
                  Confirm Resolution
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
