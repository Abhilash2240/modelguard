import React, { useEffect, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle, ExternalLink, RefreshCw, Zap } from 'lucide-react'
import { api } from '../api'
import type { DriftReport, QualityReport } from '../types'

export const MonitoringView: React.FC = () => {
  const [reports, setReports] = useState<DriftReport[]>([])
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [evaluating, setEvaluating] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        const [reps, qual] = await Promise.all([
          api.getReports().catch(() => []),
          api.getQuality().catch(() => null),
        ])
        if (!ignore) {
          setReports(reps)
          setQuality(qual)
        }
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [])

  async function refreshSignals() {
    try {
      setLoading(true)
      const [reps, qual] = await Promise.all([
        api.getReports().catch(() => []),
        api.getQuality().catch(() => null),
      ])
      setReports(reps)
      setQuality(qual)
    } finally {
      setLoading(false)
    }
  }

  async function handleTriggerInputDrift() {
    try {
      setEvaluating('input')
      setMessage(null)
      const r = await api.triggerInputDrift()
      setMessage(`Input drift analysis complete: ${r.is_drift_detected ? 'DRIFT DETECTED!' : 'All features stable.'}`)
      await refreshSignals()
    } catch (err: any) {
      setMessage(`Failed to run input drift: ${err.message}`)
    } finally {
      setEvaluating(null)
    }
  }

  async function handleTriggerPredictionDrift() {
    try {
      setEvaluating('prediction')
      setMessage(null)
      const r = await api.triggerPredictionDrift()
      setMessage(`Prediction drift analysis complete: ${r.is_drift_detected ? 'OUTPUT DRIFT DETECTED!' : 'Predictions stable.'}`)
      await refreshSignals()
    } catch (err: any) {
      setMessage(`Failed to run prediction drift: ${err.message}`)
    } finally {
      setEvaluating(null)
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Monitoring, Drift & Statistical Safeguards</h1>
          <p className="page-description">Covariate input drift, output prediction shifts, delayed ground-truth accuracy, and Prometheus telemetry.</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <a
            href="http://127.0.0.1:3001"
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
          >
            <ExternalLink size={14} /> Open Grafana Dashboard
          </a>
          <button className="btn btn-secondary" onClick={refreshSignals} disabled={loading}>
            <RefreshCw size={14} /> Refresh Signals
          </button>
        </div>
      </div>

      {message && (
        <div className="card badge-info" style={{ marginBottom: '1rem', padding: '0.75rem 1rem' }}>
          {message}
        </div>
      )}

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-header">
            <span>Input Feature Drift</span>
            <Activity size={16} />
          </div>
          <div className="stat-value">
            {reports.find((r) => r.report_type === 'DATA_DRIFT')?.is_drift_detected ? (
              <span style={{ color: '#f87171' }}>DRIFT</span>
            ) : (
              <span style={{ color: '#34d399' }}>STABLE</span>
            )}
          </div>
          <div className="stat-footer">Kolmogorov-Smirnov & Population Stability Index (PSI)</div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>Prediction Drift</span>
            <Zap size={16} />
          </div>
          <div className="stat-value">
            {reports.find((r) => r.report_type === 'PREDICTION_DRIFT')?.is_drift_detected ? (
              <span style={{ color: '#fbbf24' }}>SHIFT</span>
            ) : (
              <span style={{ color: '#34d399' }}>STABLE</span>
            )}
          </div>
          <div className="stat-footer">Output distribution shift independent of input drift</div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>Delayed Label Accuracy</span>
            <CheckCircle size={16} />
          </div>
          <div className="stat-value">
            {quality?.mae ? `${quality.mae.toFixed(1)} MAE` : quality?.status || 'NO DATA'}
          </div>
          <div className="stat-footer">Coverage: {quality?.label_coverage ? `${quality.label_coverage.toFixed(1)}%` : '0%'}</div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span>On-Demand Analysis</span>
            <Activity size={16} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.5rem' }}>
            <button
              className="btn btn-primary"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
              onClick={handleTriggerInputDrift}
              disabled={evaluating !== null}
            >
              {evaluating === 'input' ? 'Analyzing...' : 'Run Input Drift'}
            </button>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
              onClick={handleTriggerPredictionDrift}
              disabled={evaluating !== null}
            >
              {evaluating === 'prediction' ? 'Analyzing...' : 'Run Prediction Drift'}
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Drift Analysis Reports & Statistical Evidence</span>
          <AlertTriangle size={16} />
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Report Type</th>
                <th>Model Version</th>
                <th>Status</th>
                <th>Detection Method</th>
                <th>Reference Samples</th>
                <th>Current Samples</th>
                <th>Drifting Features / Stats</th>
                <th>Generated At</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.id}>
                  <td>
                    <span className={`badge ${r.report_type === 'DATA_DRIFT' ? 'badge-info' : 'badge-neutral'}`}>
                      {r.report_type}
                    </span>
                  </td>
                  <td>v{r.model_version}</td>
                  <td>
                    {r.is_drift_detected ? (
                      <span className="badge badge-error">DRIFT DETECTED</span>
                    ) : (
                      <span className="badge badge-success">STABLE</span>
                    )}
                  </td>
                  <td><code>{r.detection_method}</code></td>
                  <td>{r.reference_sample_count}</td>
                  <td>{r.current_sample_count}</td>
                  <td style={{ fontSize: '0.75rem' }}>
                    {r.summary_json.drifting_features && r.summary_json.drifting_features.length > 0 ? (
                      <span style={{ color: '#f87171' }}>{r.summary_json.drifting_features.join(', ')}</span>
                    ) : r.summary_json.ks_statistic !== undefined ? (
                      <span>KS={r.summary_json.ks_statistic.toFixed(3)}, p={r.summary_json.p_value?.toExponential(2)}</span>
                    ) : (
                      'None'
                    )}
                  </td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
              {reports.length === 0 && (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No drift reports generated yet. Click "Run Input Drift" above to initiate a statistical check.
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
