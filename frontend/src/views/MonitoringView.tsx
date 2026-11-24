import React, { useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Layers,
  Play,
  RefreshCw,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { CardSkeleton, StatGridSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatCard } from '../components/StatCard'
import { StatusBadge } from '../components/StatusBadge'
import { useToast } from '../components/Toast'
import type { DriftReport, QualityReport } from '../types'

interface MonitoringViewProps {
  refreshTrigger?: number
}

export const MonitoringView: React.FC<MonitoringViewProps> = ({ refreshTrigger }) => {
  const [reports, setReports] = useState<DriftReport[]>([])
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [evaluating, setEvaluating] = useState<string | null>(null)
  const { addToast } = useToast()

  const loadSignals = async () => {
    try {
      setLoading(true)
      setError(null)
      const [reps, qual] = await Promise.all([
        api.getReports(),
        api.getQuality().catch(() => null),
      ])
      setReports(reps)
      setQuality(qual)
    } catch (err: any) {
      setError(err?.message || 'Failed to load monitoring signals.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSignals()
  }, [refreshTrigger])

  const handleTriggerInputDrift = async () => {
    try {
      setEvaluating('input')
      const r = await api.triggerInputDrift()
      addToast({
        type: r.is_drift_detected ? 'warning' : 'success',
        title: 'Input Drift Evaluation Complete',
        message: r.is_drift_detected
          ? 'Covariate feature drift detected across monitored inputs!'
          : 'All input feature distributions are statistically stable.',
      })
      await loadSignals()
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Drift Evaluation Failed',
        message: err.message || 'Failed to trigger input drift check.',
      })
    } finally {
      setEvaluating(null)
    }
  }

  const handleTriggerPredictionDrift = async () => {
    try {
      setEvaluating('prediction')
      const r = await api.triggerPredictionDrift()
      addToast({
        type: r.is_drift_detected ? 'warning' : 'success',
        title: 'Prediction Drift Evaluation Complete',
        message: r.is_drift_detected
          ? 'Prediction distribution shift detected!'
          : 'Live prediction distribution matches reference baseline.',
      })
      await loadSignals()
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Drift Evaluation Failed',
        message: err.message || 'Failed to trigger prediction drift check.',
      })
    } finally {
      setEvaluating(null)
    }
  }

  const latestInputReport = reports.find(
    (r) => r.report_type === 'DATA_DRIFT' || (r.report_type as string) === 'INPUT_DRIFT'
  )
  const latestPredictionReport = reports.find((r) => r.report_type === 'PREDICTION_DRIFT')

  if (loading && reports.length === 0) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Monitoring, Drift & Statistical Safeguards</h1>
            <p className="page-description">Covariate input drift, output prediction shifts, delayed ground-truth accuracy, and Prometheus telemetry.</p>
          </div>
        </div>
        <StatGridSkeleton count={3} />
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
          <h1 className="page-title">Monitoring, Drift & Statistical Safeguards</h1>
          <p className="page-description">
            Kolmogorov-Smirnov statistical hypothesis testing for covariate shifts, prediction drift, delayed ground truth evaluation, and live Grafana telemetry.
          </p>
        </div>
        <div className="header-actions">
          <a
            href="http://127.0.0.1:3001"
            target="_blank"
            rel="noreferrer"
            className="btn btn-primary"
            title="Open Grafana Dashboards"
          >
            <Activity size={14} />
            <span>Open Grafana</span>
            <ExternalLink size={12} />
          </a>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={loadSignals}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'spin-icon' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.85rem 1.2rem', borderRadius: 'var(--radius-md)', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <AlertTriangle size={18} className="text-red" />
            <span style={{ fontSize: '0.88rem' }}>{error}</span>
          </div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={loadSignals}>
            <RefreshCw size={12} />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* KPI Cards Grid - Specifically Formatted to Prevent Overflow */}
      <div className="stat-grid">
        <StatCard
          title="Input Feature Drift"
          value={
            latestInputReport
              ? latestInputReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestInputReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
                ? 'INSUFFICIENT DATA'
                : latestInputReport.is_drift_detected
                ? 'DRIFT DETECTED'
                : 'STABLE'
              : 'NOT EVALUATED'
          }
          icon={Activity}
          badge={{
            text: !latestInputReport
              ? 'NOT EVALUATED'
              : latestInputReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestInputReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
              ? 'INSUFFICIENT'
              : latestInputReport.is_drift_detected
              ? 'ATTENTION'
              : 'NOMINAL',
            type: !latestInputReport
              ? 'neutral'
              : latestInputReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestInputReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
              ? 'warning'
              : latestInputReport.is_drift_detected
              ? 'error'
              : 'success',
          }}
          footer={
            latestInputReport
              ? `Last evaluated ${new Date(latestInputReport.created_at).toLocaleTimeString()}`
              : 'Run analysis to compute'
          }
        />

        <StatCard
          title="Prediction Output Drift"
          value={
            latestPredictionReport
              ? latestPredictionReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestPredictionReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
                ? 'INSUFFICIENT DATA'
                : latestPredictionReport.is_drift_detected
                ? 'SHIFT DETECTED'
                : 'STABLE'
              : 'NOT EVALUATED'
          }
          icon={Zap}
          badge={{
            text: !latestPredictionReport
              ? 'NOT EVALUATED'
              : latestPredictionReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestPredictionReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
              ? 'INSUFFICIENT'
              : latestPredictionReport.is_drift_detected
              ? 'ATTENTION'
              : 'NOMINAL',
            type: !latestPredictionReport
              ? 'neutral'
              : latestPredictionReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestPredictionReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
              ? 'warning'
              : latestPredictionReport.is_drift_detected
              ? 'error'
              : 'success',
          }}
          footer={
            latestPredictionReport
              ? latestPredictionReport.summary_json?.p_value != null
                ? `p-val: ${latestPredictionReport.summary_json.p_value.toFixed(3)} (PSI: ${(latestPredictionReport.summary_json.psi ?? 0).toFixed(3)})`
                : latestPredictionReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA'
                ? 'Insufficient predictions (< 10)'
                : `Evaluated ${new Date(latestPredictionReport.created_at).toLocaleTimeString()}`
              : 'Run analysis to compute'
          }
        />

        {/* Rolling Quality Card with Clean StatusBadge formatting to prevent text clipping */}
        <div className="kpi-card">
          <div className="kpi-header">
            <span className="kpi-title">Delayed-Label Model Quality</span>
            <div className="kpi-icon-wrap">
              <TrendingUp size={18} />
            </div>
          </div>
          <div className="kpi-content">
            <div className="quality-status-badge-wrap">
              <StatusBadge status={quality?.status || 'INSUFFICIENT_LABELS'} />
            </div>
            <div className="stat-metric-subtext" style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {quality?.mae != null ? (
                <span>Rolling MAE: <strong>{quality.mae.toFixed(1)}</strong></span>
              ) : (
                <span>Requires &ge; {quality?.details_json?.min_samples_required ?? 10} ground-truth feedback labels</span>
              )}
            </div>
          </div>
          <div className="kpi-footer">
            <span>{quality?.labeled_sample_count ?? 0} feedback samples recorded</span>
          </div>
        </div>
      </div>

      {/* Trigger Drift Checks Row */}
      <div className="two-column-grid" style={{ marginBottom: '1.5rem' }}>
        {/* Input Drift Check Box */}
        <div className="card">
          <div className="card-title">
            <div className="title-with-badge">
              <Layers size={16} className="text-accent" />
              <span>Input Covariate Shift Analysis</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleTriggerInputDrift}
              disabled={evaluating === 'input'}
            >
              <Play size={12} />
              <span>{evaluating === 'input' ? 'Running KS Tests...' : 'Run Input Drift Check'}</span>
            </button>
          </div>
          <p className="section-description-text">
            Performs statistical drift tests (KS-test, PSI, and TVD) comparing live production feature telemetry against the baseline training dataset.
          </p>

          {latestInputReport?.metrics_json?.features && latestInputReport.metrics_json.features.length > 0 ? (
            <div className="feature-drift-summary-list">
              {latestInputReport.metrics_json.features.map((feat) => {
                const statLabel =
                  feat.p_value != null
                    ? `p = ${feat.p_value.toFixed(3)}`
                    : feat.psi != null
                    ? `PSI = ${feat.psi.toFixed(3)}`
                    : feat.test_statistic != null
                    ? `TVD = ${feat.test_statistic.toFixed(3)}`
                    : ''
                return (
                  <div key={feat.feature_name} className="feature-row">
                    <span className="feature-name">
                      {feat.feature_name} ({feat.feature_type || 'numeric'})
                    </span>
                    <span className={`badge ${feat.is_drift_detected ? 'badge-error' : 'badge-success'}`}>
                      {feat.is_drift_detected ? 'Drift Detected' : 'Stable'}
                      {statLabel ? ` (${statLabel})` : ''}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : latestInputReport?.detection_method === 'INSUFFICIENT_DATA' || latestInputReport?.summary_json?.evidence_source === 'INSUFFICIENT_DATA' ? (
            <div className="empty-notice-box" style={{ padding: '0.75rem', background: 'var(--bg-surface-2)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              <AlertTriangle size={15} className="text-amber" />
              <span>Insufficient live inference events (requires &ge; 5 events, 10 records) to compute feature drift.</span>
            </div>
          ) : (
            <div className="empty-notice-box" style={{ padding: '0.75rem', background: 'var(--bg-surface-2)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              <Activity size={15} className="text-muted" />
              <span>No input drift analysis run yet. Click "Run Input Drift Check" above to compute.</span>
            </div>
          )}
        </div>

        {/* Prediction Drift Check Box */}
        <div className="card">
          <div className="card-title">
            <div className="title-with-badge">
              <Zap size={16} className="text-amber" />
              <span>Prediction Distribution Shift</span>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleTriggerPredictionDrift}
              disabled={evaluating === 'prediction'}
            >
              <Play size={12} />
              <span>{evaluating === 'prediction' ? 'Evaluating Shift...' : 'Run Output Shift Check'}</span>
            </button>
          </div>
          <p className="section-description-text">
            Compares sliding-window model predictions with baseline inference distribution to catch distribution shifts prior to receiving ground-truth labels.
          </p>

          <div className="drift-metrics-summary-box">
            <div className="drift-meta-item">
              <span className="drift-label">Statistical Test:</span>
              <strong>Kolmogorov-Smirnov &amp; PSI</strong>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Significance Threshold (&alpha;):</span>
              <code>0.05</code>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Current KS p-value:</span>
              <strong className={latestPredictionReport?.summary_json?.p_value != null ? 'text-emerald' : 'text-muted'}>
                {latestPredictionReport?.summary_json?.p_value != null
                  ? latestPredictionReport.summary_json.p_value.toFixed(3)
                  : '—'}
              </strong>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Distribution Status:</span>
              {!latestPredictionReport ? (
                <span className="badge badge-neutral">Not Evaluated</span>
              ) : latestPredictionReport.detection_method === 'INSUFFICIENT_DATA' ||
                latestPredictionReport.summary_json?.evidence_source === 'INSUFFICIENT_DATA' ? (
                <span className="badge badge-warning">Insufficient Predictions</span>
              ) : latestPredictionReport.is_drift_detected ? (
                <span className="badge badge-error">Shift Detected</span>
              ) : (
                <span className="badge badge-success">Consistent with Baseline</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Historical Drift Reports Table */}
      <div className="card">
        <div className="card-title">
          <div className="title-with-badge">
            <Activity size={16} className="text-accent" />
            <span>Audited Statistical Drift Reports</span>
            <span className="counter-pill">{reports.length}</span>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Report ID</th>
                <th>Report Type</th>
                <th>Drift Status</th>
                <th>P-Value / Metric</th>
                <th>Model</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.id}>
                  <td>
                    <div className="id-with-copy">
                      <code>{r.id.slice(0, 8)}</code>
                      <CopyButton text={r.id} tooltip="Copy Report UUID" />
                    </div>
                  </td>
                  <td>
                    <span className="badge badge-neutral">{r.report_type}</span>
                  </td>
                  <td>
                    {r.detection_method === 'INSUFFICIENT_DATA' ||
                    r.summary_json?.evidence_source === 'INSUFFICIENT_DATA' ? (
                      <span className="badge badge-warning">
                        <AlertTriangle size={11} />
                        <span>Insufficient Data</span>
                      </span>
                    ) : r.summary_json?.evidence_source === 'SYNTHETIC_TEST' ? (
                      <span className={`badge ${r.is_drift_detected ? 'badge-error' : 'badge-neutral'}`}>
                        {r.is_drift_detected ? <AlertTriangle size={11} /> : <CheckCircle2 size={11} />}
                        <span>Synthetic {r.is_drift_detected ? 'Drift' : 'Baseline'}</span>
                      </span>
                    ) : r.is_drift_detected ? (
                      <span className="badge badge-error">
                        <AlertTriangle size={11} />
                        <span>Drift Detected</span>
                      </span>
                    ) : (
                      <span className="badge badge-success">
                        <CheckCircle2 size={11} />
                        <span>Stable</span>
                      </span>
                    )}
                  </td>
                  <td>
                    <code>{r.summary_json?.p_value != null ? r.summary_json.p_value.toFixed(4) : '—'}</code>
                  </td>
                  <td>
                    <code>{r.model_name} v{r.model_version}</code>
                  </td>
                  <td className="text-muted">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}

              {reports.length === 0 && (
                <tr>
                  <td colSpan={6} className="empty-table-cell">
                    No drift reports recorded yet. Click one of the drift check buttons above to execute an analysis.
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
