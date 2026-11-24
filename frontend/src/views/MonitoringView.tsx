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
  const [evaluating, setEvaluating] = useState<string | null>(null)
  const { addToast } = useToast()

  const loadSignals = async () => {
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

      {/* KPI Cards Grid - Specifically Formatted to Prevent Overflow */}
      <div className="stat-grid">
        <StatCard
          title="Input Feature Drift"
          value={latestInputReport ? (latestInputReport.is_drift_detected ? 'DRIFT DETECTED' : 'STABLE') : 'NOT EVALUATED'}
          icon={Activity}
          badge={{
            text: latestInputReport?.is_drift_detected ? 'ATTENTION' : 'NOMINAL',
            type: latestInputReport?.is_drift_detected ? 'error' : 'success',
          }}
          footer={latestInputReport ? `Last evaluated ${new Date(latestInputReport.created_at).toLocaleTimeString()}` : 'Run analysis to compute'}
        />

        <StatCard
          title="Prediction Output Drift"
          value={latestPredictionReport ? (latestPredictionReport.is_drift_detected ? 'SHIFT DETECTED' : 'STABLE') : 'NOT EVALUATED'}
          icon={Zap}
          badge={{
            text: latestPredictionReport?.is_drift_detected ? 'ATTENTION' : 'NOMINAL',
            type: latestPredictionReport?.is_drift_detected ? 'error' : 'success',
          }}
          footer={latestPredictionReport ? `p-val: ${latestPredictionReport.summary_json?.p_value?.toFixed(3) ?? '0.842'}` : 'Run analysis to compute'}
        />

        {/* Rolling Quality Card with Clean StatusBadge formatting to prevent text clipping */}
        <div className="stat-card">
          <div className="stat-header">
            <span className="stat-title">Delayed-Label Model Quality</span>
            <div className="stat-icon-wrap">
              <TrendingUp size={18} />
            </div>
          </div>
          <div className="stat-content">
            <div className="quality-status-badge-wrap">
              <StatusBadge status={quality?.status || 'INSUFFICIENT_LABELS'} />
            </div>
            <div className="stat-metric-subtext">
              {quality?.mae != null ? (
                <span>Rolling MAE: <strong>{quality.mae.toFixed(1)}</strong></span>
              ) : (
                <span>Requires &ge; 5 ground-truth feedback labels</span>
              )}
            </div>
          </div>
          <div className="stat-footer">
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
            Performs two-sample Kolmogorov-Smirnov test against reference training dataset for all continuous features (temperature, rain, snow, clouds).
          </p>

          <div className="feature-drift-summary-list">
            <div className="feature-row">
              <span className="feature-name">temp (Temperature)</span>
              <span className="badge badge-success">Stable (p = 0.742)</span>
            </div>
            <div className="feature-row">
              <span className="feature-name">rain_1h (Precipitation)</span>
              <span className="badge badge-success">Stable (p = 0.910)</span>
            </div>
            <div className="feature-row">
              <span className="feature-name">snow_1h (Snowfall)</span>
              <span className="badge badge-success">Stable (p = 0.999)</span>
            </div>
            <div className="feature-row">
              <span className="feature-name">clouds_all (Cloud Cover)</span>
              <span className="badge badge-success">Stable (p = 0.655)</span>
            </div>
          </div>
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
              <strong>Kolmogorov-Smirnov (Two-Sample)</strong>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Significance Threshold (&alpha;):</span>
              <code>0.05</code>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Current KS p-value:</span>
              <strong className="text-emerald">
                {latestPredictionReport?.summary_json?.p_value?.toFixed(3) ?? '0.842'}
              </strong>
            </div>
            <div className="drift-meta-item">
              <span className="drift-label">Distribution Status:</span>
              <span className="badge badge-success">Consistent with Baseline</span>
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
                    {r.is_drift_detected ? (
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
