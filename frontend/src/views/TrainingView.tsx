import React, { useEffect, useMemo, useState } from 'react'
import {
  CheckCircle2,
  ExternalLink,
  Filter,
  Sparkles,
  TrendingDown,
  Zap,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { StatGridSkeleton, TableSkeleton } from '../components/Skeleton'
import { StatCard } from '../components/StatCard'
import { StatusBadge } from '../components/StatusBadge'
import type { TrainingJob } from '../types'

interface TrainingViewProps {
  refreshTrigger?: number
}

export const TrainingView: React.FC<TrainingViewProps> = ({ refreshTrigger }) => {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('ALL')

  useEffect(() => {
    let ignore = false
    async function load() {
      try {
        setLoading(true)
        setError(null)
        const res = await api.getTrainingJobs()
        if (!ignore) setJobs(res)
      } catch (err: any) {
        if (!ignore) setError(err.message || 'Failed to load training jobs.')
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [refreshTrigger])

  const filteredJobs = useMemo(() => {
    if (statusFilter === 'ALL') return jobs
    return jobs.filter((j) => j.status === statusFilter)
  }, [jobs, statusFilter])

  // Aggregate training KPIs
  const successfulJobs = jobs.filter((j) => j.status === 'SUCCEEDED')
  const candidateMaes = successfulJobs
    .map((j) => j.training_run?.metrics_json?.candidate_val_mae)
    .filter((mae): mae is number => typeof mae === 'number' && !isNaN(mae))

  const bestCandidateMae = candidateMaes.length > 0 ? Math.min(...candidateMaes) : null

  const improvements = successfulJobs
    .map((j) => {
      const c = j.training_run?.metrics_json?.candidate_val_mae
      const b = j.training_run?.metrics_json?.baseline_val_mae
      return c && b ? ((b - c) / b) * 100 : null
    })
    .filter((imp): imp is number => typeof imp === 'number' && !isNaN(imp))

  const maxImprovement = improvements.length > 0 ? Math.max(...improvements) : null

  if (loading && jobs.length === 0) {
    return (
      <div className="view-container">
        <div className="page-header">
          <div>
            <h1 className="page-title">Training Jobs & Experiment Tracking</h1>
            <p className="page-description">Reproducible model training pipelines, baseline comparisons, and MLflow run provenance.</p>
          </div>
        </div>
        <StatGridSkeleton count={4} />
        <TableSkeleton rows={5} />
      </div>
    )
  }

  return (
    <div className="view-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Training Jobs & Experiment Tracking</h1>
          <p className="page-description">
            Reproducible pipelines, deterministic random seeds, MLflow experiment provenance, and candidate vs baseline superiority tracking.
          </p>
        </div>
        <div className="header-actions">
          <a
            href="http://127.0.0.1:5000"
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
            title="Inspect MLflow Run Artifacts and Parameters"
          >
            <ExternalLink size={14} />
            <span>Open MLflow UI</span>
          </a>
        </div>
      </div>

      {error && (
        <div className="card badge-error" style={{ padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
          <strong>Error loading training jobs:</strong> {error}
        </div>
      )}

      {/* KPI Grid */}
      <div className="stat-grid">
        <StatCard
          title="Total Pipeline Runs"
          value={String(jobs.length)}
          icon={Zap}
          badge={{ text: 'AUDITED', type: 'info' }}
          footer="Triggered via CLI / Automated Orchestration"
        />

        <StatCard
          title="Successful Completions"
          value={String(successfulJobs.length)}
          icon={CheckCircle2}
          badge={{
            text: `${jobs.length > 0 ? Math.round((successfulJobs.length / jobs.length) * 100) : 100}% SUCCESS`,
            type: 'success',
          }}
          footer="All artifacts stored in MLflow registry"
        />

        <StatCard
          title="Best Candidate MAE"
          value={bestCandidateMae != null ? bestCandidateMae.toFixed(1) : '—'}
          icon={TrendingDown}
          badge={{ text: 'SUPERIOR', type: 'success' }}
          footer="Target metric: lower is superior"
        />

        <StatCard
          title="Max Baseline Improvement"
          value={maxImprovement != null ? `+${maxImprovement.toFixed(1)}%` : '—'}
          icon={Sparkles}
          badge={{ text: 'OUTPERFORMING', type: 'success' }}
          footer="Compared against historical baseline"
        />
      </div>

      {/* Main Table Card */}
      <div className="card">
        <div className="card-title">
          <div className="title-with-badge">
            <Zap size={16} className="text-amber" />
            <span>Executed Training Pipelines</span>
            <span className="counter-pill">{filteredJobs.length}</span>
          </div>

          <div className="filter-pill-group">
            <span className="filter-label">
              <Filter size={12} />
              <span>Status:</span>
            </span>
            {['ALL', 'SUCCEEDED', 'RUNNING', 'FAILED'].map((st) => (
              <button
                key={st}
                type="button"
                className={`filter-pill ${statusFilter === st ? 'active' : ''}`}
                onClick={() => setStatusFilter(st)}
              >
                {st}
              </button>
            ))}
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Status</th>
                <th>Dataset Version</th>
                <th>MLflow Run ID</th>
                <th>Candidate MAE</th>
                <th>Baseline MAE</th>
                <th>Superiority</th>
                <th>Seed</th>
                <th>Created At</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => {
                const run = job.training_run
                const candMae = run?.metrics_json?.candidate_val_mae
                const baseMae = run?.metrics_json?.baseline_val_mae
                const improvement =
                  candMae && baseMae ? ((baseMae - candMae) / baseMae) * 100 : null

                return (
                  <tr key={job.id}>
                    <td>
                      <div className="id-with-copy">
                        <code>{job.id.slice(0, 8)}</code>
                        <CopyButton text={job.id} tooltip="Copy Training Job UUID" />
                      </div>
                    </td>
                    <td>
                      <StatusBadge status={job.status} />
                    </td>
                    <td>
                      <div className="id-with-copy">
                        <code>{job.dataset_version_id.slice(0, 8)}</code>
                        <CopyButton text={job.dataset_version_id} tooltip="Copy Dataset Version UUID" />
                      </div>
                    </td>
                    <td>
                      {run?.mlflow_run_id ? (
                        <div className="id-with-copy">
                          <a
                            href={`http://127.0.0.1:5000/#/experiments/0/runs/${run.mlflow_run_id}`}
                            target="_blank"
                            rel="noreferrer"
                            className="mlflow-link"
                            title="Inspect in MLflow UI"
                          >
                            <code>{run.mlflow_run_id.slice(0, 8)}</code>
                            <ExternalLink size={11} />
                          </a>
                          <CopyButton text={run.mlflow_run_id} tooltip="Copy MLflow Run ID" />
                        </div>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                    <td>
                      <strong className="metric-highlight">
                        {candMae != null ? candMae.toFixed(1) : '—'}
                      </strong>
                    </td>
                    <td>
                      <span className="text-muted">
                        {baseMae != null ? baseMae.toFixed(1) : '—'}
                      </span>
                    </td>
                    <td>
                      {improvement != null ? (
                        <span className={`badge ${improvement > 0 ? 'badge-success' : 'badge-error'}`}>
                          {improvement > 0 ? `+${improvement.toFixed(1)}%` : `${improvement.toFixed(1)}%`}
                        </span>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                    <td>
                      <code className="seed-badge">{run?.random_seed ?? 42}</code>
                    </td>
                    <td className="text-muted">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                  </tr>
                )
              })}

              {filteredJobs.length === 0 && (
                <tr>
                  <td colSpan={9} className="empty-table-cell">
                    No training jobs match the selected filter.
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
