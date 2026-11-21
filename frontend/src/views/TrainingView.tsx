import React, { useEffect, useState } from 'react'
import { ExternalLink, Zap } from 'lucide-react'
import { api } from '../api'
import type { TrainingJob } from '../types'

export const TrainingView: React.FC = () => {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const res = await api.getTrainingJobs()
        setJobs(res)
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div className="card">Loading training jobs...</div>
  if (error) return <div className="card badge-error">Failed to load training jobs: {error}</div>

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Training Jobs & Experiment Tracking</h1>
          <p className="page-description">Reproducible model training pipelines, baseline comparisons, and MLflow run provenance.</p>
        </div>
        <a
          href="http://127.0.0.1:5000"
          target="_blank"
          rel="noreferrer"
          className="btn btn-secondary"
        >
          <ExternalLink size={16} /> Open MLflow UI
        </a>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Executed Training Pipelines</span>
          <Zap size={16} color="#fbbf24" />
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Status</th>
                <th>Dataset Version ID</th>
                <th>MLflow Run ID</th>
                <th>Candidate MAE</th>
                <th>Baseline MAE</th>
                <th>Superiority</th>
                <th>Random Seed</th>
                <th>Execution Time</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const run = job.training_run
                const candMae = run?.metrics_json?.candidate_val_mae
                const baseMae = run?.metrics_json?.baseline_val_mae
                const improvement = candMae && baseMae ? ((baseMae - candMae) / baseMae) * 100 : null

                return (
                  <tr key={job.id}>
                    <td><code>{job.id.slice(0, 8)}</code></td>
                    <td>
                      <span className={`badge ${job.status === 'SUCCEEDED' ? 'badge-success' : job.status === 'RUNNING' ? 'badge-info' : 'badge-error'}`}>
                        {job.status}
                      </span>
                    </td>
                    <td><code>{job.dataset_version_id.slice(0, 8)}</code></td>
                    <td>
                      {run?.mlflow_run_id ? (
                        <a
                          href={`http://127.0.0.1:5000/#/experiments/0/runs/${run.mlflow_run_id}`}
                          target="_blank"
                          rel="noreferrer"
                          style={{ color: '#818cf8', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                        >
                          <code>{run.mlflow_run_id.slice(0, 8)}</code>
                          <ExternalLink size={12} />
                        </a>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td><strong>{candMae ? candMae.toFixed(1) : '—'}</strong></td>
                    <td style={{ color: 'var(--text-muted)' }}>{baseMae ? baseMae.toFixed(1) : '—'}</td>
                    <td>
                      {improvement !== null ? (
                        <span className="badge badge-success">+{improvement.toFixed(1)}%</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>{run?.random_seed ?? 42}</td>
                    <td>{new Date(job.created_at).toLocaleString()}</td>
                  </tr>
                )
              })}
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No training jobs found.
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
