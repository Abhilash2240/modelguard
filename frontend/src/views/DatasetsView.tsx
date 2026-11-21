import React, { useEffect, useState } from 'react'
import { Database, FileText, Hash } from 'lucide-react'
import { api } from '../api'
import type { Dataset } from '../types'

export const DatasetsView: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const d = await api.getDatasets()
        setDatasets(d)
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div className="card">Loading registered datasets...</div>
  if (error) return <div className="card badge-error">Failed to load datasets: {error}</div>

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Dataset Provenance & Verification</h1>
          <p className="page-description">Immutable data versions, SHA-256 provenance hashes, and contract validation audit logs.</p>
        </div>
      </div>

      {datasets.map((dataset) => (
        <div key={dataset.id} className="card" style={{ marginBottom: '2rem' }}>
          <div className="card-title">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Database size={20} color="#818cf8" />
              <span>{dataset.name}</span>
              <span className="badge badge-info">{dataset.task_type}</span>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>ID: {dataset.id}</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SOURCE URI</div>
              <div style={{ fontSize: '0.875rem', wordBreak: 'break-all' }}>{dataset.source_uri}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>LICENCE</div>
              <div style={{ fontSize: '0.875rem' }}>{dataset.licence}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>REGISTERED AT</div>
              <div style={{ fontSize: '0.875rem' }}>{new Date(dataset.created_at).toLocaleString()}</div>
            </div>
          </div>

          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={16} /> Immutable Dataset Versions & Checksums
          </h3>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Rows</th>
                  <th>Size</th>
                  <th>SHA-256 Provenance Digest</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {dataset.versions && dataset.versions.length > 0 ? (
                  dataset.versions.map((v) => (
                    <tr key={v.id}>
                      <td><span className="badge badge-neutral">{v.version_tag}</span></td>
                      <td>{v.row_count.toLocaleString()}</td>
                      <td>{(v.byte_size / 1024).toFixed(1)} KB</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontFamily: 'monospace', fontSize: '0.75rem', color: '#06b6d4' }}>
                          <Hash size={12} />
                          <span>{v.sha256_hash}</span>
                        </div>
                      </td>
                      <td>{new Date(v.created_at).toLocaleString()}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No versions found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {datasets.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
          No datasets currently registered in ModelGuard.
        </div>
      )}
    </div>
  )
}
