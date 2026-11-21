import React, { useState } from 'react'
import { Cpu, MessageSquare, Play, Send } from 'lucide-react'
import { api } from '../api'
import type { PredictionResponse } from '../types'

const DEFAULT_SAMPLE_PAYLOAD = [
  {
    holiday: 'None',
    temp: 288.15,
    rain_1h: 0.0,
    snow_1h: 0.0,
    clouds_all: 40,
    weather_main: 'Clouds',
    weather_description: 'scattered clouds',
    date_time: '2026-11-21 09:00:00',
  },
]

export const InferenceView: React.FC = () => {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(DEFAULT_SAMPLE_PAYLOAD, null, 2))
  const [requestedVersion, setRequestedVersion] = useState('')
  const [loading, setLoading] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Delayed feedback state
  const [actualValue, setActualValue] = useState<string>('')
  const [feedbackSuccess, setFeedbackSuccess] = useState<string | null>(null)
  const [submittingFeedback, setSubmittingFeedback] = useState(false)

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault()
    setErrorMsg(null)
    setPredictionResult(null)
    setFeedbackSuccess(null)
    setLoading(true)

    try {
      let parsedInputs: any
      try {
        parsedInputs = JSON.parse(jsonInput)
      } catch {
        throw new Error('Input payload must be valid JSON array of feature records.')
      }

      if (!Array.isArray(parsedInputs)) {
        throw new Error('Input payload must be a JSON array.')
      }

      const res = await api.predict({
        inputs: parsedInputs,
        model_version: requestedVersion.trim() || undefined,
      })
      setPredictionResult(res)
    } catch (err: any) {
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleFeedbackSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!predictionResult || !actualValue) return
    setSubmittingFeedback(true)
    setFeedbackSuccess(null)
    setErrorMsg(null)

    try {
      const val = parseFloat(actualValue)
      if (isNaN(val)) throw new Error('Actual value must be a valid number.')

      await api.submitFeedback({
        inference_id: predictionResult.inference_id,
        actual_value: val,
        submitted_by: 'dashboard_operator',
        notes: 'Manual feedback submission from inference UI',
      })
      setFeedbackSuccess(`Successfully recorded ground truth (${val}) for inference ${predictionResult.inference_id.slice(0, 8)}!`)
      setActualValue('')
    } catch (err: any) {
      setErrorMsg(`Feedback submission failed: ${err.message}`)
    } finally {
      setSubmittingFeedback(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Version-Aware Inference & Telemetry</h1>
          <p className="page-description">Execute real-time tabular model predictions with schema contract validation and privacy-safe logging.</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem' }}>
        <div className="card">
          <div className="card-title">
            <span>Inference Request Payload</span>
            <Cpu size={16} />
          </div>

          <form onSubmit={handlePredict}>
            <div className="form-group">
              <label className="form-label">Target Model Version (Optional — defaults to active Production Champion)</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. 1 (leave blank for active champion)"
                value={requestedVersion}
                onChange={(e) => setRequestedVersion(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Tabular Features JSON (Array of Records)</label>
              <textarea
                className="form-textarea"
                style={{ height: '220px' }}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setJsonInput(JSON.stringify(DEFAULT_SAMPLE_PAYLOAD, null, 2))}
              >
                Reset Sample
              </button>
              <button type="submit" className="btn btn-primary" disabled={loading}>
                <Play size={14} /> {loading ? 'Running Estimator...' : 'Execute Prediction'}
              </button>
            </div>
          </form>
        </div>

        <div>
          {errorMsg && (
            <div className="card badge-error" style={{ marginBottom: '1rem' }}>
              <strong>Error:</strong> {errorMsg}
            </div>
          )}

          {predictionResult ? (
            <div className="card">
              <div className="card-title">
                <span>Inference Output & Telemetry</span>
                <span className="badge badge-success">200 OK</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem', background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '6px' }}>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>INFERENCE ID</div>
                  <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#06b6d4' }}>
                    {predictionResult.inference_id}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MODEL VERSION</div>
                  <div style={{ fontWeight: 600 }}>v{predictionResult.model_version}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SERVING LATENCY</div>
                  <div style={{ fontWeight: 600, color: '#34d399' }}>{predictionResult.latency_ms.toFixed(2)} ms</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TIMESTAMP</div>
                  <div style={{ fontSize: '0.75rem' }}>{new Date(predictionResult.timestamp).toLocaleTimeString()}</div>
                </div>
              </div>

              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.5rem' }}>Predicted Traffic Volume</div>
                <div style={{ background: '#0b0f19', padding: '0.75rem', borderRadius: '6px', fontFamily: 'monospace', fontSize: '1.25rem', color: '#818cf8', fontWeight: 700 }}>
                  {predictionResult.predictions.map((p, idx) => (
                    <div key={idx}>Record {idx + 1}: {p.toFixed(1)} vehicles/hr</div>
                  ))}
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <MessageSquare size={14} /> Submit Delayed Ground Truth Feedback
                </div>
                <form onSubmit={handleFeedbackSubmit} style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    type="number"
                    step="any"
                    className="form-input"
                    placeholder="Actual traffic volume"
                    value={actualValue}
                    onChange={(e) => setActualValue(e.target.value)}
                    required
                  />
                  <button type="submit" className="btn btn-secondary" disabled={submittingFeedback}>
                    <Send size={12} /> {submittingFeedback ? 'Submitting...' : 'Submit'}
                  </button>
                </form>
                {feedbackSuccess && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#34d399' }}>
                    {feedbackSuccess}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
              Submit an inference request to view live prediction results, model version resolution, and latency telemetry.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
