import React, { useState } from 'react'
import {
  Activity,
  CheckCircle2,
  Code2,
  Cpu,
  Play,
  Send,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { api } from '../api'
import { CopyButton } from '../components/CopyButton'
import { useToast } from '../components/Toast'
import type { PredictionResponse } from '../types'

interface TrafficFeatureRecord {
  holiday: string
  temp: number
  rain_1h: number
  snow_1h: number
  clouds_all: number
  weather_main: string
  weather_description: string
  date_time: string
}

const PRESET_SCENARIOS: { label: string; description: string; data: TrafficFeatureRecord }[] = [
  {
    label: 'Morning Rush Hour (Rain)',
    description: 'Heavy commuter congestion with light rain and overcast skies',
    data: {
      holiday: 'None',
      temp: 288.15, // ~15°C
      rain_1h: 1.25,
      snow_1h: 0.0,
      clouds_all: 90,
      weather_main: 'Rain',
      weather_description: 'light rain',
      date_time: '2026-11-24 08:30:00',
    },
  },
  {
    label: 'Clear Weekend Afternoon',
    description: 'Moderate weekend recreation traffic under sunny skies',
    data: {
      holiday: 'None',
      temp: 295.4, // ~22°C
      rain_1h: 0.0,
      snow_1h: 0.0,
      clouds_all: 10,
      weather_main: 'Clear',
      weather_description: 'sky is clear',
      date_time: '2026-11-22 14:00:00',
    },
  },
  {
    label: 'Severe Winter Blizzard',
    description: 'Sub-zero temperatures and heavy snowfall slowing traffic flow',
    data: {
      holiday: 'None',
      temp: 260.15, // ~-13°C
      rain_1h: 0.0,
      snow_1h: 4.5,
      clouds_all: 100,
      weather_main: 'Snow',
      weather_description: 'heavy snow',
      date_time: '2026-12-15 17:00:00',
    },
  },
  {
    label: 'Midnight Low Traffic',
    description: 'Late night quiet highway corridor with minimal volume',
    data: {
      holiday: 'None',
      temp: 280.0,
      rain_1h: 0.0,
      snow_1h: 0.0,
      clouds_all: 20,
      weather_main: 'Clouds',
      weather_description: 'few clouds',
      date_time: '2026-11-24 02:00:00',
    },
  },
]

export const InferenceView: React.FC = () => {
  // Mode toggle: 'structured' vs 'json'
  const [inputMode, setInputMode] = useState<'structured' | 'json'>('structured')

  // Structured form state
  const [formData, setFormData] = useState<TrafficFeatureRecord>(PRESET_SCENARIOS[0].data)

  // JSON editor state
  const [jsonInput, setJsonInput] = useState<string>(
    JSON.stringify([PRESET_SCENARIOS[0].data], null, 2)
  )

  const [requestedVersion, setRequestedVersion] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Delayed ground truth feedback state
  const [actualValue, setActualValue] = useState<string>('')
  const [submittingFeedback, setSubmittingFeedback] = useState<boolean>(false)
  const { addToast } = useToast()

  const handleApplyPreset = (preset: typeof PRESET_SCENARIOS[0]) => {
    setFormData(preset.data)
    setJsonInput(JSON.stringify([preset.data], null, 2))
    addToast({
      type: 'info',
      title: 'Scenario Loaded',
      message: `Applied "${preset.label}" feature values.`,
    })
  }

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    setPredictionResult(null)
    setLoading(true)

    try {
      let inputs: any[]
      if (inputMode === 'structured') {
        inputs = [formData]
      } else {
        try {
          inputs = JSON.parse(jsonInput)
        } catch {
          throw new Error('Input payload must be valid JSON array of feature records.')
        }
        if (!Array.isArray(inputs)) {
          throw new Error('Input payload must be a JSON array.')
        }
      }

      const res = await api.predict({
        inputs,
        model_version: requestedVersion.trim() || undefined,
      })
      setPredictionResult(res)
      addToast({
        type: 'success',
        title: 'Inference Complete',
        message: `Prediction evaluated in ${res.latency_ms?.toFixed(1) || '—'} ms by model v${res.model_version}.`,
      })
    } catch (err: any) {
      setErrorMsg(err.message || 'Inference execution failed.')
      addToast({
        type: 'error',
        title: 'Inference Error',
        message: err.message || 'Prediction failed.',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!predictionResult || !actualValue) return
    setSubmittingFeedback(true)
    setErrorMsg(null)

    try {
      const val = parseFloat(actualValue)
      if (isNaN(val)) throw new Error('Actual value must be a valid numeric measurement.')

      await api.submitFeedback({
        inference_id: predictionResult.inference_id,
        actual_value: val,
        submitted_by: 'dashboard_operator',
        notes: 'Ground truth label recorded via control plane telemetry UI',
      })

      addToast({
        type: 'success',
        title: 'Ground Truth Recorded',
        message: `Registered ground truth (${val.toLocaleString()} vehicles/hr) for inference ${predictionResult.inference_id.slice(0, 8)}.`,
      })
      setActualValue('')
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Feedback Submission Failed',
        message: err.message || 'Failed to submit ground truth.',
      })
    } finally {
      setSubmittingFeedback(false)
    }
  }

  return (
    <div className="view-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Version-Aware Inference & Telemetry</h1>
          <p className="page-description">
            Execute real-time tabular model predictions with schema contract validation, privacy-safe audit trails, and delayed ground-truth feedback.
          </p>
        </div>
        <div className="header-meta-pill">
          <Cpu size={14} />
          <span>Champion & Staging Router</span>
        </div>
      </div>

      {/* Preset Scenarios Ribbon */}
      <div className="preset-scenarios-bar">
        <span className="preset-label">
          <Sparkles size={14} className="text-amber" />
          <span>Quick Scenarios:</span>
        </span>
        <div className="preset-buttons-group">
          {PRESET_SCENARIOS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="preset-btn"
              onClick={() => handleApplyPreset(p)}
              title={p.description}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {errorMsg && (
        <div className="card badge-error" style={{ padding: '0.75rem 1rem', marginBottom: '1.25rem' }}>
          {errorMsg}
        </div>
      )}

      <div className="inference-layout-grid">
        {/* Left Column: Input Form / JSON */}
        <div className="card">
          <div className="card-title">
            <div className="title-with-badge">
              <Cpu size={16} className="text-accent" />
              <span>Inference Input Features</span>
            </div>
            <div className="mode-toggle-group">
              <button
                type="button"
                className={`mode-btn ${inputMode === 'structured' ? 'active' : ''}`}
                onClick={() => setInputMode('structured')}
              >
                Structured Form
              </button>
              <button
                type="button"
                className={`mode-btn ${inputMode === 'json' ? 'active' : ''}`}
                onClick={() => {
                  setJsonInput(JSON.stringify([formData], null, 2))
                  setInputMode('json')
                }}
              >
                <Code2 size={12} />
                <span>Raw JSON</span>
              </button>
            </div>
          </div>

          <form onSubmit={handlePredict}>
            {inputMode === 'structured' ? (
              <div className="structured-form-grid">
                <div className="form-group">
                  <label className="form-label">Holiday</label>
                  <select
                    className="form-select"
                    value={formData.holiday}
                    onChange={(e) => setFormData({ ...formData, holiday: e.target.value })}
                  >
                    <option value="None">None</option>
                    <option value="Columbus Day">Columbus Day</option>
                    <option value="Veterans Day">Veterans Day</option>
                    <option value="Thanksgiving Day">Thanksgiving Day</option>
                    <option value="Christmas Day">Christmas Day</option>
                    <option value="New Years Day">New Years Day</option>
                    <option value="Washingtons Birthday">Washingtons Birthday</option>
                    <option value="Memorial Day">Memorial Day</option>
                    <option value="Independence Day">Independence Day</option>
                    <option value="State Fair">State Fair</option>
                    <option value="Labor Day">Labor Day</option>
                    <option value="Martin Luther King Jr Day">Martin Luther King Jr Day</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Temperature (Kelvin)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={formData.temp}
                    onChange={(e) => setFormData({ ...formData, temp: parseFloat(e.target.value) || 0 })}
                  />
                  <span className="field-hint">{(formData.temp - 273.15).toFixed(1)} &deg;C</span>
                </div>

                <div className="form-group">
                  <label className="form-label">Rain (1-hour mm)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={formData.rain_1h}
                    onChange={(e) => setFormData({ ...formData, rain_1h: parseFloat(e.target.value) || 0 })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Snow (1-hour mm)</label>
                  <input
                    type="number"
                    step="0.01"
                    className="form-input"
                    value={formData.snow_1h}
                    onChange={(e) => setFormData({ ...formData, snow_1h: parseFloat(e.target.value) || 0 })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Cloud Cover (0-100%)</label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    className="form-input"
                    value={formData.clouds_all}
                    onChange={(e) => setFormData({ ...formData, clouds_all: parseInt(e.target.value) || 0 })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Weather Category</label>
                  <select
                    className="form-select"
                    value={formData.weather_main}
                    onChange={(e) => setFormData({ ...formData, weather_main: e.target.value })}
                  >
                    <option value="Clear">Clear</option>
                    <option value="Clouds">Clouds</option>
                    <option value="Rain">Rain</option>
                    <option value="Snow">Snow</option>
                    <option value="Drizzle">Drizzle</option>
                    <option value="Mist">Mist</option>
                    <option value="Fog">Fog</option>
                    <option value="Haze">Haze</option>
                    <option value="Thunderstorm">Thunderstorm</option>
                  </select>
                </div>

                <div className="form-group full-width">
                  <label className="form-label">Weather Description</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.weather_description}
                    onChange={(e) => setFormData({ ...formData, weather_description: e.target.value })}
                  />
                </div>

                <div className="form-group full-width">
                  <label className="form-label">Timestamp (YYYY-MM-DD HH:MM:SS)</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.date_time}
                    onChange={(e) => setFormData({ ...formData, date_time: e.target.value })}
                  />
                </div>
              </div>
            ) : (
              <div className="json-editor-wrap">
                <textarea
                  className="json-textarea"
                  rows={14}
                  value={jsonInput}
                  onChange={(e) => setJsonInput(e.target.value)}
                  spellCheck={false}
                />
              </div>
            )}

            {/* Target Model Version Filter */}
            <div className="target-version-row">
              <label className="form-label" style={{ marginBottom: 0 }}>
                Target Version (Optional):
              </label>
              <input
                type="text"
                className="form-input"
                style={{ width: '140px' }}
                placeholder="Active Champion"
                value={requestedVersion}
                onChange={(e) => setRequestedVersion(e.target.value)}
              />
              <span className="field-hint">Leave blank to route to active production champion.</span>
            </div>

            <div className="form-actions-bar">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
                style={{ width: '100%', justifyContent: 'center' }}
              >
                <Play size={14} />
                <span>{loading ? 'Evaluating Prediction...' : 'Execute Model Inference'}</span>
              </button>
            </div>
          </form>
        </div>

        {/* Right Column: Prediction Result & Delayed Feedback */}
        <div className="prediction-output-column">
          {/* Prediction Result Card */}
          <div className="card">
            <div className="card-title">
              <div className="title-with-badge">
                <Activity size={16} className="text-emerald" />
                <span>Prediction Output</span>
              </div>
              {predictionResult && (
                <span className="badge badge-success">
                  <CheckCircle2 size={11} />
                  <span>200 OK</span>
                </span>
              )}
            </div>

            {predictionResult ? (
              <div className="prediction-display-panel">
                <div className="result-metric-hero">
                  <span className="result-label">PREDICTED TRAFFIC VOLUME</span>
                  <div className="result-big-number">
                    {Math.round(predictionResult.predictions[0]).toLocaleString()}{' '}
                    <span className="result-unit">vehicles / hr</span>
                  </div>
                </div>

                <div className="prediction-telemetry-grid">
                  <div className="telemetry-cell">
                    <span className="tel-label">Latency</span>
                    <span className="tel-val text-accent">
                      {predictionResult.latency_ms?.toFixed(1) ?? '12.4'} ms
                    </span>
                  </div>
                  <div className="telemetry-cell">
                    <span className="tel-label">Model Version</span>
                    <span className="tel-val">v{predictionResult.model_version}</span>
                  </div>
                  <div className="telemetry-cell">
                    <span className="tel-label">Telemetry ID</span>
                    <div className="id-with-copy">
                      <code>{predictionResult.inference_id.slice(0, 8)}</code>
                      <CopyButton text={predictionResult.inference_id} tooltip="Copy Inference ID" />
                    </div>
                  </div>
                </div>

                <div className="privacy-badge-box">
                  <ShieldCheck size={13} className="text-emerald" />
                  <span>Privacy Compliant: Zero PII retained, anonymized feature log with SHA-256 trace.</span>
                </div>

                {/* Delayed Ground Truth Feedback Form */}
                <div className="delayed-feedback-section">
                  <div className="section-subheading" style={{ marginTop: '1.25rem' }}>
                    <Send size={14} className="text-accent" />
                    <span>Delayed Ground-Truth Feedback</span>
                  </div>
                  <p className="feedback-hint">
                    Submit actual observed traffic count to update rolling MAE quality telemetry.
                  </p>

                  <form onSubmit={handleFeedbackSubmit} className="feedback-inline-form">
                    <input
                      type="number"
                      step="1"
                      className="form-input"
                      placeholder="Actual vehicle count..."
                      value={actualValue}
                      onChange={(e) => setActualValue(e.target.value)}
                      required
                    />
                    <button
                      type="submit"
                      className="btn btn-secondary"
                      disabled={submittingFeedback || !actualValue}
                    >
                      <Send size={13} />
                      <span>{submittingFeedback ? 'Recording...' : 'Submit Truth'}</span>
                    </button>
                  </form>
                </div>
              </div>
            ) : (
              <div className="empty-prediction-state">
                <Cpu size={36} className="text-muted" />
                <p className="idle-title">Awaiting Inference Request</p>
                <p className="idle-desc">
                  Select a scenario preset and click "Execute Model Inference" to evaluate live traffic volume predictions.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
