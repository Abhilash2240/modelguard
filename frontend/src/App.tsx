import React, { useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  ExternalLink,
  GitBranch,
  LayoutDashboard,
  ShieldCheck,
  Zap,
} from 'lucide-react'
import { AlertsView } from './views/AlertsView'
import { DatasetsView } from './views/DatasetsView'
import { DeploymentsView } from './views/DeploymentsView'
import { InferenceView } from './views/InferenceView'
import { ModelsView } from './views/ModelsView'
import { MonitoringView } from './views/MonitoringView'
import { OverviewView } from './views/OverviewView'
import { TrainingView } from './views/TrainingView'

type Tab = 'overview' | 'datasets' | 'training' | 'models' | 'deployments' | 'inference' | 'monitoring' | 'alerts'

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('overview')

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-icon">MG</div>
          <div>
            <div className="brand-title">ModelGuard</div>
            <div className="brand-subtitle">Reliability Control Plane</div>
          </div>
        </div>

        <nav className="nav-menu">
          <button
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <LayoutDashboard size={18} />
            <span>Overview</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'datasets' ? 'active' : ''}`}
            onClick={() => setActiveTab('datasets')}
          >
            <Database size={18} />
            <span>Datasets</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveTab('training')}
          >
            <Zap size={18} />
            <span>Training & Runs</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
          >
            <GitBranch size={18} />
            <span>Model Registry</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'deployments' ? 'active' : ''}`}
            onClick={() => setActiveTab('deployments')}
          >
            <ShieldCheck size={18} />
            <span>Deployments</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'inference' ? 'active' : ''}`}
            onClick={() => setActiveTab('inference')}
          >
            <Cpu size={18} />
            <span>Inference & Telemetry</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'monitoring' ? 'active' : ''}`}
            onClick={() => setActiveTab('monitoring')}
          >
            <Activity size={18} />
            <span>Monitoring & Drift</span>
          </button>

          <button
            className={`nav-item ${activeTab === 'alerts' ? 'active' : ''}`}
            onClick={() => setActiveTab('alerts')}
          >
            <AlertTriangle size={18} />
            <span>Alerts & Audit</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div>ModelGuard v0.1.0</div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginTop: '0.25rem' }}>
            Production Reliability Engine
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span className="badge badge-success" style={{ gap: '0.35rem' }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34d399' }} />
              CONTROL PLANE ACTIVE
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <a
              href="http://127.0.0.1:5000"
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
            >
              <ExternalLink size={12} /> MLflow Tracking
            </a>
            <a
              href="http://127.0.0.1:3001"
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
            >
              <ExternalLink size={12} /> Grafana Telemetry
            </a>
          </div>
        </header>

        <div className="content-body">
          {activeTab === 'overview' && <OverviewView />}
          {activeTab === 'datasets' && <DatasetsView />}
          {activeTab === 'training' && <TrainingView />}
          {activeTab === 'models' && <ModelsView />}
          {activeTab === 'deployments' && <DeploymentsView />}
          {activeTab === 'inference' && <InferenceView />}
          {activeTab === 'monitoring' && <MonitoringView />}
          {activeTab === 'alerts' && <AlertsView />}
        </div>
      </main>
    </div>
  )
}

export default App
