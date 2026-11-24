import React from 'react'
import { ExternalLink, Menu, RefreshCw, ShieldCheck } from 'lucide-react'

interface NavbarProps {
  onToggleMobile: () => void
  onRefresh?: () => void
  isRefreshing?: boolean
  healthStatus?: 'healthy' | 'degraded' | 'unhealthy' | string
  activeModelName?: string
  activeModelVersion?: string
}

export const Navbar: React.FC<NavbarProps> = ({
  onToggleMobile,
  onRefresh,
  isRefreshing = false,
  healthStatus = 'healthy',
  activeModelName = 'metro-traffic-volume',
  activeModelVersion = '1',
}) => {
  return (
    <header className="app-navbar" aria-label="Global Controls">
      <div className="navbar-left">
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={onToggleMobile}
          aria-label="Toggle navigation menu"
        >
          <Menu size={20} />
        </button>

        <div className="env-indicator">
          <span className="env-dot" />
          <span className="env-name">PRODUCTION CONTROL PLANE</span>
        </div>

        <div className="status-pill-wrap">
          <span className={`health-pill ${healthStatus === 'healthy' ? 'pill-healthy' : 'pill-alert'}`}>
            <ShieldCheck size={13} />
            <span>{healthStatus === 'healthy' ? 'CONTROL PLANE NOMINAL' : 'HEALTH DEGRADED'}</span>
          </span>
        </div>

        {activeModelVersion && (
          <div className="active-champion-tag">
            <span className="champion-label">CHAMPION:</span>
            <span className="champion-value">{activeModelName} v{activeModelVersion}</span>
          </div>
        )}
      </div>

      <div className="navbar-right">
        {onRefresh && (
          <button
            type="button"
            className="navbar-btn refresh-btn"
            onClick={onRefresh}
            disabled={isRefreshing}
            title="Refresh control plane state"
          >
            <RefreshCw size={13} className={isRefreshing ? 'spin-icon' : ''} />
            <span className="btn-label">{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        )}

        <div className="nav-divider" />

        <a
          href="http://127.0.0.1:5000"
          target="_blank"
          rel="noreferrer"
          className="navbar-link-btn"
          title="Open MLflow Tracking Server UI"
        >
          <span>MLflow</span>
          <ExternalLink size={12} />
        </a>

        <a
          href="http://127.0.0.1:3001"
          target="_blank"
          rel="noreferrer"
          className="navbar-link-btn"
          title="Open Grafana Dashboards"
        >
          <span>Grafana</span>
          <ExternalLink size={12} />
        </a>
      </div>
    </header>
  )
}
