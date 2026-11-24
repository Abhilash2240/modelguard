import React from 'react'
import { ExternalLink, LogIn, LogOut, Menu, RefreshCw, ShieldCheck, User } from 'lucide-react'
import type { UserProfile } from '../types'

interface NavbarProps {
  onToggleMobile: () => void
  onRefresh?: () => void
  isRefreshing?: boolean
  healthStatus?: 'healthy' | 'degraded' | 'unhealthy' | string
  activeModelName?: string
  activeModelVersion?: string
  currentUser?: UserProfile | null
  onOpenLogin?: () => void
  onLogout?: () => void
}

export const Navbar: React.FC<NavbarProps> = ({
  onToggleMobile,
  onRefresh,
  isRefreshing = false,
  healthStatus = 'healthy',
  activeModelName = 'metro-traffic-volume',
  activeModelVersion = '1',
  currentUser = null,
  onOpenLogin,
  onLogout,
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
        {currentUser ? (
          <div className="user-profile-badge">
            <span className="user-avatar-icon" title={`Authenticated as ${currentUser.username}`}>
              <User size={13} />
            </span>
            <span className="user-name" title={currentUser.username}>
              {currentUser.username}
            </span>
            <span className={`role-badge role-${currentUser.role.toLowerCase()}`} title={`Role: ${currentUser.role}`}>
              {currentUser.role}
            </span>
            {onLogout && (
              <button
                type="button"
                className="navbar-btn logout-btn"
                onClick={onLogout}
                title="Sign out of control plane"
              >
                <LogOut size={13} />
                <span className="btn-label">Logout</span>
              </button>
            )}
          </div>
        ) : (
          onOpenLogin && (
            <button
              type="button"
              className="navbar-btn login-nav-btn"
              onClick={onOpenLogin}
              title="Sign in to control plane"
            >
              <LogIn size={13} />
              <span className="btn-label">Sign In</span>
            </button>
          )
        )}

        <div className="nav-divider" />

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
