import React, { useState } from 'react'
import { AlertCircle, Key, Lock, LogIn, Shield, User } from 'lucide-react'
import { api } from '../api'
import type { UserProfile } from '../types'
import { Modal } from './Modal'

interface LoginModalProps {
  isOpen: boolean
  onClose: () => void
  onLoginSuccess: (user: UserProfile) => void
}

export const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose, onLoginSuccess }) => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) {
      setError('Username and password are required.')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const res = await api.login({ username: username.trim(), password })
      onLoginSuccess(res.user)
      setUsername('')
      setPassword('')
    } catch (err: any) {
      setError(err?.message || 'Authentication failed. Please verify your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal isOpen={isOpen} title="Control Plane Authentication" onClose={onClose} maxWidth="460px">
      <form onSubmit={handleSubmit} className="login-form">
        <div className="login-intro">
          <div className="login-icon-badge">
            <Shield size={28} className="shield-icon" />
          </div>
          <p className="login-desc">
            Sign in with your ModelGuard credentials to access protected endpoints, trigger drift analyses, and approve model promotions.
          </p>
        </div>

        {error && (
          <div className="login-error-alert" role="alert">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <div className="form-group">
          <label htmlFor="login-username" className="form-label">
            <User size={14} />
            <span>Username</span>
          </label>
          <input
            id="login-username"
            type="text"
            className="form-input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. admin or operator"
            autoComplete="username"
            autoFocus
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="login-password" className="form-label">
            <Lock size={14} />
            <span>Password</span>
          </label>
          <input
            id="login-password"
            type="password"
            className="form-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter password"
            autoComplete="current-password"
            disabled={loading}
          />
        </div>

        <div className="login-rbac-hint">
          <div className="hint-title">
            <Key size={12} />
            <span>Role-Based Access Control</span>
          </div>
          <div className="hint-grid">
            <div className="hint-item">
              <span className="role-tag role-admin">ADMIN</span>
              <span className="hint-text">Full lifecycle + user admin</span>
            </div>
            <div className="hint-item">
              <span className="role-tag role-approver">APPROVER</span>
              <span className="hint-text">Stage, approve & promote</span>
            </div>
            <div className="hint-item">
              <span className="role-tag role-operator">OPERATOR</span>
              <span className="hint-text">Monitoring, drift & rollback</span>
            </div>
            <div className="hint-item">
              <span className="role-tag role-viewer">VIEWER</span>
              <span className="hint-text">Read-only + inference</span>
            </div>
          </div>
        </div>

        <div className="login-actions">
          <button type="button" className="btn btn-secondary" onClick={onClose} disabled={loading}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            <LogIn size={14} />
            <span>{loading ? 'Authenticating...' : 'Sign In'}</span>
          </button>
        </div>
      </form>
    </Modal>
  )
}
