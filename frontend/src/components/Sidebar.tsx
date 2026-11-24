import React from 'react'
import {
  Activity,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  GitBranch,
  LayoutDashboard,
  Shield,
  ShieldCheck,
  Zap,
} from 'lucide-react'

export type TabKey =
  | 'overview'
  | 'datasets'
  | 'training'
  | 'models'
  | 'deployments'
  | 'inference'
  | 'monitoring'
  | 'alerts'

interface SidebarProps {
  activeTab: TabKey
  onSelectTab: (tab: TabKey) => void
  collapsed: boolean
  onToggleCollapse: () => void
  mobileOpen: boolean
  onCloseMobile: () => void
  badgeCounts?: {
    alerts?: number
    deployments?: number
  }
}

interface NavItemConfig {
  key: TabKey
  label: string
  icon: React.ComponentType<{ size: number; className?: string }>
  badgeKey?: 'alerts' | 'deployments'
  badgeType?: 'warning' | 'info' | 'error'
}

const NAV_ITEMS: NavItemConfig[] = [
  { key: 'overview', label: 'Overview', icon: LayoutDashboard },
  { key: 'datasets', label: 'Datasets', icon: Database },
  { key: 'training', label: 'Training & Runs', icon: Zap },
  { key: 'models', label: 'Model Registry', icon: GitBranch },
  { key: 'deployments', label: 'Deployments', icon: ShieldCheck, badgeKey: 'deployments', badgeType: 'info' },
  { key: 'inference', label: 'Inference & Telemetry', icon: Cpu },
  { key: 'monitoring', label: 'Monitoring & Drift', icon: Activity },
  { key: 'alerts', label: 'Alerts & Audit', icon: AlertTriangle, badgeKey: 'alerts', badgeType: 'error' },
]

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
  badgeCounts = {},
}) => {
  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {mobileOpen && (
        <div
          className="sidebar-mobile-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside
        className={`app-sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}
        aria-label="Primary Navigation"
      >
        {/* Brand Header */}
        <div className="sidebar-brand">
          <div className="brand-logo">
            <div className="logo-symbol">
              <Shield size={18} className="symbol-icon" />
              <span className="symbol-core">MG</span>
            </div>
            {!collapsed && (
              <div className="brand-meta">
                <span className="brand-name">ModelGuard</span>
                <span className="brand-tagline">ML Reliability Control Plane</span>
              </div>
            )}
          </div>
          <button
            type="button"
            className="collapse-toggle-btn"
            onClick={onToggleCollapse}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="sidebar-nav">
          <div className="nav-group">
            {!collapsed && <span className="nav-group-label">WORKSPACE</span>}
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.key
              const count = item.badgeKey ? badgeCounts[item.badgeKey] : undefined

              return (
                <button
                  key={item.key}
                  type="button"
                  className={`nav-link ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    onSelectTab(item.key)
                    if (mobileOpen) onCloseMobile()
                  }}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon size={18} className="nav-icon" />
                  {!collapsed && <span className="nav-label">{item.label}</span>}
                  {!collapsed && count !== undefined && count > 0 && (
                    <span className={`nav-badge badge-${item.badgeType || 'neutral'}`}>
                      {count}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </nav>

        {/* Sidebar Footer with Environment & Version Info */}
        <div className="sidebar-footer">
          {!collapsed ? (
            <div className="footer-expanded">
              <div className="env-badge-row">
                <span className="env-pill">PRODUCTION</span>
                <span className="version-tag">v0.1.0</span>
              </div>
              <div className="host-info">Docker Cluster • Host Mode</div>
            </div>
          ) : (
            <div className="footer-compact" title="ModelGuard v0.1.0 (Production)">
              <span className="compact-dot" />
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
