import React, { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { LoginModal } from './components/LoginModal'
import { Navbar } from './components/Navbar'
import { Sidebar, type TabKey } from './components/Sidebar'
import { ToastProvider, useToast } from './components/Toast'
import type { Deployment, HealthResponse, UserProfile } from './types'
import { AlertsView } from './views/AlertsView'
import { DatasetsView } from './views/DatasetsView'
import { DeploymentsView } from './views/DeploymentsView'
import { InferenceView } from './views/InferenceView'
import { ModelsView } from './views/ModelsView'
import { MonitoringView } from './views/MonitoringView'
import { OverviewView } from './views/OverviewView'
import { TrainingView } from './views/TrainingView'

const MainContent: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    return localStorage.getItem('mg_sidebar_collapsed') === 'true'
  })
  const [mobileOpen, setMobileOpen] = useState<boolean>(false)
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0)
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)

  // Auth state
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null)
  const [loginModalOpen, setLoginModalOpen] = useState<boolean>(false)

  // Global control-plane state for Navbar and Sidebar
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [openAlertsCount, setOpenAlertsCount] = useState<number>(0)
  const { addToast } = useToast()

  // Restore user session on mount
  useEffect(() => {
    let isMounted = true
    api.getMe()
      .then((user) => {
        if (isMounted) setCurrentUser(user)
      })
      .catch(() => {
        if (isMounted) setCurrentUser(null)
      })
    return () => {
      isMounted = false
    }
  }, [])

  // Listen for unauthorized 401 events across the app
  useEffect(() => {
    const handleUnauthorized = () => {
      setCurrentUser(null)
      setLoginModalOpen(true)
      addToast({
        type: 'warning',
        title: 'Authentication Required',
        message: 'Your session has expired or requires authorization. Please sign in.',
      })
    }
    window.addEventListener('mg:unauthorized', handleUnauthorized)
    return () => {
      window.removeEventListener('mg:unauthorized', handleUnauthorized)
    }
  }, [addToast])

  const loadGlobalState = useCallback(async () => {
    try {
      const [h, deps, alerts] = await Promise.all([
        api.getHealth().catch(() => null),
        api.getDeployments().catch(() => []),
        api.getAlerts('OPEN').catch(() => []),
      ])
      setHealth(h)
      setDeployments(deps)
      setOpenAlertsCount(alerts.length)
    } catch {
      // Degraded or unreachable
    }
  }, [])

  useEffect(() => {
    loadGlobalState()
  }, [loadGlobalState, refreshTrigger])

  const handleToggleCollapse = () => {
    setCollapsed((prev) => {
      const next = !prev
      localStorage.setItem('mg_sidebar_collapsed', String(next))
      return next
    })
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    setRefreshTrigger((prev) => prev + 1)
    addToast({
      type: 'info',
      title: 'Control Plane Refresh',
      message: 'Reloading operational metrics across all services...',
    })
    setTimeout(() => {
      setIsRefreshing(false)
    }, 600)
  }

  const handleLogout = async () => {
    try {
      await api.logout()
      setCurrentUser(null)
      addToast({
        type: 'info',
        title: 'Signed Out',
        message: 'You have been signed out of the control plane.',
      })
    } catch {
      setCurrentUser(null)
    }
  }

  const handleLoginSuccess = (user: UserProfile) => {
    setCurrentUser(user)
    setLoginModalOpen(false)
    addToast({
      type: 'success',
      title: 'Authenticated',
      message: `Signed in as ${user.username} (${user.role}).`,
    })
    setRefreshTrigger((prev) => prev + 1)
  }

  // Find active champion deployment
  const prodDeployment = deployments.find((d) => d.status === 'PROMOTED')
  const stagingDeployment = deployments.find((d) => d.status === 'STAGED' || d.status === 'AWAITING_APPROVAL')

  const badgeCounts = {
    alerts: openAlertsCount,
    deployments: stagingDeployment ? 1 : 0,
  }

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={(tab) => {
          setActiveTab(tab)
          setMobileOpen(false)
        }}
        collapsed={collapsed}
        onToggleCollapse={handleToggleCollapse}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
        badgeCounts={badgeCounts}
      />

      {/* Main Body */}
      <div className={`main-body ${collapsed ? 'sidebar-collapsed' : ''}`}>
        {/* Top Navbar */}
        <Navbar
          onToggleMobile={() => setMobileOpen((prev) => !prev)}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          healthStatus={health?.status || 'healthy'}
          activeModelName={prodDeployment?.candidate_id ? 'metro-traffic-volume' : 'metro-traffic-volume'}
          activeModelVersion={prodDeployment?.candidate_version || '1'}
          currentUser={currentUser}
          onOpenLogin={() => setLoginModalOpen(true)}
          onLogout={handleLogout}
        />

        {/* Viewport content */}
        <main className="content-area">
          {activeTab === 'overview' && (
            <OverviewView
              refreshTrigger={refreshTrigger}
              onNavigate={(tab) => setActiveTab(tab as TabKey)}
            />
          )}
          {activeTab === 'datasets' && <DatasetsView refreshTrigger={refreshTrigger} />}
          {activeTab === 'training' && <TrainingView refreshTrigger={refreshTrigger} />}
          {activeTab === 'models' && <ModelsView refreshTrigger={refreshTrigger} />}
          {activeTab === 'deployments' && <DeploymentsView refreshTrigger={refreshTrigger} />}
          {activeTab === 'inference' && <InferenceView />}
          {activeTab === 'monitoring' && <MonitoringView refreshTrigger={refreshTrigger} />}
          {activeTab === 'alerts' && <AlertsView refreshTrigger={refreshTrigger} />}
        </main>
      </div>

      {/* Login Modal */}
      <LoginModal
        isOpen={loginModalOpen}
        onClose={() => setLoginModalOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />
    </div>
  )
}

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <MainContent />
    </ToastProvider>
  )
}

export default App
