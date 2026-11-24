import React, { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { Navbar } from './components/Navbar'
import { Sidebar, type TabKey } from './components/Sidebar'
import { ToastProvider, useToast } from './components/Toast'
import type { Deployment, HealthResponse } from './types'
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

  // Global control-plane state for Navbar and Sidebar
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [openAlertsCount, setOpenAlertsCount] = useState<number>(0)
  const { addToast } = useToast()

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
