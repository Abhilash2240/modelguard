import React from 'react'

export interface StatCardBadgeConfig {
  text: string
  type?: 'success' | 'warning' | 'error' | 'info' | 'neutral'
}

export interface StatCardProps {
  title: string
  value: React.ReactNode
  icon?: React.ReactNode | React.ComponentType<{ size?: number; className?: string }>
  subtitle?: React.ReactNode
  footer?: React.ReactNode
  badge?: React.ReactNode | StatCardBadgeConfig
  className?: string
  onClick?: () => void
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  subtitle,
  footer,
  badge,
  className = '',
  onClick,
}) => {
  const renderIcon = () => {
    if (!icon) return null
    if (typeof icon === 'function') {
      const IconComp = icon as React.ComponentType<{ size?: number; className?: string }>
      return <IconComp size={18} />
    }
    return icon
  }

  const renderBadge = () => {
    if (!badge) return null
    if (React.isValidElement(badge)) return badge
    if (typeof badge === 'object' && 'text' in badge) {
      const badgeObj = badge as StatCardBadgeConfig
      const typeClass =
        badgeObj.type === 'success'
          ? 'badge-success'
          : badgeObj.type === 'error'
          ? 'badge-error'
          : badgeObj.type === 'warning'
          ? 'badge-warning'
          : badgeObj.type === 'info'
          ? 'badge-info'
          : 'badge-neutral'
      return <span className={`badge ${typeClass}`}>{badgeObj.text}</span>
    }
    return null
  }

  return (
    <div
      className={`kpi-card ${onClick ? 'interactive' : ''} ${className}`}
      onClick={onClick}
    >
      <div className="kpi-header">
        <span className="kpi-title">{title}</span>
        <div className="kpi-icon-wrap">{renderIcon()}</div>
      </div>
      <div className="kpi-body">
        <div className="kpi-value">{value}</div>
        {badge && <div className="kpi-badge-wrap">{renderBadge()}</div>}
      </div>
      {(footer || subtitle) && (
        <div className="kpi-footer">{footer || subtitle}</div>
      )}
    </div>
  )
}
