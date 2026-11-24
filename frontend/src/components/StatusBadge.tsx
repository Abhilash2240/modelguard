import React from 'react'

interface StatusBadgeProps {
  status: string | null | undefined
  size?: 'sm' | 'md'
  showDot?: boolean
  customLabel?: string
  tooltip?: string
  className?: string
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'md',
  showDot = true,
  customLabel,
  tooltip,
  className = '',
}) => {
  if (!status) {
    return <span className="status-badge badge-neutral">Unknown</span>
  }

  const normalized = status.toUpperCase().trim()

  let theme = 'badge-neutral'
  let label = customLabel || status
  let defaultTooltip = tooltip

  switch (normalized) {
    case 'PROMOTED':
    case 'HEALTHY':
    case 'PASSED':
    case 'SUCCEEDED':
    case 'SUFFICIENT_LABELS':
    case 'STABLE':
    case 'RESOLVED':
    case 'APPROVED':
      theme = 'badge-emerald'
      if (normalized === 'SUFFICIENT_LABELS') label = 'Sufficient Labels'
      if (normalized === 'PROMOTED') label = 'Champion (Promoted)'
      break

    case 'STAGED':
    case 'AWAITING_APPROVAL':
    case 'WARNING':
    case 'INSUFFICIENT_LABELS':
    case 'ACKNOWLEDGED':
    case 'SHIFT':
      theme = 'badge-amber'
      if (normalized === 'INSUFFICIENT_LABELS') {
        label = 'Insufficient Labels'
        defaultTooltip = defaultTooltip || 'Requires >= 10 labeled ground truth feedback records for statistical confidence'
      }
      if (normalized === 'AWAITING_APPROVAL') label = 'Awaiting Approval'
      break

    case 'FAILED':
    case 'CRITICAL':
    case 'REJECTED':
    case 'DRIFT':
    case 'SMOKE_FAILED':
    case 'UNHEALTHY':
      theme = 'badge-rose'
      if (normalized === 'SMOKE_FAILED') label = 'Smoke Test Failed'
      break

    case 'RUNNING':
    case 'GATE_PASSED':
    case 'REGISTERED':
    case 'QUEUED':
    case 'OPEN':
    case 'INFO':
      theme = 'badge-indigo'
      if (normalized === 'GATE_PASSED') label = 'Gate Passed'
      break

    default:
      theme = 'badge-neutral'
      break
  }

  return (
    <span
      className={`status-badge ${theme} size-${size} ${className}`}
      title={defaultTooltip || label}
    >
      {showDot && <span className="status-dot" />}
      <span className="status-text">{label}</span>
    </span>
  )
}
