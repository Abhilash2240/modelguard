import React from 'react'

export const Skeleton: React.FC<{
  className?: string
  width?: string
  height?: string
}> = ({ className = '', width = '100%', height = '1.25rem' }) => {
  return (
    <div
      className={`skeleton-box ${className}`}
      style={{ width, height }}
      aria-hidden="true"
    />
  )
}

export const CardSkeleton: React.FC = () => {
  return (
    <div className="card loading-card">
      <Skeleton width="40%" height="1.5rem" className="mb-3" />
      <Skeleton width="100%" height="3rem" className="mb-2" />
      <Skeleton width="75%" height="1rem" />
    </div>
  )
}

export const StatGridSkeleton: React.FC<{ count?: number }> = ({ count = 4 }) => {
  return (
    <div className="stat-grid">
      {Array.from({ length: count }).map((_, i) => (
        <CardSkeleton key={i} />
      ))}
    </div>
  )
}

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 4 }) => {
  return (
    <div className="table-container">
      <table className="table">
        <thead>
          <tr>
            <th><Skeleton width="60%" /></th>
            <th><Skeleton width="70%" /></th>
            <th><Skeleton width="50%" /></th>
            <th><Skeleton width="80%" /></th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i}>
              <td><Skeleton width="90%" /></td>
              <td><Skeleton width="80%" /></td>
              <td><Skeleton width="60%" /></td>
              <td><Skeleton width="85%" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
