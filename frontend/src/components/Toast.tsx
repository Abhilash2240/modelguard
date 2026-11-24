import React, { createContext, useCallback, useContext, useState } from 'react'
import { AlertCircle, CheckCircle, Info, X, XCircle } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface ToastMessage {
  id: string
  type: ToastType
  message: string
  details?: string
}

export interface ToastOptions {
  type?: ToastType
  title?: string
  message: string
  details?: string
}

export interface ToastContextValue {
  addToast: (opts: ToastOptions) => void
  showToast: (message: string, type?: ToastType, details?: string) => void
  success: (message: string, details?: string) => void
  error: (message: string, details?: string) => void
  warning: (message: string, details?: string) => void
  info: (message: string, details?: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export const useToast = (): ToastContextValue => {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within ToastProvider')
  }
  return ctx
}

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const showToast = useCallback(
    (message: string, type: ToastType = 'info', details?: string) => {
      const id = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
      setToasts((prev) => [...prev.slice(-4), { id, type, message, details }])
      setTimeout(() => removeToast(id), 5000)
    },
    [removeToast]
  )

  const addToast = useCallback(
    (opts: ToastOptions) => {
      const displayMsg = opts.title ? `${opts.title}: ${opts.message}` : opts.message
      showToast(displayMsg, opts.type || 'info', opts.details)
    },
    [showToast]
  )

  const success = useCallback((msg: string, details?: string) => showToast(msg, 'success', details), [showToast])
  const error = useCallback((msg: string, details?: string) => showToast(msg, 'error', details), [showToast])
  const warning = useCallback((msg: string, details?: string) => showToast(msg, 'warning', details), [showToast])
  const info = useCallback((msg: string, details?: string) => showToast(msg, 'info', details), [showToast])

  return (
    <ToastContext.Provider value={{ addToast, showToast, success, error, warning, info }}>
      {children}
      <div className="toast-viewport" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast-notification toast-${t.type}`}>
            <div className="toast-icon">
              {t.type === 'success' && <CheckCircle size={16} />}
              {t.type === 'error' && <XCircle size={16} />}
              {t.type === 'warning' && <AlertCircle size={16} />}
              {t.type === 'info' && <Info size={16} />}
            </div>
            <div className="toast-content">
              <div className="toast-message">{t.message}</div>
              {t.details && <div className="toast-details">{t.details}</div>}
            </div>
            <button
              type="button"
              className="toast-close"
              onClick={() => removeToast(t.id)}
              aria-label="Close notification"
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
