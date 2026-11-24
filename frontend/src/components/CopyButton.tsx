import React, { useState } from 'react'
import { Check, Copy } from 'lucide-react'

interface CopyButtonProps {
  text: string
  label?: string
  tooltip?: string
  className?: string
  truncateLength?: number
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  label,
  tooltip,
  className = '',
  truncateLength,
}) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback if clipboard API is restricted
      const textarea = document.createElement('textarea')
      textarea.value = text
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const displayText = truncateLength && text.length > truncateLength
    ? `${text.slice(0, truncateLength)}…`
    : text

  const titleText = copied
    ? 'Copied to clipboard!'
    : tooltip || `Copy: ${text}`

  return (
    <button
      type="button"
      className={`copy-btn ${copied ? 'copied' : ''} ${className}`}
      onClick={handleCopy}
      title={titleText}
      aria-label="Copy to clipboard"
    >
      <span className="mono-text">{label || displayText}</span>
      {copied ? <Check size={12} className="copy-icon check-icon" /> : <Copy size={12} className="copy-icon" />}
    </button>
  )
}
