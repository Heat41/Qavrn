import { KeyboardEvent, useRef, useState } from 'react'

const MODELS = ['qwen2.5:3b']

interface Props {
  onSubmit: (question: string) => void
  isDisabled: boolean
  model: string
  onModelChange: (model: string) => void
}

export function SearchBar({ onSubmit, isDisabled, model, onModelChange }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const q = value.trim()
    if (!q || isDisabled) return
    onSubmit(q)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }

  return (
    <div className="flex items-end gap-2 rounded-xl p-2" style={{ background: '#161b27', border: '1px solid #2a3244' }}>
      {/* Model selector */}
      <select
        value={model}
        onChange={e => onModelChange(e.target.value)}
        disabled={isDisabled}
        className="shrink-0 text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer disabled:opacity-40"
        style={{ background: '#1e2535', color: '#D1D5DB', border: '1px solid #2a3244' }}
        title="Ollama model"
      >
        {MODELS.map(m => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>

      {/* Textarea */}
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={e => { setValue(e.target.value); handleInput() }}
        onKeyDown={handleKeyDown}
        disabled={isDisabled}
        placeholder={isDisabled ? 'Thinking…' : 'Ask your documents…'}
        className="flex-1 resize-none bg-transparent outline-none text-sm leading-relaxed py-1.5 px-1 disabled:opacity-60 placeholder-slate-500"
        style={{ color: '#ffffff', maxHeight: '160px' }}
      />

      {/* Send button */}
      <button
        onClick={submit}
        disabled={isDisabled || !value.trim()}
        className="shrink-0 flex items-center justify-center w-9 h-9 rounded-lg transition-colors disabled:opacity-30"
        style={{ background: value.trim() && !isDisabled ? '#4f8ef7' : '#1e2535' }}
        title="Send (Enter)"
      >
        {isDisabled ? (
          <SpinnerIcon className="w-4 h-4 text-blue-400 animate-spin" />
        ) : (
          <SendIcon className="w-4 h-4 text-white" />
        )}
      </button>
    </div>
  )
}

function SendIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  )
}
