import { useState } from 'react'
import type { Source } from '../types'

interface Props {
  source: Source
  rank: number
}

function scoreColor(score: number): string {
  if (score >= 0.70) return '#4ade80'   // green
  if (score >= 0.45) return '#facc15'   // yellow
  return '#f87171'                       // red
}

export function SourceCard({ source, rank }: Props) {
  const [expanded, setExpanded] = useState(false)
  const color = scoreColor(source.score)
  const pct = Math.round(source.score * 100)
  const preview = source.chunk_text.replace(/\s+/g, ' ').trim()
  const short = preview.length > 180 ? preview.slice(0, 180) + '…' : preview
  const extMatch = source.filename.match(/\.(\w+)$/)
  const ext = extMatch ? extMatch[1].toUpperCase() : 'DOC'

  const openFile = async () => {
    try {
      const response = await fetch('/api/open-file', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: source.file_path,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        alert(data.detail || 'Gagal membuka file')
      }
    } catch {
      alert('Gagal terhubung ke server')
    }
  }

  return (
    <div
      className="rounded-lg p-3 text-sm transition-colors"
      style={{ background: '#161b27', border: '1px solid #2a3244' }}
    >
      {/* Header row */}
      <div className="flex items-start gap-2 mb-2">
        <span
          className="shrink-0 text-xs font-bold px-1.5 py-0.5 rounded"
          style={{ background: '#1e2535', color: '#A0AEC0' }}
        >
          #{rank}
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate" style={{ color: '#ffffff' }} title={source.filename}>
            {source.filename}
          </p>
          {/* Score bar */}
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-1 rounded-full" style={{ background: '#2a3244' }}>
              <div
                className="h-1 rounded-full transition-all"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
            <span className="text-xs font-mono shrink-0" style={{ color }}>
              {pct}%
            </span>
            <span
              className="text-xs px-1 rounded shrink-0"
              style={{ background: '#1e2535', color: '#A0AEC0' }}
            >
              {ext}
            </span>
          </div>
        </div>
      </div>

      <button
          onClick={async () => {
            try {
              const response = await fetch('/api/open-file', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  file_path: source.file_path,
                }),
              })

              if (!response.ok) {
                const error = await response.json()
                alert(error.detail || 'Gagal membuka file')
              }
            } catch (error) {
              alert('Gagal terhubung ke server')
            }
          }}
          className="mt-2 text-xs px-2 py-1 rounded hover:opacity-80"
          style={{
            background: '#1e3a5f',
            color: '#ffffff',
          }}
      >
        📂 Buka File
      </button>

      {/* Chunk text */}
      <p className="text-xs leading-relaxed" style={{ color: '#D1D5DB' }}>
        {expanded ? preview : short}
      </p>

      {preview.length > 180 && (
        <button
          onClick={() => setExpanded(e => !e)}
          className="mt-1.5 text-xs hover:underline"
          style={{ color: '#4f8ef7' }}
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}
