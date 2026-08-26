import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage, IndexedDocument, IndexSummary, Stats } from './types'
import { ChatMessageView } from './components/ChatMessage'
import { SearchBar } from './components/SearchBar'
import { Sidebar } from './components/Sidebar'

// In Tauri the page loads from tauri://localhost so relative URLs won't work.
// Detect the Tauri runtime (v1 uses __TAURI__, v2 uses __TAURI_INTERNALS__).
const isTauri =
  typeof window !== 'undefined' &&
  ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
const API = isTauri ? 'http://localhost:8000' : ''  // Vite proxy handles /api in dev

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + url, init)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState<ChatMessage | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const [stats, setStats] = useState<Stats | null>(null)
  const [documents, setDocuments] = useState<IndexedDocument[]>([])
  const [isIndexing, setIsIndexing] = useState(false)
  const [watchedFolders, setWatchedFolders] = useState<string[]>([])
  const [isWatching, setIsWatching] = useState(false)

  const [scanningDocument, setScanningDocument] = useState<string | null>(null)
  const [reindexingDocument, setReindexingDocument] = useState<string | null>(null)

  const [selectedModel, setSelectedModel] = useState('qwen2.5:3b')
  const bottomRef = useRef<HTMLDivElement>(null)

  // ── Data fetching ───────────────────────────────────────────────────────

  const refreshStats = useCallback(async () => {
    try {
      const [healthData, docsData, watchedData] = await Promise.all([
        fetchJson<Stats>('/api/health'),
        fetchJson<{ documents: IndexedDocument[] }>('/api/documents'),
        fetchJson<{ folders: string[] }>('/api/watched'),
      ])
      setStats(healthData)
      setDocuments(docsData.documents)
      setWatchedFolders(watchedData.folders)
    } catch {
      // Backend not ready yet — silently ignore
    }
  }, [])

  useEffect(() => {
    refreshStats()
    const interval = setInterval(refreshStats, 15_000)
    return () => clearInterval(interval)
  }, [refreshStats])

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming?.answer])

  // ── SSE streaming ───────────────────────────────────────────────────────

  const handleAsk = useCallback(async (question: string) => {
    if (isLoading) return

    const id = crypto.randomUUID()
    const placeholder: ChatMessage = {
      id,
      question,
      answer: '',
      sources: [],
      isStreaming: true,
    }
    setStreaming(placeholder)
    setIsLoading(true)

    try {
      const response = await fetch(API +'/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5, model: selectedModel }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let answer = ''
      let sources = placeholder.sources

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') continue

          try {
            const event = JSON.parse(raw)
            if (event.type === 'token') {
              answer += event.content as string
              setStreaming(prev => prev ? { ...prev, answer } : prev)
            } else if (event.type === 'sources') {
              sources = event.sources
              setStreaming(prev => prev ? { ...prev, sources } : prev)
            } else if (event.type === 'error') {
              throw new Error(event.message as string)
            }
          } catch (parseErr) {
            if (parseErr instanceof SyntaxError) continue
            throw parseErr
          }
        }
      }

      const final: ChatMessage = { id, question, answer, sources, isStreaming: false }
      setMessages(prev => [...prev, final])
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err)
      const final: ChatMessage = {
        id,
        question,
        answer: '',
        sources: [],
        isStreaming: false,
        error: errMsg,
      }
      setMessages(prev => [...prev, final])
    } finally {
      setStreaming(null)
      setIsLoading(false)
    }
  }, [isLoading, selectedModel])

  // ── Index folder ────────────────────────────────────────────────────────

  const handleIndexFolder = useCallback(async (folderPath: string) => {
    if (!folderPath.trim() || isIndexing) return
    setIsIndexing(true)
    try {
      await fetchJson<IndexSummary>('/api/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath }),
      })
      await refreshStats()
    } catch (err) {
      console.error('Indexing failed:', err)
    } finally {
      setIsIndexing(false)
    }
  }, [isIndexing, refreshStats])

  // ── Scan single document ───────────────────────────────────────────────

  const handleScanDocument = useCallback(async (filePath: string) => {
    if (!filePath || scanningDocument) return

    setScanningDocument(filePath)

    try {
      const result = await fetchJson<{
        success: boolean
        file_path: string
        filename: string
        indexed: boolean
        message: string
        documents: number
        chunks: number
      }>('/api/documents/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath }),
      })

      console.log('Hasil scan:', result)

      await refreshStats()
    } catch (err) {
      console.error('Scan dokumen gagal:', err)
    } finally {
      setScanningDocument(null)
    }
  }, [scanningDocument, refreshStats])

  // ── Re-index single document ──────────────────────────────────────────

  const handleReindexDocument = useCallback(async (filePath: string) => {
    if (!filePath || reindexingDocument) return

    setReindexingDocument(filePath)

    try {
      const result = await fetchJson<{
        success: boolean
        file_path: string
        filename: string
        indexed: boolean
        message: string
        documents: number
        chunks: number
      }>('/api/documents/reindex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath }),
      })

      console.log('Hasil re-index:', result)

      await refreshStats()
    } catch (err) {
      console.error('Re-index dokumen gagal:', err)
    } finally {
      setReindexingDocument(null)
    }
  }, [reindexingDocument, refreshStats])

  // ── Watch folder ─────────────────────────────────────────────────────────

  const handleWatchFolder = useCallback(async (folderPath: string) => {
    if (!folderPath.trim() || isWatching) return
    setIsWatching(true)
    try {
      await fetchJson('/api/watch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath }),
      })
      await refreshStats()
    } catch (err) {
      console.error('Watch failed:', err)
    } finally {
      setIsWatching(false)
    }
  }, [isWatching, refreshStats])

  const handleUnwatchFolder = useCallback(async (folderPath: string) => {
    try {
      await fetchJson('/api/watch', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath }),
      })
      await refreshStats()
    } catch (err) {
      console.error('Unwatch failed:', err)
    }
  }, [refreshStats])

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#0f1117' }}>
      {/* Sidebar */}
      <Sidebar
        stats={stats}
        documents={documents}
        isIndexing={isIndexing}
        onIndexFolder={handleIndexFolder}

        watchedFolders={watchedFolders}
        isWatching={isWatching}
        onWatchFolder={handleWatchFolder}
        onUnwatchFolder={handleUnwatchFolder}

        scanningDocument={scanningDocument}
        reindexingDocument={reindexingDocument}
        onScanDocument={handleScanDocument}
        onReindexDocument={handleReindexDocument}
      />

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center gap-3 px-6 py-4 border-b" style={{ borderColor: '#2a3244' }}>
          <img src="/logo.svg" alt="Qavrn" className="w-7 h-7 shrink-0" />
          <h1 className="text-lg font-semibold text-slate-100 tracking-tight">Qavrn</h1>
          <span className="text-xs text-slate-500 ml-1">private AI research assistant</span>
          {stats && (
            <span
              className={`ml-auto text-xs px-2 py-0.5 rounded-full font-medium ${
                stats.ollama_available
                  ? 'bg-green-900/50 text-green-400'
                  : 'bg-red-900/40 text-red-400'
              }`}
            >
              {stats.ollama_available ? 'Ollama connected' : 'Ollama offline'}
            </span>
          )}
        </header>

        {/* Chat area */}
        <main className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 && !streaming && (
            <EmptyState />
          )}
          {messages.map(msg => (
            <ChatMessageView key={msg.id} message={msg} />
          ))}
          {streaming && (
            <ChatMessageView message={streaming} />
          )}
          <div ref={bottomRef} />
        </main>

        {/* Search bar */}
        <div className="px-4 pb-4 pt-2 border-t" style={{ borderColor: '#2a3244' }}>
          <SearchBar
            onSubmit={handleAsk}
            isDisabled={isLoading}
            model={selectedModel}
            onModelChange={setSelectedModel}
          />
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ──────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-64 text-center px-4 select-none">
      <div className="w-20 h-20 rounded-2xl flex items-center justify-center mb-4" style={{ background: '#1e2535' }}>
        <img src="/logo.svg" alt="Qavrn" className="w-12 h-12" />
      </div>
      <h2 className="text-xl font-semibold text-slate-200 mb-2">Ask your documents anything</h2>
      <p className="text-slate-500 text-sm max-w-xs">
        Index a folder using the sidebar, then ask questions in natural language.
        Answers are grounded in your local files — nothing leaves your machine.
      </p>
    </div>
  )
}
