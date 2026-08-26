import { useState } from 'react'
import type { IndexedDocument, Stats } from '../types'

interface Props {
  stats: Stats | null
  documents: IndexedDocument[]

  isIndexing: boolean
  onIndexFolder: (path: string) => void

  watchedFolders: string[]
  isWatching: boolean
  onWatchFolder: (path: string) => void
  onUnwatchFolder: (path: string) => void

  scanningDocument: string | null
  reindexingDocument: string | null

  onScanDocument: (path: string) => void
  onReindexDocument: (path: string) => void
}

const EXT_COLOR: Record<string, string> = {
  pdf: '#f87171',
  docx: '#60a5fa',
  md: '#a78bfa',
  txt: '#94a3b8',
  csv: '#4ade80',
  html: '#fb923c',
  json: '#fbbf24',
  xml: '#f472b6',
  py: '#34d399',
  js: '#fde68a',
  ts: '#60a5fa',
  rs: '#f97316',
  go: '#67e8f9',
  java: '#fb923c',
  eml: '#c084fc',
  epub: '#a78bfa',
}

function folderName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path
}

export function Sidebar({
                          stats,
                          documents,
                          isIndexing,
                          onIndexFolder,
                          watchedFolders,
                          isWatching,
                          onWatchFolder,
                          onUnwatchFolder,
                          scanningDocument,
                          reindexingDocument,
                          onScanDocument,
                          onReindexDocument,
                        }: Props) {
  const [indexInput, setIndexInput] = useState('')
  const [showIndexInput, setShowIndexInput] = useState(false)

  const [watchInput, setWatchInput] = useState('')
  const [showWatchInput, setShowWatchInput] = useState(false)

  const handleIndex = () => {
    if (!indexInput.trim()) return

    onIndexFolder(indexInput.trim())
    setIndexInput('')
    setShowIndexInput(false)
  }

  const handleWatch = () => {
    if (!watchInput.trim()) return

    onWatchFolder(watchInput.trim())
    setWatchInput('')
    setShowWatchInput(false)
  }

  return (
      <aside
          className="flex flex-col w-64 shrink-0 border-r overflow-hidden"
          style={{
            background: '#0d1117',
            borderColor: '#2a3244',
          }}
      >
        {/* Logo */}
        <div
            className="flex items-center gap-2.5 px-4 py-4 border-b"
            style={{ borderColor: '#2a3244' }}
        >
          <img
              src="/logo.svg"
              alt="Qavrn"
              className="w-6 h-6 shrink-0"
          />

          <span className="font-bold text-slate-100 tracking-tight">
          Qavrn
        </span>
        </div>

        {/* Statistik */}
        <div
            className="px-4 py-3 border-b"
            style={{ borderColor: '#2a3244' }}
        >
          <p
              className="text-xs uppercase tracking-wider mb-2"
              style={{ color: '#06B6D4' }}
          >
            Indeks
          </p>

          {stats ? (
              <div className="grid grid-cols-2 gap-2">
                <StatPill
                    label="Dokumen"
                    value={stats.documents}
                />

                <StatPill
                    label="Potongan"
                    value={stats.chunks}
                />
              </div>
          ) : (
              <p
                  className="text-xs"
                  style={{ color: '#A0AEC0' }}
              >
                Memuat...
              </p>
          )}
        </div>

        {/* Indeks Folder */}
        <div
            className="px-4 py-3 border-b"
            style={{ borderColor: '#2a3244' }}
        >
          {showIndexInput ? (
              <div className="space-y-2">
                <input
                    autoFocus
                    type="text"
                    value={indexInput}
                    onChange={e => setIndexInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        handleIndex()
                      }

                      if (e.key === 'Escape') {
                        setShowIndexInput(false)
                      }
                    }}
                    placeholder="D:\folder\dokumen"
                    className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none placeholder-slate-500"
                    style={{
                      background: '#1e2535',
                      border: '1px solid #2a3244',
                      color: '#ffffff',
                    }}
                />

                <div className="flex gap-1.5">
                  <button
                      onClick={handleIndex}
                      disabled={isIndexing || !indexInput.trim()}
                      className="flex-1 text-xs py-1 rounded-lg font-medium transition-colors disabled:opacity-40"
                      style={{
                        background: '#4f8ef7',
                        color: 'white',
                      }}
                  >
                    {isIndexing ? 'Mengindeks...' : 'Indeks'}
                  </button>

                  <button
                      onClick={() => setShowIndexInput(false)}
                      className="text-xs px-2 py-1 rounded-lg"
                      style={{
                        background: '#1e2535',
                        color: '#64748b',
                      }}
                  >
                    Batal
                  </button>
                </div>
              </div>
          ) : (
              <button
                  onClick={() => setShowIndexInput(true)}
                  disabled={isIndexing}
                  className="w-full flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-lg font-medium transition-colors disabled:opacity-40"
                  style={{
                    background: '#1e2535',
                    color: '#94a3b8',
                    border: '1px solid #2a3244',
                  }}
              >
                {isIndexing ? <SpinIcon /> : <FolderPlusIcon />}
                {isIndexing ? 'Mengindeks...' : 'Indeks Folder'}
              </button>
          )}
        </div>

        {/* Folder yang Dipantau */}
        <div
            className="px-4 py-3 border-b"
            style={{ borderColor: '#2a3244' }}
        >
          <div className="flex items-center justify-between mb-2">
            <p
                className="text-xs uppercase tracking-wider flex items-center gap-1.5"
                style={{ color: '#06B6D4' }}
            >
              Pemantauan

              {watchedFolders.length > 0 && (
                  <span
                      className="px-1 rounded text-green-400 font-bold"
                      style={{ background: '#14291a' }}
                  >
                {watchedFolders.length}
              </span>
              )}
            </p>

            {!showWatchInput && (
                <button
                    onClick={() => setShowWatchInput(true)}
                    disabled={isWatching}
                    className="text-xs px-1.5 py-0.5 rounded transition-colors disabled:opacity-40"
                    style={{
                      background: '#1e2535',
                      color: '#4f8ef7',
                    }}
                    title="Pantau folder untuk perubahan otomatis"
                >
                  + Tambah
                </button>
            )}
          </div>

          {showWatchInput && (
              <div className="space-y-2 mb-2">
                <input
                    autoFocus
                    type="text"
                    value={watchInput}
                    onChange={e => setWatchInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        handleWatch()
                      }

                      if (e.key === 'Escape') {
                        setShowWatchInput(false)
                      }
                    }}
                    placeholder="D:\folder\dokumen"
                    className="w-full text-xs px-2.5 py-1.5 rounded-lg outline-none placeholder-slate-500"
                    style={{
                      background: '#1e2535',
                      border: '1px solid #2a3244',
                      color: '#ffffff',
                    }}
                />

                <div className="flex gap-1.5">
                  <button
                      onClick={handleWatch}
                      disabled={isWatching || !watchInput.trim()}
                      className="flex-1 text-xs py-1 rounded-lg font-medium transition-colors disabled:opacity-40"
                      style={{
                        background: '#059669',
                        color: 'white',
                      }}
                  >
                    {isWatching ? 'Memantau...' : 'Pantau & Indeks'}
                  </button>

                  <button
                      onClick={() => setShowWatchInput(false)}
                      className="text-xs px-2 py-1 rounded-lg"
                      style={{
                        background: '#1e2535',
                        color: '#64748b',
                      }}
                  >
                    Batal
                  </button>
                </div>
              </div>
          )}

          {watchedFolders.length === 0 ? (
              <p
                  className="text-xs leading-relaxed"
                  style={{ color: '#A0AEC0' }}
              >
                Belum ada folder yang dipantau. Tambahkan folder untuk
                mengindeks ulang otomatis ketika terjadi perubahan.
              </p>
          ) : (
              <ul className="space-y-1">
                {watchedFolders.map(folder => (
                    <li key={folder}>
                      <div
                          className="flex items-center gap-2 px-2 py-1.5 rounded-lg group"
                          style={{ background: '#0d1f12' }}
                      >
                  <span
                      className="w-2 h-2 rounded-full shrink-0 animate-pulse"
                      style={{ background: '#4ade80' }}
                  />

                        <span
                            className="text-xs text-green-300 truncate flex-1 min-w-0"
                            title={folder}
                        >
                    {folderName(folder)}
                  </span>

                        <button
                            onClick={() => onUnwatchFolder(folder)}
                            className="shrink-0 text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                            style={{ color: '#64748b' }}
                            title={`Berhenti memantau ${folder}`}
                        >
                          ✕
                        </button>
                      </div>
                    </li>
                ))}
              </ul>
          )}
        </div>

        {/* Daftar Dokumen */}
        <div className="flex-1 overflow-y-auto px-4 py-3">
          <p
              className="text-xs uppercase tracking-wider mb-2"
              style={{ color: '#06B6D4' }}
          >
            Dokumen Terindeks{' '}
            {documents.length > 0 && `(${documents.length})`}
          </p>

          {documents.length === 0 ? (
              <p
                  className="text-xs leading-relaxed"
                  style={{ color: '#A0AEC0' }}
              >
                Belum ada dokumen yang terindeks.
                Gunakan "Indeks Folder" untuk memulai.
              </p>
          ) : (
              <ul className="space-y-2">
                {documents.map(doc => {
                  const isScanning =
                      scanningDocument === doc.file_path

                  const isReindexing =
                      reindexingDocument === doc.file_path

                  const isBusy =
                      isScanning || isReindexing

                  return (
                      <li key={doc.document_id}>
                        <div
                            className="rounded-lg p-2"
                            style={{ background: '#161b27' }}
                        >
                          {/* Nama file */}
                          <div className="flex items-center gap-2 min-w-0">
                      <span
                          className="shrink-0 text-xs font-bold px-1 rounded"
                          style={{
                            background: '#1e2535',
                            color:
                                EXT_COLOR[doc.file_type] ??
                                '#94a3b8',
                          }}
                      >
                        {doc.file_type.toUpperCase()}
                      </span>

                            <span
                                className="text-xs truncate"
                                style={{ color: '#ffffff' }}
                                title={doc.file_path}
                            >
                        {doc.filename}
                      </span>
                          </div>

                          {/* Tombol aksi dokumen */}
                          <div className="flex gap-1.5 mt-2">
                            <button
                                onClick={() =>
                                    onScanDocument(doc.file_path)
                                }
                                disabled={isBusy}
                                className="flex-1 text-xs py-1 rounded-md font-medium transition-colors disabled:opacity-40"
                                style={{
                                  background: '#1e2535',
                                  color: '#60a5fa',
                                  border: '1px solid #2a3244',
                                }}
                                title="Periksa file dan indeks jika terjadi perubahan"
                            >
                              {isScanning
                                  ? 'Memindai...'
                                  : 'Pindai'}
                            </button>

                            <button
                                onClick={() =>
                                    onReindexDocument(doc.file_path)
                                }
                                disabled={isBusy}
                                className="flex-1 text-xs py-1 rounded-md font-medium transition-colors disabled:opacity-40"
                                style={{
                                  background: '#1e2535',
                                  color: '#fbbf24',
                                  border: '1px solid #2a3244',
                                }}
                                title="Paksa file untuk diindeks ulang"
                            >
                              {isReindexing
                                  ? 'Mengindeks...'
                                  : 'Indeks Ulang'}
                            </button>
                          </div>
                        </div>
                      </li>
                  )
                })}
              </ul>
          )}
        </div>
      </aside>
  )
}

function StatPill({
                    label,
                    value,
                  }: {
  label: string
  value: number
}) {
  return (
      <div
          className="rounded-lg p-2 text-center"
          style={{ background: '#161b27' }}
      >
        <p
            className="text-base font-bold"
            style={{ color: '#ffffff' }}
        >
          {value.toLocaleString()}
        </p>

        <p
            className="text-xs"
            style={{ color: '#A0AEC0' }}
        >
          {label}
        </p>
      </div>
  )
}

function FolderPlusIcon() {
  return (
      <svg
          className="w-3.5 h-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
      >
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        <line x1="12" y1="11" x2="12" y2="17" />
        <line x1="9" y1="14" x2="15" y2="14" />
      </svg>
  )
}

function SpinIcon() {
  return (
      <svg
          className="w-3.5 h-3.5 animate-spin"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
      >
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
      </svg>
  )
}