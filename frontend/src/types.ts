export interface Source {
  filename: string
  file_path: string
  chunk_text: string
  score: number
}

export interface ChatMessage {
  id: string
  question: string
  answer: string
  sources: Source[]
  isStreaming: boolean
  error?: string
}

export interface Stats {
  documents: number
  chunks: number
  storage_mb: number
  ollama_available: boolean
}

export interface IndexedDocument {
  document_id: string
  filename: string
  file_path: string
  file_type: string
  total_chunks: string
}

export interface IndexSummary {
  total: number
  indexed: number
  skipped: number
  failed: number
  documents: number
  chunks: number
}
