import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage } from '../types'
import { SourceCard } from './SourceCard'

interface Props {
  message: ChatMessage
}

export function ChatMessageView({ message }: Props) {
  return (
    <div className="max-w-3xl mx-auto space-y-3">
      {/* Question bubble */}
      <div className="flex justify-end">
        <div
          className="px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm max-w-lg"
          style={{ background: '#1e3a5f', color: '#ffffff' }}
        >
          {message.question}
        </div>
      </div>

      {/* Answer */}
      <div
        className="rounded-2xl rounded-tl-sm px-4 py-3"
        style={{ background: '#161b27', border: '1px solid #2a3244' }}
      >
        {message.error ? (
          <p className="text-red-400 text-sm flex items-start gap-2">
            <span className="shrink-0">⚠</span>
            <span>{message.error}</span>
          </p>
        ) : message.answer ? (
          <div className={`prose-dark text-sm leading-relaxed ${message.isStreaming ? 'cursor-blink' : ''}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.answer}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm py-1" style={{ color: '#A0AEC0' }}>
            <SearchingDots />
            <span>Searching your documents…</span>
          </div>
        )}
      </div>

      {/* Sources */}
      {message.sources.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wider mb-2 px-1" style={{ color: '#06B6D4' }}>
            Sources
          </p>
          <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
              {message.sources
                  .slice(
                      0,
                      /\.(pdf|docx?|xlsx?|csv|txt|md|html?|json)\b/i.test(message.question)
                          ? 1
                          : 5
                  )
                  .map((src, i) => (
                      <SourceCard key={i} source={src} rank={i + 1} />
                  ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SearchingDots() {
  return (
    <span className="flex gap-1 items-center">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={{
            background: '#4f8ef7',
            animation: `blink 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </span>
  )
}
