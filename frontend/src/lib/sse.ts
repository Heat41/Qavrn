import type { AskStreamEvent } from '../types'

export interface SSEParserResult {
  events: AskStreamEvent[]
  done: boolean
}

export function createSSEParser() {
  let buffer = ''

  return {
    feed(chunk: string): SSEParserResult {
      buffer += chunk

      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''

      const events: AskStreamEvent[] = []
      let done = false

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue

        const raw = line.slice(6).trim()

        if (raw === '[DONE]') {
          done = true
          continue
        }

        try {
          const event = JSON.parse(raw) as AskStreamEvent

          if (
            event.type === 'token'
            || event.type === 'sources'
            || event.type === 'error'
          ) {
            events.push(event)
          }
        } catch {
          // Keep streaming even when a malformed SSE line is received.
        }
      }

      return { events, done }
    },

    flush(): SSEParserResult {
      if (!buffer.trim()) {
        return { events: [], done: false }
      }

      const line = buffer
      buffer = ''
      return this.feed(line + '\n')
    },
  }
}
