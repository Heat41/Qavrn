import test from 'node:test'
import assert from 'node:assert/strict'

import { createSSEParser } from '../src/lib/sse.ts'

test('parses token events in order', () => {
  const parser = createSSEParser()

  const result = parser.feed(
    'data: {"type":"token","content":"Halo"}\n'
    + 'data: {"type":"token","content":" dunia"}\n',
  )

  assert.deepEqual(result.events, [
    { type: 'token', content: 'Halo' },
    { type: 'token', content: ' dunia' },
  ])
  assert.equal(result.done, false)
})

test('preserves fragmented SSE lines until complete', () => {
  const parser = createSSEParser()

  const first = parser.feed(
    'data: {"type":"token","content":"Ha',
  )
  const second = parser.feed(
    'lo"}\n',
  )

  assert.deepEqual(first.events, [])
  assert.deepEqual(second.events, [
    { type: 'token', content: 'Halo' },
  ])
})

test('parses sources event using backend schema', () => {
  const parser = createSSEParser()

  const result = parser.feed(
    'data: {"type":"sources","sources":'
    + '[{"filename":"A.pdf","file_path":"C:/A.pdf",'
    + '"chunk_text":"isi","score":1.2346}]}\n',
  )

  assert.equal(result.events.length, 1)
  assert.equal(result.events[0].type, 'sources')
  assert.equal(result.events[0].sources[0].filename, 'A.pdf')
  assert.equal(result.events[0].sources[0].score, 1.2346)
})

test('marks DONE without creating a normal event', () => {
  const parser = createSSEParser()

  const result = parser.feed('data: [DONE]\n')

  assert.equal(result.done, true)
  assert.deepEqual(result.events, [])
})

test('parses backend error event', () => {
  const parser = createSSEParser()

  const result = parser.feed(
    'data: {"type":"error","message":"retrieval gagal"}\n',
  )

  assert.deepEqual(result.events, [
    { type: 'error', message: 'retrieval gagal' },
  ])
})

test('ignores malformed SSE JSON and continues', () => {
  const parser = createSSEParser()

  const result = parser.feed(
    'data: {invalid}\n'
    + 'data: {"type":"token","content":"ok"}\n',
  )

  assert.deepEqual(result.events, [
    { type: 'token', content: 'ok' },
  ])
})
