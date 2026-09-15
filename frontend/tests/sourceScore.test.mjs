import test from 'node:test'
import assert from 'node:assert/strict'

import { scorePercentage } from '../src/lib/sourceScore.ts'

test('converts normal similarity score to percentage', () => {
  assert.equal(scorePercentage(0.704), 70)
})

test('clamps reranking scores above one to 100 percent', () => {
  assert.equal(scorePercentage(3.228), 100)
})

test('clamps negative scores to zero', () => {
  assert.equal(scorePercentage(-0.5), 0)
})

test('returns zero for non-finite scores', () => {
  assert.equal(scorePercentage(Number.NaN), 0)
  assert.equal(scorePercentage(Number.POSITIVE_INFINITY), 0)
})
