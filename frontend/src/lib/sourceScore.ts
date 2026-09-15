export function scorePercentage(score: number): number {
  if (!Number.isFinite(score)) return 0

  const percentage = Math.round(score * 100)

  return Math.max(
    0,
    Math.min(100, percentage),
  )
}
