# Stage 9A - CPU Performance Baseline

This benchmark measures the current Qvarn-RAG runtime before any production
optimization is applied.

It uses the real local Chroma index and the production `Indexer`,
`Embedder`, and `RAGEngine`.

## Baseline run

From the repository root:

```powershell
python -m tests.performance.benchmark_cpu
```

The default run measures:

- Indexer initialization
- index stats access
- first/cold embedding call
- repeated/warm embedding calls
- exact-file search
- deterministic director query
- deterministic turnover query
- metadata-only SPT search

The real Ollama benchmark is intentionally optional:

```powershell
python -m tests.performance.benchmark_cpu --include-ai
```

Use `--include-ai` only when Ollama is running and the default model is
available.

For a meaningful cold-start comparison, run the benchmark in a fresh Python
process each time. Do not treat one process's warm timings as cold timings.

This Stage 9A runner is diagnostic and is not part of normal unit-test
discovery.
