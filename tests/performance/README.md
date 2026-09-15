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


## Stage 9C - Indexing performance

The indexing benchmark uses a **temporary Chroma database**. It does not write
to the production `data/chroma` directory.

From the repository root:

```powershell
python -m tests.performance.benchmark_indexing
```

The default run uses files under `test-data`, warms MiniLM once, and reports
per-file timings for:

- parsing (including OCR when the parser decides OCR is required)
- chunking
- batch embedding
- Chroma write
- total indexing pipeline

It also runs the whole folder twice against a separate temporary index:

1. first indexing pass
2. unchanged-file pass

This makes the cost of startup scans with no changed files visible.

To benchmark specific real documents without changing the production index:

```powershell
python -m tests.performance.benchmark_indexing \
  --file "C:\path\native-text.pdf" \
  --file "C:\path\scanned.pdf" \
  --file "C:\path\workbook.xlsx" \
  --skip-folder
```

For meaningful comparisons, use the same files between runs.
