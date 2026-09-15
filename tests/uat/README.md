# Stage 8 - Real Document UAT

Stage 8 runs acceptance tests against the real local Chroma index and the
production `RAGEngine`. It is intentionally separate from the normal unit
test suite because results depend on locally indexed office documents.

## Baseline

The first baseline covers deterministic/search-only PT AEP cases:

- Director 2024
- Annual turnover 2024
- Company address 2024
- Exact filename/location
- Strict-year negative turnover
- Strict-year negative director

These baseline cases should not require Ollama generation. They still use the
real embedding model and real Chroma index where retrieval is required.

## Run

From the repository root:

```powershell
python -m tests.uat.run_real_document_uat
```

To use another Chroma directory:

```powershell
python -m tests.uat.run_real_document_uat --chroma-dir "D:\path\to\chroma"
```

Run selected cases only:

```powershell
python -m tests.uat.run_real_document_uat --only UAT_01 UAT_04
```

Exit code is `0` only when every selected case passes.

The assertions avoid absolute document paths so the UAT can be reused when
Qvarn-RAG is moved to the production PC.
