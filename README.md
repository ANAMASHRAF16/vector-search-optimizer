# Vector Search Optimizer

A vector search system using FAISS that demonstrates the performance difference between rebuilding the index on every request vs. loading it once at startup with a background refresh mechanism.

## Problem

The baseline search system rebuilds the FAISS index from scratch on every search request:
- Each query re-reads all documents, re-computes embeddings, and re-builds the index
- Latency exceeds **500ms per search** even for small datasets
- CPU usage spikes on every request
- Completely unusable at scale

## Fix

| Change | Before | After |
|---|---|---|
| Index lifecycle | Rebuilt on every request | Loaded once at startup, kept in memory |
| Search latency | >500ms | <100ms |
| Refresh strategy | None (always full rebuild) | Background thread refreshes every N seconds |
| Thread safety | Not needed (no shared state) | Read-write lock protects index during refresh |

## Architecture

```
Before (broken):
  Request -> load docs -> compute embeddings -> build FAISS index -> search -> return
  (repeated every single request, >500ms each time)

After (fixed):
  Startup  -> load docs -> compute embeddings -> build FAISS index (once)
  Request  -> search in-memory index -> return (<100ms)
  Background thread -> refresh index every 30s (non-blocking)
```

## Run

```bash
pip install -r requirements.txt

# Generate sample documents
python src/generate_data.py

# Run broken version (slow — rebuilds index every request)
python src/search_broken.py

# Run fixed version (fast — in-memory index with refresh)
python src/search_fixed.py

# Run benchmark comparison
python tests/test_benchmark.py
```
