# HAT Benchmark Reproducibility Package

This directory contains everything needed to reproduce the benchmark results from the HAT paper.

## Quick Start

```bash
# Run all benchmarks
./run_all_benchmarks.sh

# Run abbreviated version (faster)
./run_all_benchmarks.sh --quick
```

## Benchmark Suite

### Phase 3.1: HAT vs HNSW Comparison

**Test file**: `tests/phase31_hat_vs_hnsw.rs`

Compares HAT against HNSW on hierarchically-structured data (AI conversation patterns).

**Expected Results**:

| Metric | HAT | HNSW |
|--------|-----|------|
| Recall@10 | 100% | ~70% |
| Build Time | 30ms | 2100ms |
| Query Latency | 1.4ms | 0.5ms |

**Key finding**: HAT achieves 30% higher recall while building 70x faster.

### Phase 3.2: Real Embedding Dimensions

**Test file**: `tests/phase32_real_embeddings.rs`

Tests HAT with production embedding sizes.

**Expected Results**:

| Dimensions | Model | Recall@10 |
|------------|-------|-----------|
| 384 | MiniLM | 100% |
| 768 | BERT-base | 100% |
| 1536 | OpenAI ada-002 | 100% |

### Phase 3.3: Persistence Layer

**Test file**: `tests/phase33_persistence.rs`

Validates serialization/deserialization correctness and performance.

**Expected Results**:

| Metric | Value |
|--------|-------|
| Serialize throughput | 300+ MB/s |
| Deserialize throughput | 100+ MB/s |
| Recall after restore | 100% |

### Phase 4.2: Attention State Format

**Test file**: `tests/phase42_attention_state.rs`

Tests the attention state serialization format.

**Expected Results**:
- All 9 tests pass
- Role types roundtrip correctly
- Metadata preserved
- KV cache support working

### Phase 4.3: End-to-End Demo

**Script**: `examples/demo_hat_memory.py`

Full integration with sentence-transformers and optional LLM.

**Expected Results**:

| Metric | Value |
|--------|-------|
| Messages | 2000 |
| Tokens | ~60,000 |
| Recall accuracy | 100% |
| Retrieval latency | <5ms |

## Running Individual Benchmarks

### Rust Benchmarks

```bash
# HAT vs HNSW
cargo test --test phase31_hat_vs_hnsw -- --nocapture

# Real embeddings
cargo test --test phase32_real_embeddings -- --nocapture

# Persistence
cargo test --test phase33_persistence -- --nocapture

# Attention state
cargo test --test phase42_attention_state -- --nocapture
```

### Python Tests

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install maturin pytest sentence-transformers

# Build extension
maturin develop --features python

# Run tests
pytest python/tests/ -v

# Run demo
python examples/demo_hat_memory.py
```

## Hardware Requirements

- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB RAM for large-scale tests
- **Storage**: ~2GB for full benchmark suite

## Expected Runtime

| Mode | Time |
|------|------|
| Quick (`--quick`) | ~2 minutes |
| Full | ~10 minutes |
| With LLM demo | ~15 minutes |

## Interpreting Results

### Key Metrics

1. **Recall@k**: Percentage of true nearest neighbors found
   - HAT target: 100% on hierarchical data
   - HNSW baseline: ~70% on hierarchical data

2. **Build Time**: Time to construct the index
   - HAT target: <100ms for 1000 points
   - Should be 50-100x faster than HNSW

3. **Query Latency**: Time per query
   - HAT target: <5ms
   - Acceptable to be 2-3x slower than HNSW (recall matters more)

4. **Throughput**: Serialization/deserialization speed
   - Target: 100+ MB/s

### Success Criteria

The benchmarks validate the paper's claims if:

1. HAT recall@10 ≥ 99% on hierarchical data
2. HAT recall significantly exceeds HNSW on hierarchical data
3. HAT builds faster than HNSW
4. Persistence preserves 100% recall
5. Python bindings pass all tests
6. End-to-end demo achieves ≥95% retrieval accuracy

## Troubleshooting

### Build Errors

```bash
# Update Rust
rustup update

# Clean build
cargo clean && cargo build --release
```

### Python Issues

```bash
# Ensure venv is activated
source venv/bin/activate

# Rebuild extension
maturin develop --features python --release
```

### Memory Issues

For large-scale tests, ensure sufficient RAM:

```bash
# Check available memory
free -h

# Run with limited parallelism
RAYON_NUM_THREADS=2 cargo test --test phase31_hat_vs_hnsw
```

## Output Files

Results are saved to `benchmarks/results/`:

```
results/
  benchmark_results_YYYYMMDD_HHMMSS.txt  # Full output
```

## Citation

If you use these benchmarks, please cite:

```bibtex
@article{hat2026,
  title={Hierarchical Attention Tree: Extending LLM Context Through Structural Memory},
  author={AI Research Lab},
  year={2026}
}
```
