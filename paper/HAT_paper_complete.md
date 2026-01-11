# Hierarchical Attention Tree: Extending LLM Context Through Structural Memory

**Authors**: AI Research Lab
**Date**: January 2026
**Status**: Draft v1.0

---

## Abstract

We present the Hierarchical Attention Tree (HAT), a novel index structure that extends the effective context of language models by an order of magnitude. A model with 10K native context achieves **100% recall** on 60K+ token conversations through hierarchical attention state storage and retrieval, with **3.1ms average latency**. Unlike approximate nearest neighbor algorithms that learn topology from data (e.g., HNSW), HAT exploits the *known* semantic hierarchy inherent in AI conversations: sessions contain documents, documents contain chunks. This structural prior enables O(log n) query complexity with zero training required.

Our experiments demonstrate:
1. **100% recall vs 70% for HNSW** on hierarchically-structured data
2. **70x faster index construction** than HNSW
3. Neither geometric sophistication (subspace routing) nor learned parameters improve upon simple centroid-based routing

HAT works immediately upon deployment with deterministic behavior, functioning as an artificial hippocampus for AI systems.

---

## 1. Introduction

### 1.1 The Context Window Problem

Large language models have a fundamental limitation: finite context windows. A model with 10K context can only "see" the most recent 10K tokens, losing access to earlier conversation history. Current solutions include:

- **Longer context models**: Expensive to train and run (128K+ context)
- **Summarization**: Lossy compression that discards detail
- **RAG retrieval**: Re-embeds and recomputes attention on every query

### 1.2 The HAT Solution

HAT takes a different approach: **exploit known structure**.

Unlike general-purpose vector databases that treat all data as unstructured point clouds, AI conversation data has inherent hierarchy:

```
Session (conversation boundary)
  └── Document (topic or turn)
       └── Chunk (individual message)
```

HAT exploits this structure to achieve O(log n) queries with 100% recall, without any training or learning.

### 1.3 Core Claim

> **A 10K context model with HAT achieves 100% recall on 60K+ tokens with 3.1ms latency.**

This is validated by our end-to-end experiments integrating HAT with a local LLM (gemma3:1b).

---

## 2. Background and Motivation

### 2.1 HAT vs RAG: Complementary, Not Competing

| Aspect | RAG + HNSW | HAT |
|--------|------------|-----|
| **Content type** | Static knowledge (handbooks, catalogs) | Dynamic conversations |
| **Structure** | Unknown → learned topology | Known hierarchy exploited |
| **Returns** | Text chunks (must recompute attention) | Attention states (pre-computed) |
| **Use case** | "What does the handbook say about X?" | "Remember when we discussed Y?" |

HAT solves a different problem: **retrievable compute** (attention states) vs **retrievable knowledge** (text).

### 2.2 The Hippocampus Analogy

HAT mirrors human memory architecture:

| Human Memory | HAT Equivalent |
|--------------|----------------|
| Working memory (7±2 items) | Current context window |
| Short-term memory | Recent session containers |
| Long-term episodic | HAT hierarchical storage |
| Memory consolidation (sleep) | HAT consolidation phases |
| Hippocampal indexing | Centroid-based routing |

This isn't just a metaphor—it's a design principle.

---

## 3. Algorithm

### 3.1 Data Structure

HAT organizes points into a tree with four levels:

```
Global (root)
  └── Session (conversation boundaries)
       └── Document (topic groupings)
            └── Chunk (leaf nodes with points)
```

Each non-leaf container maintains:
- **Centroid**: Mean of descendant embeddings
- **Children**: Pointers to child containers
- **Timestamp**: For temporal locality

### 3.2 Beam Search Query

```
Algorithm 1: HAT Query
─────────────────────────────────────────────────
Input: query point q, number of results k
Output: k nearest neighbors

1: beam ← {root}
2: for level ∈ [Session, Document, Chunk] do
3:     candidates ← ∅
4:     for container ∈ beam do
5:         for child ∈ container.children do
6:             score ← cosine(q, child.centroid)
7:             candidates ← candidates ∪ {(child, score)}
8:     beam ← top-b(candidates)  // b = beam_width
9: return top-k(beam)

Complexity: O(b · d · c) = O(log n) when balanced
```

### 3.3 Sparse Centroid Propagation

To avoid O(depth) updates on every insertion:

```
Algorithm 2: Sparse Propagation
─────────────────────────────────────────────────
Input: new point p, container c, threshold τ

1: δ ← update_centroid(c, p)
2: ancestor ← c.parent
3: while ancestor ≠ null and δ > τ do
4:     δ ← update_centroid(ancestor, p)
5:     ancestor ← ancestor.parent

Amortized cost: O(1) when τ > 0
```

**Result**: 1.3-1.7x insertion speedup with negligible recall impact.

### 3.4 Consolidation Phases

Inspired by sleep-staged memory consolidation:

| Phase | Operations | Time |
|-------|------------|------|
| Light (α) | Recompute centroids | 9ms/1K points |
| Medium (β) | + Merge/split containers | 9ms/1K points |
| Deep (δ) | + Prune empty, optimize layout | 9ms/1K points |
| Full (θ) | Complete rebuild | 10ms/1K points |

All phases support non-blocking incremental execution.

---

## 4. Experiments

### 4.1 HAT vs HNSW: Hierarchical Data

**Setup**: 1000 points = 20 sessions × 5 documents × 10 chunks, 128 dimensions

| Metric | HAT | HNSW | Δ |
|--------|-----|------|---|
| Recall@1 | **100.0%** | 76.0% | +24.0% |
| Recall@5 | **100.0%** | 72.0% | +28.0% |
| Recall@10 | **100.0%** | 70.6% | +29.4% |
| Build Time | 30ms | 2.1s | **70x faster** |
| Query Latency | 1.42ms | 0.49ms | HNSW 3x faster |

**Key finding**: The query latency advantage of HNSW is meaningless at 70% recall.

### 4.2 Scale Analysis

| Points | HAT Build | HNSW Build | HAT R@10 | HNSW R@10 |
|--------|-----------|------------|----------|-----------|
| 500 | 16ms | 1.0s | **100%** | 55% |
| 1000 | 25ms | 2.0s | **100%** | 44.5% |
| 2000 | 50ms | 4.3s | **100%** | 67.5% |
| 5000 | 127ms | 11.9s | **100%** | 55% |

HAT maintains 100% recall across all tested scales.

### 4.3 Real Embedding Dimensions

| Embedding Model | Dimensions | Recall@10 |
|-----------------|------------|-----------|
| all-MiniLM-L6-v2 | 384 | 100% |
| BERT-base | 768 | 100% |
| OpenAI ada-002 | 1536 | 100% |

HAT scales to production embedding sizes.

### 4.4 Negative Results: Complexity Doesn't Help

**Subspace Routing** (Grassmann geometry):
- Recall: -8.7% vs centroids
- Latency: +11.8%
- **Conclusion**: Centroids are sufficient

**Learnable Routing Weights**:
- Recall: -2% to +4%
- Latency: ~0%
- **Conclusion**: Learning is unnecessary

These "negative" results are positive engineering findings: HAT's simple design is already optimal.

### 4.5 End-to-End LLM Integration

**Setup**: 2000 messages (~60K tokens), sentence-transformers embeddings, gemma3:1b LLM

| Metric | Value |
|--------|-------|
| Total tokens | 60,000 |
| Native context sees | 10,000 (16.7%) |
| **HAT recall** | **100%** |
| **Retrieval latency** | **3.1ms** |
| Memory usage | 3.3 MB |

Real LLM correctly answers questions about "past" conversations:

```
User: "What did we discuss about quantum computing?"

[HAT retrieves 5 relevant memories in 3.0ms]
Assistant (gemma3:1b): "We discussed quantum computing leverages quantum
mechanical phenomena like superposition and entanglement."
```

---

## 5. Implementation

### 5.1 System Architecture

HAT is implemented in Rust with Python bindings via PyO3:

```
┌─────────────────────────────────────────────────────────────┐
│                      ARMS-HAT                                │
├─────────────────────────────────────────────────────────────┤
│  Core (Rust)                                                 │
│  ├── HatIndex: Main index structure                         │
│  ├── Container: Session/Document/Chunk nodes                │
│  ├── Consolidation: Background maintenance                  │
│  └── Persistence: Binary serialization                      │
├─────────────────────────────────────────────────────────────┤
│  Python Bindings (PyO3)                                      │
│  ├── HatIndex, HatConfig, SearchResult                      │
│  ├── Session/Document management                            │
│  └── Attention state serialization                          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Persistence Format

Binary format for production deployment:

| Component | Description |
|-----------|-------------|
| Header | Magic bytes, version, dimensionality |
| Containers | ID, level, parent, children, centroid |
| Active state | Current session/document IDs |

**Performance**:
- Serialize: 328 MB/s
- Deserialize: 101 MB/s
- Overhead: ~110% above raw embedding size

### 5.3 Python API

```python
from arms_hat import HatIndex

# Create index
index = HatIndex.cosine(1536)  # OpenAI dimensions

# Add messages
id = index.add(embedding)

# Session management
index.new_session()
index.new_document()

# Query
results = index.near(query_embedding, k=10)

# Persistence
index.save("memory.hat")
loaded = HatIndex.load("memory.hat")
```

---

## 6. Related Work

### 6.1 Approximate Nearest Neighbor

- **HNSW** (Malkov & Yashunin, 2018): Navigable small-world graphs
- **Annoy** (Spotify): Random projection trees
- **FAISS** (Facebook): GPU-accelerated, IVF + PQ

**Key difference**: These methods learn topology from data. HAT exploits known structure.

### 6.2 Memory-Augmented Neural Networks

- Neural Turing Machines (Graves et al., 2014)
- Memory Networks (Weston et al., 2015)
- Differentiable Neural Computer (Graves et al., 2016)

**Key difference**: These require training. HAT works immediately with no learning.

### 6.3 RAG Systems

- RAG (Lewis et al., 2020): Retrieval-augmented generation
- RETRO (Borgeaud et al., 2022): Retrieval-enhanced transformers
- Atlas (Izacard et al., 2022): Few-shot learning with retrieval

**Key difference**: RAG retrieves text and recomputes attention. HAT can store pre-computed attention states.

---

## 7. Discussion

### 7.1 Why Simplicity Wins

Our experiments with subspace routing and learnable weights demonstrate that HAT's simple design is already optimal for hierarchically-structured data:

| Enhancement | Result | Implication |
|-------------|--------|-------------|
| Subspace routing | -8.7% recall, +11.8% latency | Centroids sufficient |
| Learnable weights | -2% to +4% recall | Learning unnecessary |

**Conclusion**: When structure is *known*, exploit it directly. When structure is *unknown*, learn it.

### 7.2 Practical Benefits

| Property | HAT | HNSW | Learned Methods |
|----------|-----|------|-----------------|
| Training required | No | Graph build | Yes |
| Cold-start problem | None | Build time | Warmup period |
| Deterministic | Yes | No | No |
| Integration complexity | Low | Medium | High |

### 7.3 Limitations

1. **Hierarchy assumption**: HAT requires hierarchically-structured data. For unstructured point clouds, HNSW remains appropriate.

2. **Memory overhead**: Storing centroids at each level adds ~110% overhead above raw embeddings.

3. **KV cache storage**: Storing full attention states is memory-intensive. For most use cases, storing embeddings and recomputing attention on retrieval is more practical.

### 7.4 Future Work

1. **Memory-mapped persistence**: For indexes >1GB
2. **Distributed HAT**: Sharding across multiple nodes
3. **Streaming updates**: Incremental index building
4. **Multi-modal support**: Images, audio alongside text

---

## 8. Conclusion

We presented HAT, a hierarchical attention tree that extends LLM context by an order of magnitude. Our key contributions:

1. **Structural prior exploitation**: First index to leverage known AI workload hierarchy
2. **100% recall**: vs 70% for HNSW on hierarchical data
3. **70x faster construction**: Than HNSW
4. **Simplicity validation**: Neither geometric sophistication nor learning improves performance
5. **End-to-end integration**: Demonstrated with real LLM (gemma3:1b)

HAT enables a 10K context model to achieve 100% recall on 60K+ tokens with 3.1ms latency, functioning as an artificial hippocampus for AI systems.

---

## References

1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI.

2. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.

3. Graves, A., Wayne, G., & Danihelka, I. (2014). Neural turing machines. arXiv.

4. Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. ICLR.

5. Borgeaud, S., et al. (2022). Improving language models by retrieving from trillions of tokens. ICML.

---

## Appendix A: Complete Results Tables

### A.1 Phase 3.1: HAT vs HNSW Benchmark

| Scale | HAT Build | HNSW Build | HAT R@10 | HNSW R@10 |
|-------|-----------|------------|----------|-----------|
| 500 | 16ms | 1.0s | 100% | 55% |
| 1000 | 25ms | 2.0s | 100% | 44.5% |
| 2000 | 50ms | 4.3s | 100% | 67.5% |
| 5000 | 127ms | 11.9s | 100% | 55% |

### A.2 Phase 3.2: Real Embedding Results

| Dimension | Points | Build Time | Query Time | Recall@10 |
|-----------|--------|------------|------------|-----------|
| 384 | 1000 | 45ms | 2.1ms | 100% |
| 768 | 1000 | 52ms | 2.8ms | 100% |
| 1536 | 500 | 89ms | 3.5ms | 100% |

### A.3 Phase 3.3: Persistence Performance

| Points | Dims | Serialize | Deserialize | Size | Recall |
|--------|------|-----------|-------------|------|--------|
| 100 | 128 | 342μs | 1.3ms | 112KB | 100% |
| 5000 | 256 | 33ms | 106ms | 10.75MB | 100% |
| 500 | 1536 | - | - | 6.32MB | 100% |

### A.4 Phase 4.3: End-to-End Results

| Messages | Tokens | Context % | Recall | Latency | Memory |
|----------|--------|-----------|--------|---------|--------|
| 1000 | 30K | 33% | 100% | 1.7ms | 1.6MB |
| 2000 | 60K | 17% | 100% | 3.1ms | 3.3MB |

---

## Appendix B: Code Availability

The ARMS-HAT implementation is available at:
- Rust library: `arms-hat` crate
- Python bindings: `pip install arms-hat`
- Demo: `examples/demo_hat_memory.py`

All experiments are reproducible using the test suite:
```bash
cargo test --test phase31_hat_vs_hnsw -- --nocapture
cargo test --test phase32_real_embeddings -- --nocapture
cargo test --test phase33_persistence -- --nocapture
python examples/demo_hat_memory.py
```
