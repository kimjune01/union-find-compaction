# Eviction Experiment Lessons

Findings from v4 experiments: Anderson-inspired value-based eviction vs force-merge-closest-pair, tested on 200-message synthetic conversations with 40 planted factual recall questions.

## Results

| Config | Force-merge | Eviction | Gap | Storage reduction |
|--------|-------------|----------|-----|-------------------|
| TF-IDF, batch ingest, cap=10 | 92% | 8% | -84pp | 38% |
| TF-IDF, interleaved, cap=10 | 85% | 50% | -35pp | 38% |
| BGE-small, batch ingest, cap=20 | 75% | 5% | -70pp | 72% |
| BGE-small, interleaved, cap=20 | 75% | 42% | -33pp | 66% |
| BGE-small, interleaved, cap=200 | 75% | 57% | -18pp | 0% (no eviction fired) |

The 18pp gap at cap=200 (zero evictions) is LLM noise from independent summarizer calls, not eviction damage.

## Lessons

### 1. Eviction is a cross-session mechanism, not within-session

Within a single conversation, set the cap high enough that eviction doesn't fire. The policy's value is bounding storage growth across hundreds of sessions, not optimizing a single conversation. A user's 200-message session should never trigger eviction. Their 200th session should.

**Recommendation:** Set `max_cold_clusters` to 100+ for single-user CLI. Eviction fires only when accumulated cross-session clusters exceed this. Within one session, all clusters survive.

### 2. Embedding quality determines clustering quality determines eviction quality

Eviction can only be as smart as the clustering it operates on. Clustering can only be as smart as the embeddings.

| Embedder | Same-topic similarity | Cross-topic similarity | Result |
|----------|----------------------|----------------------|--------|
| TF-IDF (bag of words) | 0.03 - 0.21 | 0.00 - 0.15 | Topics indistinguishable. Fact-bearing messages become orphan singletons. |
| BGE-small-en-v1.5 (384d) | 0.52 - 0.80 | 0.20 - 0.45 | Clear topical separation. Facts cluster by topic. |

With TF-IDF, "MySQL migration details" and "PostgreSQL version target" score 0.14 against each other. Both are database facts; TF-IDF can't tell. They become singletons, decay, and get evicted.

With BGE, they score 0.75. They merge into a database cluster, accumulate write-absorb boosts, and survive.

**Recommendation:** Use a dense embedding model (BGE-small, Voyage, or the host LLM's embeddings). TF-IDF is deterministic and free but too weak for topical clustering. For production, consider ONNX export of BGE-small (~50MB) to avoid the 2GB PyTorch dependency.

### 3. The merge threshold must match the embedding space

The threshold that produces right-sized clusters depends entirely on the similarity distribution of the embedder.

| Embedder | Threshold | Clusters (200 msgs, cap=10) | Largest cluster | Notes |
|----------|-----------|----------------------------|-----------------|-------|
| TF-IDF | 0.15 | 10 | ~20 | Good cluster sizes, bad topic separation |
| BGE | 0.15 | 1 | 190 | Everything collapses into one mega-cluster |
| BGE | 0.70 | 10 | 63 | One mega-cluster dominates |
| BGE | 0.75 | 10 | 40 | Reasonable sizes |
| BGE | 0.80 | 10 | 8 | Good sizes but too many singletons evicted |
| BGE | 0.85 | 10 | 4 | Tight topic clusters, most messages are singletons |

**Recommendation:** Calibrate the threshold empirically for each embedder. Target: largest cluster < 20 members, most clusters 3-10 members. For BGE-small, 0.75 is the sweet spot.

### 4. Batch ingest is structurally hostile to eviction

Ingesting 200 messages with zero retrieval means every cluster starts at strength=1.0 and decays from there. By message 200, clusters from message 2 have decayed through 200 turns. They're evicted before any question is asked.

Interleaved retrieval (asking questions during the conversation) recovers 37 percentage points for both TF-IDF and BGE. Retrieval boosts keep accessed clusters alive.

This isn't a bug in the policy. Real conversations interleave messages and queries. The batch-ingest-then-query pattern is an artifact of the experiment design.

**Recommendation:** No action needed. Real usage naturally interleaves retrieval with ingestion.

### 5. Storage grows O(active topics), not O(total messages)

Force-merge keeps every message node forever. After N messages, the forest contains N cold nodes packed into `cap` clusters. Cluster summaries become increasingly lossy as clusters grow.

Eviction deletes entire clusters when they're no longer needed. After N messages, the forest contains only the nodes in surviving clusters. Storage is bounded by `cap × avg_cluster_size`, not by N.

Over many sessions:
- Force-merge: 100 sessions × 200 messages = 20,000 nodes, growing linearly
- Eviction: storage stabilizes at `cap` clusters regardless of session count

**Recommendation:** This is the primary value proposition. Eviction isn't about saving space in one conversation. It's about preventing unbounded growth across the lifetime of a user's cold storage.

### 6. Don't call on_write_absorb before on_merge

Implementation bug found by Gemini: when a new message merges into an existing cluster, calling both `on_write_absorb` (on the existing cluster) and then `on_merge` (combining both clusters) applies decay twice. The merge operation itself combines the strengths; the write-absorb is redundant.

**Recommendation:** Union triggers `on_merge` only. No separate `on_write_absorb` on merge. Reserve `on_write_absorb` for cases where write-path clustering reinforces an existing cluster without creating a new node.

### 7. The cap controls eviction aggressiveness, not recall quality

Recall is determined by retrieval (top-k nearest centroids). Eviction only removes clusters that are never in the top-k. Setting cap too low forces eviction of clusters that would have been retrieved.

| Cap | Eviction fires? | Storage | Recall gap vs force-merge |
|-----|-----------------|---------|--------------------------|
| 20 | Yes, aggressively | 64 nodes | -33pp |
| 200 | No | 190 nodes | -18pp (noise) |

**Recommendation:** Cap should be large enough that eviction doesn't fire within a single session. Cross-session eviction is the intended use case. Start with cap=100, tune based on observed cluster count after 10+ sessions.

## What the experiment did NOT test

- Multi-session storage growth (the primary value proposition)
- Cross-session topic decay (does a project from 3 months ago get evicted?)
- Multi-user shared cold storage
- Dense embeddings with the host LLM's own embedding API
- Interaction with context rot (does evicting stale clusters improve LLM output quality by reducing noise?)

These are the next experiments.

## Code

- Implementation: `compaction.py` on branch `value-eviction`
- Tests: `test_compaction.py` (41 tests, all passing)
- Experiments: `experiment_v4.py`, `experiment_v4_interleaved.py`
- Results: `results-v4-*.json`
- Design doc: `/Users/junekim/Documents/petricode/docs/union-find-v2.md`
