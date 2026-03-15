# Tuning Guide

Nine tunable parameters. Each has a failure mode, an interaction with other parameters, and a best guess.

## 1. Merge function

The LLM call inside `union()`. Takes N source messages, returns one summary.

### Parameters

| Param | Default | Range | Notes |
|---|---|---|---|
| `max_summary_tokens` | 150 | 50–500 | Compression ratio. Lower = lossier. |
| `recency_weight` | 0.7 | 0.0–1.0 | When messages contradict, weight toward newer. 0.5 = equal. |
| `idempotency_check` | true | bool | Deduplicate near-identical messages before summarizing. |

### Failure modes

- **Too terse** (max_summary_tokens too low): specific facts vanish. "PostgreSQL 16.2 on port 5433" becomes "database setup."
- **Too verbose** (too high): summaries approach the length of the originals. No compression.
- **Contradiction ignored** (recency_weight too low): "port 5432" and "port 5433" coexist in the summary. The model hallucinates a reconciliation.
- **Bloat from duplication** (no idempotency): merging "Got it" with "Got it, makes sense" produces a summary longer than either input.

### Prompt template

```
Summarize these conversation messages into one paragraph of at most
{max_summary_tokens} tokens. When messages contradict, prefer the
more recent one (later index = more recent). Preserve all specific
details: version numbers, ports, file names, line numbers, IPs,
exact commands, function names, threshold values. Drop filler and
acknowledgments. If two messages say the same thing, include the
fact once.
```

### Interaction

`max_summary_tokens` × `max_cold_clusters` = total cold context budget. If you lower one, raise the other.

---

## 2. Union threshold

When a message graduates from hot to cold, merge into nearest e-class if similarity exceeds this threshold. Otherwise create a new singleton.

| Param | Default | Range | Notes |
|---|---|---|---|
| `merge_threshold` | 0.15 | 0.0–1.0 | Cosine similarity floor for merging. |

### Failure modes

- **Too strict** (> 0.4): most messages become singletons. Trial 4 had 66 e-classes from 190 messages at threshold 0.3. Retrieval drowns in noise.
- **Too loose** (< 0.05): unrelated messages merge. "Database port 5433" merges with "PagerDuty escalation policy." Summary is incoherent.
- **TF-IDF specific**: short filler messages have near-zero cosine with everything. Threshold 0.3 is strict for TF-IDF; 0.15 is reasonable. With dense embeddings (e.g., text-embedding-3-small), 0.5+ is typical.

### Best guess

0.15 for TF-IDF. Target: ~10–15 e-classes from 190 messages (roughly one per topic + a filler bucket).

### Interaction

Loose union + strict retrieval: few large classes, retrieve the right one. Tight union + loose retrieval: many small classes, retrieve several hoping one is right. Loose union is preferable — fewer LLM calls, denser summaries, cleaner retrieval.

---

## 3. Retrieval threshold

When injecting cold context for a query, only include e-classes above this similarity threshold.

| Param | Default | Range | Notes |
|---|---|---|---|
| `retrieve_k` | 3 | 1–10 | Max e-classes to retrieve. |
| `retrieve_min_sim` | 0.05 | 0.0–1.0 | Minimum cosine to include. |

### Failure modes

- **k too low**: misses cross-topic questions. "What webhook endpoint receives payment events?" needs both the billing and API clusters.
- **k too high**: injects irrelevant clusters. Noise dilutes the signal.
- **min_sim too high**: question phrasing doesn't match any centroid. Returns nothing.

### Best guess

k=3 with ~10 e-classes (30% coverage per query). k=5 with ~20+ e-classes.

### Interaction

With loose union (few large classes), k=3 covers most queries. With tight union (many small classes), need k=5+. The product `k / cluster_count` is the coverage ratio — aim for 20–30%.

---

## 4. Window boundary

How many recent messages stay verbatim in the hot zone.

| Param | Default | Range | Notes |
|---|---|---|---|
| `hot_size` | 10 | 5–50 | Fixed count. |
| `hot_budget` | — | tokens | Alternative: token-based budget. |
| `graduation` | immediate | immediate / gradual | What happens at the boundary. |

### Failure modes

- **Too small** (< 5): the model loses conversational continuity. The last few messages don't make sense without their predecessors.
- **Too large** (> 30): cold zone is tiny. No compression happens.
- **Fixed count ignores message length**: 10 one-word messages waste the hot zone. 10 multi-paragraph messages overflow the token budget.

### Best guess

10 messages for a typical chat. Token-based budget (e.g., 4000 tokens) is more principled but harder to implement.

### Graduation strategy

- **Immediate** (current): oldest hot message moves to cold in one step. Simple.
- **Gradual**: message stays in a "warm" zone where it's still verbatim but eligible for retrieval-based eviction. More complex, unclear benefit.

Immediate is fine. Gradual is overengineering for now.

---

## 5. Cold start

The first N messages have no e-classes to merge into.

| Param | Default | Range | Notes |
|---|---|---|---|
| `min_classes_before_merge` | 2 | 1–5 | Don't attempt merge until this many e-classes exist. |

### Failure modes

- **First graduated message always becomes a singleton.** This is correct — there's nothing to merge with.
- **First few messages form bad clusters** if they're all filler ("Hey", "Got it", "Let me check"). The first substantive message then merges into the filler cluster.

### Best guess

No special handling needed. The first few singletons will merge naturally once similar messages graduate. The filler-bucket problem is addressed by the merge threshold, not by cold start logic.

---

## 6. Class fragmentation

If the merge threshold is too strict, e-classes proliferate. The forest becomes a flat list of singletons — no better than raw messages.

| Param | Default | Range | Notes |
|---|---|---|---|
| `max_cold_clusters` | 10 | 5–20 | Hard cap. If exceeded, force-merge the closest pair. |
| `rebalance_interval` | 0 | 0–100 | Every N graduations, run a batch merge pass. 0 = disabled. |

### Failure modes

- **No cap, no rebalance**: cluster count grows unboundedly. At 200 messages with strict threshold, 66 e-classes (trial 4).
- **Cap too low**: forces merges between dissimilar messages. Summary quality degrades.
- **Rebalance too frequent**: unnecessary LLM calls. Each merge costs one summarizer call.

### Best guess

Hard cap at 10. If incremental merging keeps the count under 10, no batch pass needed. If it doesn't, the cap forces the closest pair to merge. This is the hybrid: incremental compaction with a batch fallback.

### Implementation

```python
def _graduate(self, msg):
    self._forest.insert(msg.id, msg.content, msg.embedding)
    # Incremental: merge into nearest if similar
    ...
    # Fallback: enforce hard cap
    while self._forest.cluster_count() > self._max_cold_clusters:
        pair = find_closest_pair(self._forest)
        if pair is None:
            break
        self._forest.union(*pair)
```

---

## 7. Embedding consistency

Embeddings must be stable across sessions for cross-session e-class lookup.

| Param | Default | Range | Notes |
|---|---|---|---|
| `embed_model` | TF-IDF | any | Must be deterministic or versioned. |
| `embed_version` | stored | string | Recorded in forest metadata. |

### Failure modes

- **Model version change**: old centroids and new query embeddings live in different vector spaces. Nearest-neighbor returns garbage.
- **TF-IDF vocabulary drift**: if the vocabulary is fit on one session's corpus, a new session's messages may have OOV tokens. Centroids shift.

### Best guess

For cross-session: use a frozen embedding model (text-embedding-3-small or similar). Store the model version in the forest metadata. On load, check version match. If mismatch, re-embed all stored messages (expensive but correct).

For single-session (current experiment): TF-IDF fit on the full conversation is fine. Deterministic and reproducible.

---

## 8. Summary staleness

A class summary may become outdated when new contradicting messages arrive.

| Param | Default | Range | Notes |
|---|---|---|---|
| `recency_weight` | 0.7 | 0.0–1.0 | In merge prompt, weight toward newer messages. |
| `max_summary_age` | — | int | Re-summarize if summary is older than N merges. |

### Failure modes

- **Stale summary persists**: "Port 5432" was correct yesterday. Today it's 5433. The class summary still says 5432 because no re-summarization was triggered.
- **Re-summarize too eagerly**: every new merge re-summarizes the entire class. Expensive for large classes.

### Best guess

The current design re-summarizes on every `union()` call — so staleness only matters if a new message SHOULD merge into an existing class but doesn't (because the threshold check uses the centroid, not the summary). Recency weighting in the prompt handles contradictions during merge. No separate staleness timer needed if every merge re-summarizes.

The risk: a message correcting an earlier fact might not be similar enough to merge into the same class. It becomes a separate singleton. The old class keeps the wrong fact. **This is the hardest failure mode to address.** The correction is semantically related but lexically different ("Actually, use port 5433 not 5432" vs "Database is on port 5432").

Potential fix: on each graduation, check if the new message contradicts any existing summary (not just centroid similarity). This requires an LLM call per graduation — expensive.

---

## 9. Storage and pruning

The forest grows across sessions without bound.

| Param | Default | Range | Notes |
|---|---|---|---|
| `max_total_messages` | 10000 | 1000–100000 | Hard cap on total stored messages. |
| `prune_strategy` | oldest-first | oldest / smallest / stalest | Which classes to drop when at capacity. |
| `prune_threshold` | — | int | Drop classes with fewer than N members. |

### Failure modes

- **No pruning**: storage grows linearly with conversation length. After months of use, the forest has 50K messages and retrieval is slow.
- **Prune too aggressively**: useful long-term schemas are dropped.
- **Prune smallest**: drops singletons, which may contain the one specific fact the user needs later.

### Best guess

For MVP: no pruning. Storage is cheap. A forest of 10K messages with 100 e-classes is ~10MB of JSON.

For production: prune by access pattern. Track last-retrieved timestamp per class. Classes not retrieved in N sessions are candidates for archival (move to cold-cold storage, not deletion).

---

## Parameter interaction matrix

| | merge_threshold | retrieve_k | hot_size | max_cold_clusters | max_summary_tokens |
|---|---|---|---|---|---|
| **merge_threshold** | — | Loose→low k. Tight→high k. | Independent. | Loose→fewer classes, cap rarely hit. | Independent. |
| **retrieve_k** | | — | Independent. | k should be < cluster_count/3. | Independent. |
| **hot_size** | | | — | Larger hot→fewer cold messages→fewer classes. | Independent. |
| **max_cold_clusters** | | | | — | tokens × clusters = cold budget. |
| **max_summary_tokens** | | | | | — |

## Starting configuration

```python
ContextWindow(
    embedder=TFIDFEmbedder(corpus),
    summarizer=ClaudeSummarizer(client, "claude-haiku-4-5-20251001"),
    hot_size=10,
    max_cold_clusters=10,
    merge_threshold=0.15,  # lowered from 0.3
)
# retrieve_k=3, retrieve_min_sim=0.05
# max_summary_tokens=150, recency_weight=0.7
```
