# Union-Find Context Compaction

Context compaction for chatbots that tracks cluster provenance, enabling consolidation instead of summarization.

## Problem

Chatbot context windows fill up. The current fix is **compaction**: summarize old messages, discard originals, continue. This is batched cache eviction — it reorganizes the window without changing how the system processes the next turn. [The Natural Framework](https://june.kim/the-natural-framework) calls this an arbitrary self-map on cache: useful, but not consolidation.

The failure mode is well-documented. [Diagnosis LLM](https://june.kim/diagnosis-llm) identifies it as the top dysfunction in deployed agents:

> **Agent consolidate: nil.** Has machinery. No initiative.

Compaction produces vague summaries that lose specificity. Schemas never form. The agent starts every session equally ignorant of what mattered in the last one. The [consolidation post](https://june.kim/consolidation) names the constraint: *if you can't trace a schema back to the episodes that formed it, the merge was lossy and wrong.*

## Thesis

Union-find is the right data structure for context compaction that preserves provenance. It tracks which messages belong to which cluster, supports incremental merging as context grows, and maintains the traceability constraint that distinguishes consolidation from summarization.

## How It Works

### Compound cache

The context window has two zones:

```
┌──────────────────────────────────────────────────────────────┐
│                     Context Window                           │
│                                                              │
│  ┌─────────────────────────┐  ┌───────────────────────────┐  │
│  │      COLD (forest)      │  │      HOT (deque)          │  │
│  │                         │  │                           │  │
│  │  Compacted clusters,    │  │  Recent messages, raw,    │  │
│  │  union-find managed,    │  │  never compacted.         │  │
│  │  summaries + originals. │  │  FIFO: oldest graduates   │  │
│  │                         │  │  to cold on overflow.     │  │
│  └─────────────────────────┘  └───────────────────────────┘  │
│         ↑ graduate                    ← append()             │
└──────────────────────────────────────────────────────────────┘
```

**Hot zone** — A fixed-size deque of the most recent messages. Served raw. Never touched by the union-find. This preserves conversational recency: the last N turns are always intact, exactly as the user and model produced them.

**Cold zone** — Everything older. Managed by the union-find forest. When hot overflows, the oldest message graduates to cold. When cold has too many clusters, the closest pair is merged and summarized by a cheap LLM.

`render()` returns cold summaries + hot messages, in order. The model sees compressed history followed by full recent context.

### Data model

```
Message {
    id:        uint
    content:   string
    embedding: vector
    parent:    *Message   // union-find parent pointer
    rank:      int        // union-find rank for balancing
    summary:   string     // cluster representative text (set on root only)
    children:  []uint     // source message IDs (set on root only)
}
```

### Operations

**1. Append** — A new message enters the hot zone. If hot exceeds capacity, the oldest message graduates to cold (FIFO).

**2. Graduate** — The oldest hot message is inserted into the cold forest as a singleton set. If cold now exceeds its cluster budget, the closest pair is merged.

**3. Union** — Find the two closest cold clusters by centroid similarity and `union()` them. The merged cluster gets a new summary from the cheap LLM. The original messages remain addressable through the root.

**4. Compact** — Replace a cluster's individual messages in the rendered context with the cluster's summary. The originals stay in cold storage, linked to the root via `children`.

**5. Expand** — On cache miss (the model needs detail that was compacted away), `find()` the cluster root, retrieve `children`, and reinflate the originals into context. This is the operation current compaction cannot do — once summarized, the originals are gone.

**6. Consolidate** — After N compaction rounds, scan for clusters that keep getting merged together across sessions. These become schema candidates. A schema is a new node that links to its source clusters, inserted into the union-find as a persistent root. This is the bridge from compaction (cache eviction) to consolidation (changes future processing).

### The six-step mapping

| Pipeline step | Union-find operation |
|---|---|
| **Perceive** | `append()`: message enters hot zone |
| **Cache** | Hot deque holds recent; cold forest indexes older messages |
| **Filter** | Threshold similarity check decides merge candidates in cold |
| **Attend** | Select which cold clusters to compact (diversity-aware: keep dissimilar clusters expanded) |
| **Consolidate** | Schema formation from repeatedly co-merged clusters |
| **Remember** | Schemas persist across sessions; source messages remain traceable |

### Invariants

1. **Recency** — The last `hot_size` messages are always raw and intact. Compaction never touches them.
2. **Provenance** — Every compacted summary traces back to source messages via `find()`. No orphaned summaries.
3. **Reversibility** — Compaction is reversible (`expand()`). Consolidation is not (schemas are additive, lossy, and change future processing). The data structure distinguishes between the two.
4. **Bounded render** — `render()` returns at most `hot_size + max_cold_clusters` entries. Context budget is predictable.
5. **Amortized cost** — Union-find with path compression and union by rank gives near-O(1) amortized per operation. Context management stays cheap.
6. **Traceability** — The constraint from [Consolidation](https://june.kim/consolidation): if you can't trace a schema back to the episodes that formed it, the merge was lossy and wrong. Union-find enforces this structurally.

## Why Not Just Summarize

Current compaction (Claude, GPT, Gemini) is a flat summarize-and-replace. It's an irreversible lossy operation with no structure. You can't:

- **Expand** a summary back to its sources when the model needs detail
- **Track** which messages contributed to which summary
- **Detect** cross-session patterns because the summaries are opaque strings
- **Distinguish** compaction from consolidation — the system has one operation (summarize) for two different needs

Union-find adds one layer of structure — the parent pointer — and gets all four.

## Comparison to Prior Art

| System | Compaction strategy | Provenance | Expandable | Consolidation |
|---|---|---|---|---|
| Claude Code context compression | Flat summarization | No | No | No |
| [Mem0](https://github.com/mem0ai/mem0) | Entity extraction + graph | Partial (entity links) | No | No |
| [Zep Graphiti](https://github.com/getzep/graphiti) | Temporal knowledge graph | Yes (graph edges) | Partial | No |
| [GraphRAG](https://arxiv.org/abs/2404.16130) | Community detection + summarization | Partial (communities) | No | No |
| **This** | Union-find forest | Yes (parent pointers) | Yes (`expand`) | Yes (schema formation) |

## Architecture

```
                          append("new msg")
                                │
                                ▼
┌──────────────────────────────────────────────────────────┐
│  HOT ZONE (deque, size=20)                               │
│                                                          │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  ...  ┌─────────┐  │
│  │m81 │ │m82 │ │m83 │ │m84 │ │m85 │       │ new msg │  │
│  └────┘ └────┘ └────┘ └────┘ └────┘       └─────────┘  │
│    ↓ graduate (FIFO: oldest out when full)               │
└──┬───────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│  COLD ZONE (union-find forest, max_clusters=10)          │
│                                                          │
│  ┌─────────────┐  ┌──────┐  ┌─────────────┐  ┌──────┐  │
│  │ s[1-8]      │  │ m42  │  │ s[50-60]    │  │ m71  │  │
│  │ "setup and  │  │(solo)│  │ "debugging  │  │(solo)│  │
│  │  deploy..." │  │      │  │  the auth..." │  │      │  │
│  └──────┬──────┘  └──────┘  └──────┬──────┘  └──────┘  │
│         │                          │                     │
│    ┌────┴────┐                ┌────┴────┐                │
│    │ sources │  expand()      │ sources │  expand()      │
│    │ 1,2,3,  │◄──────────    │ 50,51,  │◄──────────    │
│    │ 4,5,6,  │                │ 52,...  │                │
│    │ 7,8     │                │ 59,60   │                │
│    └─────────┘                └─────────┘                │
└──────────────────────────────────────────────────────────┘
   │
   │ consolidate() — after repeated co-merging across sessions
   ▼
┌──────────────────────────────────────────────────────────┐
│  SCHEMA STORE (persistent)                               │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Schema: "copyleft propagates through compilation"  │  │
│  │ sources: [s[1-8], s[50-60], s[120-125]]            │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

`render()` returns: cold summaries (compressed history) + hot messages (recent, raw).

## Implementation Plan

### Phase 1: Union-find with compound cache

- [x] Union-find data structure with path compression and union by rank
- [x] Compound cache: hot deque (recent, raw) + cold forest (compacted)
- [x] FIFO graduation from hot to cold on overflow
- [x] Embedding-based similarity for merge candidate selection
- [x] LLM-generated summaries for cluster roots (Summarizer protocol)
- [x] Expand operation to reinflate compacted clusters
- [x] Bounded render: cold summaries + hot messages
- [ ] Integration with a chat loop (stdin/stdout MVP)

### Phase 2: Consolidation

- [ ] Co-merge frequency tracking across compaction rounds
- [ ] Schema candidate detection from repeated co-activation
- [ ] Schema insertion as persistent roots with source links
- [ ] Schema-aware retrieval: queries match schemas first, then expand to sources

### Phase 3: Cross-session persistence

- [ ] Serialize union-find forest to disk (SQLite)
- [ ] Load schemas from prior sessions into new context
- [ ] Measure: does schema-primed context outperform flat compaction on downstream tasks?

## References

- [The Natural Framework](https://june.kim/the-natural-framework) — the six-step pipeline and why compaction is not consolidation
- [Consolidation](https://june.kim/consolidation) — complementary learning systems and the traceability constraint
- [Diagnosis LLM](https://june.kim/diagnosis-llm) — SOAP notes on agent consolidation dysfunction
- [General Intelligence](https://june.kim/general-intelligence) — complementation: human + agent closing the learning loop
- [The Handshake](https://june.kim/the-handshake) — morphisms with postconditions; compaction is an arbitrary self-map, consolidation is contract-preserving
- [The Parts Bin](https://june.kim/the-parts-bin) — catalog of operations by contract; consolidation candidates include K-means, PCA, prototype condensation
- [McClelland, McNaughton, & O'Reilly (1995)](https://doi.org/10.1037/0033-295X.102.3.419) — Complementary Learning Systems: hippocampus for fast storage, neocortex for slow consolidation
- [Tompary & Davachi (2017)](https://doi.org/10.1016/j.cub.2017.05.041) — Overlapping memories reorganize into shared schemas
- [Tarjan (1975)](https://doi.org/10.1145/321879.321884) — Union-find with path compression: near-O(1) amortized

## Repository map

| File | What it is |
|---|---|
| [`compaction.py`](compaction.py) | Core implementation: `Forest` (union-find with path compression), `ContextWindow` (compound cache with hot/cold zones), `Embedder`/`Summarizer` protocols |
| [`test_compaction.py`](test_compaction.py) | Unit tests (pytest). Stub embedder/summarizer, no API calls |
| [`experiment.py`](experiment.py) | Experiment harness: builds both conditions (flat vs UF), asks questions, scores with LLM judge, runs McNemar's test. Supports `--model`, `--summarizer-model`, `--long` |
| [`fixtures.py`](fixtures.py) | Short synthetic conversation (50 messages, 5 topics, 20 planted facts) |
| [`fixtures_long.py`](fixtures_long.py) | Long synthetic conversation (200 messages, 8 topics, 40 planted facts, timestamps) |
| [`EXPERIMENT.md`](EXPERIMENT.md) | Pre-registration and lab notebook. Hypotheses, design, and per-trial observations for all 7 trials |
| [`DISCUSSION.md`](DISCUSSION.md) | Cross-trial analysis, limitations, and recommendations for what to build next |
| [`TUNING.md`](TUNING.md) | Notes on parameter tuning (merge threshold, cluster cap, retrieval k) |
| `results-*.json` | Raw trial data: per-question answers, scores, and McNemar tables |

Start with [EXPERIMENT.md](EXPERIMENT.md) for the research narrative, [DISCUSSION.md](DISCUSSION.md) for conclusions.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
