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

### Data model

Each message entering the context window becomes an element in a union-find forest.

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

**1. Insert** — A new message arrives. It starts as its own singleton set.

**2. Merge** — When context pressure exceeds a threshold, find the two closest clusters by centroid similarity and `union()` them. The merged cluster gets a new summary generated from all member messages. The original messages remain addressable through the root.

**3. Compact** — Replace a cluster's individual messages in the active context with the cluster's summary. The originals move to cold storage but remain linked to the root via `children`. The context window shrinks; provenance survives.

**4. Expand** — On cache miss (the model needs detail that was compacted away), `find()` the cluster root, retrieve `children`, and reinflate the originals into context. This is the operation current compaction cannot do — once summarized, the originals are gone.

**5. Consolidate** — After N compaction rounds, scan for clusters that keep getting merged together across sessions. These become schema candidates. A schema is a new node that links to its source clusters, inserted into the union-find as a persistent root. This is the bridge from compaction (cache eviction) to consolidation (changes future processing).

### The six-step mapping

| Pipeline step | Union-find operation |
|---|---|
| **Perceive** | New message arrives, becomes singleton |
| **Cache** | Indexed in the union-find forest |
| **Filter** | Threshold similarity check decides merge candidates |
| **Attend** | Select which clusters to compact (diversity-aware: keep dissimilar clusters expanded) |
| **Consolidate** | Schema formation from repeatedly co-merged clusters |
| **Remember** | Schemas persist across sessions; source messages remain traceable |

### Invariants

1. **Provenance** — Every compacted summary traces back to source messages via `find()`. No orphaned summaries.
2. **Reversibility** — Compaction is reversible (expand). Consolidation is not (schemas are additive, lossy, and change future processing). The data structure distinguishes between the two.
3. **Amortized cost** — Union-find with path compression and union by rank gives near-O(1) amortized per operation. Context management stays cheap.
4. **Traceability** — The constraint from [Consolidation](https://june.kim/consolidation): if you can't trace a schema back to the episodes that formed it, the merge was lossy and wrong. Union-find enforces this structurally.

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
┌─────────────────────────────────────────────┐
│              Active Context Window           │
│  ┌───┐ ┌───┐ ┌─────────┐ ┌───┐ ┌───┐      │
│  │m12│ │m13│ │ s[7-11] │ │m14│ │m15│      │
│  └───┘ └───┘ └────┬────┘ └───┘ └───┘      │
│                    │                         │
│         ┌──────────┴──────────┐              │
│         │  Compacted cluster  │              │
│         │  root: s[7-11]      │              │
│         │  children: 7,8,9,   │              │
│         │           10,11     │              │
│         └─────────────────────┘              │
└─────────────────────────────────────────────┘
                     │ expand()
                     ▼
┌─────────────────────────────────────────────┐
│              Cold Storage                    │
│  ┌───┐ ┌───┐ ┌───┐ ┌────┐ ┌────┐          │
│  │m7 │ │m8 │ │m9 │ │m10 │ │m11 │          │
│  └───┘ └───┘ └───┘ └────┘ └────┘          │
└─────────────────────────────────────────────┘
                     │ consolidate() — after repeated co-merging
                     ▼
┌─────────────────────────────────────────────┐
│              Schema Store                    │
│  ┌──────────────────────────────────┐       │
│  │ Schema: "copyleft propagates     │       │
│  │  through compilation"            │       │
│  │ sources: [s[7-11], s[22-25],     │       │
│  │           s[41-43]]              │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Union-find with compaction

- [ ] Union-find data structure with path compression and union by rank
- [ ] Embedding-based similarity for merge candidate selection
- [ ] LLM-generated summaries for cluster roots
- [ ] Expand operation to reinflate compacted clusters
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

## License

AGPL-3.0 — see [LICENSE](LICENSE).
