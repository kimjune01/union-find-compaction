# Discussion

Five trials. One statistically significant result. A consistent directional effect across all others. Here's what the data says, what it doesn't say, and what to do next.

## The claim

Union-find compaction preserves more retrievable detail than flat summarization when compression pressure is high and the summarizer is cheap.

The evidence supports this claim directionally across all trials with 200 messages. It reaches statistical significance once (trial 2, p=0.039). The remaining trials hover at p=0.065–0.45, underpowered by the 40-question design.

## What the data shows

### Compression pressure is the gating variable

| Conversation length | Cold messages | Flat accuracy | UF accuracy | Difference |
|---|---|---|---|---|
| 50 messages | 40 | 90% | 90% | 0pp |
| 200 messages | 190 | 62–70% | 78–82% | 12–18pp |

At 40 cold messages, a single LLM call preserves everything. There is nothing to gain from structured compaction. The flat summary is ~300 tokens and comfortably holds 20 planted facts.

At 190 cold messages, the flat summary must compress ~4000 tokens into ~500. Facts that appear once — scrape intervals, webhook paths, cron schedules, filterable attribute counts — get dropped. These are exactly the details a developer asks about mid-session.

Union-find's per-cluster summaries each cover ~20–40 messages instead of 190. Each summary has room for its local facts. The structural ceiling is higher.

### Summarizer quality narrows the gap

| Summarizer | Flat | UF | Gap |
|---|---|---|---|
| Haiku (cheap) | 62–65% | 80–82% | 17pp |
| Sonnet (expensive) | 70% | 78% | 8pp |

A better model makes a better flat summary. Sonnet's flat preserved facts that Haiku's flat dropped (freshness_score, webhook path, scrape interval, Avro format). The gap shrank from 17pp to 8pp.

A perfect summarizer would preserve everything in a single pass, and the gap would vanish. Union-find's advantage is structural: it reduces the per-summary compression ratio. When the model is the bottleneck, structure compensates. When the model is sufficient, structure is redundant.

The realistic case for production context management is cheap models. You don't run Sonnet on background compaction. You run Haiku or smaller. That's where the gap is widest.

### What flat drops

Across trials 2 and 5, the facts flat consistently missed:

- CI pipeline details (test shards, artifact retention, deploy approvals, build timeout)
- Monitoring config (scrape intervals, error thresholds, PagerDuty escalation)
- Data pipeline specifics (Avro format, cron schedules, partition counts)
- Fine-grained API details (webhook endpoints, filterable attribute counts)

The pattern: flat preserves "headline" facts (database version, auth algorithm, search engine) and drops "footnote" facts (configuration values, thresholds, schedules). Both categories matter. The footnotes are what you look up.

### What UF drops

UF's failure modes are different from flat's:

1. **Retrieval misses** (v2 only): The question phrasing doesn't match the right cluster centroid. "What payment processor?" doesn't retrieve the billing cluster if the centroid is dominated by invoice/dunning terms. This is a retrieval problem, not a compaction problem.

2. **Intra-cluster compression**: A fact inside a large merged cluster still gets compressed during summarization. Q15 (filterable attributes 41/64) lived in a search cluster with denser facts that the summarizer prioritized. More clusters = less intra-cluster compression, but also more retrieval noise.

3. **Filler pollution** (trial 4): With loose thresholds, filler messages ("Got it", "Makes sense") merge into substantive clusters and dilute the summary. The hard cap (trial 5) forced these to merge together, creating a filler bucket that retrieval ignores.

## Retrieval vs dump-all

| Strategy | UF accuracy | p-value | Failure mode |
|---|---|---|---|
| Dump all cold clusters | 82% | 0.039 | Injects irrelevant clusters, wastes tokens |
| Retrieve top-3 | 80–82% | 0.065–0.18 | Misses cross-topic questions |

The dump-all approach (v1) gave the strongest statistical signal because it has no retrieval misses — every fact is always in context. The cost: injecting all clusters wastes tokens on irrelevant history.

The retrieval approach (v2) is more efficient but introduces a new failure mode. A question about billing infrastructure might need facts from both the billing and monitoring clusters. k=3 doesn't always cover cross-topic queries.

The practical resolution: dump when under token budget, retrieve when over. The e-class structure supports both — it's an index you can query or enumerate.

## Limitations

### Synthetic conversation

The 200-message conversation is synthetic: 8 topics, interleaved, with planted facts at known positions. Real conversations are messier — topics overlap, facts get corrected, context shifts without clear boundaries.

The synthetic design gives us controlled comparison but may overstate UF's advantage. Real topics don't cluster as cleanly. TF-IDF centroids on natural conversation may produce worse clusters than on our structured fixture.

Conversely, real conversations are longer. Production coding sessions routinely exceed 500 messages. The compression pressure on flat would be even higher.

### TF-IDF embeddings

All clustering used TF-IDF cosine similarity — deterministic and API-free, but lexically shallow. "What port is the database on?" has low TF-IDF similarity to "PostgreSQL 16.3 running on port 5433" because the word overlap is low.

Dense embeddings (text-embedding-3-small, etc.) would produce tighter semantic clusters and better retrieval. The TF-IDF choice was deliberate — repeatability and zero API cost for clustering — but it handicaps the retrieval path. Trial 4's 66 singletons were partly a TF-IDF problem: short filler messages have sparse vectors with low cosine to everything.

### LLM judge variance

Scoring uses an LLM judge with a strict rubric ("does the answer contain the specific detail?"). The judge is the same model as the answerer (Haiku or Sonnet). Judge variance is uncontrolled — a different judge model might score differently.

The rubric is designed to be strict: "PostgreSQL" without "16.3" is a miss. This favors UF, which preserves version numbers in per-cluster summaries. A looser rubric would narrow the gap.

### 40 questions is underpowered

McNemar's exact test on 11 discordant pairs (trial 5) needs 10/11 or 11/11 to favor UF for p < 0.05. We got 9/11. With 80 questions and the same effect size, we'd likely clear alpha.

The pre-registration acknowledged this: "if the effect isn't large, the practical value is questionable anyway." The effect is medium (Cohen's g ≈ 0.3), which is practically meaningful but statistically borderline at n=40.

## What this means for production

### When to use union-find compaction

1. **Cheap summarizer**: Haiku-class models where flat summaries lose detail.
2. **Long sessions**: 200+ messages where compression pressure is high.
3. **Detail-sensitive domains**: Coding, ops, medical — where footnote facts matter.
4. **Multi-topic conversations**: Interleaved topics that benefit from per-topic clustering.

### When flat is fine

1. **Short conversations**: Under 50 messages, there's nothing to gain.
2. **Expensive summarizer**: Sonnet/Opus flat summaries preserve enough.
3. **Coarse recall is sufficient**: If "Stripe" is enough and "Stripe Connect for marketplace payouts" isn't needed.

### The real architectural advantage

The experiment tested recall accuracy on planted facts. This is the narrowest possible test of the thesis. The broader claim from the [README](README.md) is about provenance, expandability, and consolidation:

- **Expand**: When the model gets a vague answer from a summary, it can reinflate the cluster to source messages. Flat can't do this. Not tested.
- **Provenance**: Every summary traces to its sources. Flat summaries are opaque. Not tested.
- **Consolidation**: Cross-session schema formation from repeatedly co-merged clusters. Not tested. This is the eventual goal — the bridge from compaction (cache eviction) to consolidation (changes future processing).

The recall experiment establishes the foundation: the data structure works, the per-cluster summaries preserve more, and the architecture supports retrieval. The interesting questions are in Phase 2.

## Next steps

1. **Increase N**: 80 questions across 400 messages. The effect should clear alpha with the same magnitude.
2. **Dense embeddings**: Replace TF-IDF with text-embedding-3-small for clustering and retrieval. Expected: tighter clusters, better retrieval, fewer singletons.
3. **Adaptive injection**: Dump all when under token budget, retrieve when over. Best of both strategies.
4. **Expand experiment**: Deliberately ask questions that require detail beyond the summary. Measure whether `expand()` recovers facts that `compact()` lost.
5. **Real conversations**: Export a long coding session and run the same comparison. No planted facts — use the LLM judge to evaluate answer quality on naturally occurring questions.
6. **Cross-session consolidation**: The actual thesis. Do clusters that repeatedly co-merge form useful schemas? Does schema-primed context outperform cold-start?
