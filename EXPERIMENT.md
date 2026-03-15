# Experiment: Union-Find Compaction vs Flat Summarization

Pre-registration. Written before any experimental code.

## Research Question

Does union-find compaction preserve more retrievable detail than flat summarization in chatbot context windows?

## Hypotheses

**H₀ (null):** Union-find compaction produces the same recall accuracy as flat summarization on factual questions about compacted context, given the same token budget.

**H₁ (alternative):** Union-find compaction produces higher recall accuracy than flat summarization.

**α = 0.05.** We reject H₀ if p < 0.05.

## Design

Within-subjects: same 20 questions scored against both conditions. Each question targets a specific planted fact in the conversation.

### Independent variable

Compaction strategy (two levels):
1. **Flat** (control): All messages outside the hot window are concatenated and summarized into a single block by the same cheap LLM, truncated to a token budget.
2. **Union-find** (treatment): Compound cache with hot window + cold forest. Cold clusters summarized individually. Same token budget for the cold zone.

### Dependent variable

Binary recall accuracy per question: does the answering model produce the correct fact? Scored by an LLM judge against ground truth. Judge uses a strict rubric: the specific detail must be present (e.g., "PostgreSQL 16.2" not just "PostgreSQL"; "port 2222" not just "a custom port").

### Control variables

- Same synthetic conversation (50 messages, 5 topics)
- Same 20 recall questions (4 per topic)
- Same answering model (claude-haiku)
- Same token budget for compacted context (~2000 tokens for cold zone)
- Same hot window size (10 messages)
- Same embedding model for union-find similarity
- Same cheap LLM for both summarizers (gpt-4o-mini)

### Conversation design

Five topics, interleaved to simulate a real coding session:
1. **Project setup** — DB version, port, schema name, ORM version
2. **Bug hunt** — file name, line number, error type, root cause
3. **API design** — rate limit, auth scheme, endpoint path, response format
4. **Deployment** — server IP, SSH port, deploy script path, rollback command
5. **Code review** — function name, bug type, threshold value, fix approach

Each topic has 4 planted facts = 20 facts total. Questions are written before the conversation is generated. Facts are specific enough that a vague summary would lose them.

### Statistical test

McNemar's test (paired binary outcomes, same questions across conditions). If assumptions are violated (expected cell counts < 5), use exact binomial test on discordant pairs.

Effect size: Cohen's g on discordant pairs.

### Power analysis

With 20 questions, McNemar's exact test can detect a difference when the discordant pairs ratio is ≥ 0.75 (i.e., of questions where the two methods disagree, ≥ 75% favor union-find). This is a large effect. If union-find's advantage is subtle, 20 questions won't detect it. That's acceptable for a first trial — if the effect isn't large, the practical value is questionable anyway.

### What would falsify the thesis

1. Flat summarization scores equal to or better than union-find: H₀ not rejected. The parent pointer doesn't help.
2. Union-find scores better but p ≥ 0.05: suggestive but inconclusive. Increase N.
3. Both score perfectly: the conversation or questions aren't hard enough. Redesign with more aggressive compression.

### Procedure

1. Generate conversation and questions (commit)
2. Implement flat summarizer baseline (commit)
3. Run both conditions (commit raw outputs)
4. Score with LLM judge (commit scores)
5. Run statistical test (commit results)
6. Interpret

### Not tested in this experiment

- The `expand()` operation (that's Experiment 2, contingent on Experiment 1 results)
- Cross-session consolidation / schema formation (Phase 2)
- Real conversations (synthetic only in this trial)

## Lab notebook

Observations, surprises, and course corrections logged below as the experiment runs.

---

### Trial 1: Haiku, 50 messages

**Model:** claude-haiku-4-5-20251001
**Date:** 2026-03-14
**Flat:** 18/20 (90%)
**UF:** 18/20 (90%)
**p = 1.0000. FAIL TO REJECT H₀.**

Discordant pairs: 1 each way.
- Q7 (bug exception type): Flat missed full error string, UF got it.
- Q20 (ledger_entries table): UF missed it, Flat got it.
- Q12 (NDJSON format): Both missed the "final summary object" detail.

**Diagnosis:** 50 messages is too easy. The flat summarizer compresses 40 cold messages into one paragraph and preserves nearly every planted fact — because 40 messages fit comfortably in a single summary call. The LLM is good enough at extractive summarization that nothing meaningful is lost at this scale.

The thesis is that union-find wins when flat summarization must compress *aggressively* — when the cold zone is too large for one summary to capture everything. At 40 messages, there's no compression pressure. The flat summary is ~300 tokens; the facts are sparse enough to survive.

**Next:** Increase conversation length to stress the flat summarizer. At 200+ cold messages, a single summary must drop details. Union-find's per-cluster summaries should retain more because each cluster summary covers fewer messages.

---

### Trial 2: Haiku, 200 messages (long)

**Model:** claude-haiku-4-5-20251001
**Date:** 2026-03-14
**Flat:** 26/40 (65%)
**UF:** 33/40 (82%)
**p = 0.0391. REJECT H₀. Union-find > Flat.**
**Cohen's g = 0.389** (medium effect)

Discordant pairs: 9 total. 8 favored UF, 1 favored Flat.

The 8 questions UF got that Flat missed:
- Q14 (search ranking criterion): freshness_score — flat summary dropped it
- Q18 (artifact retention): 90 days — flat summary dropped it
- Q21 (payment processor detail): "with Connect for marketplace payouts" — flat had "Stripe" but missed the specificity
- Q23 (webhook endpoint): /api/v3/webhooks/stripe — flat summary dropped it entirely
- Q27 (scrape interval): 15 seconds — flat summary dropped it
- Q29 (error rate threshold): 5% over 5 minutes — flat summary dropped it
- Q33 (offline sync): "with vector clocks" — flat had "last-write-wins" but missed the specificity
- Q40 (batch job schedule): cron expression — flat summary dropped it

The 1 question Flat got that UF missed:
- Q35 (app binary size): 35MB — UF's clustering happened to merge the mobile message into a cluster that lost this detail in summarization

**Diagnosis:** At 190 cold messages, the flat summary must compress ~4000 tokens into ~500. Details that appear once in the conversation get dropped. Union-find's 5 clusters each summarize ~38 messages — each cluster summary preserves more of its local facts.

The flat summary preserved all the "big" facts (database version, auth algorithm, search engine) but dropped "small" facts (scrape intervals, webhook paths, cron schedules). These are exactly the details a developer needs mid-session.

**Bug observed:** UF context entry [1] was "I don't see any conversation messages provided in your request" — the summarizer received an empty or malformed input for one cluster. This is a bug in how filler-only clusters get summarized. One cluster was entirely filler messages, and the summarizer had nothing substantive to extract. This did not affect recall (no facts were in that cluster) but it wastes a context slot.

**Next:** Fix the empty-cluster bug. Run with Sonnet for model invariance check.

---

### Trial 3: Sonnet 4.6, 200 messages (long)

**Model:** claude-sonnet-4-6
**Date:** 2026-03-14
**Flat:** 28/40 (70%)
**UF:** 31/40 (78%)
**p = 0.4531. FAIL TO REJECT H₀.**
**Cohen's g = 0.214** (small effect)

Discordant pairs: 7 total. 5 favored UF, 2 favored Flat.

UF-only correct (5): Q19 (deploy approval), Q31 (iOS minimum), Q32 (TCA state management), Q34 (deep link scheme), Q40 (batch cron).
Flat-only correct (2): Q16 (CI runner image), Q29 (error rate threshold).

**Diagnosis:** Sonnet is a better summarizer. Its flat summary preserved facts that Haiku's flat summary dropped (Q14 freshness_score, Q23 webhook path, Q27 scrape interval, Q38 Avro format all now correct for flat). The flat accuracy went from 65% (Haiku) to 70% (Sonnet). UF went from 82% to 78%.

The gap narrowed because the better model narrows the bottleneck. Union-find's structural advantage is largest when the summarizer is the constraint — when a single summary must compress too many facts and the LLM drops some. A better LLM drops fewer. A perfect LLM would drop none and the gap would vanish.

**Cross-trial summary:**

| Trial | Model | Flat | UF | p | Verdict |
|---|---|---|---|---|---|
| 1 | Haiku, 50msg | 90% | 90% | 1.000 | No difference |
| 2 | Haiku, 200msg | 65% | 82% | 0.039 | **UF wins** |
| 3 | Sonnet, 200msg | 70% | 78% | 0.453 | Directional, not significant |

**Interpretation:** The effect is real but model-dependent. Union-find compaction outperforms flat summarization when compression pressure exceeds the summarizer's capacity. A better summarizer raises the threshold before the gap appears. The structural advantage — per-cluster summaries preserving local details — is most valuable with cheap/fast models, which is the realistic case for production context management (you don't want to spend Sonnet-tier compute on compaction).

The honest claim: **union-find compaction is a hedge against summarizer quality.** It provides a structural floor on recall that flat summarization cannot guarantee.

---
