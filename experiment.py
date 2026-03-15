"""Experiment: Union-find compaction vs flat summarization.

Measures recall accuracy on 20 planted facts after compacting
a 50-message conversation to the same token budget.

Two conditions:
  1. Flat: all cold messages summarized into one block.
  2. Union-find: compound cache with per-cluster summaries.

Same hot window (last 10 messages), same answering model,
same summarizer, same token budget for cold zone.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import anthropic
import openai

from compaction import ContextWindow, Forest, _cosine_similarity, find_closest_pair
from fixtures import CONVERSATION, FACTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOT_SIZE = 10  # last 10 messages stay raw
MAX_COLD_CLUSTERS = 5  # union-find compacts cold to ≤5 clusters
ANSWERING_MODEL = "claude-haiku-4-5-20251001"
SUMMARIZE_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

anthropic_client = anthropic.Anthropic()
openai_client = openai.OpenAI()


# ---------------------------------------------------------------------------
# Embedder + Summarizer (real implementations)
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    def embed(self, text: str) -> list[float]:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding


class GPTSummarizer:
    def summarize(self, messages: list[str]) -> str:
        numbered = "\n".join(f"[{i}] {m}" for i, m in enumerate(messages))
        resp = openai_client.chat.completions.create(
            model=SUMMARIZE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following conversation messages into a concise paragraph. Preserve all specific details: version numbers, port numbers, file names, line numbers, IP addresses, exact commands, function names, and threshold values. Do not omit any concrete facts.",
                },
                {"role": "user", "content": numbered},
            ],
            max_tokens=500,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Condition 1: Flat summarization
# ---------------------------------------------------------------------------


def run_flat(conversation: list[str]) -> list[str]:
    """Summarize all cold messages into one block, keep hot raw."""
    hot = conversation[-HOT_SIZE:]
    cold = conversation[:-HOT_SIZE]

    if not cold:
        return hot

    summarizer = GPTSummarizer()
    summary = summarizer.summarize(cold)
    return [f"[Summary of earlier conversation]\n{summary}"] + hot


# ---------------------------------------------------------------------------
# Condition 2: Union-find compound cache
# ---------------------------------------------------------------------------


def run_unionfind(conversation: list[str]) -> list[str]:
    """Compound cache: hot window + cold forest."""
    embedder = OpenAIEmbedder()
    summarizer = GPTSummarizer()
    window = ContextWindow(embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS)

    for msg in conversation:
        window.append(msg)

    return window.render()


# ---------------------------------------------------------------------------
# Answering model
# ---------------------------------------------------------------------------


def ask_question(context: list[str], question: str) -> str:
    """Ask the answering model a factual question given the context."""
    context_block = "\n\n".join(f"[{i}] {m}" for i, m in enumerate(context))
    resp = anthropic_client.messages.create(
        model=ANSWERING_MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"Here is a conversation history:\n\n{context_block}\n\nAnswer this question using ONLY the conversation above. Be specific and exact. If the answer is not in the conversation, say 'NOT FOUND'.\n\nQuestion: {question}",
            }
        ],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_answer(question: str, expected: str, actual: str) -> bool:
    """LLM judge: does the answer contain the specific expected fact?"""
    resp = anthropic_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\nExpected answer (must contain this specific detail): {expected}\nActual answer: {actual}\n\nDoes the actual answer contain the specific detail from the expected answer? Reply ONLY 'YES' or 'NO'.",
            }
        ],
    )
    return resp.content[0].text.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    question: str
    topic: str
    expected: str
    flat_answer: str
    flat_correct: bool
    uf_answer: str
    uf_correct: bool


def run_experiment() -> list[TrialResult]:
    print("=" * 60)
    print("EXPERIMENT: Union-Find vs Flat Summarization")
    print("=" * 60)
    print(f"Conversation: {len(CONVERSATION)} messages")
    print(f"Hot window: {HOT_SIZE} messages")
    print(f"Cold clusters (UF): {MAX_COLD_CLUSTERS}")
    print(f"Questions: {len(FACTS)}")
    print()

    # Build contexts
    print("Building flat context...")
    t0 = time.time()
    flat_ctx = run_flat(CONVERSATION)
    flat_time = time.time() - t0
    print(f"  Done in {flat_time:.1f}s. Entries: {len(flat_ctx)}")
    print()

    print("Building union-find context...")
    t0 = time.time()
    uf_ctx = run_unionfind(CONVERSATION)
    uf_time = time.time() - t0
    print(f"  Done in {uf_time:.1f}s. Entries: {len(uf_ctx)}")
    print()

    # Log rendered contexts
    print("-" * 40)
    print("FLAT CONTEXT:")
    print("-" * 40)
    for i, entry in enumerate(flat_ctx):
        preview = entry[:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    print()

    print("-" * 40)
    print("UNION-FIND CONTEXT:")
    print("-" * 40)
    for i, entry in enumerate(uf_ctx):
        preview = entry[:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    print()

    # Ask questions
    results: list[TrialResult] = []
    for i, fact in enumerate(FACTS):
        q = fact["question"]
        expected = fact["answer"]
        topic = fact["topic"]
        print(f"Q{i+1:2d} [{topic:6s}] {q}")

        flat_ans = ask_question(flat_ctx, q)
        uf_ans = ask_question(uf_ctx, q)

        flat_ok = judge_answer(q, expected, flat_ans)
        uf_ok = judge_answer(q, expected, uf_ans)

        mark_flat = "✓" if flat_ok else "✗"
        mark_uf = "✓" if uf_ok else "✗"
        print(f"     Flat {mark_flat}: {flat_ans[:80]}")
        print(f"     UF   {mark_uf}: {uf_ans[:80]}")
        print()

        results.append(
            TrialResult(
                question=q,
                topic=topic,
                expected=expected,
                flat_answer=flat_ans,
                flat_correct=flat_ok,
                uf_answer=uf_ans,
                uf_correct=uf_ok,
            )
        )

    return results


def analyze(results: list[TrialResult]) -> None:
    """Print results and run McNemar's test."""
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    flat_score = sum(1 for r in results if r.flat_correct)
    uf_score = sum(1 for r in results if r.uf_correct)
    n = len(results)

    print(f"Flat accuracy:       {flat_score}/{n} = {flat_score/n:.0%}")
    print(f"Union-find accuracy: {uf_score}/{n} = {uf_score/n:.0%}")
    print()

    # McNemar contingency table
    # a = both correct, b = flat correct & UF wrong,
    # c = flat wrong & UF correct, d = both wrong
    a = sum(1 for r in results if r.flat_correct and r.uf_correct)
    b = sum(1 for r in results if r.flat_correct and not r.uf_correct)
    c = sum(1 for r in results if not r.flat_correct and r.uf_correct)
    d = sum(1 for r in results if not r.flat_correct and not r.uf_correct)

    print("McNemar contingency table:")
    print(f"  Both correct:         {a}")
    print(f"  Flat only correct:    {b}")
    print(f"  UF only correct:      {c}")
    print(f"  Both wrong:           {d}")
    print()

    discordant = b + c
    if discordant == 0:
        print("No discordant pairs. Cannot compute McNemar's test.")
        print("H₀ cannot be rejected (no disagreement between methods).")
        return

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0

    # For exact test: under H₀, discordant pairs are Binomial(n=b+c, p=0.5)
    # P(X >= c) where X ~ Binom(b+c, 0.5)
    from math import comb

    # Two-sided exact p-value
    k = max(b, c)
    n_disc = b + c
    p_exact = 0.0
    for i in range(k, n_disc + 1):
        p_exact += comb(n_disc, i) * (0.5**n_disc)
    p_exact *= 2  # two-sided
    p_exact = min(p_exact, 1.0)

    print(f"McNemar's χ² (corrected): {chi2:.3f}")
    print(f"Exact p-value (two-sided): {p_exact:.4f}")
    print()

    if p_exact < 0.05:
        if c > b:
            print(f"REJECT H₀ (p={p_exact:.4f} < 0.05). Union-find > Flat.")
        else:
            print(f"REJECT H₀ (p={p_exact:.4f} < 0.05). Flat > Union-find.")
    else:
        print(f"FAIL TO REJECT H₀ (p={p_exact:.4f} >= 0.05).")

    # Effect size: Cohen's g
    if discordant > 0:
        prop = c / discordant
        cohens_g = prop - 0.5
        print(f"Cohen's g: {cohens_g:.3f}")

    # Per-topic breakdown
    print()
    print("Per-topic breakdown:")
    topics = sorted(set(r.topic for r in results))
    for topic in topics:
        topic_results = [r for r in results if r.topic == topic]
        f_ok = sum(1 for r in topic_results if r.flat_correct)
        u_ok = sum(1 for r in topic_results if r.uf_correct)
        tn = len(topic_results)
        print(f"  {topic:8s}  Flat: {f_ok}/{tn}  UF: {u_ok}/{tn}")

    # Save raw data
    raw = []
    for r in results:
        raw.append(
            {
                "question": r.question,
                "topic": r.topic,
                "expected": r.expected,
                "flat_answer": r.flat_answer,
                "flat_correct": r.flat_correct,
                "uf_answer": r.uf_answer,
                "uf_correct": r.uf_correct,
            }
        )
    with open("results.json", "w") as f:
        json.dump(raw, f, indent=2)
    print()
    print("Raw data saved to results.json")


if __name__ == "__main__":
    results = run_experiment()
    analyze(results)
