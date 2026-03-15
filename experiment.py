"""Experiment: Union-find compaction vs flat summarization.

Repeatability: clustering is TF-IDF cosine (deterministic, no API).
Model-invariant: pass --model to swap the LLM. If UF wins, it should
win regardless of which model summarizes, answers, and judges.

Usage:
    python3 experiment.py                          # default: haiku
    python3 experiment.py --model claude-sonnet-4-5-20250929
    python3 experiment.py --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from math import comb

import anthropic

from compaction import ContextWindow, Forest, find_closest_pair
from fixtures import CONVERSATION, FACTS
from fixtures_long import CONVERSATION_LONG, FACTS_LONG

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOT_SIZE = 10
MAX_COLD_CLUSTERS = 5

# ---------------------------------------------------------------------------
# Deterministic TF-IDF embedder (no API, reproducible)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFEmbedder:
    """Bag-of-words TF-IDF. Deterministic. No API calls.

    Fit on the full conversation vocabulary so embeddings are
    stable regardless of insertion order.
    """

    def __init__(self, corpus: list[str]) -> None:
        # Build vocabulary from corpus
        self._vocab: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        n_docs = len(corpus)

        for doc in corpus:
            tokens = set(_tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

        self._idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

        self._dim = len(self._vocab)

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        tf: Counter[str] = Counter(tokens)
        vec = [0.0] * self._dim
        for term, count in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                vec[idx] = count * self._idf.get(term, 1.0)
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ---------------------------------------------------------------------------
# LLM summarizer (uses whatever model is passed)
# ---------------------------------------------------------------------------


class ClaudeSummarizer:
    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def summarize(self, messages: list[str]) -> str:
        numbered = "\n".join(f"[{i}] {m}" for i, m in enumerate(messages))
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize these conversation messages into one concise paragraph. "
                        "PRESERVE ALL specific details: version numbers, port numbers, file names, "
                        "line numbers, IP addresses, exact commands, function names, threshold values. "
                        "Do not omit any concrete fact.\n\n" + numbered
                    ),
                }
            ],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Condition 1: Flat summarization
# ---------------------------------------------------------------------------


def run_flat(conversation: list[str], summarizer: ClaudeSummarizer) -> list[str]:
    hot = conversation[-HOT_SIZE:]
    cold = conversation[:-HOT_SIZE]
    if not cold:
        return hot
    summary = summarizer.summarize(cold)
    return [f"[Summary of earlier conversation]\n{summary}"] + hot


# ---------------------------------------------------------------------------
# Condition 2: Union-find compound cache
# ---------------------------------------------------------------------------


def run_unionfind(
    conversation: list[str],
    embedder: TFIDFEmbedder,
    summarizer: ClaudeSummarizer,
) -> list[str]:
    window = ContextWindow(embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS)
    for msg in conversation:
        window.append(msg)
    return window.render()


# ---------------------------------------------------------------------------
# Answering + judging
# ---------------------------------------------------------------------------


def ask_question(
    client: anthropic.Anthropic, model: str, context: list[str], question: str
) -> str:
    context_block = "\n\n".join(f"[{i}] {m}" for i, m in enumerate(context))
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is a conversation history:\n\n{context_block}\n\n"
                    f"Answer this question using ONLY the conversation above. "
                    f"Be specific and exact. If not found, say 'NOT FOUND'.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    return resp.content[0].text


def judge_answer(
    client: anthropic.Anthropic, model: str, question: str, expected: str, actual: str
) -> bool:
    resp = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected (must contain this detail): {expected}\n"
                    f"Actual: {actual}\n\n"
                    f"Does the actual answer contain the specific detail? "
                    f"Reply ONLY 'YES' or 'NO'."
                ),
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


def run_experiment(model: str, long: bool = False) -> list[TrialResult]:
    conversation = CONVERSATION_LONG if long else CONVERSATION
    facts = FACTS_LONG if long else FACTS
    client = anthropic.Anthropic()
    embedder = TFIDFEmbedder(conversation)
    summarizer = ClaudeSummarizer(client, model)

    print("=" * 60)
    print("EXPERIMENT: Union-Find vs Flat Summarization")
    print("=" * 60)
    print(f"Model:       {model}")
    print(f"Conversation: {len(conversation)} messages ({'long' if long else 'short'})")
    print(f"Hot window:  {HOT_SIZE} messages")
    print(f"Cold clusters: {MAX_COLD_CLUSTERS}")
    print(f"Embedding:   TF-IDF (deterministic, {embedder._dim} dims)")
    print(f"Questions:   {len(facts)}")
    print()

    # Build contexts
    print("Building flat context...")
    t0 = time.time()
    flat_ctx = run_flat(conversation, summarizer)
    print(f"  Done in {time.time() - t0:.1f}s. Entries: {len(flat_ctx)}")

    print("Building union-find context...")
    t0 = time.time()
    uf_ctx = run_unionfind(conversation, embedder, summarizer)
    print(f"  Done in {time.time() - t0:.1f}s. Entries: {len(uf_ctx)}")
    print()

    # Log rendered contexts (full, for reproducibility)
    print("-" * 60)
    print("FLAT CONTEXT:")
    print("-" * 60)
    for i, entry in enumerate(flat_ctx):
        print(f"  [{i}] {entry[:150].replace(chr(10), ' ')}")
    print()

    print("-" * 60)
    print("UNION-FIND CONTEXT:")
    print("-" * 60)
    for i, entry in enumerate(uf_ctx):
        print(f"  [{i}] {entry[:150].replace(chr(10), ' ')}")
    print()

    # Ask questions
    results: list[TrialResult] = []
    for i, fact in enumerate(facts):
        q = fact["question"]
        expected = fact["answer"]
        topic = fact["topic"]
        print(f"Q{i + 1:2d} [{topic:6s}] {q}")

        flat_ans = ask_question(client, model, flat_ctx, q)
        uf_ans = ask_question(client, model, uf_ctx, q)

        flat_ok = judge_answer(client, model, q, expected, flat_ans)
        uf_ok = judge_answer(client, model, q, expected, uf_ans)

        mark = lambda ok: "+" if ok else "-"
        print(f"     Flat {mark(flat_ok)}: {flat_ans[:100]}")
        print(f"     UF   {mark(uf_ok)}: {uf_ans[:100]}")
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


def analyze(results: list[TrialResult], tag: str) -> None:
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    flat_score = sum(1 for r in results if r.flat_correct)
    uf_score = sum(1 for r in results if r.uf_correct)
    n = len(results)

    print(f"Flat accuracy:       {flat_score}/{n} = {flat_score / n:.0%}")
    print(f"Union-find accuracy: {uf_score}/{n} = {uf_score / n:.0%}")
    print()

    a = sum(1 for r in results if r.flat_correct and r.uf_correct)
    b = sum(1 for r in results if r.flat_correct and not r.uf_correct)
    c = sum(1 for r in results if not r.flat_correct and r.uf_correct)
    d = sum(1 for r in results if not r.flat_correct and not r.uf_correct)

    print("McNemar contingency table:")
    print(f"  Both correct:      {a}")
    print(f"  Flat only:         {b}")
    print(f"  UF only:           {c}")
    print(f"  Both wrong:        {d}")
    print()

    discordant = b + c
    if discordant == 0:
        print("No discordant pairs. Cannot run McNemar's test.")
        verdict = "H0_NO_DISCORDANT"
        p_exact = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / discordant
        k = max(b, c)
        p_exact = sum(comb(discordant, i) * (0.5**discordant) for i in range(k, discordant + 1))
        p_exact = min(p_exact * 2, 1.0)

        print(f"McNemar chi2 (corrected): {chi2:.3f}")
        print(f"Exact p-value (two-sided): {p_exact:.4f}")

        if p_exact < 0.05:
            winner = "UF" if c > b else "Flat"
            print(f"REJECT H0 (p={p_exact:.4f} < 0.05). Winner: {winner}.")
            verdict = f"REJECT_H0_{winner}"
        else:
            print(f"FAIL TO REJECT H0 (p={p_exact:.4f} >= 0.05).")
            verdict = "FAIL_TO_REJECT"

        cohens_g = (c / discordant) - 0.5
        print(f"Cohen's g: {cohens_g:.3f}")

    print()
    print("Per-topic breakdown:")
    for topic in sorted(set(r.topic for r in results)):
        tr = [r for r in results if r.topic == topic]
        f_ok = sum(1 for r in tr if r.flat_correct)
        u_ok = sum(1 for r in tr if r.uf_correct)
        print(f"  {topic:8s}  Flat: {f_ok}/{len(tr)}  UF: {u_ok}/{len(tr)}")

    # Save raw data
    outfile = f"results-{tag}.json"
    raw = {
        "model": model,
        "hot_size": HOT_SIZE,
        "max_cold_clusters": MAX_COLD_CLUSTERS,
        "flat_accuracy": flat_score / n,
        "uf_accuracy": uf_score / n,
        "mcnemar": {"a": a, "b": b, "c": c, "d": d},
        "p_value": p_exact,
        "verdict": verdict,
        "trials": [
            {
                "question": r.question,
                "topic": r.topic,
                "expected": r.expected,
                "flat_answer": r.flat_answer,
                "flat_correct": r.flat_correct,
                "uf_answer": r.uf_answer,
                "uf_correct": r.uf_correct,
            }
            for r in results
        ],
    }
    with open(outfile, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw data saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for summarize/answer/judge",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use 200-message conversation (40 facts) instead of 50-message (20 facts)",
    )
    args = parser.parse_args()
    tag = f"{args.model}{'-long' if args.long else ''}"
    results = run_experiment(args.model, args.long)
    analyze(results, tag)
