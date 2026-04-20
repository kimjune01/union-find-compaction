"""Experiment v4: Value-based eviction vs force-merge vs flat.

Smoke test: same fixtures as the original experiment, adds a third
condition (UF with Anderson eviction). Measures:
1. Factual recall accuracy (same as v1)
2. Cold storage size over time (nodes retained in forest)

Usage:
    python3 experiment_v4.py
    python3 experiment_v4.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import anthropic

from compaction import ContextWindow, Forest, AndersonEviction, find_closest_pair
from fixtures_long import CONVERSATION_LONG, FACTS_LONG, TIMESTAMPS_LONG

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOT_SIZE = 10
MAX_COLD_CLUSTERS = 10
MERGE_THRESHOLD = 0.15
RETRIEVE_K = 3
RETRIEVE_MIN_SIM = 0.05

# ---------------------------------------------------------------------------
# Deterministic TF-IDF embedder (same as original experiment)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFEmbedder:
    def __init__(self, corpus: list[str]) -> None:
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
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ---------------------------------------------------------------------------
# LLM summarizer
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
            messages=[{
                "role": "user",
                "content": (
                    "Summarize these conversation messages into one concise paragraph. "
                    "Messages may include timestamps in [ISO-8601] format. When messages "
                    "contradict each other, ALWAYS prefer the more recent timestamp. "
                    "PRESERVE ALL specific details: version numbers, port numbers, file names, "
                    "line numbers, IP addresses, exact commands, function names, threshold values. "
                    "Do not omit any concrete fact. Drop filler and acknowledgments.\n\n"
                    + numbered
                ),
            }],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Three conditions
# ---------------------------------------------------------------------------


def run_flat(conversation: list[str], summarizer: ClaudeSummarizer) -> list[str]:
    hot = conversation[-HOT_SIZE:]
    cold = conversation[:-HOT_SIZE]
    if not cold:
        return hot
    summary = summarizer.summarize(cold)
    return [f"[Summary of earlier conversation]\n{summary}"] + hot


def build_force_merge_window(
    conversation: list[str],
    embedder: TFIDFEmbedder,
    summarizer: ClaudeSummarizer,
    timestamps: list[str] | None = None,
) -> tuple[ContextWindow, list[int]]:
    """Original UF with force-merge closest pair. Returns (window, storage_trace)."""
    # Use a subclass that force-merges instead of evicting
    window = ForceMergeWindow(embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS, MERGE_THRESHOLD)
    storage_trace = []
    for i, msg in enumerate(conversation):
        ts = timestamps[i] if timestamps else None
        window.append(msg, timestamp=ts, is_user=True)
        storage_trace.append(window.forest.size())
    return window, storage_trace


def build_eviction_window(
    conversation: list[str],
    embedder: TFIDFEmbedder,
    summarizer: ClaudeSummarizer,
    timestamps: list[str] | None = None,
) -> tuple[ContextWindow, list[int]]:
    """UF with value-based eviction. Returns (window, storage_trace)."""
    window = ContextWindow(
        embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS, MERGE_THRESHOLD,
        eviction_policy=AndersonEviction(),
    )
    storage_trace = []
    for i, msg in enumerate(conversation):
        ts = timestamps[i] if timestamps else None
        window.append(msg, timestamp=ts, is_user=True)
        storage_trace.append(window.forest.size())
    return window, storage_trace


class ForceMergeWindow(ContextWindow):
    """Original v3 behavior: force-merge closest pair on cap overflow."""

    def _graduate(self, msg):
        from compaction import _cosine_similarity, Message
        self._forest.insert(msg.id, msg.content, msg.embedding, msg.timestamp)

        # Initialize meta for consistency
        meta = self._forest._meta.get(msg.id)
        if meta:
            self._policy.on_graduate(meta, self._turn)

        if self._forest.cluster_count() <= 1:
            return

        match = self._forest.nearest_root(msg.embedding)
        if match is None:
            return

        nearest_root, sim = match
        if nearest_root == msg.id:
            scored = []
            for root in self._forest.roots():
                if root == msg.id:
                    continue
                centroid = self._forest._centroids.get(root)
                if centroid:
                    s = _cosine_similarity(msg.embedding, centroid)
                    scored.append((s, root))
            if not scored:
                return
            scored.sort(reverse=True)
            sim, nearest_root = scored[0]

        if sim >= self._merge_threshold:
            self._forest.union(msg.id, nearest_root)

        # Force-merge closest pair (original behavior)
        while self._forest.cluster_count() > self._max_cold_clusters:
            pair = find_closest_pair(self._forest)
            if pair is None:
                break
            self._forest.union(*pair)


# ---------------------------------------------------------------------------
# Answering + judging
# ---------------------------------------------------------------------------


def ask_question(client, model, context, question):
    context_block = "\n\n".join(f"[{i}] {m}" for i, m in enumerate(context))
    resp = client.messages.create(
        model=model, max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Here is a conversation history:\n\n{context_block}\n\n"
                f"Answer this question using ONLY the conversation above. "
                f"Be specific and exact. If not found, say 'NOT FOUND'.\n\n"
                f"Question: {question}"
            ),
        }],
    )
    return resp.content[0].text


def judge_answer(client, model, question, expected, actual):
    resp = client.messages.create(
        model=model, max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Expected (must contain this detail): {expected}\n"
                f"Actual: {actual}\n\n"
                f"Does the actual answer contain the specific detail? "
                f"Reply ONLY 'YES' or 'NO'."
            ),
        }],
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
    fm_answer: str  # force-merge
    fm_correct: bool
    ev_answer: str  # eviction
    ev_correct: bool


def run_experiment(model: str) -> tuple[list[TrialResult], list[int], list[int]]:
    conversation = CONVERSATION_LONG
    timestamps = TIMESTAMPS_LONG
    facts = FACTS_LONG
    client = anthropic.Anthropic()
    embedder = TFIDFEmbedder(conversation)
    summarizer = ClaudeSummarizer(client, model)

    print("=" * 60)
    print("EXPERIMENT v4: Flat vs Force-Merge vs Value-Eviction")
    print("=" * 60)
    print(f"Model:        {model}")
    print(f"Conversation: {len(conversation)} messages")
    print(f"Hot window:   {HOT_SIZE}")
    print(f"Cold cap:     {MAX_COLD_CLUSTERS}")
    print(f"Questions:    {len(facts)}")
    print()

    # Build all three conditions
    print("Building flat context...")
    t0 = time.time()
    flat_ctx = run_flat(conversation, summarizer)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Building force-merge UF...")
    t0 = time.time()
    fm_window, fm_trace = build_force_merge_window(conversation, embedder, summarizer, timestamps)
    print(f"  Done in {time.time() - t0:.1f}s. Clusters: {fm_window.cold_cluster_count}, Cold nodes: {fm_window.forest.size()}")

    print("Building eviction UF...")
    t0 = time.time()
    ev_window, ev_trace = build_eviction_window(conversation, embedder, summarizer, timestamps)
    print(f"  Done in {time.time() - t0:.1f}s. Clusters: {ev_window.cold_cluster_count}, Cold nodes: {ev_window.forest.size()}")

    print()
    print(f"Storage comparison:")
    print(f"  Force-merge final cold nodes: {fm_window.forest.size()}")
    print(f"  Eviction final cold nodes:    {ev_window.forest.size()}")
    print(f"  Reduction:                    {fm_window.forest.size() - ev_window.forest.size()} nodes ({(1 - ev_window.forest.size() / max(fm_window.forest.size(), 1)) * 100:.0f}%)")
    print()

    # Ask questions
    results: list[TrialResult] = []
    for i, fact in enumerate(facts):
        q = fact["question"]
        expected = fact["answer"]
        topic = fact["topic"]
        print(f"Q{i + 1:2d} [{topic:8s}] {q}")

        fm_ctx = fm_window.render(query=q, k=RETRIEVE_K, min_sim=RETRIEVE_MIN_SIM)
        ev_ctx = ev_window.render(query=q, k=RETRIEVE_K, min_sim=RETRIEVE_MIN_SIM)

        flat_ans = ask_question(client, model, flat_ctx, q)
        fm_ans = ask_question(client, model, fm_ctx, q)
        ev_ans = ask_question(client, model, ev_ctx, q)

        flat_ok = judge_answer(client, model, q, expected, flat_ans)
        fm_ok = judge_answer(client, model, q, expected, fm_ans)
        ev_ok = judge_answer(client, model, q, expected, ev_ans)

        mark = lambda ok: "+" if ok else "-"
        print(f"     Flat {mark(flat_ok)}: {flat_ans[:80]}")
        print(f"     FM   {mark(fm_ok)}: {fm_ans[:80]}")
        print(f"     EV   {mark(ev_ok)}: {ev_ans[:80]}")
        print()

        results.append(TrialResult(
            question=q, topic=topic, expected=expected,
            flat_answer=flat_ans, flat_correct=flat_ok,
            fm_answer=fm_ans, fm_correct=fm_ok,
            ev_answer=ev_ans, ev_correct=ev_ok,
        ))

    return results, fm_trace, ev_trace


def analyze(results: list[TrialResult], fm_trace: list[int], ev_trace: list[int], model: str):
    n = len(results)
    flat_score = sum(1 for r in results if r.flat_correct)
    fm_score = sum(1 for r in results if r.fm_correct)
    ev_score = sum(1 for r in results if r.ev_correct)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Flat accuracy:          {flat_score}/{n} = {flat_score / n:.0%}")
    print(f"Force-merge accuracy:   {fm_score}/{n} = {fm_score / n:.0%}")
    print(f"Eviction accuracy:      {ev_score}/{n} = {ev_score / n:.0%}")
    print()
    print(f"Eviction vs Force-merge: {ev_score - fm_score:+d} questions")
    print(f"Eviction vs Flat:        {ev_score - flat_score:+d} questions")
    print()

    # Storage trace summary
    print("Storage trace (cold nodes over message count):")
    checkpoints = [0, 25, 50, 75, 100, 125, 150, 175, len(fm_trace) - 1]
    print(f"  {'Msg':>4s}  {'FM nodes':>8s}  {'EV nodes':>8s}  {'Reduction':>9s}")
    for cp in checkpoints:
        if cp < len(fm_trace):
            fm_n = fm_trace[cp]
            ev_n = ev_trace[cp]
            red = f"{(1 - ev_n / max(fm_n, 1)) * 100:.0f}%" if fm_n > 0 else "n/a"
            print(f"  {cp:4d}  {fm_n:8d}  {ev_n:8d}  {red:>9s}")
    print()

    # Per-topic breakdown
    print("Per-topic breakdown:")
    for topic in sorted(set(r.topic for r in results)):
        tr = [r for r in results if r.topic == topic]
        f_ok = sum(1 for r in tr if r.flat_correct)
        m_ok = sum(1 for r in tr if r.fm_correct)
        e_ok = sum(1 for r in tr if r.ev_correct)
        print(f"  {topic:8s}  Flat: {f_ok}/{len(tr)}  FM: {m_ok}/{len(tr)}  EV: {e_ok}/{len(tr)}")

    # Save
    outfile = f"results-v4-{model}.json"
    raw = {
        "model": model,
        "hot_size": HOT_SIZE,
        "max_cold_clusters": MAX_COLD_CLUSTERS,
        "flat_accuracy": flat_score / n,
        "force_merge_accuracy": fm_score / n,
        "eviction_accuracy": ev_score / n,
        "fm_final_cold_nodes": fm_trace[-1] if fm_trace else 0,
        "ev_final_cold_nodes": ev_trace[-1] if ev_trace else 0,
        "fm_storage_trace": fm_trace,
        "ev_storage_trace": ev_trace,
        "trials": [
            {
                "question": r.question,
                "topic": r.topic,
                "expected": r.expected,
                "flat_answer": r.flat_answer, "flat_correct": r.flat_correct,
                "fm_answer": r.fm_answer, "fm_correct": r.fm_correct,
                "ev_answer": r.ev_answer, "ev_correct": r.ev_correct,
            }
            for r in results
        ],
    }
    with open(outfile, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw data saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()
    results, fm_trace, ev_trace = run_experiment(args.model)
    analyze(results, fm_trace, ev_trace, args.model)
