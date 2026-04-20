"""Experiment v4 interleaved: questions asked during conversation.

Instead of batch-ingest-then-query, asks each question ~20 messages
after the fact was planted. This simulates real conversation where
retrieval and ingestion are interleaved — the intended use case for
value-based eviction.

Usage:
    python3 experiment_v4_interleaved.py
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

from compaction import ContextWindow, AndersonEviction, _need, find_closest_pair
from fixtures_long import CONVERSATION_LONG, FACTS_LONG, TIMESTAMPS_LONG

HOT_SIZE = 10
MAX_COLD_CLUSTERS = 10
MERGE_THRESHOLD = 0.15
RETRIEVE_K = 3
RETRIEVE_MIN_SIM = 0.05
QUESTION_DELAY = 20  # ask each question this many messages after fact was planted


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


class ClaudeSummarizer:
    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def summarize(self, messages: list[str]) -> str:
        numbered = "\n".join(f"[{i}] {m}" for i, m in enumerate(messages))
        resp = self._client.messages.create(
            model=self._model, max_tokens=500,
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


class ForceMergeWindow(ContextWindow):
    """Original v3 behavior: force-merge closest pair on cap overflow."""
    def _graduate(self, msg):
        from compaction import _cosine_similarity
        self._forest.insert(msg.id, msg.content, msg.embedding, msg.timestamp)
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
        while self._forest.cluster_count() > self._max_cold_clusters:
            pair = find_closest_pair(self._forest)
            if pair is None:
                break
            self._forest.union(*pair)


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


@dataclass
class TrialResult:
    question: str
    topic: str
    expected: str
    ask_at_msg: int
    fm_answer: str
    fm_correct: bool
    ev_answer: str
    ev_correct: bool


def run_experiment(model: str):
    conversation = CONVERSATION_LONG
    timestamps = TIMESTAMPS_LONG
    facts = FACTS_LONG
    client = anthropic.Anthropic()
    embedder = TFIDFEmbedder(conversation)
    summarizer = ClaudeSummarizer(client, model)

    # Build schedule: ask each question QUESTION_DELAY messages after its fact
    schedule: dict[int, list[dict]] = {}  # msg_index -> [facts to ask]
    for fact in facts:
        ask_at = min(fact["msg_index"] + QUESTION_DELAY, len(conversation) - 1)
        schedule.setdefault(ask_at, []).append(fact)

    print("=" * 60)
    print("EXPERIMENT v4 INTERLEAVED")
    print("=" * 60)
    print(f"Model:        {model}")
    print(f"Conversation: {len(conversation)} messages")
    print(f"Hot window:   {HOT_SIZE}")
    print(f"Cold cap:     {MAX_COLD_CLUSTERS}")
    print(f"Questions:    {len(facts)} (asked {QUESTION_DELAY} msgs after fact)")
    print()

    # Build both windows simultaneously, asking questions at scheduled points
    fm_window = ForceMergeWindow(embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS, MERGE_THRESHOLD)
    ev_window = ContextWindow(
        embedder, summarizer, HOT_SIZE, MAX_COLD_CLUSTERS, MERGE_THRESHOLD,
        eviction_policy=AndersonEviction(),
    )

    results: list[TrialResult] = []
    fm_trace = []
    ev_trace = []

    print("Ingesting conversation with interleaved questions...\n")

    for i, msg in enumerate(conversation):
        ts = timestamps[i] if i < len(timestamps) else None
        fm_window.append(msg, timestamp=ts, is_user=True)
        ev_window.append(msg, timestamp=ts, is_user=True)
        fm_trace.append(fm_window.forest.size())
        ev_trace.append(ev_window.forest.size())

        # Ask scheduled questions at this message index
        if i in schedule:
            for fact in schedule[i]:
                q = fact["question"]
                expected = fact["answer"]
                topic = fact["topic"]

                fm_ctx = fm_window.render(query=q, k=RETRIEVE_K, min_sim=RETRIEVE_MIN_SIM)
                ev_ctx = ev_window.render(query=q, k=RETRIEVE_K, min_sim=RETRIEVE_MIN_SIM)

                fm_ans = ask_question(client, model, fm_ctx, q)
                ev_ans = ask_question(client, model, ev_ctx, q)

                fm_ok = judge_answer(client, model, q, expected, fm_ans)
                ev_ok = judge_answer(client, model, q, expected, ev_ans)

                mark = lambda ok: "+" if ok else "-"
                print(f"  @msg {i:3d} [{topic:8s}] {q[:55]}")
                print(f"    FM {mark(fm_ok)}: {fm_ans[:70]}")
                print(f"    EV {mark(ev_ok)}: {ev_ans[:70]}")
                print()

                results.append(TrialResult(
                    question=q, topic=topic, expected=expected, ask_at_msg=i,
                    fm_answer=fm_ans, fm_correct=fm_ok,
                    ev_answer=ev_ans, ev_correct=ev_ok,
                ))

    # Summary
    n = len(results)
    fm_score = sum(1 for r in results if r.fm_correct)
    ev_score = sum(1 for r in results if r.ev_correct)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Force-merge accuracy:   {fm_score}/{n} = {fm_score / n:.0%}")
    print(f"Eviction accuracy:      {ev_score}/{n} = {ev_score / n:.0%}")
    print(f"Eviction vs Force-merge: {ev_score - fm_score:+d} questions")
    print()
    print(f"Storage:")
    print(f"  Force-merge final cold nodes: {fm_trace[-1]}")
    print(f"  Eviction final cold nodes:    {ev_trace[-1]}")
    print(f"  Reduction:                    {fm_trace[-1] - ev_trace[-1]} nodes ({(1 - ev_trace[-1] / max(fm_trace[-1], 1)) * 100:.0f}%)")
    print()

    # Storage trace
    print("Storage trace:")
    print(f"  {'Msg':>4s}  {'FM':>4s}  {'EV':>4s}  {'Red':>5s}")
    for cp in [0, 25, 50, 75, 100, 125, 150, 175, len(fm_trace) - 1]:
        if cp < len(fm_trace):
            fm_n = fm_trace[cp]
            ev_n = ev_trace[cp]
            red = f"{(1 - ev_n / max(fm_n, 1)) * 100:.0f}%" if fm_n > 0 else "n/a"
            print(f"  {cp:4d}  {fm_n:4d}  {ev_n:4d}  {red:>5s}")
    print()

    # Per-topic
    print("Per-topic:")
    for topic in sorted(set(r.topic for r in results)):
        tr = [r for r in results if r.topic == topic]
        m_ok = sum(1 for r in tr if r.fm_correct)
        e_ok = sum(1 for r in tr if r.ev_correct)
        print(f"  {topic:8s}  FM: {m_ok}/{len(tr)}  EV: {e_ok}/{len(tr)}")

    # Save
    outfile = f"results-v4-interleaved-{model}.json"
    raw = {
        "model": model,
        "hot_size": HOT_SIZE,
        "max_cold_clusters": MAX_COLD_CLUSTERS,
        "question_delay": QUESTION_DELAY,
        "force_merge_accuracy": fm_score / n,
        "eviction_accuracy": ev_score / n,
        "fm_final_cold_nodes": fm_trace[-1],
        "ev_final_cold_nodes": ev_trace[-1],
        "fm_storage_trace": fm_trace,
        "ev_storage_trace": ev_trace,
        "trials": [
            {
                "question": r.question, "topic": r.topic, "expected": r.expected,
                "ask_at_msg": r.ask_at_msg,
                "fm_answer": r.fm_answer, "fm_correct": r.fm_correct,
                "ev_answer": r.ev_answer, "ev_correct": r.ev_correct,
            }
            for r in results
        ],
    }
    with open(outfile, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()
    run_experiment(args.model)
