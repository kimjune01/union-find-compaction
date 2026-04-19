"""Union-find context compaction with LLM-generated cluster summaries.

v4: value-based eviction (Anderson-inspired frequency × recency).

E-class roots are cache keys for summaries. Union = cheap LLM merge.
Graduation merges into nearest e-class if similar enough, else creates
a new singleton. Retrieval embeds a query and returns top-k e-classes.
Timestamps enable structural contradiction resolution in merge.

Eviction replaces force-merge-closest-pair when cluster count exceeds cap.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


E = math.e  # Euler's number, used in decay shift
D = 0.5     # decay exponent (ACT-R canonical default)


class Embedder(Protocol):
    """Maps text to a dense vector. Any embedding model."""

    def embed(self, text: str) -> list[float]: ...


class Summarizer(Protocol):
    """Generates a cluster summary from member messages. Cheap LLM."""

    def summarize(self, messages: list[str]) -> str: ...


@dataclass
class Message:
    id: int
    content: str
    embedding: list[float] = field(default_factory=list)
    timestamp: str | None = None  # ISO-8601, for recency-aware merge
    _parent: int | None = None  # union-find parent (None = self)
    _rank: int = 0  # union-find rank


@dataclass
class ClusterMeta:
    """Per-root eviction metadata. strength is a latent value;
    the eviction score is always strength × decay(elapsed)."""
    strength: float = 1.0
    last_access_turn: int = 0
    created_at_turn: int = 0
    last_retrieve_turn: int = -1  # for same-turn retrieval dedup only


# ---------------------------------------------------------------------------
# Eviction policy
# ---------------------------------------------------------------------------


def _decay(elapsed: int) -> float:
    """Power-law decay with Euler shift: (elapsed + e)^(-d)."""
    return (elapsed + E) ** (-D)


def _need(meta: ClusterMeta, current_turn: int) -> float:
    """Eviction score. Lower = evict first."""
    elapsed = current_turn - meta.last_access_turn
    return meta.strength * _decay(elapsed)


class EvictionPolicy(Protocol):
    """Swappable eviction scoring. Default: Anderson-inspired."""

    def score(self, meta: ClusterMeta, current_turn: int) -> float: ...
    def on_graduate(self, meta: ClusterMeta, current_turn: int) -> None: ...
    def on_retrieve(self, meta: ClusterMeta, current_turn: int) -> None: ...
    def on_write_absorb(self, meta: ClusterMeta, current_turn: int) -> None: ...
    def on_merge(self, winner: ClusterMeta, loser: ClusterMeta, current_turn: int) -> None: ...


class AndersonEviction:
    """Frequency × recency with bounded running strength."""

    def score(self, meta: ClusterMeta, current_turn: int) -> float:
        return _need(meta, current_turn)

    def on_graduate(self, meta: ClusterMeta, current_turn: int) -> None:
        meta.strength = 1.0
        meta.last_access_turn = current_turn
        meta.created_at_turn = current_turn

    def on_retrieve(self, meta: ClusterMeta, current_turn: int) -> None:
        if meta.last_retrieve_turn == current_turn:
            return  # dedup same-turn retrieval
        meta.last_retrieve_turn = current_turn
        elapsed = current_turn - meta.last_access_turn
        meta.strength = meta.strength * _decay(elapsed) + 1.0
        meta.last_access_turn = current_turn

    def on_write_absorb(self, meta: ClusterMeta, current_turn: int) -> None:
        elapsed = current_turn - meta.last_access_turn
        meta.strength = meta.strength * _decay(elapsed) + 0.5
        meta.last_access_turn = current_turn

    def on_merge(self, winner: ClusterMeta, loser: ClusterMeta, current_turn: int) -> None:
        w_elapsed = current_turn - winner.last_access_turn
        l_elapsed = current_turn - loser.last_access_turn
        winner.strength = (
            winner.strength * _decay(w_elapsed)
            + loser.strength * _decay(l_elapsed)
        )
        winner.last_access_turn = max(winner.last_access_turn, loser.last_access_turn)
        winner.created_at_turn = min(winner.created_at_turn, loser.created_at_turn)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Forest:
    """Union-find forest over chat messages.

    Each message starts as a singleton set. Union merges two sets and
    generates a summary for the merged cluster via a cheap LLM. Find
    returns the cluster root with path compression.

    E-class = equivalence class = the set of messages sharing a root.
    The root ID is the cache key for the summary.
    """

    def __init__(self, embedder: Embedder, summarizer: Summarizer) -> None:
        self._nodes: dict[int, Message] = {}
        self._summaries: dict[int, str] = {}  # root_id -> summary
        self._children: dict[int, list[int]] = {}  # root_id -> member ids
        self._centroids: dict[int, list[float]] = {}  # root_id -> centroid
        self._meta: dict[int, ClusterMeta] = {}  # root_id -> eviction metadata
        self._embedder = embedder
        self._summarizer = summarizer

    # -- Core operations --

    def insert(
        self,
        msg_id: int,
        content: str,
        embedding: list[float] | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Add a message to the forest as a singleton set."""
        if embedding is None:
            embedding = self._embedder.embed(content)
        msg = Message(id=msg_id, content=content, embedding=embedding, timestamp=timestamp)
        self._nodes[msg_id] = msg
        self._children[msg_id] = [msg_id]
        self._centroids[msg_id] = list(embedding)
        self._meta[msg_id] = ClusterMeta()
        return msg_id

    def find(self, msg_id: int) -> int:
        """Find cluster root with path compression."""
        node = self._nodes[msg_id]
        if node._parent is None:
            return msg_id
        root = self.find(node._parent)
        node._parent = root  # path compression
        return root

    def union(self, id_a: int, id_b: int) -> int:
        """Merge two e-classes. Summarizer generates the new summary.

        Returns the root of the merged e-class.
        """
        root_a = self.find(id_a)
        root_b = self.find(id_b)
        if root_a == root_b:
            return root_a

        node_a = self._nodes[root_a]
        node_b = self._nodes[root_b]

        # Union by rank — winner is determined by rank, not argument order
        if node_a._rank < node_b._rank:
            root_a, root_b = root_b, root_a
            node_a, node_b = node_b, node_a
        node_b._parent = root_a
        if node_a._rank == node_b._rank:
            node_a._rank += 1

        # Merge children lists
        members_b = self._children.pop(root_b, [])
        self._children.setdefault(root_a, []).extend(members_b)

        # Update centroid (weighted average)
        ca = self._centroids.pop(root_a, [])
        cb = self._centroids.pop(root_b, [])
        na = len(self._children[root_a]) - len(members_b)
        nb = len(members_b)
        if ca and cb:
            dim = len(ca)
            self._centroids[root_a] = [
                (ca[i] * na + cb[i] * nb) / (na + nb) for i in range(dim)
            ]
        elif ca:
            self._centroids[root_a] = ca

        # Eviction metadata merge is handled by ContextWindow._graduate()
        # which calls policy.on_merge() with both metas before popping the loser.
        # Forest.union() does NOT touch _meta — ownership belongs to the caller.

        # Summarize merged e-class via cheap LLM
        # Sort by timestamp so summarizer sees chronological order
        member_ids = self._children[root_a]
        sorted_ids = sorted(
            member_ids,
            key=lambda mid: self._nodes[mid].timestamp or "",
        )
        member_texts = []
        for mid in sorted_ids:
            node = self._nodes[mid]
            if node.timestamp:
                member_texts.append(f"[{node.timestamp}] {node.content}")
            else:
                member_texts.append(node.content)
        self._summaries[root_a] = self._summarizer.summarize(member_texts)

        return root_a

    def evict(self, root_id: int) -> None:
        """Delete an entire e-class: all members, summary, centroid, metadata."""
        root = self.find(root_id)
        member_ids = self._children.pop(root, [])
        for mid in member_ids:
            self._nodes.pop(mid, None)
        self._summaries.pop(root, None)
        self._centroids.pop(root, None)
        self._meta.pop(root, None)

    def compact(self, root_id: int) -> str:
        """Return e-class summary for injection into context."""
        root = self.find(root_id)
        if root not in self._summaries:
            return self._nodes[root].content
        return self._summaries[root]

    def expand(self, root_id: int) -> list[str]:
        """Reinflate an e-class to its source messages."""
        root = self.find(root_id)
        member_ids = self._children.get(root, [root])
        return [self._nodes[mid].content for mid in member_ids]

    # -- Retrieval --

    def nearest(
        self, query_embedding: list[float], k: int = 3, min_sim: float = 0.0
    ) -> list[int]:
        """Return top-k e-class roots by cosine similarity to query.

        Only returns roots with similarity >= min_sim.
        """
        scored = []
        for root in self._children:
            centroid = self._centroids.get(root)
            if centroid:
                sim = _cosine_similarity(query_embedding, centroid)
                if sim >= min_sim:
                    scored.append((sim, root))
        scored.sort(reverse=True)
        return [root for _, root in scored[:k]]

    def nearest_root(self, query_embedding: list[float]) -> tuple[int, float] | None:
        """Return the single nearest e-class root and its similarity."""
        best_sim = -1.0
        best_root = None
        for root in self._children:
            centroid = self._centroids.get(root)
            if centroid:
                sim = _cosine_similarity(query_embedding, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_root = root
        if best_root is None:
            return None
        return best_root, best_sim

    # -- Queries --

    def roots(self) -> list[int]:
        """All current e-class roots."""
        return list(self._children.keys())

    def members(self, root_id: int) -> list[int]:
        """Source message IDs for an e-class."""
        root = self.find(root_id)
        return list(self._children.get(root, [root]))

    def summary(self, root_id: int) -> str | None:
        """E-class summary, or None if singleton."""
        root = self.find(root_id)
        return self._summaries.get(root)

    def size(self) -> int:
        return len(self._nodes)

    def cluster_count(self) -> int:
        return len(self._children)

    # -- Persistence --

    def save(self, path: str | Path) -> None:
        """Serialize forest to JSON for cross-session persistence."""
        data = {
            "nodes": {
                str(mid): {
                    "content": m.content,
                    "embedding": m.embedding,
                    "timestamp": m.timestamp,
                    "parent": m._parent,
                    "rank": m._rank,
                }
                for mid, m in self._nodes.items()
            },
            "summaries": {str(k): v for k, v in self._summaries.items()},
            "children": {str(k): v for k, v in self._children.items()},
            "centroids": {str(k): v for k, v in self._centroids.items()},
            "meta": {
                str(k): {
                    "strength": m.strength,
                    "last_access_turn": m.last_access_turn,
                    "created_at_turn": m.created_at_turn,
                    "last_retrieve_turn": m.last_retrieve_turn,
                }
                for k, m in self._meta.items()
            },
        }
        Path(path).write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        """Deserialize forest from JSON."""
        data = json.loads(Path(path).read_text())
        self._nodes = {}
        for mid_str, info in data["nodes"].items():
            mid = int(mid_str)
            self._nodes[mid] = Message(
                id=mid,
                content=info["content"],
                embedding=info["embedding"],
                timestamp=info.get("timestamp"),
                _parent=info["parent"],
                _rank=info["rank"],
            )
        self._summaries = {int(k): v for k, v in data["summaries"].items()}
        self._children = {int(k): v for k, v in data["children"].items()}
        self._centroids = {int(k): v for k, v in data["centroids"].items()}
        # Migration: existing forests without meta get default values
        self._meta = {}
        if "meta" in data:
            for k, m in data["meta"].items():
                self._meta[int(k)] = ClusterMeta(
                    strength=m["strength"],
                    last_access_turn=m["last_access_turn"],
                    created_at_turn=m["created_at_turn"],
                    last_retrieve_turn=m.get("last_retrieve_turn", -1),
                )
        else:
            # v1 migration: treat all roots as freshly graduated
            for root_id in self._children:
                self._meta[root_id] = ClusterMeta()


# ---------------------------------------------------------------------------
# Compound cache: hot window + cold forest
# ---------------------------------------------------------------------------


class ContextWindow:
    """Compound cache for a conversation.

    Hot: most recent messages, verbatim. Never compacted.
    Cold: older messages in a union-find forest of e-classes.

    v4: value-based eviction replaces force-merge-closest-pair.
    """

    def __init__(
        self,
        embedder: Embedder,
        summarizer: Summarizer,
        hot_size: int = 30,
        max_cold_clusters: int = 10,
        merge_threshold: float = 0.3,
        eviction_policy: EvictionPolicy | None = None,
    ) -> None:
        self._embedder = embedder
        self._summarizer = summarizer
        self._forest = Forest(embedder, summarizer)
        self._hot: deque[Message] = deque()
        self._hot_size = hot_size
        self._max_cold_clusters = max_cold_clusters
        self._merge_threshold = merge_threshold
        self._policy: EvictionPolicy = eviction_policy or AndersonEviction()
        self._turn = 0
        self._next_id = 0

    def append(self, content: str, timestamp: str | None = None, is_user: bool = False) -> int:
        """Add a message. Enters hot; oldest graduates to cold.

        Set is_user=True for user messages to increment the turn counter.
        """
        if is_user:
            self._turn += 1

        msg_id = self._next_id
        self._next_id += 1
        embedding = self._embedder.embed(content)
        msg = Message(id=msg_id, content=content, embedding=embedding, timestamp=timestamp)
        self._hot.append(msg)

        while len(self._hot) > self._hot_size:
            graduated = self._hot.popleft()
            self._graduate(graduated)

        return msg_id

    def _graduate(self, msg: Message) -> None:
        """Incremental compaction with value-based eviction.

        1. Insert as singleton, initialize eviction metadata.
        2. Merge into nearest e-class if similarity > threshold.
        3. Enforce hard cap: evict lowest-need cluster until under limit.
        """
        self._forest.insert(msg.id, msg.content, msg.embedding, msg.timestamp)

        # Initialize eviction metadata for the new singleton
        meta = self._forest._meta.get(msg.id)
        if meta:
            self._policy.on_graduate(meta, self._turn)

        if self._forest.cluster_count() > 1:
            self._try_merge(msg)

        # Value-based eviction: evict lowest-need clusters
        while self._forest.cluster_count() > self._max_cold_clusters:
            self._evict_lowest()

    def _try_merge(self, msg: Message) -> None:
        """Try to merge the newly inserted singleton into the nearest existing cluster."""
        match = self._forest.nearest_root(msg.embedding)
        if match is None:
            return

        nearest_root, sim = match
        # Don't merge with self
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
            # Merge eviction metadata before union (union may change roots)
            msg_root = self._forest.find(msg.id)
            near_root = self._forest.find(nearest_root)
            meta_msg = self._forest._meta.get(msg_root)
            meta_near = self._forest._meta.get(near_root)
            # Write-absorb: the existing cluster is absorbing new content
            if meta_near:
                self._policy.on_write_absorb(meta_near, self._turn)

            winner = self._forest.union(msg.id, nearest_root)

            # After union, winner root has the merged metadata
            # on_merge combines the two metas
            winner_meta = self._forest._meta.get(winner)
            # The loser's meta was already popped by forest — we need to
            # handle merge before union consumes it. Restructure:
            # Actually, forest doesn't pop meta in union. We handle it here.
            loser_root = near_root if winner != near_root else msg_root
            loser_meta = self._forest._meta.pop(loser_root, None)
            if winner_meta and loser_meta:
                self._policy.on_merge(winner_meta, loser_meta, self._turn)

    def _evict_lowest(self) -> None:
        """Evict the cluster with the lowest need score."""
        roots = self._forest.roots()
        if not roots:
            return

        scored = []
        for root in roots:
            meta = self._forest._meta.get(root)
            if meta:
                score = self._policy.score(meta, self._turn)
                scored.append((score, meta.created_at_turn, root))
            else:
                scored.append((0.0, 0, root))

        # Sort: lowest score first. Tie-break: higher created_at first
        # (newer evicted before older), then higher root ID.
        scored.sort(key=lambda x: (x[0], -x[1], -x[2]))
        _, _, victim = scored[0]
        self._forest.evict(victim)

    def render(
        self, query: str | None = None, k: int = 3, min_sim: float = 0.05
    ) -> list[str]:
        """Return context: retrieved cold e-classes + hot messages.

        If query is provided, embed it and retrieve top-k cold e-classes
        with similarity >= min_sim. Tracks retrieval hits for eviction.
        """
        if query is not None and self._forest.cluster_count() > 0:
            query_emb = self._embedder.embed(query)
            top_roots = self._forest.nearest(query_emb, k, min_sim=min_sim)

            # Track retrieval hits
            for root in top_roots:
                meta = self._forest._meta.get(root)
                if meta:
                    self._policy.on_retrieve(meta, self._turn)

            cold = [self._forest.compact(r) for r in top_roots]
        else:
            cold = [self._forest.compact(r) for r in self._forest.roots()]

        hot = [m.content for m in self._hot]
        return cold + hot

    def expand(self, root_id: int) -> list[str]:
        return self._forest.expand(root_id)

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def hot_count(self) -> int:
        return len(self._hot)

    @property
    def cold_cluster_count(self) -> int:
        return self._forest.cluster_count()

    @property
    def total_messages(self) -> int:
        return self._forest.size() + len(self._hot)

    @property
    def forest(self) -> Forest:
        return self._forest


# ---------------------------------------------------------------------------
# Compat: v1 helpers still used by some tests
# ---------------------------------------------------------------------------


def find_closest_pair(forest: Forest) -> tuple[int, int] | None:
    """Find the two closest e-class roots by centroid similarity."""
    roots = forest.roots()
    if len(roots) < 2:
        return None

    best_sim = -1.0
    best_pair = (roots[0], roots[1])
    for i, ra in enumerate(roots):
        ca = forest._centroids.get(ra)
        if not ca:
            continue
        for rb in roots[i + 1 :]:
            cb = forest._centroids.get(rb)
            if not cb:
                continue
            sim = _cosine_similarity(ca, cb)
            if sim > best_sim:
                best_sim = sim
                best_pair = (ra, rb)

    return best_pair
