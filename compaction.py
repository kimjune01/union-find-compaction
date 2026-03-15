"""Union-find context compaction with LLM-generated cluster summaries.

v2: incremental compaction + retrieval path.

E-class roots are cache keys for summaries. Union = cheap LLM merge.
Graduation merges into nearest e-class if similar enough, else creates
a new singleton. Retrieval embeds a query and returns top-k e-classes.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


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
    _parent: int | None = None  # union-find parent (None = self)
    _rank: int = 0  # union-find rank


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
        self._embedder = embedder
        self._summarizer = summarizer

    # -- Core operations --

    def insert(self, msg_id: int, content: str, embedding: list[float] | None = None) -> int:
        """Add a message to the forest as a singleton set."""
        if embedding is None:
            embedding = self._embedder.embed(content)
        msg = Message(id=msg_id, content=content, embedding=embedding)
        self._nodes[msg_id] = msg
        self._children[msg_id] = [msg_id]
        self._centroids[msg_id] = list(embedding)
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

        # Union by rank
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

        # Summarize merged e-class via cheap LLM
        member_texts = [self._nodes[mid].content for mid in self._children[root_a]]
        self._summaries[root_a] = self._summarizer.summarize(member_texts)

        return root_a

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
                    "parent": m._parent,
                    "rank": m._rank,
                }
                for mid, m in self._nodes.items()
            },
            "summaries": {str(k): v for k, v in self._summaries.items()},
            "children": {str(k): v for k, v in self._children.items()},
            "centroids": {str(k): v for k, v in self._centroids.items()},
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
                _parent=info["parent"],
                _rank=info["rank"],
            )
        self._summaries = {int(k): v for k, v in data["summaries"].items()}
        self._children = {int(k): v for k, v in data["children"].items()}
        self._centroids = {int(k): v for k, v in data["centroids"].items()}


# ---------------------------------------------------------------------------
# Compound cache: hot window + cold forest
# ---------------------------------------------------------------------------


class ContextWindow:
    """Compound cache for a conversation.

    Hot: most recent messages, verbatim. Never compacted.
    Cold: older messages in a union-find forest of e-classes.

    v2 changes from v1:
    - Incremental compaction: graduating message merges into nearest
      e-class if similarity > threshold, else creates new singleton.
    - Retrieval: render(query) embeds the query and returns top-k
      relevant cold e-classes + hot messages.
    """

    def __init__(
        self,
        embedder: Embedder,
        summarizer: Summarizer,
        hot_size: int = 20,
        max_cold_clusters: int = 10,
        merge_threshold: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._summarizer = summarizer
        self._forest = Forest(embedder, summarizer)
        self._hot: deque[Message] = deque()
        self._hot_size = hot_size
        self._max_cold_clusters = max_cold_clusters
        self._merge_threshold = merge_threshold
        self._next_id = 0

    def append(self, content: str) -> int:
        """Add a message. Enters hot; oldest graduates to cold."""
        msg_id = self._next_id
        self._next_id += 1
        embedding = self._embedder.embed(content)
        msg = Message(id=msg_id, content=content, embedding=embedding)
        self._hot.append(msg)

        while len(self._hot) > self._hot_size:
            graduated = self._hot.popleft()
            self._graduate(graduated)

        return msg_id

    def _graduate(self, msg: Message) -> None:
        """Incremental compaction with batch fallback.

        1. Insert as singleton.
        2. Merge into nearest e-class if similarity > threshold.
        3. Enforce hard cap: force-merge closest pairs until under limit.
        """
        self._forest.insert(msg.id, msg.content, msg.embedding)

        if self._forest.cluster_count() <= 1:
            return

        # Find nearest existing e-class (excluding the singleton we just inserted)
        match = self._forest.nearest_root(msg.embedding)
        if match is None:
            return

        nearest_root, sim = match
        # Don't merge with self
        if nearest_root == msg.id:
            # Find second-nearest
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

        # Batch fallback: enforce hard cap on cluster count
        while self._forest.cluster_count() > self._max_cold_clusters:
            pair = find_closest_pair(self._forest)
            if pair is None:
                break
            self._forest.union(*pair)

    def render(
        self, query: str | None = None, k: int = 3, min_sim: float = 0.05
    ) -> list[str]:
        """Return context: retrieved cold e-classes + hot messages.

        If query is provided, embed it and retrieve top-k cold e-classes
        with similarity >= min_sim. If query is None, return all cold
        summaries (v1 behavior).
        """
        if query is not None and self._forest.cluster_count() > 0:
            query_emb = self._embedder.embed(query)
            top_roots = self._forest.nearest(query_emb, k, min_sim=min_sim)
            cold = [self._forest.compact(r) for r in top_roots]
        else:
            cold = [self._forest.compact(r) for r in self._forest.roots()]

        hot = [m.content for m in self._hot]
        return cold + hot

    def expand(self, root_id: int) -> list[str]:
        return self._forest.expand(root_id)

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
