"""Union-find context compaction with LLM-generated cluster summaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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


class Forest:
    """Union-find forest over chat messages.

    Each message starts as a singleton set. Union merges two sets and
    generates a summary for the merged cluster via a cheap LLM. Find
    returns the cluster root with path compression.
    """

    def __init__(self, embedder: Embedder, summarizer: Summarizer) -> None:
        self._nodes: dict[int, Message] = {}
        self._summaries: dict[int, str] = {}  # root_id -> summary
        self._children: dict[int, list[int]] = {}  # root_id -> member ids
        self._embedder = embedder
        self._summarizer = summarizer
        self._next_id = 0

    # -- Core operations --

    def insert(self, content: str) -> int:
        """Perceive: new message enters as singleton set."""
        msg_id = self._next_id
        self._next_id += 1
        embedding = self._embedder.embed(content)
        msg = Message(id=msg_id, content=content, embedding=embedding)
        self._nodes[msg_id] = msg
        self._children[msg_id] = [msg_id]
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
        """Merge two clusters. Summarizer generates the new cluster summary.

        Returns the root of the merged cluster.
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

        # Summarize merged cluster via cheap LLM
        member_texts = [self._nodes[mid].content for mid in self._children[root_a]]
        self._summaries[root_a] = self._summarizer.summarize(member_texts)

        return root_a

    def compact(self, root_id: int) -> str:
        """Return cluster summary for injection into active context.

        The originals remain in cold storage, addressable via expand().
        """
        root = self.find(root_id)
        if root not in self._summaries:
            return self._nodes[root].content
        return self._summaries[root]

    def expand(self, root_id: int) -> list[str]:
        """Reinflate a compacted cluster to its source messages.

        This is the operation flat summarization cannot do.
        """
        root = self.find(root_id)
        member_ids = self._children.get(root, [root])
        return [self._nodes[mid].content for mid in member_ids]

    # -- Queries --

    def roots(self) -> list[int]:
        """All current cluster roots."""
        return list(self._children.keys())

    def members(self, root_id: int) -> list[int]:
        """Source message IDs for a cluster."""
        root = self.find(root_id)
        return list(self._children.get(root, [root]))

    def summary(self, root_id: int) -> str | None:
        """Cluster summary, or None if singleton."""
        root = self.find(root_id)
        return self._summaries.get(root)

    def size(self) -> int:
        """Total messages in the forest."""
        return len(self._nodes)

    def cluster_count(self) -> int:
        """Number of disjoint clusters."""
        return len(self._children)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_closest_pair(forest: Forest) -> tuple[int, int] | None:
    """Find the two closest cluster roots by centroid similarity.

    This is the merge candidate selector — the Filter step.
    """
    roots = forest.roots()
    if len(roots) < 2:
        return None

    centroids: dict[int, list[float]] = {}
    for root in roots:
        member_ids = forest.members(root)
        embeddings = [forest._nodes[mid].embedding for mid in member_ids]
        dim = len(embeddings[0])
        centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
        centroids[root] = centroid

    best_sim = -1.0
    best_pair = (roots[0], roots[1])
    for i, ra in enumerate(roots):
        for rb in roots[i + 1 :]:
            sim = _cosine_similarity(centroids[ra], centroids[rb])
            if sim > best_sim:
                best_sim = sim
                best_pair = (ra, rb)

    return best_pair


def compact_context(
    forest: Forest,
    max_clusters: int,
) -> list[str]:
    """Compact the forest until at most max_clusters remain.

    Returns the active context: summaries for merged clusters,
    raw content for singletons.
    """
    while forest.cluster_count() > max_clusters:
        pair = find_closest_pair(forest)
        if pair is None:
            break
        forest.union(*pair)

    return [forest.compact(root) for root in forest.roots()]
