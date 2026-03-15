"""Tests for union-find context compaction."""

from compaction import Forest, compact_context, find_closest_pair


class StubEmbedder:
    """Returns a deterministic embedding based on content hash."""

    def embed(self, text: str) -> list[float]:
        # Simple: use char ordinals mod 10 as a 4-dim vector
        h = [ord(c) % 10 for c in text[:4]]
        while len(h) < 4:
            h.append(0)
        return [float(x) for x in h]


class StubSummarizer:
    """Concatenates messages with ' + ' as a fake summary."""

    def summarize(self, messages: list[str]) -> str:
        return " + ".join(messages)


def make_forest() -> Forest:
    return Forest(StubEmbedder(), StubSummarizer())


def test_insert_creates_singleton():
    f = make_forest()
    mid = f.insert("hello")
    assert mid == 0
    assert f.size() == 1
    assert f.cluster_count() == 1
    assert f.members(0) == [0]


def test_find_returns_self_for_singleton():
    f = make_forest()
    mid = f.insert("hello")
    assert f.find(mid) == mid


def test_union_merges_two_singletons():
    f = make_forest()
    a = f.insert("AGPL requires derivative works to share source")
    b = f.insert("Copyleft is irrevocable")
    root = f.union(a, b)
    assert f.cluster_count() == 1
    assert f.find(a) == root
    assert f.find(b) == root
    assert set(f.members(root)) == {a, b}


def test_union_generates_summary():
    f = make_forest()
    a = f.insert("message one")
    b = f.insert("message two")
    root = f.union(a, b)
    summary = f.summary(root)
    assert summary is not None
    assert "message one" in summary
    assert "message two" in summary


def test_compact_returns_summary():
    f = make_forest()
    a = f.insert("alpha")
    b = f.insert("beta")
    root = f.union(a, b)
    text = f.compact(root)
    assert "alpha" in text
    assert "beta" in text


def test_compact_singleton_returns_content():
    f = make_forest()
    a = f.insert("standalone message")
    assert f.compact(a) == "standalone message"


def test_expand_returns_originals():
    f = make_forest()
    a = f.insert("first")
    b = f.insert("second")
    c = f.insert("third")
    root = f.union(a, b)
    root = f.union(root, c)
    originals = f.expand(root)
    assert set(originals) == {"first", "second", "third"}


def test_path_compression():
    f = make_forest()
    ids = [f.insert(f"msg {i}") for i in range(5)]
    # Chain: 0 <- 1 <- 2 <- 3 <- 4
    f.union(ids[0], ids[1])
    f.union(ids[1], ids[2])
    f.union(ids[2], ids[3])
    f.union(ids[3], ids[4])
    root = f.find(ids[4])
    # After find with path compression, ids[4] should point directly to root
    assert f._nodes[ids[4]]._parent == root or f._nodes[ids[4]]._parent is None


def test_find_closest_pair():
    """Use a controlled embedder to guarantee which pair is closest."""

    class ControlledEmbedder:
        def __init__(self):
            self._map: dict[str, list[float]] = {}

        def set(self, text: str, vec: list[float]):
            self._map[text] = vec

        def embed(self, text: str) -> list[float]:
            return self._map[text]

    emb = ControlledEmbedder()
    emb.set("north", [1.0, 0.0, 0.0])
    emb.set("northeast", [0.9, 0.1, 0.0])  # close to north
    emb.set("south", [-1.0, 0.0, 0.0])  # far from both

    f = Forest(emb, StubSummarizer())
    f.insert("north")  # id 0
    f.insert("northeast")  # id 1
    f.insert("south")  # id 2
    pair = find_closest_pair(f)
    assert pair is not None
    assert set(pair) == {0, 1}


def test_compact_context_reduces_clusters():
    f = make_forest()
    for i in range(10):
        f.insert(f"message {i}")
    assert f.cluster_count() == 10
    result = compact_context(f, max_clusters=3)
    assert len(result) == 3
    assert f.cluster_count() == 3


def test_union_idempotent():
    f = make_forest()
    a = f.insert("x")
    b = f.insert("y")
    root1 = f.union(a, b)
    root2 = f.union(a, b)
    assert root1 == root2
    assert f.cluster_count() == 1


def test_expand_after_compact_context():
    f = make_forest()
    ids = [f.insert(f"item {i}") for i in range(6)]
    compact_context(f, max_clusters=2)
    # All original messages should still be expandable
    all_expanded = []
    for root in f.roots():
        all_expanded.extend(f.expand(root))
    assert len(all_expanded) == 6
    for i in range(6):
        assert f"item {i}" in all_expanded
