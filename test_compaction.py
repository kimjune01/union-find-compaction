"""Tests for union-find context compaction."""

from compaction import ContextWindow, Forest, find_closest_pair


class StubEmbedder:
    """Returns a deterministic embedding based on content hash."""

    def embed(self, text: str) -> list[float]:
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


def make_window(hot_size: int = 5, max_cold: int = 3) -> ContextWindow:
    return ContextWindow(StubEmbedder(), StubSummarizer(), hot_size, max_cold)


# ---------------------------------------------------------------------------
# Forest tests
# ---------------------------------------------------------------------------


def test_insert_creates_singleton():
    f = make_forest()
    mid = f.insert(0, "hello")
    assert mid == 0
    assert f.size() == 1
    assert f.cluster_count() == 1
    assert f.members(0) == [0]


def test_find_returns_self_for_singleton():
    f = make_forest()
    mid = f.insert(0, "hello")
    assert f.find(mid) == mid


def test_union_merges_two_singletons():
    f = make_forest()
    a = f.insert(0, "AGPL requires derivative works to share source")
    b = f.insert(1, "Copyleft is irrevocable")
    root = f.union(a, b)
    assert f.cluster_count() == 1
    assert f.find(a) == root
    assert f.find(b) == root
    assert set(f.members(root)) == {a, b}


def test_union_generates_summary():
    f = make_forest()
    a = f.insert(0, "message one")
    b = f.insert(1, "message two")
    root = f.union(a, b)
    summary = f.summary(root)
    assert summary is not None
    assert "message one" in summary
    assert "message two" in summary


def test_compact_returns_summary():
    f = make_forest()
    a = f.insert(0, "alpha")
    b = f.insert(1, "beta")
    root = f.union(a, b)
    text = f.compact(root)
    assert "alpha" in text
    assert "beta" in text


def test_compact_singleton_returns_content():
    f = make_forest()
    f.insert(0, "standalone message")
    assert f.compact(0) == "standalone message"


def test_expand_returns_originals():
    f = make_forest()
    a = f.insert(0, "first")
    b = f.insert(1, "second")
    c = f.insert(2, "third")
    root = f.union(a, b)
    root = f.union(root, c)
    originals = f.expand(root)
    assert set(originals) == {"first", "second", "third"}


def test_path_compression():
    f = make_forest()
    ids = [f.insert(i, f"msg {i}") for i in range(5)]
    f.union(ids[0], ids[1])
    f.union(ids[1], ids[2])
    f.union(ids[2], ids[3])
    f.union(ids[3], ids[4])
    root = f.find(ids[4])
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
    emb.set("northeast", [0.9, 0.1, 0.0])
    emb.set("south", [-1.0, 0.0, 0.0])

    f = Forest(emb, StubSummarizer())
    f.insert(0, "north")
    f.insert(1, "northeast")
    f.insert(2, "south")
    pair = find_closest_pair(f)
    assert pair is not None
    assert set(pair) == {0, 1}


def test_union_idempotent():
    f = make_forest()
    a = f.insert(0, "x")
    b = f.insert(1, "y")
    root1 = f.union(a, b)
    root2 = f.union(a, b)
    assert root1 == root2
    assert f.cluster_count() == 1


# ---------------------------------------------------------------------------
# ContextWindow tests
# ---------------------------------------------------------------------------


def test_messages_enter_hot():
    w = make_window(hot_size=5)
    for i in range(5):
        w.append(f"msg {i}")
    assert w.hot_count == 5
    assert w.cold_cluster_count == 0


def test_overflow_graduates_to_cold():
    w = make_window(hot_size=3, max_cold=10)
    for i in range(5):
        w.append(f"msg {i}")
    # 3 in hot, 2 graduated to cold
    assert w.hot_count == 3
    assert w.cold_cluster_count == 2
    assert w.total_messages == 5


def test_hot_preserves_recency():
    w = make_window(hot_size=3, max_cold=10)
    for i in range(6):
        w.append(f"msg {i}")
    rendered = w.render()
    # Last 3 messages are hot, appear at end in order
    assert rendered[-3:] == ["msg 3", "msg 4", "msg 5"]


def test_cold_compacts_when_over_budget():
    w = make_window(hot_size=2, max_cold=2)
    for i in range(10):
        w.append(f"msg {i}")
    # 2 hot, 8 graduated, cold compacted to <= 2 clusters
    assert w.hot_count == 2
    assert w.cold_cluster_count <= 2


def test_render_has_cold_then_hot():
    w = make_window(hot_size=3, max_cold=10)
    for i in range(6):
        w.append(f"msg {i}")
    rendered = w.render()
    # 3 cold singletons + 3 hot messages = 6 entries
    assert len(rendered) == 6
    # Hot tail is intact
    assert rendered[-1] == "msg 5"
    assert rendered[-2] == "msg 4"
    assert rendered[-3] == "msg 3"


def test_expand_cold_cluster():
    w = make_window(hot_size=2, max_cold=1)
    for i in range(6):
        w.append(f"msg {i}")
    # Everything except last 2 is cold, merged into 1 cluster
    roots = w.forest.roots()
    assert len(roots) == 1
    originals = w.expand(roots[0])
    # All 4 graduated messages are recoverable
    assert len(originals) == 4
    for i in range(4):
        assert f"msg {i}" in originals


def test_total_messages_tracks_all():
    w = make_window(hot_size=3, max_cold=5)
    for i in range(20):
        w.append(f"msg {i}")
    assert w.total_messages == 20


def test_render_length_bounded():
    """Context window render should never exceed hot_size + max_cold_clusters."""
    w = make_window(hot_size=5, max_cold=3)
    for i in range(100):
        w.append(f"msg {i}")
    rendered = w.render()
    assert len(rendered) <= 5 + 3


def test_single_message_stays_hot():
    w = make_window(hot_size=5)
    w.append("only message")
    assert w.hot_count == 1
    assert w.cold_cluster_count == 0
    assert w.render() == ["only message"]


def test_graduation_order_is_fifo():
    """Oldest hot messages graduate first."""
    w = make_window(hot_size=2, max_cold=10)
    w.append("first")
    w.append("second")
    w.append("third")  # pushes "first" to cold
    # "first" should be in cold, "second" and "third" in hot
    hot_contents = [m.content for m in w._hot]
    assert hot_contents == ["second", "third"]
    cold_contents = [w.forest._nodes[mid].content for mid in w.forest._nodes]
    assert "first" in cold_contents
