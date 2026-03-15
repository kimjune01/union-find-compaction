"""Tests for union-find context compaction v2."""

from compaction import ContextWindow, Forest, _cosine_similarity, find_closest_pair


class StubEmbedder:
    def embed(self, text: str) -> list[float]:
        h = [ord(c) % 10 for c in text[:4]]
        while len(h) < 4:
            h.append(0)
        return [float(x) for x in h]


class StubSummarizer:
    def summarize(self, messages: list[str]) -> str:
        return " + ".join(messages)


def make_forest() -> Forest:
    return Forest(StubEmbedder(), StubSummarizer())


def make_window(hot_size: int = 5, max_cold: int = 3, threshold: float = 0.3) -> ContextWindow:
    return ContextWindow(StubEmbedder(), StubSummarizer(), hot_size, max_cold, threshold)


# ---------------------------------------------------------------------------
# Forest tests
# ---------------------------------------------------------------------------


def test_insert_creates_singleton():
    f = make_forest()
    f.insert(0, "hello")
    assert f.size() == 1
    assert f.cluster_count() == 1
    assert f.members(0) == [0]


def test_find_returns_self_for_singleton():
    f = make_forest()
    mid = f.insert(0, "hello")
    assert f.find(mid) == mid


def test_union_merges_two():
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
    s = f.summary(root)
    assert s is not None
    assert "message one" in s


def test_compact_returns_summary():
    f = make_forest()
    a = f.insert(0, "alpha")
    b = f.insert(1, "beta")
    root = f.union(a, b)
    assert "alpha" in f.compact(root)


def test_expand_returns_originals():
    f = make_forest()
    a = f.insert(0, "first")
    b = f.insert(1, "second")
    c = f.insert(2, "third")
    root = f.union(a, b)
    root = f.union(root, c)
    assert set(f.expand(root)) == {"first", "second", "third"}


def test_union_idempotent():
    f = make_forest()
    a = f.insert(0, "x")
    b = f.insert(1, "y")
    r1 = f.union(a, b)
    r2 = f.union(a, b)
    assert r1 == r2
    assert f.cluster_count() == 1


def test_centroid_updated_on_union():
    f = make_forest()
    f.insert(0, "aa", [1.0, 0.0])
    f.insert(1, "bb", [0.0, 1.0])
    root = f.union(0, 1)
    centroid = f._centroids[root]
    assert abs(centroid[0] - 0.5) < 0.01
    assert abs(centroid[1] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------


def test_nearest_returns_closest():
    f = make_forest()
    f.insert(0, "a", [1.0, 0.0, 0.0])
    f.insert(1, "b", [0.9, 0.1, 0.0])
    f.insert(2, "c", [-1.0, 0.0, 0.0])
    results = f.nearest([1.0, 0.0, 0.0], k=1)
    assert results == [0]


def test_nearest_returns_k():
    f = make_forest()
    f.insert(0, "a", [1.0, 0.0])
    f.insert(1, "b", [0.9, 0.1])
    f.insert(2, "c", [-1.0, 0.0])
    results = f.nearest([1.0, 0.0], k=2)
    assert len(results) == 2
    assert 0 in results
    assert 1 in results


def test_nearest_root_returns_best():
    f = make_forest()
    f.insert(0, "a", [1.0, 0.0])
    f.insert(1, "b", [-1.0, 0.0])
    result = f.nearest_root([0.9, 0.1])
    assert result is not None
    root, sim = result
    assert root == 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    f = make_forest()
    f.insert(0, "hello")
    f.insert(1, "world")
    f.union(0, 1)

    path = tmp_path / "forest.json"
    f.save(path)

    f2 = make_forest()
    f2.load(path)
    assert f2.size() == 2
    assert f2.find(0) == f2.find(1)
    assert "hello" in f2.compact(f2.find(0))


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
    w = make_window(hot_size=3, max_cold=10, threshold=0.0)
    for i in range(5):
        w.append(f"msg {i}")
    assert w.hot_count == 3
    assert w.total_messages == 5


def test_hot_preserves_recency():
    w = make_window(hot_size=3, max_cold=10, threshold=0.0)
    for i in range(6):
        w.append(f"msg {i}")
    rendered = w.render()
    assert rendered[-3:] == ["msg 3", "msg 4", "msg 5"]


def test_render_with_query_retrieves_relevant():
    """With a query, only top-k cold e-classes are returned."""

    class ControlledEmbedder:
        def __init__(self):
            self._map: dict[str, list[float]] = {}

        def set(self, text: str, vec: list[float]):
            self._map[text] = vec

        def embed(self, text: str) -> list[float]:
            return self._map.get(text, [0.0, 0.0, 0.0])

    emb = ControlledEmbedder()
    # Hot messages
    emb.set("hot1", [0.0, 0.0, 0.0])
    emb.set("hot2", [0.0, 0.0, 0.0])
    # Cold messages — two distinct topics
    emb.set("database setup on port 5433", [1.0, 0.0, 0.0])
    emb.set("database migration from mysql", [0.9, 0.1, 0.0])
    emb.set("auth bug in handler", [0.0, 1.0, 0.0])
    emb.set("auth token expiry", [0.0, 0.9, 0.1])
    # Query about databases
    emb.set("what port is the database on?", [0.95, 0.05, 0.0])

    w = ContextWindow(emb, StubSummarizer(), hot_size=2, max_cold_clusters=10, merge_threshold=0.0)
    # Insert cold messages first (will graduate when hot overflows)
    for msg in ["database setup on port 5433", "database migration from mysql",
                "auth bug in handler", "auth token expiry",
                "hot1", "hot2"]:
        w.append(msg)

    # Without query: all cold clusters returned
    all_ctx = w.render()
    assert len(all_ctx) > 2  # cold + hot

    # With query: only top-1 relevant cold cluster
    db_ctx = w.render(query="what port is the database on?", k=1)
    assert len(db_ctx) == 3  # 1 cold + 2 hot
    # The cold entry should be database-related
    assert "database" in db_ctx[0].lower() or "5433" in db_ctx[0]


def test_incremental_compaction_merges_similar():
    """Similar messages should auto-merge on graduation."""

    class ControlledEmbedder:
        def __init__(self):
            self._map: dict[str, list[float]] = {}

        def set(self, text: str, vec: list[float]):
            self._map[text] = vec

        def embed(self, text: str) -> list[float]:
            return self._map.get(text, [0.0, 0.0])

    emb = ControlledEmbedder()
    emb.set("msg A1", [1.0, 0.0])
    emb.set("msg A2", [0.95, 0.05])  # very similar to A1
    emb.set("msg B1", [0.0, 1.0])  # different
    emb.set("hot1", [0.5, 0.5])
    emb.set("hot2", [0.5, 0.5])

    w = ContextWindow(emb, StubSummarizer(), hot_size=2, max_cold_clusters=10, merge_threshold=0.8)
    w.append("msg A1")
    w.append("msg A2")
    w.append("msg B1")
    w.append("hot1")
    w.append("hot2")

    # A1 and A2 should have merged (sim=0.95 > 0.8), B1 stays separate
    assert w.cold_cluster_count <= 2  # at most A-class + B-class


def test_total_messages_tracks_all():
    w = make_window(hot_size=3, max_cold=5, threshold=0.0)
    for i in range(20):
        w.append(f"msg {i}")
    assert w.total_messages == 20


def test_single_message_stays_hot():
    w = make_window(hot_size=5)
    w.append("only message")
    assert w.hot_count == 1
    assert w.cold_cluster_count == 0
    assert w.render() == ["only message"]


def test_graduation_order_is_fifo():
    w = make_window(hot_size=2, max_cold=10, threshold=0.0)
    w.append("first")
    w.append("second")
    w.append("third")
    hot_contents = [m.content for m in w._hot]
    assert hot_contents == ["second", "third"]


def test_hard_cap_enforced():
    """Batch fallback force-merges when cluster count exceeds cap."""
    w = make_window(hot_size=2, max_cold=3, threshold=1.0)  # threshold=1.0: never merge incrementally
    # Graduate 6 messages — each becomes a singleton, but cap=3 forces merges
    for i in range(8):
        w.append(f"topic_{i} message")
    assert w.cold_cluster_count <= 3


def test_retrieve_min_sim_filters():
    """Clusters below min_sim should not be returned."""

    class ControlledEmbedder:
        def __init__(self):
            self._map: dict[str, list[float]] = {}

        def set(self, text: str, vec: list[float]):
            self._map[text] = vec

        def embed(self, text: str) -> list[float]:
            return self._map.get(text, [0.0, 0.0])

    emb = ControlledEmbedder()
    emb.set("close", [1.0, 0.0])
    emb.set("far", [-1.0, 0.0])
    emb.set("hot1", [0.5, 0.5])
    emb.set("hot2", [0.5, 0.5])
    emb.set("query", [0.9, 0.1])

    w = ContextWindow(emb, StubSummarizer(), hot_size=2, max_cold_clusters=10, merge_threshold=1.0)
    w.append("close")
    w.append("far")
    w.append("hot1")
    w.append("hot2")

    # With high min_sim, "far" should be filtered out
    ctx = w.render(query="query", k=10, min_sim=0.5)
    # Should have 1 cold (close) + 2 hot = 3
    assert len(ctx) == 3
