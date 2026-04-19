"""Tests for union-find context compaction v4 (value-based eviction)."""

import math
from compaction import (
    ContextWindow, Forest, ClusterMeta, AndersonEviction,
    _cosine_similarity, _decay, _need, find_closest_pair,
)


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
    """Value-based eviction keeps cluster count at or below cap."""
    w = make_window(hot_size=2, max_cold=3, threshold=1.0)  # threshold=1.0: never merge incrementally
    # Graduate 6 messages — each becomes a singleton, cap=3 forces eviction
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


# ---------------------------------------------------------------------------
# Eviction policy unit tests
# ---------------------------------------------------------------------------


def test_decay_at_zero():
    """decay(0) = e^(-0.5) ≈ 0.607."""
    assert abs(_decay(0) - math.e ** (-0.5)) < 1e-6


def test_decay_monotonic():
    """Decay decreases with elapsed time."""
    assert _decay(0) > _decay(1) > _decay(10) > _decay(100)


def test_need_applies_decay():
    """Score always applies decay, even at elapsed=0."""
    meta = ClusterMeta(strength=1.0, last_access_turn=5)
    score = _need(meta, 5)  # same turn
    assert score < 1.0
    assert abs(score - 1.0 * _decay(0)) < 1e-6


def test_graduation_metadata():
    """Newly graduated clusters get strength=1.0 at current turn."""
    policy = AndersonEviction()
    meta = ClusterMeta()
    policy.on_graduate(meta, current_turn=42)
    assert meta.strength == 1.0
    assert meta.last_access_turn == 42
    assert meta.created_at_turn == 42


def test_retrieval_boost():
    """Retrieval adds 1.0 after decaying existing strength."""
    policy = AndersonEviction()
    meta = ClusterMeta(strength=1.0, last_access_turn=0)
    policy.on_retrieve(meta, current_turn=10)
    expected = 1.0 * _decay(10) + 1.0
    assert abs(meta.strength - expected) < 1e-6
    assert meta.last_access_turn == 10


def test_retrieval_dedup_same_turn():
    """Same-turn retrieval is skipped."""
    policy = AndersonEviction()
    meta = ClusterMeta(strength=1.0, last_access_turn=0)
    policy.on_retrieve(meta, current_turn=5)
    strength_after_first = meta.strength
    policy.on_retrieve(meta, current_turn=5)
    assert meta.strength == strength_after_first


def test_retrieval_after_write_absorb_same_turn():
    """Retrieval after write-absorb in the same turn should still count."""
    policy = AndersonEviction()
    meta = ClusterMeta(strength=1.0, last_access_turn=0)
    policy.on_write_absorb(meta, current_turn=5)
    strength_after_absorb = meta.strength
    policy.on_retrieve(meta, current_turn=5)
    # Retrieval should add a boost on top of the write-absorb
    assert meta.strength > strength_after_absorb


def test_write_absorb_boost():
    """Write absorption adds 0.5 after decaying."""
    policy = AndersonEviction()
    meta = ClusterMeta(strength=1.0, last_access_turn=0)
    policy.on_write_absorb(meta, current_turn=10)
    expected = 1.0 * _decay(10) + 0.5
    assert abs(meta.strength - expected) < 1e-6


def test_strength_bounded_under_sustained_access():
    """Consecutive-turn retrievals converge to ceiling ≈ 2.08."""
    policy = AndersonEviction()
    meta = ClusterMeta(strength=1.0, last_access_turn=0)
    for t in range(1, 200):
        policy.on_retrieve(meta, current_turn=t)
    ceiling = 1.0 / (1.0 - _decay(1))
    assert meta.strength < ceiling + 0.1
    assert meta.strength > ceiling - 0.5


def test_merge_combines_decayed_strengths():
    """on_merge sums both decayed strengths."""
    policy = AndersonEviction()
    winner = ClusterMeta(strength=2.0, last_access_turn=10, created_at_turn=5)
    loser = ClusterMeta(strength=1.5, last_access_turn=8, created_at_turn=3)
    policy.on_merge(winner, loser, current_turn=20)
    expected = 2.0 * _decay(10) + 1.5 * _decay(12)
    assert abs(winner.strength - expected) < 1e-6
    assert winner.last_access_turn == 10  # max of originals, not current_turn
    assert winner.created_at_turn == 3  # min of originals


def test_eviction_chooses_lowest_need():
    """Eviction removes the lowest-scoring cluster, not the closest pair."""
    class FixedEmbedder:
        def __init__(self):
            self._vecs = {}
        def set(self, text, vec):
            self._vecs[text] = vec
        def embed(self, text):
            return self._vecs.get(text, [0.0, 0.0, 0.0])

    emb = FixedEmbedder()
    # 3 cold clusters with distinct embeddings, cap=2
    emb.set("old_topic", [1.0, 0.0, 0.0])
    emb.set("new_topic_A", [0.0, 1.0, 0.0])
    emb.set("new_topic_B", [0.0, 0.0, 1.0])
    # Hot filler
    emb.set("hot1", [0.3, 0.3, 0.3])
    emb.set("hot2", [0.3, 0.3, 0.3])

    w = ContextWindow(emb, StubSummarizer(), hot_size=2, max_cold_clusters=2, merge_threshold=1.0)

    # old_topic graduates first (earliest, lowest turn)
    w.append("old_topic", is_user=True)
    w.append("new_topic_A", is_user=True)
    w.append("new_topic_B", is_user=True)
    w.append("hot1", is_user=True)
    w.append("hot2", is_user=True)

    # Cap=2, 3 cold clusters → one evicted. old_topic graduated earliest, most decayed.
    assert w.cold_cluster_count <= 2
    # old_topic should have been evicted (lowest need)
    remaining_roots = w.forest.roots()
    remaining_content = []
    for r in remaining_roots:
        remaining_content.extend(w.forest.expand(r))
    # old_topic should be gone
    assert "old_topic" not in remaining_content


def test_eviction_deletes_all_cluster_data():
    """After eviction, all member data is purged."""
    f = make_forest()
    a = f.insert(0, "first", [1.0, 0.0])
    b = f.insert(1, "second", [0.9, 0.1])
    root = f.union(a, b)

    f.evict(root)
    assert f.size() == 0
    assert f.cluster_count() == 0
    assert root not in f._summaries
    assert root not in f._centroids
    assert root not in f._meta


def test_eviction_preserves_other_clusters():
    """Evicting one cluster doesn't affect others."""
    f = make_forest()
    a = f.insert(0, "cluster_a", [1.0, 0.0])
    b = f.insert(1, "cluster_b", [0.0, 1.0])

    f.evict(a)
    assert f.size() == 1
    assert f.cluster_count() == 1
    assert f.find(b) == b
    assert f.compact(b) == "cluster_b"


def test_save_load_roundtrip_with_meta(tmp_path):
    """Metadata survives serialization."""
    f = make_forest()
    f.insert(0, "hello")
    f._meta[0].strength = 2.5
    f._meta[0].last_access_turn = 10
    f._meta[0].created_at_turn = 3

    path = tmp_path / "forest.json"
    f.save(path)

    f2 = make_forest()
    f2.load(path)
    assert abs(f2._meta[0].strength - 2.5) < 1e-6
    assert f2._meta[0].last_access_turn == 10
    assert f2._meta[0].created_at_turn == 3


def test_v1_migration_creates_default_meta(tmp_path):
    """Loading a v1 forest without meta creates default ClusterMeta."""
    import json
    # Simulate v1 save (no meta key)
    data = {
        "nodes": {"0": {"content": "hello", "embedding": [1.0], "parent": None, "rank": 0}},
        "summaries": {},
        "children": {"0": [0]},
        "centroids": {"0": [1.0]},
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(data))

    f = make_forest()
    f.load(path)
    assert 0 in f._meta
    assert f._meta[0].strength == 1.0


# ---------------------------------------------------------------------------
# Gemini bug-hunt findings
# ---------------------------------------------------------------------------


def test_no_double_decay_on_merge():
    """on_write_absorb should NOT be called before on_merge (double decay)."""
    class SpyPolicy:
        def __init__(self):
            self.absorbs = 0
            self.merges = 0
        def score(self, meta, turn): return 0.0
        def on_graduate(self, meta, turn): pass
        def on_retrieve(self, meta, turn): pass
        def on_write_absorb(self, meta, turn):
            self.absorbs += 1
        def on_merge(self, winner, loser, turn):
            self.merges += 1

    class UniformEmbedder:
        def embed(self, text): return [1.0, 0.0]

    policy = SpyPolicy()
    w = ContextWindow(
        UniformEmbedder(), StubSummarizer(),
        hot_size=1, max_cold_clusters=10,
        merge_threshold=0.5, eviction_policy=policy
    )
    w.append("A", is_user=True)
    w.append("B", is_user=True)  # A graduates
    w.append("C", is_user=True)  # B graduates and merges into A

    assert policy.merges == 1
    assert policy.absorbs == 0, "on_write_absorb was called before on_merge, causing double decay"


def test_merge_preserves_last_retrieve_turn():
    """on_merge should transfer last_retrieve_turn from loser."""
    policy = AndersonEviction()
    loser = ClusterMeta(strength=2.0, last_access_turn=10, created_at_turn=1)
    loser.last_retrieve_turn = 10
    winner = ClusterMeta(strength=1.0, last_access_turn=12, created_at_turn=12)
    winner.last_retrieve_turn = -1

    policy.on_merge(winner, loser, 12)
    assert winner.last_retrieve_turn == 10


def test_union_cleans_loser_summary():
    """Forest.union() should remove the loser's stale summary."""
    f = make_forest()
    a = f.insert(0, "A")
    b = f.insert(1, "B")
    f._summaries[a] = "summary A"
    f._summaries[b] = "summary B"

    winner = f.union(a, b)
    loser = b if winner == a else a
    assert loser not in f._summaries, "Loser summary leaked"


def test_nearest_root_at_negative_one():
    """nearest_root should work even when best similarity is exactly -1.0."""
    f = make_forest()
    f.insert(0, "A", [-1.0, 0.0])
    result = f.nearest_root([1.0, 0.0])
    assert result is not None
    assert result[0] == 0
