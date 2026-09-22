"""Tests for vector_novelty.receipts — the echogram engine over ledger rows."""

from vector_novelty.receipts import (
    canonical_json,
    find_fish,
    fnv1a64,
    jaccard,
    lineage_clusters,
    shingle_payload,
    sign_rows,
    sweep,
    window_novelty,
)


def _row(ref, op, tick, payload):
    return {"row_ref": ref, "op": op, "tick": tick, "payload": payload}


def _draw(ref, tick, draw_index, result):
    return _row(ref, "EFFECT", tick, {
        "kind": "draw/v1", "distribution": "svd_init_uniform",
        "draw_index": draw_index, "result": result,
    })


class TestCanonical:
    def test_key_order_irrelevant(self):
        a = shingle_payload({"alpha": 1, "beta": "two", "gamma": [3]})
        b = shingle_payload({"gamma": [3], "alpha": 1, "beta": "two"})
        assert a == b

    def test_canonical_matches_family_contract(self):
        # sorted keys, tight separators, raw UTF-8 — same as substrate-rng TS
        assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'

    def test_fnv1a64_known_vector(self):
        assert fnv1a64(b"") == 0xCBF29CE484222325
        assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C


class TestJaccard:
    def test_identical_is_one(self):
        p = {"kind": "draw/v1", "result": 0.5}
        assert jaccard(shingle_payload(p), shingle_payload(p)) == 1.0

    def test_disjoint_is_zero(self):
        a = shingle_payload({"completely": "different", "shape": 1})
        b = shingle_payload({"zzz": "qqq", "yyy": 0})
        assert jaccard(a, b) == 0.0

    def test_empty_empty_is_one(self):
        assert jaccard(frozenset(), frozenset()) == 1.0


class TestSweep:
    def test_nearest_neighbour_is_self(self):
        rows = sign_rows([
            _draw("r1", 1, 0, 0.11),
            _draw("r2", 2, 1, 0.22),
            _draw("r3", 3, 2, 0.33),
        ])
        echo = sweep(rows[1], rows)
        assert echo[0] == ("r2", 1.0)
        assert {r for r, _ in echo} == {"r1", "r2", "r3"}

    def test_similar_draws_cluster_above_dissimilar(self):
        # two draws of the same distribution sit closer to each other
        # than to a refusal row with unrelated payload
        rows = sign_rows([
            _draw("d1", 1, 0, 0.111),
            _draw("d2", 2, 1, 0.222),
            _row("x1", "REFUSED", 3, {"reason": "rate_limited"}),
        ])
        echo = sweep(rows[0], rows)
        scores = dict(echo)
        assert scores["d2"] > scores["x1"]


class TestWindowNovelty:
    def test_radar_catches_the_anomaly(self):
        # an agent books the same decision shape repeatedly, then one
        # radically different row — the radar must spike exactly there
        stream = sign_rows([
            _draw(f"s{i}", i, i, 0.5 + i * 1e-4) for i in range(6)
        ] + [_row("BOOM", "EFFECT", 7, {
            "kind": "agent_blocked/v1",
            "url": "https://example.invalid",
            "raw_html_sha256": "deadbeef" * 8,
        })])
        novelty = dict(window_novelty(stream))
        baseline = [novelty[f"s{i}"] for i in range(1, 6)]
        assert novelty["s0"] == 1.0  # no history: maximally novel
        assert all(n < 0.9 for n in baseline)
        assert novelty["BOOM"] > max(baseline)

    def test_history_counts_as_past(self):
        hist = sign_rows([_draw("h1", 1, 0, 0.5)])
        stream = sign_rows([_draw("s0", 2, 1, 0.5001)])
        novelty = dict(window_novelty(stream, history=hist))
        assert novelty["s0"] < novelty.get("h1", 1.0)


class TestFish:
    def test_recurring_low_novelty_is_a_fish(self):
        # a new decision kind (agent_blocked) appears repeatedly, never
        # resembling the mainstream draws — the fish finder surfaces it
        rows = [_draw(f"d{i}", i, i, 0.5) for i in range(8)]
        blocked = [
            _row(f"b{i}", "REFUSED", 10 + i, {
                "reason": "agent_blocked/v1",
                "url": f"https://host{i}.invalid",
                "raw_html_sha256": f"{i:064x}",
            })
            for i in range(3)
        ]
        fish = find_fish(sign_rows(rows + blocked), min_recurrence=3)
        refs = {f.row_ref for f in fish}
        assert refs == {"b0", "b1", "b2"}

    def test_one_off_noise_is_not_a_fish(self):
        rows = [_draw(f"d{i}", i, i, 0.5) for i in range(8)]
        rows.append(_row("odd", "REFUSED", 99, {
            "reason": "agent_blocked/v1", "url": "https://x.invalid",
        }))
        assert find_fish(sign_rows(rows), min_recurrence=3) == []


class TestLineage:
    def test_sibling_streams_cluster(self):
        # two agents making "the same kind" of decision (same payload
        # shape with different values) share lineage; a stranger doesn't
        a = [_row(f"a{i}", "EFFECT", i, {"kind": "choose/v1", "candidates": [i, i + 1], "choice": i}) for i in range(3)]
        b = [_row(f"b{i}", "EFFECT", i + 10, {"kind": "choose/v1", "candidates": [9, i], "choice": 9}) for i in range(3)]
        stranger = [_row("z9", "EFFECT", 99, {"kind": "shutdown/v1", "reason": "thermal", "celsius": 91.5})]
        clusters = lineage_clusters(sign_rows(a + b + stranger), threshold=0.35)
        top = clusters[0]
        refs = {s.row_ref for s in top}
        assert "z9" not in refs
        assert len(refs) >= 5  # the six choose rows mostly fuse
