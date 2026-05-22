"""Test suite for vector-novelty."""

from __future__ import annotations

import numpy as np
import pytest

from vector_novelty import (
    AgentVector,
    VectorTable,
    batch_novelty,
    compute_novelty,
    cosine_distance,
)


class TestCosineDistance:
    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_degenerate_zero_vector(self) -> None:
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        # defined to return 1.0 (orthogonal) when either vector is zero
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_list_input(self) -> None:
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


class TestComputeNovelty:
    def test_empty_population(self) -> None:
        """Novelty against empty pop should be 0.0 (nothing to diverge from)."""
        score = compute_novelty(1, [1.0, 0.0], [])
        assert score == 0.0

    def test_single_item_population(self) -> None:
        """Population of one → centroid is that vector → distance = 0."""
        vec = [0.5, -0.5]
        score = compute_novelty(1, vec, [vec])
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_divergent_agent(self) -> None:
        """An agent pointing opposite to the population should score near 2.0."""
        pop = [[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]
        outlier = [-1.0, 0.0]
        score = compute_novelty(99, outlier, pop)
        assert score > 1.5

    def test_average_agent(self) -> None:
        """An agent close to the population centroid should score near 0.0."""
        pop = [[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]]
        average = [1.0, 0.0]
        score = compute_novelty(1, average, pop)
        assert score < 0.2

    def test_with_numpy_arrays(self) -> None:
        pop = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        vec = np.array([1.0, 1.0], dtype=np.float32)
        score = compute_novelty(1, vec, pop)
        centroid = np.mean(pop, axis=0)
        expected = cosine_distance(vec, centroid)
        assert score == pytest.approx(expected, abs=1e-5)

    def test_api_symmetry_ignores_agent_id(self) -> None:
        """agent_id parameter exists for API symmetry but must not affect result."""
        pop = [[1.0, 0.0], [0.0, 1.0]]
        s1 = compute_novelty(1, [1.0, 1.0], pop)
        s2 = compute_novelty(999, [1.0, 1.0], pop)
        assert s1 == s2


class TestBatchNovelty:
    def test_empty_raises(self) -> None:
        """Empty vectors array should not crash — shape (0, dim) is valid."""
        empty = np.empty((0, 4), dtype=np.float32)
        result = batch_novelty(empty)
        assert result.shape == (0,)

    def test_single_agent_against_self(self) -> None:
        """One agent in pop → centroid is itself → distance 0."""
        vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = batch_novelty(vec)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_two_agents_opposite(self) -> None:
        """Two opposite agents; centroid is origin, both get distance 1.0."""
        vecs = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        scores = batch_novelty(vecs)
        # centroid = (0,0), norm=0 → degenerate path returns 1.0 for both
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[1] == pytest.approx(1.0, abs=1e-6)

    def test_performance_1000_agents(self) -> None:
        """Smoke / perf test: 1000 agents should complete in well under 100 ms."""
        import time

        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((1000, 128), dtype=np.float32)
        t0 = time.perf_counter()
        scores = batch_novelty(vecs)
        elapsed = time.perf_counter() - t0
        assert scores.shape == (1000,)
        assert elapsed < 0.1  # generous bound; usually ~2 ms

    def test_1d_population_input(self) -> None:
        """Pass a 1-D array as population_vectors should still work."""
        # compute_novelty path reshapes 1-D pop
        pop = np.array([1.0, 0.0], dtype=np.float32)
        score = compute_novelty(1, [1.0, 0.0], pop)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_batch_novelty_with_external_population(self) -> None:
        """Override population used for centroid."""
        agents = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        external = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        scores = batch_novelty(agents, population=external)
        # centroid of external is (1,0), so agents[0] distance ≈ 0
        assert scores[0] == pytest.approx(0.0, abs=1e-5)
        # agents[1] is orthogonal to (1,0)
        assert scores[1] == pytest.approx(1.0, abs=1e-5)


class TestVectorTable:
    def test_insert_and_len(self) -> None:
        table = VectorTable(dim=3)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0, 0.0]))
        assert len(table) == 1
        assert 1 in table

    def test_insert_wrong_dim(self) -> None:
        table = VectorTable(dim=3)
        with pytest.raises(ValueError, match="dim"):
            table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))

    def test_delete(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        assert table.delete(1) is True
        assert table.delete(1) is False
        assert len(table) == 0

    def test_get(self) -> None:
        table = VectorTable(dim=2)
        av = AgentVector(agent_id=42, vector=[0.5, -0.5], fitness=0.9)
        table.insert(av)
        retrieved = table.get(42)
        assert retrieved is not None
        assert retrieved.fitness == pytest.approx(0.9)
        assert table.get(99) is None

    def test_vectors_matrix(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        table.insert(AgentVector(agent_id=2, vector=[0.0, 1.0]))
        mat = table.vectors()
        assert mat.shape == (2, 2)
        assert np.allclose(mat, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    def test_novelty_scores_empty(self) -> None:
        table = VectorTable(dim=4)
        scores = table.novelty_scores()
        assert scores.shape == (0,)

    def test_novelty_scores_single(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        scores = table.novelty_scores()
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_most_novel(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        table.insert(AgentVector(agent_id=2, vector=[-1.0, 0.0]))
        table.insert(AgentVector(agent_id=3, vector=[1.0, 0.1]))

        most = table.most_novel(k=1)
        # agent 2 points opposite to centroid → most novel
        assert most[0][0] == 2
        assert most[0][1] > 1.0

    def test_least_novel(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        table.insert(AgentVector(agent_id=2, vector=[1.0, 0.05]))
        table.insert(AgentVector(agent_id=3, vector=[-1.0, 0.0]))

        least = table.least_novel(k=2)
        # agents 1 and 2 are close to each other / centroid
        ids = {aid for aid, _ in least}
        assert ids == {1, 2}

    def test_search_by_vector(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0]))
        table.insert(AgentVector(agent_id=2, vector=[0.0, 1.0]))
        table.insert(AgentVector(agent_id=3, vector=[-1.0, 0.0]))

        results = table.search_by_vector([1.0, 0.0], k=2)
        # closest should be agent 1 (identical), then agent 3 is opposite (furthest)
        # actually we ask for k=2 nearest: agent 1 is distance 0, agent 2 is distance 1
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)
        assert results[1][0] == 2  # orthogonal
        assert results[1][1] == pytest.approx(1.0, abs=1e-6)

    def test_search_by_vector_empty_table(self) -> None:
        table = VectorTable(dim=3)
        assert table.search_by_vector([1.0, 0.0, 0.0], k=3) == []

    def test_repr(self) -> None:
        table = VectorTable(dim=256)
        assert repr(table) == "VectorTable(dim=256, agents=0)"

    def test_overwrite_insert(self) -> None:
        table = VectorTable(dim=2)
        table.insert(AgentVector(agent_id=1, vector=[1.0, 0.0], fitness=0.1))
        table.insert(AgentVector(agent_id=1, vector=[0.0, 1.0], fitness=0.9))
        av = table.get(1)
        assert av is not None
        assert av.fitness == pytest.approx(0.9)
        assert np.allclose(av.to_numpy(), np.array([0.0, 1.0], dtype=np.float32))

    def test_numpy_vector_input(self) -> None:
        table = VectorTable(dim=3)
        vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        table.insert(AgentVector(agent_id=1, vector=vec))
        mat = table.vectors()
        assert np.allclose(mat[0], vec)


class TestAgentVector:
    def test_to_numpy(self) -> None:
        av = AgentVector(agent_id=1, vector=[0.1, 0.2, 0.3])
        arr = av.to_numpy()
        assert arr.dtype == np.float32
        assert np.allclose(arr, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_defaults(self) -> None:
        av = AgentVector(agent_id=7, vector=[1.0, 0.0])
        assert av.fitness == 0.0
        assert av.generation == 0
        assert av.capability_mask == 0xFFFF
        assert av.thermal_pressure == 0.0
