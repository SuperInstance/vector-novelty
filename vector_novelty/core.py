"""vector-novelty — centroid-based novelty computation for agent populations.

Fast O(n) diversity scoring via cosine distance from population centroid.
No external vector-search libraries required; pure NumPy.
"""

from __future__ import annotations

__all__ = [
    "AgentVector",
    "VectorTable",
    "compute_novelty",
    "batch_novelty",
    "cosine_distance",
]

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AgentVector:
    """A single agent's latent vector + fleet metadata.

    Attributes:
        agent_id: Unique agent identifier (int or uint64-compatible).
        vector: Flat float array of dimension ``dim``.  Accepts ``list[float]``
            or ``np.ndarray`` and normalises internally to ``float32``.
        fitness: Trinity product (ethos × pathos × logos) in [0, 1].
        generation: Breeding generation this agent belongs to.
        capability_mask: 16-bit capability mask (default 0xFFFF = all).
        thermal_pressure: Current thermal load on the agent [0, 1].
    """

    agent_id: int
    vector: list[float] | np.ndarray
    fitness: float = 0.0
    generation: int = 0
    capability_mask: int = 0xFFFF
    thermal_pressure: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.vector)

    def to_numpy(self) -> np.ndarray:
        """Return a float32 NumPy array of shape (dim,)."""
        return np.asarray(self.vector, dtype=np.float32)


# ── pure functions ──────────────────────────────────────────


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors, in [0, 2].

    0  → identical direction
    1  → orthogonal
    2  → opposite direction
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0  # degenerate: treat as orthogonal
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))  # clamp for fp safety
    return 1.0 - sim


def compute_novelty(
    agent_id: int,
    vector: list[float] | np.ndarray,
    population_vectors: list[list[float]] | list[np.ndarray] | np.ndarray,
) -> float:
    """Compute novelty as cosine distance from the population centroid.

    Instead of expensive pairwise comparisons (O(n²)), this computes
    a single centroid of the population and measures how far the agent
    deviates from it — O(n) and fully vectorised.

    Args:
        agent_id: Included for API symmetry; not used in calculation.
        vector: The agent's vector.
        population_vectors: Vectors of all other agents in the population.

    Returns:
        Cosine distance in [0, 2].  0 = identical to the population average,
        higher values = more divergent / novel.
    """
    # Suppress unused argument (API symmetry with AgentVector.agent_id)
    _ = agent_id

    if len(population_vectors) == 0:
        return 0.0

    vec = np.asarray(vector, dtype=np.float32)
    pop = np.asarray(population_vectors, dtype=np.float32)

    if pop.ndim == 1:
        # single other agent passed as a flat vector
        pop = pop.reshape(1, -1)

    centroid = np.mean(pop, axis=0)
    return cosine_distance(vec, centroid)


def batch_novelty(
    vectors: np.ndarray,
    population: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised novelty for an entire population.

    Computes the population centroid once, then returns the cosine distance
    of every vector from that centroid.  ~2 ms for 1 000 agents on a laptop.

    Args:
        vectors: Array of shape ``(n_agents, dim)`` — the population itself.
        population: Optional override population.  If ``None``, *vectors* is
            used as its own population (each agent scored against the centroid
            of the whole set, excluding itself when n > 1).

    Returns:
        1-D float32 array of length ``n_agents`` with novelty scores in [0, 2].
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")

    pop = np.asarray(population, dtype=np.float32) if population is not None else vectors
    if pop.ndim != 2:
        raise ValueError(f"population must be 2-D, got shape {pop.shape}")

    n, dim = vectors.shape
    if n == 0:
        return np.array([], dtype=np.float32)

    centroid = np.mean(pop, axis=0)
    cn = np.linalg.norm(centroid)

    # norms of every vector
    vnorms = np.linalg.norm(vectors, axis=1)

    # dot products: (n, dim) · (dim,) → (n,)
    dots = vectors @ centroid

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(
            (vnorms == 0) | (cn == 0),
            0.0,  # degenerate → orthogonal (distance 1.0 after 1-sim)
            dots / (vnorms * cn),
        )
    sims = np.clip(sims, -1.0, 1.0)
    distances = 1.0 - sims
    return distances.astype(np.float32)


# ── container class ─────────────────────────────────────────


class VectorTable:
    """In-memory manager for a population of agent vectors.

    Pure-NumPy storage — no turbovec, no FAISS, no IVF.  Designed for
    small-to-medium populations (tens of thousands) where centroid-based
    diversity is the primary metric.

    Example::

        table = VectorTable(dim=256)
        table.insert(AgentVector(agent_id=1, vector=[0.1, -0.2, ...]))
        scores = table.novelty_scores()          # array of divergences
        most_novel = table.most_novel(k=3)         # top-3 divergent agents
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._agents: dict[int, AgentVector] = {}
        self._matrix: np.ndarray | None = None   # lazy (n, dim) cache
        self._dirty: bool = True

    # ── mutation ────────────────────────────────────────────

    def insert(self, av: AgentVector) -> None:
        """Add or overwrite an agent vector."""
        if av.dim != self.dim:
            raise ValueError(
                f"AgentVector dim {av.dim} != table dim {self.dim}"
            )
        self._agents[av.agent_id] = av
        self._dirty = True

    def delete(self, agent_id: int) -> bool:
        """Remove an agent.  Returns ``True`` if it existed."""
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        self._dirty = True
        return True

    # ── queries ─────────────────────────────────────────────

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def get(self, agent_id: int) -> AgentVector | None:
        """Retrieve an ``AgentVector`` by ID, or ``None``."""
        return self._agents.get(agent_id)

    def ids(self) -> list[int]:
        """All stored agent IDs."""
        return list(self._agents.keys())

    def vectors(self) -> np.ndarray:
        """Return the full population matrix ``(n, dim)`` as float32."""
        self._rebuild_matrix()
        assert self._matrix is not None
        return self._matrix

    def novelty_scores(self) -> np.ndarray:
        """Novelty score for every agent in the table.

        Returns a float32 array aligned with ``self.ids()``.
        """
        if len(self._agents) == 0:
            return np.array([], dtype=np.float32)
        return batch_novelty(self.vectors(), population=self.vectors())

    def most_novel(self, k: int = 1) -> list[tuple[int, float]]:
        """Return the *k* most divergent agents as ``(agent_id, score)``."""
        if k <= 0:
            return []
        scores = self.novelty_scores()
        ids = self.ids()
        if len(ids) == 0:
            return []
        # argsort descending
        idx = np.argsort(scores)[::-1][:k]
        return [(ids[i], float(scores[i])) for i in idx]

    def least_novel(self, k: int = 1) -> list[tuple[int, float]]:
        """Return the *k* least divergent (most average) agents."""
        if k <= 0:
            return []
        scores = self.novelty_scores()
        ids = self.ids()
        if len(ids) == 0:
            return []
        idx = np.argsort(scores)[:k]
        return [(ids[i], float(scores[i])) for i in idx]

    def search_by_vector(
        self,
        query: list[float] | np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Brute-force cosine-distance search for the *k* nearest neighbours.

        No index — O(n) per query.  Fast enough for n < 10 000.
        """
        if k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != table dim {self.dim}")

        mat = self.vectors()
        if mat.shape[0] == 0:
            return []

        # compute cosine distance from query to every row
        qn = np.linalg.norm(q)
        mnorms = np.linalg.norm(mat, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (mat @ q) / (mnorms * qn)
        sims = np.where((mnorms == 0) | (qn == 0), 0.0, sims)
        sims = np.clip(sims, -1.0, 1.0)
        dists = 1.0 - sims

        idx = np.argsort(dists)[:k]
        ids = self.ids()
        return [(ids[i], float(dists[i])) for i in idx]

    # ── internals ───────────────────────────────────────────

    def _rebuild_matrix(self) -> None:
        if not self._dirty and self._matrix is not None:
            return
        if len(self._agents) == 0:
            self._matrix = np.empty((0, self.dim), dtype=np.float32)
            self._dirty = False
            return
        self._matrix = np.vstack([av.to_numpy() for av in self._agents.values()])
        self._dirty = False

    def __repr__(self) -> str:
        return f"VectorTable(dim={self.dim}, agents={len(self)})"
