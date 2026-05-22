# vector-novelty

> Centroid-based novelty for agent populations — O(n) diversity scoring with pure NumPy.

[![CI](https://github.com/SuperInstance/vector-novelty/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperInstance/vector-novelty/actions)

## Install

```bash
pip install vector-novelty
```

Or in development mode:

```bash
git clone https://github.com/SuperInstance/vector-novelty.git
cd vector-novelty
pip install -e .
```

Requires **Python 3.10+** and **NumPy**.

## Quickstart

Find the most creative design in a population of 1 000 agents:

```python
import numpy as np
from vector_novelty import VectorTable, AgentVector

# 1 000 agents, 128-dimensional latent vectors
rng = np.random.default_rng(42)
table = VectorTable(dim=128)

for i in range(1000):
    vec = rng.standard_normal(128)
    table.insert(AgentVector(
        agent_id=i,
        vector=vec,
        fitness=rng.random(),
        generation=3,
    ))

# Score every agent for divergence from the population centroid
scores = table.novelty_scores()        # float32 array, shape (1000,)

# Top 3 most divergent (most "novel") agents
most_novel = table.most_novel(k=3)
for agent_id, score in most_novel:
    print(f"agent {agent_id}: novelty = {score:.4f}")
```

**Typical runtime:** `batch_novelty()` on 1 000 agents × 128 dims takes **~2 ms** on a laptop.

## Why centroid-based?

Pairwise diversity is O(n²).  For 1 000 agents that's 500 000 comparisons.

`vector-novelty` computes a **single population centroid** and measures each agent's cosine distance from it — **O(n)**, fully vectorised with NumPy.  Same biological intuition (how different is this individual from the group average?) but without the quadratic blow-up.

## API Reference

### `AgentVector`

```python
from vector_novelty import AgentVector

av = AgentVector(
    agent_id=42,                # int — unique identifier
    vector=[0.1, -0.2, ...],    # list[float] | np.ndarray
    fitness=0.85,               # [0, 1] — trinity product
    generation=3,               # int — breeding generation
    capability_mask=0x00FF,      # int — 16-bit capability mask
    thermal_pressure=0.2,       # [0, 1] — current thermal load
)
```

### `compute_novelty(agent_id, vector, population_vectors)` → `float`

Scalar novelty for one agent against a population.

```python
from vector_novelty import compute_novelty

score = compute_novelty(
    agent_id=99,
    vector=my_agent_vec,
    population_vectors=[v1, v2, v3, ...],
)
# score in [0, 2]; higher = more divergent from the population centroid
```

### `batch_novelty(vectors, population=None)` → `np.ndarray`

Vectorised novelty for an entire population.  Returns a 1-D float32 array.

```python
from vector_novelty import batch_novelty

scores = batch_novelty(agents_matrix)   # (n_agents, dim) → (n_agents,)
```

If `population` is given, the centroid is computed from that matrix instead of `vectors` itself.

### `VectorTable(dim)`

In-memory manager with insert, delete, novelty scoring, and brute-force NN search.

| Method | Description |
|--------|-------------|
| `insert(AgentVector)` | Add or overwrite an agent |
| `delete(agent_id)` | Remove an agent |
| `get(agent_id)` | Retrieve `AgentVector` or `None` |
| `vectors()` | Full population matrix `(n, dim)` |
| `novelty_scores()` | Divergence of every agent |
| `most_novel(k=1)` | Top-k divergent agents as `[(id, score), ...]` |
| `least_novel(k=1)` | Top-k average agents |
| `search_by_vector(query, k=10)` | Brute-force cosine NN search |

## Tests

```bash
pytest tests/ -q
```

14 tests covering cosine distance, centroid computation, batch novelty, edge cases (empty population, single item), and the full `VectorTable` lifecycle.

## License

MIT — see [LICENSE](./LICENSE).
