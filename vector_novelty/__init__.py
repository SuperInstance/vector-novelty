"""vector-novelty — centroid-based novelty for agent populations.

Public API
----------
* `AgentVector`    – dataclass: id, vector, fitness, generation, mask, thermal
* `VectorTable`    – in-memory store with novelty scoring and NN search
* `compute_novelty(agent_id, vector, population_vectors)` – scalar novelty
* `batch_novelty(vectors, population)` – vectorised novelty for whole pop
* `cosine_distance(a, b)` – helper, returns [0, 2]
"""

from __future__ import annotations

from vector_novelty.core import (
    AgentVector,
    VectorTable,
    batch_novelty,
    compute_novelty,
    cosine_distance,
)
from vector_novelty.receipts import (
    ReceiptSignature,
    find_fish,
    jaccard,
    lineage_clusters,
    shingle_payload,
    sign_rows,
    sweep,
    window_novelty,
)

__version__ = "0.1.0"

__all__ = [
    "AgentVector",
    "VectorTable",
    "batch_novelty",
    "compute_novelty",
    "cosine_distance",
    "ReceiptSignature",
    "find_fish",
    "jaccard",
    "lineage_clusters",
    "shingle_payload",
    "sign_rows",
    "sweep",
    "window_novelty",
    "__version__",
]