# vector-novelty

**Novelty detection for vector embeddings** — detect when incoming embeddings are statistically novel relative to a reference distribution.

## What This Gives You

- **Distance-based detection** — k-nearest-neighbor novelty scoring
- **Statistical thresholds** — configurable false positive rates
- **Incremental updates** — update the reference distribution as data arrives
- **Embedding-agnostic** — works with any fixed-dimension vector space

## How It Fits

Used in `plato-training` to detect when agent outputs are genuinely novel vs. variations of known patterns. Feeds into `quality-gate-stream` for novelty-gated quality scoring.

## License

MIT

## Receipts sensor (the echogram engine)

`vector_novelty.receipts` treats a hash-chained receipt ledger as a
signal instead of a log: each row's payload is canonicalised (family
canonical JSON, raw UTF-8) and shingled into fnv1a-64 hash sets, so rows
relate by Jaccard — the zero-order wavefunction, an inner product over
non-negative amplitudes, scored in one pass over every cell:

- `sweep` — the ping as echogram: query vs ALL rows, one pass
- `window_novelty` — behavioural radar: 1 − nearest-precedent similarity
  (anomalies surface before outcomes degrade)
- `find_fish` — recurring low-external-similarity kinds (majority stream
  excluded): novel once is noise, novel repeatedly is a frontier
- `lineage_clusters` — decision kinship: single-linkage clustering of
  rows by payload overlap, whoever made them, whenever

Storage discipline: this module never interprets payload semantics;
producers own their vocabularies. See research/2026-09-22-jev-horizon.md
(H1) and docs/recommendations-from-kimi1/NOTE-03 (cargo-line-tycoon).
