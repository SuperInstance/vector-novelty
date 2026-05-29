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
