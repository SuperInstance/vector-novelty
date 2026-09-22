"""receipts — JEV as a sensor over receipt-ledger rows (the echogram engine).

Instead of scoring agents against a population centroid (core.py), this
module treats a hash-chained receipt ledger as a *signal*: each row's
payload is canonicalised to JSON and shingled into a set of 64-bit
hashes (fnv1a-64, the fleet's rate-limiter/canonical hash), so a row
becomes a set — and rows relate by Jaccard, the zero-order wavefunction:
an inner product over non-negative amplitudes. One sweep scores every
row against the query; a sliding window turns the stream into novelty
over time; recurring low-similarity rows are the "fish" — signal that a
threshold (argmax) instrument never sees.

Storage discipline (annals doctrine): this module NEVER interprets
payload semantics. It reads `(op, tick, payload)` shapes produced by
whatever ledger booked them; the payload's vocabulary belongs to its
producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ReceiptSignature",
    "canonical_json",
    "fnv1a64",
    "jaccard",
    "shingle_payload",
    "sweep",
    "window_novelty",
    "find_fish",
    "lineage_clusters",
]

FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    """fnv1a-64 — the fleet's canonical hash (substrate-rng ledger, canary)."""
    h = FNV64_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV64_PRIME) & MASK64
    return h


def canonical_json(obj: Any) -> str:
    """Family-canonical JSON: sorted keys, tight separators, raw UTF-8
    (ensure_ascii=False) — byte-identical contract with substrate-rng's
    TypeScript canonical() and the 4quilt Python family."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def shingle_payload(payload: Mapping[str, Any], k: int = 5) -> frozenset[int]:
    """Payload → set of fnv1a-64 char-shingle hashes over canonical JSON.

    Key order in the caller's dict is irrelevant (canonical first).
    k is in CHARACTERS of the canonical serialisation — stable across
    languages because the canonical form is.
    """
    text = canonical_json(payload)
    if len(text) <= k:
        return frozenset({fnv1a64(text.encode("utf-8"))})
    return frozenset(
        fnv1a64(text[i : i + k].encode("utf-8")) for i in range(len(text) - k + 1)
    )


@dataclass(frozen=True)
class ReceiptSignature:
    """A ledger row reduced to its wavefunction amplitude set.

    `row_ref` is opaque (row_hash, tick, whatever the producer uses) —
    this module stores it but never inspects it.
    """

    row_ref: Any
    op: str
    tick: int
    shingles: frozenset[int]


def sign_rows(rows: Iterable[Mapping[str, Any]], k: int = 5) -> list[ReceiptSignature]:
    """Sign any iterable of rows shaped like {row_ref?, op, tick, payload}."""
    out: list[ReceiptSignature] = []
    for r in rows:
        out.append(
            ReceiptSignature(
                row_ref=r.get("row_ref", r.get("row_hash", r.get("tick"))),
                op=str(r.get("op", "?")),
                tick=int(r.get("tick", 0)),
                shingles=shingle_payload(r.get("payload", {}), k=k),
            )
        )
    return out


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """|A∩B| / |A∪B|; empty∪empty = 1.0 (two empty payloads are identical)."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def sweep(query: ReceiptSignature, cells: Sequence[ReceiptSignature]) -> list[tuple[Any, float]]:
    """The ping as echogram: Jaccard of the query against EVERY cell,
    sorted by similarity desc. One pass — the waveform, not particles."""
    return sorted(
        ((c.row_ref, jaccard(query.shingles, c.shingles)) for c in cells),
        key=lambda t: t[1],
        reverse=True,
    )


def window_novelty(
    stream: Sequence[ReceiptSignature],
    history: Sequence[ReceiptSignature] = (),
) -> list[tuple[Any, float]]:
    """Sliding-window behavioural radar: for each row in `stream`, novelty
    = 1 − max Jaccard against everything BEFORE it (history + earlier
    stream rows — the row itself is never in its own pool). A spike =
    this decision has no precedent: exploring, or breaking. This is
    where anomalies surface before outcomes do."""
    past = list(history)
    out: list[tuple[Any, float]] = []
    for sig in stream:
        sims = [jaccard(sig.shingles, p.shingles) for p in past]
        nearest = max(sims) if sims else 0.0
        out.append((sig.row_ref, 1.0 - nearest))
        past.append(sig)
    return out


def find_fish(
    stream: Sequence[ReceiptSignature],
    low_threshold: float = 0.5,
    min_recurrence: int = 3,
) -> list[ReceiptSignature]:
    """Recurring low-similarity rows = emerging canon ("the fish").

    A row is a fish if its best match anywhere else in the stream stays
    BELOW low_threshold yet the same *kind* of row (same op, and Jaccard
    ≥ low_threshold among themselves) recurs at least min_recurrence
    times. Novel once is noise; novel repeatedly is a frontier.
    """
    by_kind: dict[str, list[ReceiptSignature]] = {}
    for sig in stream:
        by_kind.setdefault(sig.op, []).append(sig)

    fish: list[ReceiptSignature] = []
    n = len(stream)
    for op, group in by_kind.items():
        if len(group) < min_recurrence:
            continue
        # the majority stream IS the canon baseline, not a fish
        if len(group) * 2 > n:
            continue
        # cluster within the kind; keep clusters whose INTERNAL similarity
        # is high (a real pattern) but whose EXTERNAL best-match is low
        # (unprecedented against the rest of the stream)
        others = [s for s in stream if s.op != op]
        cluster: list[ReceiptSignature] = []
        for sig in group:
            if not cluster or max(jaccard(sig.shingles, c.shingles) for c in cluster) >= low_threshold:
                cluster.append(sig)
            if len(cluster) >= min_recurrence:
                break
        if len(cluster) < min_recurrence:
            continue
        if not others:
            fish.extend(cluster)
            continue
        external = max(
            jaccard(sig.shingles, o.shingles) for sig in cluster for o in others
        )
        if external < low_threshold:
            fish.extend(cluster)
    return fish


def lineage_clusters(
    stream: Sequence[ReceiptSignature],
    threshold: float = 0.6,
) -> list[list[ReceiptSignature]]:
    """Decision kinship: single-linkage clustering over Jaccard.
    Two rows share lineage if their payloads overlap ≥ threshold —
    "same kind of decision" regardless of who made it or when."""
    clusters: list[list[ReceiptSignature]] = []
    for sig in stream:
        home = None
        for ci, cluster in enumerate(clusters):
            if any(jaccard(sig.shingles, c.shingles) >= threshold for c in cluster):
                home = ci
                break
        if home is None:
            clusters.append([sig])
        else:
            clusters[home].append(sig)
    return sorted(clusters, key=len, reverse=True)
