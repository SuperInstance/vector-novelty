"""echogram_walk — the whole receipts instrument in one runnable file.

For the outside engineer: this is the bridge. Clone the repo, run

    PYTHONPATH=. python3 examples/echogram_walk.py

and you will see, end to end, what the sensor does with a receipt stream:
the argmax sweep (Jaccard), the interference sweep (SimHash amplitudes),
the 3-valued echogram (construct / unknown / destruct), a PENDING
transition composed under Kleene, and graded truth behind the band.

No dependencies. No network. The receipts are synthetic so the walk is
reproducible; point the same functions at your own WAL rows and the
shape of the answer is identical.

The receipts tell the truth. The instruments read it. This file shows
both, walking.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_novelty.receipts import sign_rows, sweep, sweep_simhash  # noqa: E402
from vector_novelty.ternary import (  # noqa: E402
    Trit,
    compose_transition,
    lukasiewicz_implication,
    sweep_to_trit_field,
    tutor_equal,
)


def _draw(ref, tick, draw_index, result):
    return {
        "row_ref": ref,
        "op": "EFFECT",
        "tick": tick,
        "payload": {
            "kind": "draw/v1",
            "distribution": "svd_init_uniform",
            "draw_index": draw_index,
            "result": result,
        },
    }


def main() -> None:
    # A small ledger: six draws, then a departure with no arrival (PENDING).
    stream = [
        _draw("r0", 1, 0, 0.625),
        _draw("r1", 2, 1, 0.250),
        _draw("r2", 3, 2, 0.125),
        _draw("r3", 4, 3, 0.875),
        _draw("r4", 5, 4, 0.500),
        _draw("r5", 6, 5, 0.375),
        {
            "row_ref": "r6",
            "op": "EFFECT",
            "tick": 7,
            "payload": {"kind": "departure/v1", "room": "harbor", "bound_for": "forge"},
        },
    ]
    query_payload = {"kind": "draw/v1", "distribution": "svd_init_uniform", "draw_index": 3, "result": 0.875}
    query_row = {"row_ref": "QUERY", "op": "EFFECT", "tick": 0, "payload": query_payload}

    print("== receipts (the ledger) ==")
    for r in stream:
        print(f"  {r['row_ref']}  tick={r['tick']}  {r['payload']['kind']}")

    print("\n== argmax sweep (Jaccard) — 'which cell is nearest' ==")
    cells = sign_rows(stream)
    query_sig = sign_rows([query_row])[0]
    for cell, est in sweep(query_sig, cells):
        print(f"  {cell:>3}  {est:+.3f}")

    print("\n== interference sweep (SimHash) — amplitudes, signed ==")
    cos_estimates = {}
    for cell, est in sweep_simhash(query_payload, stream):
        cos_estimates[cell] = est
        print(f"  {cell:>3}  {est:+.3f}")

    print("\n== the 3-valued echogram (construct / unknown / destruct) ==")
    field = sweep_to_trit_field(cos_estimates)
    band = {Trit.Pos: "construct", Trit.Zero: "UNKNOWN", Trit.Neg: "destruct"}
    for cell, trit in field.items():
        print(f"  {cell:>3}  {band[trit]}")

    print("\n== PENDING: departure booked (harbor -> forge), arrival absent ==")
    verdict = compose_transition(Trit.Pos, None)
    print(f"  compose_transition(Pos, None) = {verdict.name}  (the hallway persists, unresolved)")

    print("\n== graded truth behind the band ==")
    print(f"  tutor_equal(0.8750000001, 0.875) = {tutor_equal(0.8750000001, 0.875):.3f}  (TUTOR compute, 1972)")
    print(f"  tutor_equal(0.890, 0.875, tol=0.01) = {tutor_equal(0.890, 0.875, tol=0.01):.3f}  (d = 1.5x tol, half-true)")
    print(f"  lukasiewicz_implication(0.7, 0.4) = {lukasiewicz_implication(0.7, 0.4):.3f}")
    print(f"  lukasiewicz_implication(0.7, 0.9) = {lukasiewicz_implication(0.7, 0.9):.3f}")

    print("\nThe receipts tell the truth. The instruments read it.")


if __name__ == "__main__":
    main()
