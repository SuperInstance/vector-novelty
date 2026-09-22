"""ternary — GF(3) amplitudes and 3-valued composition for receipt rows.

The wavefunction reading of a ledger has three amplitude bands:
constructive (cos > +tau), destructive (cos < -tau), and the measured
noise band around zero — which is NOT "false" but UNKNOWN (Kleene's U):
the rows are unrelated, or the instrument hasn't seen enough to say.
PENDING transitions (departure booked, arrival absent) are receipts
whose truth value is U. This module gives the receipts sensor the
composition semantics for all three.

Semantics ported from the fleet's Ternary Fleet (SuperInstance, 355
crates), read by scout-γ 2026-09-22:

- GF(3) arithmetic, ternary-coordination/src/ternary.rs (wrap-around add:
  Pos+Pos=Neg — interference is literally addition with wrap)
- Kleene strong K3, ternary-logic/src/lib.rs:110-130
  (known-construct AND U = U; known-destruct AND U = F)
- Bochvar B3 toxic-U, ternary-logic/src/lib.rs:146-156 (quarantine)
- Trit::consensus(), ternary-core/src/lib.rs:40-44 (agree-or-Unknown)
- TERNARY-DESIGN-PATTERN.md is the semantic dictionary; this module does
  not over-claim physics — banding thresholds are instrument constants.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

__all__ = [
    "Trit",
    "TRIT_COS_TAU",
    "cos_to_trit",
    "compose_transition",
    "kleene_and",
    "kleene_or",
    "bochvar_and",
    "consensus",
    "sweep_to_trit_field",
]

# Instrument constant: the measured SimHash noise band for unrelated
# payloads (|est| < ~0.35 in tests). Cosines inside the band are UNKNOWN,
# not weak agreement — unknown is not false.
TRIT_COS_TAU = 0.35


class Trit(IntEnum):
    """GF(3) signed amplitude: Neg=-1, Zero=0, Pos=+1 (2 ≡ -1 mod 3)."""

    Neg = -1
    Zero = 0
    Pos = 1

    def __add__(self, other: "Trit") -> "Trit":
        r = (int(self) + int(other)) % 3  # 2 ≡ -1
        return Trit(r - 3 if r == 2 else r)

    def __mul__(self, other: "Trit") -> "Trit":
        # GF(3) field product = sign rule; Zero is absorbing
        return Trit(int(self) * int(other))


def cos_to_trit(cos_estimate: float, tau: float = TRIT_COS_TAU) -> Trit:
    """Band a sweep cosine into an amplitude trit.

    cos > +tau → Pos (constructive); cos < -tau → Neg (destructive);
    inside the band → Zero = UNKNOWN. The band is the honest instrument
    threshold, not a claim about the payloads' semantics."""
    if cos_estimate > tau:
        return Trit.Pos
    if cos_estimate < -tau:
        return Trit.Neg
    return Trit.Zero


def kleene_and(a: Trit, b: Trit) -> Trit:
    """Kleene strong conjunction for composing receipt truth values.

    Neg ∧ anything = Neg (known-destruct poisons the chain);
    Zero ∧ Pos = Zero (PENDING stays unresolved); Pos ∧ Pos = Pos.
    A chain of receipts with one unresolved PENDING link composes to U;
    one known-destruct (a REFUSED with evidence) kills the conjunction."""
    if Trit.Neg in (a, b):
        return Trit.Neg
    if Trit.Zero in (a, b):
        return Trit.Zero
    return Trit.Pos


def kleene_or(a: Trit, b: Trit) -> Trit:
    """Kleene strong disjunction: Pos ∨ anything = Pos (witnessed
    construct dominates); Zero ∨ Neg = Zero; Neg ∨ Neg = Neg."""
    if Trit.Pos in (a, b):
        return Trit.Pos
    if Trit.Zero in (a, b):
        return Trit.Zero
    return Trit.Neg


def bochvar_and(a: Trit, b: Trit) -> Trit:
    """Bochvar B3 conjunction: Unknown is toxic — any op touching Zero
    returns Zero (quarantine). Non-toxic cases are classical AND:
    both Pos → Pos, otherwise Neg."""
    if Trit.Zero in (a, b):
        return Trit.Zero
    if a is Trit.Pos and b is Trit.Pos:
        return Trit.Pos
    return Trit.Neg


def consensus(a: Trit, b: Trit) -> Trit:
    """Trit::consensus — agree or Unknown.

    Two perspectives on the same departure→arrival pair: identical
    amplitude → that amplitude; disagree → Zero (U). The arbitration
    semantics for double-entry transitions (tycoon NOTE-04) whose two
    sides were witnessed by different instruments."""
    if a is b:
        return a
    return Trit.Zero


def compose_transition(departure: Trit, arrival: Trit | None, logic: str = "kleene") -> Trit:
    """Compose a double-entry transition pair into its hallway truth value.

    arrival=None is the PENDING state (departure booked, no arrival seen):
    composes to Zero (U) under both logics — the hallway persists,
    unresolved. A witnessed arrival collapses via consensus (agree → the
    amplitude; disagree → U), or via Bochvar quarantine when the auditor
    wants contamination contained rather than propagated."""
    if arrival is None:
        return Trit.Zero
    if logic == "bochvar":
        return bochvar_and(departure, arrival)
    return consensus(departure, arrival)


def sweep_to_trit_field(cos_estimates: Mapping[Any, float], tau: float = TRIT_COS_TAU) -> dict[Any, Trit]:
    """Band a whole sweep into a trit field — the 3-valued echogram.

    The full-corpus sonar answer reduced to {construct, unknown,
    destruct} per cell: the argmax instrument's worldview and the
    interference instrument's, reconciled."""
    return {cell: cos_to_trit(c, tau) for cell, c in cos_estimates.items()}
