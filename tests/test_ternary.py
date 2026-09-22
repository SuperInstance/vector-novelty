"""Tests for the GF(3)/3-valued ternary adapter (semantics pinned against
the Ternary Fleet's own test tables per scout-γ 2026-09-22)."""

import unittest

from vector_novelty.ternary import (
    TRIT_COS_TAU,
    Trit,
    bochvar_and,
    compose_transition,
    consensus,
    cos_to_trit,
    kleene_and,
    kleene_or,
    sweep_to_trit_field,
)


class TestTritGF3(unittest.TestCase):
    def test_values(self):
        self.assertEqual((int(Trit.Neg), int(Trit.Zero), int(Trit.Pos)), (-1, 0, 1))

    def test_wraparound_add_pos_pos_is_neg(self):
        # 1 + 1 = 2 ≡ -1 (mod 3): interference is addition with wrap
        self.assertIs(Trit.Pos + Trit.Pos, Trit.Neg)

    def test_add_semantics_all(self):
        self.assertIs(Trit.Zero + Trit.Pos, Trit.Pos)
        self.assertIs(Trit.Neg + Trit.Pos, Trit.Zero)
        self.assertIs(Trit.Neg + Trit.Neg, Trit.Pos)  # -1 + -1 = -2 ≡ 1
        self.assertIs(Trit.Zero + Trit.Zero, Trit.Zero)

    def test_mul_semantics(self):
        self.assertIs(Trit.Pos * Trit.Pos, Trit.Pos)
        self.assertIs(Trit.Pos * Trit.Neg, Trit.Neg)
        self.assertIs(Trit.Neg * Trit.Neg, Trit.Pos)
        self.assertIs(Trit.Zero * Trit.Pos, Trit.Zero)


class TestCosToTrit(unittest.TestCase):
    def test_constructive_band(self):
        self.assertIs(cos_to_trit(0.9), Trit.Pos)
        self.assertIs(cos_to_trit(TRIT_COS_TAU + 0.01), Trit.Pos)

    def test_destructive_band(self):
        self.assertIs(cos_to_trit(-0.9), Trit.Neg)
        self.assertIs(cos_to_trit(-TRIT_COS_TAU - 0.01), Trit.Neg)

    def test_noise_band_is_unknown_not_false(self):
        self.assertIs(cos_to_trit(0.0), Trit.Zero)
        self.assertIs(cos_to_trit(0.2), Trit.Zero)
        self.assertIs(cos_to_trit(TRIT_COS_TAU), Trit.Zero)  # boundary → U
        self.assertIs(cos_to_trit(-0.2), Trit.Zero)


class TestKleene(unittest.TestCase):
    def test_known_construct_and_pending_is_pending(self):
        self.assertIs(kleene_and(Trit.Pos, Trit.Zero), Trit.Zero)

    def test_known_destruct_and_pending_is_destruct(self):
        # False dominates ∧ in K3
        self.assertIs(kleene_and(Trit.Neg, Trit.Zero), Trit.Neg)

    def test_pending_and_pending_is_pending(self):
        self.assertIs(kleene_and(Trit.Zero, Trit.Zero), Trit.Zero)

    def test_construct_and_construct(self):
        self.assertIs(kleene_and(Trit.Pos, Trit.Pos), Trit.Pos)

    def test_or_known_construct_wins(self):
        self.assertIs(kleene_or(Trit.Pos, Trit.Zero), Trit.Pos)
        self.assertIs(kleene_or(Trit.Zero, Trit.Neg), Trit.Zero)
        self.assertIs(kleene_or(Trit.Neg, Trit.Neg), Trit.Neg)


class TestBochvar(unittest.TestCase):
    def test_unknown_is_toxic(self):
        self.assertIs(bochvar_and(Trit.Pos, Trit.Zero), Trit.Zero)
        self.assertIs(bochvar_and(Trit.Neg, Trit.Zero), Trit.Zero)

    def test_non_toxic_is_classical_and(self):
        self.assertIs(bochvar_and(Trit.Pos, Trit.Pos), Trit.Pos)
        self.assertIs(bochvar_and(Trit.Pos, Trit.Neg), Trit.Neg)
        self.assertIs(bochvar_and(Trit.Neg, Trit.Neg), Trit.Neg)


class TestConsensusAndTransitions(unittest.TestCase):
    def test_consensus_agree(self):
        self.assertIs(consensus(Trit.Pos, Trit.Pos), Trit.Pos)
        self.assertIs(consensus(Trit.Neg, Trit.Neg), Trit.Neg)

    def test_consensus_disagree_is_unknown(self):
        self.assertIs(consensus(Trit.Pos, Trit.Neg), Trit.Zero)
        self.assertIs(consensus(Trit.Pos, Trit.Zero), Trit.Zero)

    def test_pending_transition_is_unknown(self):
        # departure booked, arrival absent: the hallway persists, unresolved
        self.assertIs(compose_transition(Trit.Pos, None), Trit.Zero)
        self.assertIs(compose_transition(Trit.Pos, None, logic="bochvar"), Trit.Zero)

    def test_witnessed_transition_collapses(self):
        self.assertIs(compose_transition(Trit.Pos, Trit.Pos), Trit.Pos)
        self.assertIs(compose_transition(Trit.Pos, Trit.Neg), Trit.Zero)  # dispute → U

    def test_bochvar_quarantine_on_dispute(self):
        # contamination contained, not propagated: Pos∧Neg = Neg under quarantine
        self.assertIs(compose_transition(Trit.Pos, Trit.Neg, logic="bochvar"), Trit.Neg)


class TestSweepToTritField(unittest.TestCase):
    def test_full_field(self):
        field = sweep_to_trit_field({"a": 0.95, "b": -0.8, "c": 0.1, "d": -0.05})
        self.assertEqual(
            field, {"a": Trit.Pos, "b": Trit.Neg, "c": Trit.Zero, "d": Trit.Zero}
        )

    def test_field_is_the_three_valued_echogram(self):
        # The argmax instrument's worldview and the interference instrument's,
        # reconciled: every cell carries construct/unknown/destruct.
        field = sweep_to_trit_field({"cell-1": 0.99, "cell-2": 0.0})
        self.assertIsInstance(field["cell-1"], Trit)
        self.assertIs(field["cell-2"], Trit.Zero)


if __name__ == "__main__":
    unittest.main()
