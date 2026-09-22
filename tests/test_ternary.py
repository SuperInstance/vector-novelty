"""Tests for the GF(3)/3-valued ternary adapter (semantics pinned against
the Ternary Fleet's own test tables per scout-γ 2026-09-22)."""

import pytest
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
    lukasiewicz_implication,
    sweep_to_trit_field,
    tutor_equal,
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


class TestGraded:
    """The graded-truth layer: what lives between the trit band and the raw
    cosine. Scout-gamma NOT FOUND (ternary-logic is 3-valued only); scout-delta
    found the historical precedent (TUTOR `compute`, 1972: equality as a
    similarity threshold)."""

    def test_lukasiewicz_tautology_bound(self):
        # a <= b implies a -> b is fully true (the implication's defining law)
        assert lukasiewicz_implication(0.7, 0.9) == pytest.approx(1.0)
        assert lukasiewicz_implication(0.7, 0.7) == pytest.approx(1.0)

    def test_lukasiewicz_partial(self):
        assert lukasiewicz_implication(0.7, 0.4) == pytest.approx(0.7)
        assert lukasiewicz_implication(1.0, 0.0) == pytest.approx(0.0)

    def test_lukasiewicz_clamps_inputs(self):
        # designated-value discipline: inputs outside [0,1] are clamped
        assert lukasiewicz_implication(1.4, 0.4) == lukasiewicz_implication(1.0, 0.4)
        assert lukasiewicz_implication(0.7, -0.5) == lukasiewicz_implication(0.7, 0.0)

    def test_lukasiewicz_collapses_to_l3_on_trit_values(self):
        # U -> U = T, T -> U = U, T -> F = F (the pinned L3 tables, on 0/0.5/1)
        assert lukasiewicz_implication(0.5, 0.5) == 1.0
        assert lukasiewicz_implication(1.0, 0.5) == 0.5
        assert lukasiewicz_implication(1.0, 0.0) == 0.0

    def test_tutor_equal_exact_and_within_roundoff(self):
        # TUTOR compute (1972): x = y TRUE for approximately-equal floats
        assert tutor_equal(0.875, 0.875) == 1.0
        assert tutor_equal(0.8750000001, 0.875) == 1.0
        assert tutor_equal(0.8751, 0.875, tol=1e-3) == 1.0

    def test_tutor_equal_decays_past_tolerance(self):
        # linear decay to 0 at twice the tolerance — a threshold as an operator
        assert tutor_equal(0.890, 0.875, tol=0.01) == pytest.approx(0.5)  # d = 1.5*tol
        assert tutor_equal(0.895, 0.875, tol=0.01) == pytest.approx(0.0)  # d = 2*tol
        assert tutor_equal(2.0, 0.875, tol=0.01) == 0.0

    def test_graded_lives_outside_the_trit_field(self):
        # the band (U) is instrument noise; the grade is instrument reading.
        # a graded value inside the noise band must not collapse to a trit.
        # same reading, two instruments: the grade says 0.75, the band says U
        g = tutor_equal(0.325, 0.30, tol=0.02)  # d = 1.25*tol -> 0.75
        assert g == pytest.approx(0.75)
        assert cos_to_trit(0.325, tau=0.35) is Trit.Zero  # and the band still says U


class TestEchogramWalkExample:
    """The bridge for outside engineers must stay runnable — the example is
    part of the contract (Casey 2026-09-22: 'make the bridges easy')."""

    def test_example_runs_end_to_end(self):
        import contextlib
        import io
        import os
        import runpy

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(repo_root, "examples", "echogram_walk.py"), run_name="__main__")
        out = buf.getvalue()
        assert "3-valued echogram" in out
        assert "construct" in out
        assert "compose_transition(Pos, None) = Zero" in out
        assert "tutor_equal" in out
