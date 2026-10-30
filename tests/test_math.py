"""Tests for mathematical calculations and transformations in Favorite Song Tournament.

Ensures that:
1. Relative to absolute transformations preserve monotonicity and scale bounds.
2. Absolute to relative transformations produce the correct geometric halving distribution.
3. Strength of schedule (Buchholz) matches theoretical sums.
4. Pearson (r) and Spearman (rho) correlation calculations are accurate.
5. Strict Pareto dominance accurately identifies strictly dominated candidates.
"""

import unittest
import numpy as np

from data_loader import (
    Playlist,
    Song,
    generate_synthetic_playlist,
    transform_absolute_to_relative,
    transform_relative_to_absolute,
)
from results import calculate_feature_correlations


class TestMathematicalCalculations(unittest.TestCase):
    """Mathematical correctness tests."""

    def test_relative_to_absolute_bounds_and_monotonicity(self) -> None:
        """Verify that relative -> absolute scaling stays in [scale_min, scale_max] and preserves order."""
        playlist = generate_synthetic_playlist(size=16, data_type="relative", random_seed=123)
        scale_min = 1.0
        scale_max = 10.0
        abs_playlist = transform_relative_to_absolute(playlist, scale_min=scale_min, scale_max=scale_max)

        for f in abs_playlist.feature_names:
            rel_scores = [s.features[f] for s in playlist]
            abs_scores = [s.features[f] for s in abs_playlist]

            # Bounds check
            for val in abs_scores:
                self.assertGreaterEqual(val, scale_min - 1e-6)
                self.assertLessEqual(val, scale_max + 1e-6)

            # Monotonicity check
            for i in range(len(rel_scores)):
                for j in range(len(rel_scores)):
                    if rel_scores[i] > rel_scores[j]:
                        self.assertGreater(abs_scores[i], abs_scores[j])
                    elif rel_scores[i] == rel_scores[j]:
                        self.assertAlmostEqual(abs_scores[i], abs_scores[j], delta=1e-2)

    def test_absolute_to_relative_geometric_distribution(self) -> None:
        """Verify that absolute -> relative produces geometric tier upgrades with a unique winner."""
        # For N=16 songs:
        # 16 songs start at 1.
        # Top 8 upgraded to 2.
        # Top 4 upgraded to 3.
        # Top 2 upgraded to 4.
        # Top 1 upgraded to 5 (unique high score).
        # Distribution: 1 at 5, 1 at 4, 2 at 3, 4 at 2, 8 at 1. Total = 16.
        songs = [
            Song(id=i + 1, title=f"S{i}", artist="A", features={"F": float(100 - i)})
            for i in range(16)
        ]
        playlist = Playlist(songs=songs, data_type="absolute", feature_names=["F"])
        rel_playlist = transform_absolute_to_relative(playlist)

        scores = [s.features["F"] for s in rel_playlist]
        unique_scores, counts = np.unique(scores, return_counts=True)
        counts_dict = dict(zip(unique_scores, counts))

        # Check unique high score
        max_score = max(scores)
        self.assertEqual(counts_dict[max_score], 1)
        self.assertEqual(max_score, 5.0)

        # Expected counts: {1.0: 8, 2.0: 4, 3.0: 2, 4.0: 1, 5.0: 1}
        expected = {1.0: 8, 2.0: 4, 3.0: 2, 4.0: 1, 5.0: 1}
        self.assertEqual(counts_dict, expected)

    def test_strength_of_schedule_calculation(self) -> None:
        """Verify Buchholz Strength of Schedule calculation."""
        from algorithms import SwissTournament

        songs = [
            Song(id=1, title="S1", artist="A", features={"F": 1.0}),
            Song(id=2, title="S2", artist="A", features={"F": 2.0}),
            Song(id=3, title="S3", artist="A", features={"F": 3.0}),
            Song(id=4, title="S4", artist="A", features={"F": 4.0}),
            Song(id=5, title="S5", artist="A", features={"F": 5.0}),
            Song(id=6, title="S6", artist="A", features={"F": 6.0}),
            Song(id=7, title="S7", artist="A", features={"F": 7.0}),
            Song(id=8, title="S8", artist="A", features={"F": 8.0}),
        ]
        playlist = Playlist(songs=songs, data_type="relative", feature_names=["F"])
        swiss = SwissTournament(playlist, match_size=2)

        # Manually set scores and opponents to test calculation
        swiss.scores = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0, 6: 0.0, 7: 0.0, 8: 0.0}
        swiss.opponents = {
            1: [2, 3],  # Opponents are S2 (score 4) and S3 (score 3) -> SOS = 7.0
            2: [1, 4],  # Opponents are S1 (score 5) and S4 (score 2) -> SOS = 7.0
            3: [1, 5],  # Opponents are S1 (score 5) and S5 (score 1) -> SOS = 6.0
        }

        self.assertEqual(swiss.calculate_strength_of_schedule(1), 7.0)
        self.assertEqual(swiss.calculate_strength_of_schedule(2), 7.0)
        self.assertEqual(swiss.calculate_strength_of_schedule(3), 6.0)

    def test_correlation_calculations(self) -> None:
        """Verify Pearson (r) and Spearman (rho) correlations on known vectors."""
        # Perfect positive correlation
        perfect_rankings = [
            {"rank": 1, "song": Song(1, "A", "A"), "features": {"X": 10.0, "Y": 1.0}},
            {"rank": 2, "song": Song(2, "B", "A"), "features": {"X": 8.0, "Y": 2.0}},
            {"rank": 3, "song": Song(3, "C", "A"), "features": {"X": 5.0, "Y": 3.0}},
            {"rank": 4, "song": Song(4, "D", "A"), "features": {"X": 2.0, "Y": 4.0}},
        ]
        corrs = calculate_feature_correlations(perfect_rankings, ["X", "Y"])

        # For X: rank 1 has highest X (10.0), rank 4 has lowest X (2.0) -> r = 1.0
        self.assertAlmostEqual(corrs["X"]["pearson_r"], 1.0, delta=0.05)
        self.assertAlmostEqual(corrs["X"]["spearman_rho"], 1.0, delta=0.05)

        # For Y: rank 1 has lowest Y (1.0), rank 4 has highest Y (4.0) -> r = -1.0
        self.assertAlmostEqual(corrs["Y"]["pearson_r"], -1.0, delta=0.05)
        self.assertAlmostEqual(corrs["Y"]["spearman_rho"], -1.0, delta=0.05)

    def test_correlation_zero_variance_safe(self) -> None:
        """Verify that zero-variance features safely return 0.0 correlation without exceptions."""
        constant_rankings = [
            {"rank": 1, "song": Song(1, "A", "A"), "features": {"Const": 5.0}},
            {"rank": 2, "song": Song(2, "B", "A"), "features": {"Const": 5.0}},
            {"rank": 3, "song": Song(3, "C", "A"), "features": {"Const": 5.0}},
        ]
        corrs = calculate_feature_correlations(constant_rankings, ["Const"])
        self.assertEqual(corrs["Const"]["pearson_r"], 0.0)
        self.assertEqual(corrs["Const"]["spearman_rho"], 0.0)

    def test_strict_pareto_dominance(self) -> None:
        """Verify strict Pareto dominance condition: B is dominated iff A[f] > B[f] for ALL f."""
        s1 = Song(1, "Dominator", "A", features={"F1": 8.0, "F2": 8.0})
        s2 = Song(2, "Dominated", "A", features={"F1": 5.0, "F2": 5.0})  # Strictly lower in both
        s3 = Song(3, "Tradeoff", "A", features={"F1": 9.0, "F2": 3.0})   # Higher in F1, lower in F2
        s4 = Song(4, "Equal", "A", features={"F1": 8.0, "F2": 5.0})      # Equal in F1 to s1, higher in F2 than s2

        p = Playlist(songs=[s1, s2, s3, s4], data_type="relative", feature_names=["F1", "F2"])
        survivors, dominated = p.filter_strictly_dominated()

        survivor_ids = [s.id for s in survivors]
        dominated_ids = [s.id for s in dominated]

        # s2 is strictly dominated by s1 (8 > 5 and 8 > 5)
        self.assertIn(2, dominated_ids)

        # s1, s3, s4 should survive
        self.assertIn(1, survivor_ids)
        self.assertIn(3, survivor_ids)
        self.assertIn(4, survivor_ids)


if __name__ == "__main__":
    unittest.main()

