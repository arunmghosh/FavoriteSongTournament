"""Tests for tournament experiment pipeline in Favorite Song Tournament.

Ensures that:
1. The experiment pipeline runs correctly across Swiss, Sorting, and Hybrid configurations.
2. Input validations (playlist size 8-128, match size 2-8) properly enforce constraints.
3. TournamentResult outputs contain all required fields, valid rankings, and statistical metrics.
4. Result display formatting generates valid winner announcements, tables, and insights.
"""

import unittest
from pathlib import Path

from experiment import TournamentExperiment
from results import TournamentResult


class TestTournamentExperiment(unittest.TestCase):
    """Pipeline and integration tests for TournamentExperiment."""

    def setUp(self) -> None:
        self.excel_file = "favorite_song_data.xlsx"
        self.has_excel = Path(self.excel_file).exists()

    def test_pipeline_validation_constraints(self) -> None:
        """Verify experiment parameter validation."""
        # Invalid format
        with self.assertRaises(ValueError):
            TournamentExperiment(tournament_format="invalid_format")

        # Invalid match size (< 2 or > 8)
        with self.assertRaises(ValueError):
            TournamentExperiment(match_size=1)
        with self.assertRaises(ValueError):
            TournamentExperiment(match_size=9)

        # Invalid playlist size (< 8 or > 128)
        exp_too_small = TournamentExperiment(playlist_size=5)
        with self.assertRaises(ValueError):
            exp_too_small.setup()

        exp_too_large = TournamentExperiment(playlist_size=150)
        with self.assertRaises(ValueError):
            exp_too_large.setup()

    def test_swiss_pipeline_execution(self) -> None:
        """Verify end-to-end Swiss tournament pipeline."""
        exp = TournamentExperiment(
            filepath=self.excel_file if self.has_excel else None,
            playlist_size=16,
            tournament_format="swiss",
            match_size=4,
            random_seed=42,
        )
        result = exp.run()

        self.assertIsInstance(result, TournamentResult)
        self.assertEqual(result.tournament_format, "swiss")
        self.assertEqual(result.match_size, 4)
        self.assertGreater(result.total_rounds, 0)
        self.assertGreater(result.total_matches, 0)
        self.assertEqual(len(result.rankings), 16)
        self.assertIsNotNone(result.winner)
        self.assertIn("Catchiness", result.correlations)
        self.assertGreater(len(result.insights), 0)

        # Test display output
        text_all = exp.display(view_mode="all")
        self.assertIn("TOURNAMENT WINNER", text_all)
        self.assertIn("COMPLETE TOURNAMENT RANKINGS", text_all)

        text_notable = exp.display(view_mode="notable", notable_count=3)
        self.assertIn("NOTABLE SONGS", text_notable)

        text_winner = exp.display(view_mode="winner")
        self.assertIn("FINAL WINNER", text_winner)

    def test_sorting_pipeline_execution(self) -> None:
        """Verify end-to-end Sorting tournament pipeline with DQN."""
        round1_called = False

        def round1_cb() -> None:
            nonlocal round1_called
            round1_called = True

        exp = TournamentExperiment(
            filepath=self.excel_file if self.has_excel else None,
            playlist_size=12,
            tournament_format="sorting",
            match_size=3,
            random_seed=42,
            on_round1_complete=round1_cb,
        )
        result = exp.run()

        self.assertTrue(round1_called, "Round 1 completion callback should be triggered.")
        self.assertIsInstance(result, TournamentResult)
        self.assertEqual(result.tournament_format, "sorting")
        self.assertEqual(len(result.rankings), 12)
        self.assertGreater(result.manual_matches, 0)
        self.assertGreater(result.automated_matches, 0)
        self.assertEqual(result.total_matches, result.manual_matches + result.automated_matches)

    def test_hybrid_pipeline_execution(self) -> None:
        """Verify end-to-end Hybrid tournament pipeline."""
        exp = TournamentExperiment(
            filepath=self.excel_file if self.has_excel else None,
            playlist_size=16,
            tournament_format="hybrid",
            hybrid_sub_format="swiss",
            match_size=2,
            random_seed=42,
        )
        result = exp.run()

        self.assertIsInstance(result, TournamentResult)
        self.assertEqual(result.tournament_format, "hybrid")
        self.assertIsNotNone(result.winner)
        # Final ranking count plus eliminated count should equal original playlist size (16)
        total_songs = len(result.rankings) + len(result.eliminated_songs)
        self.assertEqual(total_songs, 16)


if __name__ == "__main__":
    unittest.main()

