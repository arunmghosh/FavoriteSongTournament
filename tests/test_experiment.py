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

    def test_dataset_caching_and_listing(self) -> None:
        """Verify dataset caching, metadata listing, and loading."""
        from data_loader import (
            list_cached_datasets,
            load_cached_dataset,
            save_dataset_to_cache,
            Playlist,
            Song,
        )
        import json

        songs = [Song(i, f"Song_{i}", f"Artist_{i}") for i in range(1, 11)]
        playlist = Playlist(songs=songs, data_type="names_only", name="UnitTestPlaylist")

        save_path = save_dataset_to_cache(playlist, name="test_unit_cache", note="Unit test cached playlist")
        self.assertTrue(save_path.exists())

        # Verify listed in cached datasets
        cached = list_cached_datasets()
        cached_names = [d["name"] for d in cached]
        self.assertIn("test_unit_cache", cached_names)

        # Verify loaded from cache
        loaded = load_cached_dataset("test_unit_cache")
        self.assertEqual(len(loaded), 10)
        self.assertEqual(loaded.data_type, "names_only")

        # Cleanup
        save_path.unlink()
        meta_file = Path("data/metadata.json")
        if meta_file.exists():
            with open(meta_file, "r") as f:
                m = json.load(f)
            m.pop("test_unit_cache", None)
            with open(meta_file, "w") as f:
                json.dump(m, f, indent=2)

    def test_playlist_subsetting_and_shuffling(self) -> None:
        """Verify playlist random subsetting and internal shuffling (README 2b)."""
        from data_loader import generate_synthetic_playlist

        full = generate_synthetic_playlist(size=32, random_seed=42)
        self.assertEqual(len(full), 32)

        # Valid subset
        sub16 = full.get_random_subset(16, random_seed=123)
        self.assertEqual(len(sub16), 16)
        self.assertEqual(len(full), 32, "Original playlist should not be modified.")

        # Invalid subset bounds
        with self.assertRaises(ValueError):
            full.get_random_subset(5)  # Less than 8
        with self.assertRaises(ValueError):
            full.get_random_subset(50)  # Greater than dataset size

        # Shuffling
        shuffled = full.shuffled(random_seed=999)
        self.assertEqual(len(shuffled), len(full))
        # Ensure order is not strictly identical
        full_titles = [s.title for s in full]
        shuffled_titles = [s.title for s in shuffled]
        self.assertNotEqual(full_titles, shuffled_titles)

    def test_names_only_tournament_pipeline(self) -> None:
        """Verify Swiss tournament pipeline running with names-only data (no features)."""
        exp = TournamentExperiment(
            playlist_size=8,
            data_type="names_only",
            tournament_format="swiss",
            match_size=2,
            random_seed=42,
        )
        result = exp.run()

        self.assertIsInstance(result, TournamentResult)
        self.assertEqual(len(result.rankings), 8)
        self.assertEqual(result.correlations, {})
        self.assertIn("names only", result.insights[0].lower())

        output = exp.display(view_mode="all")
        self.assertIn("names only", output.lower())


if __name__ == "__main__":
    unittest.main()

