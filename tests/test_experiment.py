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

    def test_export_report_formats(self) -> None:
        """Verify report generation and export for txt, md, pdf, and all formats (Phase 2)."""
        from results import (
            export_report,
            generate_markdown_report,
            generate_text_report,
            generate_pdf_report,
        )

        exp = TournamentExperiment(
            playlist_size=8,
            tournament_format="swiss",
            match_size=2,
            random_seed=42,
        )
        result = exp.run()

        # 1. Text report
        txt_content = generate_text_report(result)
        self.assertIn("TOURNAMENT WINNER", txt_content)
        self.assertIn("COMPLETE TOURNAMENT RANKINGS", txt_content)

        txt_file = exp.export(file_format="txt", filename="test_export_txt")
        self.assertTrue(Path(txt_file).exists())
        self.assertGreater(Path(txt_file).stat().st_size, 0)
        Path(txt_file).unlink()

        # 2. Markdown report
        md_content = generate_markdown_report(result)
        self.assertIn("# 🎵 Favorite Song Tournament - Final Results", md_content)
        self.assertIn("| Rank | Song Title | Artist |", md_content)

        md_file = exp.export(file_format="md", filename="test_export_md")
        self.assertTrue(Path(md_file).exists())
        self.assertGreater(Path(md_file).stat().st_size, 0)
        Path(md_file).unlink()

        # 3. PDF report
        pdf_file = exp.export(file_format="pdf", filename="test_export_pdf")
        self.assertTrue(Path(pdf_file).exists())
        self.assertGreater(Path(pdf_file).stat().st_size, 0)
        with open(pdf_file, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")
        Path(pdf_file).unlink()

        # 4. All formats
        all_files = exp.export(file_format="all", filename="test_export_all")
        self.assertIsInstance(all_files, dict)
        self.assertIn("txt", all_files)
        self.assertIn("md", all_files)
        self.assertIn("pdf", all_files)
        for fmt, p in all_files.items():
            self.assertTrue(Path(p).exists())
            self.assertGreater(Path(p).stat().st_size, 0)
            Path(p).unlink()

    def test_delete_cached_dataset(self) -> None:
        """Verify deleting a cached dataset removes files and updates metadata.json (Phase 2)."""
        from data_loader import (
            Playlist,
            Song,
            delete_cached_dataset,
            list_cached_datasets,
            save_dataset_to_cache,
        )

        songs = [Song(i, f"Song_{i}", f"Artist_{i}") for i in range(1, 9)]
        playlist = Playlist(songs=songs, data_type="names_only", name="TempDeleteTest")

        save_path = save_dataset_to_cache(playlist, name="temp_delete_test", note="Temporary test")
        self.assertTrue(save_path.exists())

        # Check listed
        cached = list_cached_datasets()
        self.assertTrue(any(d["name"] == "temp_delete_test" for d in cached))

        # Delete it
        deleted = delete_cached_dataset("temp_delete_test")
        self.assertTrue(deleted)
        self.assertFalse(save_path.exists())

        # Check unlisted
        cached_after = list_cached_datasets()
        self.assertFalse(any(d["name"] == "temp_delete_test" for d in cached_after))

        # Second deletion should return False
        self.assertFalse(delete_cached_dataset("temp_delete_test"))

    def test_create_custom_criteria_absolute_from_existing_names(self) -> None:
        """Verify using an existing song list and creating custom criteria with absolute rubrics."""
        from unittest.mock import patch
        from data_loader import load_cached_dataset, Playlist
        from runner import (
            create_absolute_data_interactive,
            select_criteria_interactive,
        )

        base_playlist = load_cached_dataset("favorite_song_data")
        names_only = base_playlist.get_random_subset(8, random_seed=42).to_names_only()
        self.assertEqual(names_only.data_type, "names_only")
        self.assertEqual(len(names_only.feature_names), 0)

        # 1. Test custom criteria selection
        with patch("builtins.input", side_effect=["2", "Energy, Rhythm"]):
            custom_features = select_criteria_interactive()
            self.assertEqual(custom_features, ["Energy", "Rhythm"])

        # 2. Test interactive absolute scoring with standard rubric (1-10) and batch size 8
        # Inputs:
        # rubric choice: 1 (Standard 1-10 scale)
        # batch size: 8
        # 8 songs * 2 features = 16 scores (provide "7" for Energy, "8" for Rhythm)
        scores = ["7", "8"] * 8
        mock_inputs = iter(["1", "8"] + scores)
        with patch("builtins.input", side_effect=lambda *_: next(mock_inputs)):
            scored_playlist = create_absolute_data_interactive(names_only, features=custom_features)

        self.assertEqual(scored_playlist.data_type, "absolute")
        self.assertEqual(scored_playlist.feature_names, ["Energy", "Rhythm"])
        self.assertEqual(len(scored_playlist), 8)
        for song in scored_playlist:
            self.assertIn("Energy", song.features)
            self.assertIn("Rhythm", song.features)

    def test_create_custom_criteria_relative_from_existing_names(self) -> None:
        """Verify using an existing song list and creating custom criteria with relative promotions."""
        from unittest.mock import patch
        from data_loader import load_cached_dataset
        from runner import (
            create_relative_data_interactive,
            select_criteria_interactive,
        )

        base_playlist = load_cached_dataset("favorite_song_data")
        names_only = base_playlist.get_random_subset(8, random_seed=42).to_names_only()

        # 1. Test custom criteria selection
        with patch("builtins.input", side_effect=["2", "Vibe, Vocals"]):
            custom_features = select_criteria_interactive()
            self.assertEqual(custom_features, ["Vibe", "Vocals"])

        # 2. Test interactive relative scoring
        # batch_size: 4
        # upgrade promotions: empty string or "1 2"
        mock_inputs = iter(["4"] + ["1 2"] * 50)
        with patch("builtins.input", side_effect=lambda *_: next(mock_inputs)):
            rel_playlist = create_relative_data_interactive(names_only, features=custom_features)

        self.assertEqual(rel_playlist.data_type, "relative")
        self.assertEqual(rel_playlist.feature_names, ["Vibe", "Vocals"])
        self.assertEqual(len(rel_playlist), 8)
        for song in rel_playlist:
            self.assertIn("Vibe", song.features)
            self.assertIn("Vocals", song.features)
            self.assertGreaterEqual(song.features["Vibe"], 1.0)
            self.assertGreaterEqual(song.features["Vocals"], 1.0)


if __name__ == "__main__":
    unittest.main()

