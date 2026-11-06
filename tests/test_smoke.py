"""Rapid end-to-end integration smoke tests for Favorite Song Tournament.

Executes rapid sanity checks across all primary components, ensuring that the full
Phase 1 pipeline functions without exceptions.
"""

import unittest
from pathlib import Path

from experiment import TournamentExperiment
from runner import main as runner_main


class TestSmokeIntegration(unittest.TestCase):
    """End-to-end integration smoke tests."""

    def test_smoke_swiss_tournament(self) -> None:
        """Rapid smoke test of Swiss tournament on real dataset."""
        excel_path = "favorite_song_data.xlsx"
        filepath = excel_path if Path(excel_path).exists() else None

        exp = TournamentExperiment(
            filepath=filepath,
            playlist_size=16,
            tournament_format="swiss",
            match_size=2,
            random_seed=42,
        )
        res = exp.run()
        self.assertIsNotNone(res.winner)
        self.assertEqual(len(res.rankings), 16)
        output = exp.display(view_mode="winner")
        self.assertIn("TOURNAMENT WINNER", output)

    def test_smoke_sorting_tournament_with_dqn(self) -> None:
        """Rapid smoke test of Sorting tournament with DQN learning on real dataset."""
        excel_path = "favorite_song_data.xlsx"
        filepath = excel_path if Path(excel_path).exists() else None

        exp = TournamentExperiment(
            filepath=filepath,
            playlist_size=12,
            tournament_format="sorting",
            match_size=3,
            random_seed=42,
        )
        res = exp.run()
        self.assertIsNotNone(res.winner)
        self.assertEqual(len(res.rankings), 12)
        self.assertGreater(res.automated_matches, 0)

    def test_smoke_hybrid_tournament(self) -> None:
        """Rapid smoke test of Hybrid tournament with dominance filtering."""
        excel_path = "favorite_song_data.xlsx"
        filepath = excel_path if Path(excel_path).exists() else None

        exp = TournamentExperiment(
            filepath=filepath,
            playlist_size=16,
            tournament_format="hybrid",
            hybrid_sub_format="sorting",
            match_size=2,
            random_seed=42,
        )
        res = exp.run()
        self.assertIsNotNone(res.winner)
        self.assertGreater(len(res.rankings), 0)

    def test_smoke_runner_cli_entrypoints(self) -> None:
        """Rapid smoke test of runner CLI entrypoint in demo mode."""
        try:
            runner_main(["--demo", "--format", "swiss", "--size", "8", "--match-size", "2", "--view", "winner"])
            runner_main(["--demo", "--format", "sorting", "--size", "8", "--match-size", "2", "--view", "notable"])
            runner_main(["--demo", "--format", "hybrid", "--size", "8", "--match-size", "2", "--view", "all"])
            runner_main(["--demo", "--cached", "favorite_song_data", "--format", "swiss", "--size", "8", "--match-size", "2"])
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def test_user_quit_with_q_at_prompts(self) -> None:
        """Verify that entering 'Q' at any prompt raises UserQuitTournament and runner exits gracefully."""
        from unittest.mock import patch
        from runner import (
            UserQuitTournament,
            safe_input,
            prompt_int,
            prompt_choice,
            prompt_yes_no,
            console_judge_callback,
        )
        from data_loader import Song

        # Test safe_input with 'q' and 'Q'
        with patch("builtins.input", return_value="Q"):
            with self.assertRaises(UserQuitTournament):
                safe_input("Enter something: ")

        with patch("builtins.input", return_value="q"):
            with self.assertRaises(UserQuitTournament):
                prompt_int("Enter number", 1, 10)

        with patch("builtins.input", return_value="Q"):
            with self.assertRaises(UserQuitTournament):
                prompt_choice("Select option", ["Option 1", "Option 2"])

        with patch("builtins.input", return_value="q"):
            with self.assertRaises(UserQuitTournament):
                prompt_yes_no("Continue?")

        with patch("builtins.input", return_value="Q"):
            with self.assertRaises(UserQuitTournament):
                console_judge_callback([Song(1, "A", "A"), Song(2, "B", "B")], 1, 1)

        # Test runner_main cleanly exits without unhandled exception when user enters 'Q'
        with patch("builtins.input", return_value="Q"):
            runner_main([])


if __name__ == "__main__":
    unittest.main()

