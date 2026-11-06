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

    def test_user_restart_with_r_at_prompts(self) -> None:
        """Verify that entering 'R' at any prompt raises UserRestartTournament (Phase 2)."""
        from unittest.mock import patch
        from runner import (
            UserRestartTournament,
            safe_input,
            prompt_int,
            prompt_choice,
            prompt_yes_no,
        )

        with patch("builtins.input", return_value="R"):
            with self.assertRaises(UserRestartTournament):
                safe_input("Enter: ")

        with patch("builtins.input", return_value="r"):
            with self.assertRaises(UserRestartTournament):
                safe_input("Enter: ")

        with patch("builtins.input", return_value="restart"):
            with self.assertRaises(UserRestartTournament):
                safe_input("Enter: ")

        with patch("builtins.input", return_value="R"):
            with self.assertRaises(UserRestartTournament):
                prompt_int("Enter number", 1, 10)

        with patch("builtins.input", return_value="r"):
            with self.assertRaises(UserRestartTournament):
                prompt_choice("Select", ["A", "B"])

        with patch("builtins.input", return_value="R"):
            with self.assertRaises(UserRestartTournament):
                prompt_yes_no("Proceed?")

    def test_runner_restart_interactive_loop(self) -> None:
        """Verify that entering 'R' restarts the prompt loop and then 'Q' quits cleanly."""
        from unittest.mock import patch

        # First prompt receives 'R', which restarts loop, second receives 'Q' to exit cleanly
        inputs = iter(["R", "Q"])
        with patch("builtins.input", side_effect=lambda *_: next(inputs)):
            runner_main([])

    def test_smoke_runner_cli_export(self) -> None:
        """Verify runner CLI demo execution with --export flag (Phase 2)."""
        reports_dir = Path("reports")
        # Run demo with --export txt
        runner_main(["--demo", "--format", "swiss", "--size", "8", "--match-size", "2", "--export", "txt"])
        txt_files = list(reports_dir.glob("tournament_swiss_*.txt"))
        self.assertGreater(len(txt_files), 0)

        # Run demo with --export md
        runner_main(["--demo", "--format", "swiss", "--size", "8", "--match-size", "2", "--export", "md"])
        md_files = list(reports_dir.glob("tournament_swiss_*.md"))
        self.assertGreater(len(md_files), 0)

        # Run demo with --export pdf
        runner_main(["--demo", "--format", "swiss", "--size", "8", "--match-size", "2", "--export", "pdf"])
        pdf_files = list(reports_dir.glob("tournament_swiss_*.pdf"))
        self.assertGreater(len(pdf_files), 0)

        # Clean up exported test demo files
        for f in txt_files + md_files + pdf_files:
            try:
                f.unlink()
            except Exception:
                pass

    def test_smoke_manage_cache_view_full_contents(self) -> None:
        """Verify manage cache displays full cache, indices, and song contents."""
        from unittest.mock import patch
        from runner import manage_cache_interactive

        # Action 1 (View full dataset), Index 1, Enter to return, Action 3 (Return to main menu)
        inputs = iter(["1", "1", "", "3"])
        with patch("builtins.input", side_effect=lambda *_: next(inputs)):
            manage_cache_interactive()

    def test_smoke_configure_dataset_names_to_custom_criteria(self) -> None:
        """Verify configuring an existing dataset by stripping to names, then defining custom criteria."""
        import itertools
        from unittest.mock import patch
        from runner import configure_dataset_interactive

        # Flow:
        # 1. How to proceed: "1" (Draw from cache)
        # 2. Select dataset: "1" (favorite_song_data)
        # 3. Configure mode: "2" (Use only song names)
        # 4. Proceed with names: "2" (Create additional feature data)
        # 5. Type of data: "1" (Absolute feature data)
        # 6. Criteria: "2" (Enter custom criteria)
        # 7. Custom criteria input: "Vibe, Groove"
        # 8. Rubric scale: "1" (Standard 1-10)
        # 9. Batch size: "128"
        # 10. Default scores / empty string until cache prompt "n"
        specific_inputs = ["1", "1", "2", "2", "1", "2", "Vibe, Groove", "1", "128"]
        # Supply 128*2 empty strings for scores, then "n" for cache prompt, then repeat ""
        inputs = itertools.chain(specific_inputs, [""] * 256, ["n"], itertools.repeat(""))
        with patch("builtins.input", side_effect=lambda *_: next(inputs)):
            playlist = configure_dataset_interactive()

        self.assertEqual(playlist.data_type, "absolute")
        self.assertEqual(playlist.feature_names, ["Vibe", "Groove"])
        self.assertEqual(len(playlist), 128)


if __name__ == "__main__":
    unittest.main()

