"""Experiment harness for Favorite Song Tournament.

Constructs, configures, and executes the complete tournament experiment pipeline:
- Ingestion of playlist data (file or synthetic)
- Setting up tournament parameters (format, match size, criteria)
- Running tournament algorithms (Swiss, Sorting with DQN, Hybrid)
- Computing statistical metrics (correlations, feature influence)
- Producing comprehensive TournamentResult objects for display and inspection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from algorithms import (
    HybridTournament,
    JudgeCallback,
    SortingTournament,
    SwissTournament,
)
from data_loader import (
    Playlist,
    Song,
    generate_synthetic_playlist,
    load_dataset,
    transform_absolute_to_relative,
    transform_relative_to_absolute,
)
from results import (
    TournamentResult,
    calculate_feature_correlations,
    display_results,
    generate_feature_insights,
)


class TournamentExperiment:
    """Orchestrates an end-to-end Favorite Song Tournament experiment."""

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        playlist_size: Optional[int] = 16,
        tournament_format: str = "swiss",
        hybrid_sub_format: str = "swiss",
        match_size: int = 2,
        data_type: str = "relative",
        feature_names: Optional[List[str]] = None,
        judge_callback: Optional[JudgeCallback] = None,
        random_seed: Optional[int] = 42,
        max_rounds: Optional[int] = None,
        on_round1_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self.filepath = filepath
        self.playlist_size = playlist_size
        self.tournament_format = tournament_format.lower()
        if self.tournament_format not in ("swiss", "sorting", "hybrid"):
            raise ValueError(
                f"Invalid format '{tournament_format}'. Must be 'swiss', 'sorting', or 'hybrid'."
            )

        self.hybrid_sub_format = hybrid_sub_format.lower()
        self.match_size = match_size
        if self.match_size < 2 or self.match_size > 8:
            raise ValueError(f"Match size {match_size} must be between 2 and 8.")

        self.data_type = data_type.lower()
        self.feature_names = feature_names
        self.judge_callback = judge_callback
        self.random_seed = random_seed
        self.max_rounds = max_rounds
        self.on_round1_complete = on_round1_complete

        self.playlist: Optional[Playlist] = None
        self.result: Optional[TournamentResult] = None

    def setup(self) -> Playlist:
        """Loads and prepares the playlist data for the tournament."""
        if self.filepath is not None and Path(self.filepath).exists():
            loaded = load_dataset(
                filepath=self.filepath,
                data_type=self.data_type,
                feature_columns=self.feature_names,
                max_songs=self.playlist_size,
                validate_size=True,
            )
            self.playlist = loaded
        else:
            # Generate synthetic playlist if no file is provided
            size = self.playlist_size or 16
            self.playlist = generate_synthetic_playlist(
                size=size,
                feature_names=self.feature_names,
                data_type=self.data_type,
                random_seed=self.random_seed,
            )

        # Ensure correct data type format according to tournament rules
        if self.tournament_format == "sorting":
            # Sorting requires absolute feature data (README 3b)
            if self.playlist.data_type == "relative":
                self.playlist = transform_relative_to_absolute(self.playlist)
        elif self.tournament_format == "hybrid":
            # Hybrid requires relative feature data (README 3c)
            if self.playlist.data_type == "absolute":
                self.playlist = transform_absolute_to_relative(self.playlist)

        return self.playlist

    def run(self) -> TournamentResult:
        """Executes the configured tournament experiment and returns the TournamentResult."""
        if self.playlist is None:
            self.setup()
        assert self.playlist is not None

        eliminated_songs: List[Song] = []
        feature_names = list(self.playlist.feature_names)

        if self.tournament_format == "swiss":
            tournament = SwissTournament(
                playlist=self.playlist,
                match_size=self.match_size,
                judge_callback=self.judge_callback,
                max_rounds=self.max_rounds,
                random_seed=self.random_seed,
            )
            rankings, winner = tournament.run()
            total_rounds = tournament.round_num
            total_matches = len(tournament.match_history)
            manual_matches = total_matches
            automated_matches = 0

        elif self.tournament_format == "sorting":
            tournament = SortingTournament(
                playlist=self.playlist,
                match_size=self.match_size,
                user_judge_callback=self.judge_callback,
                random_seed=self.random_seed,
            )
            rankings, winner = tournament.run(on_round1_complete=self.on_round1_complete)
            total_rounds = tournament.round_num
            total_matches = len(tournament.match_history)
            manual_matches = tournament.manual_matches_count
            automated_matches = tournament.automated_matches_count

        elif self.tournament_format == "hybrid":
            tournament = HybridTournament(
                playlist=self.playlist,
                sub_format=self.hybrid_sub_format,
                match_size=self.match_size,
                judge_callback=self.judge_callback,
                random_seed=self.random_seed,
            )
            rankings, winner, eliminated_songs = tournament.run(on_round1_complete=self.on_round1_complete)
            total_rounds = tournament.engine.round_num
            total_matches = len(tournament.match_history)
            if isinstance(tournament.engine, SortingTournament):
                manual_matches = tournament.engine.manual_matches_count
                automated_matches = tournament.engine.automated_matches_count
            else:
                manual_matches = total_matches
                automated_matches = 0

        # Calculate feature correlations
        correlations = calculate_feature_correlations(rankings, feature_names)
        insights = generate_feature_insights(correlations)

        self.result = TournamentResult(
            tournament_format=self.tournament_format,
            match_size=self.match_size,
            total_rounds=total_rounds,
            total_matches=total_matches,
            manual_matches=manual_matches,
            automated_matches=automated_matches,
            rankings=rankings,
            winner=winner,
            correlations=correlations,
            insights=insights,
            eliminated_songs=eliminated_songs,
            feature_names=feature_names,
        )

        return self.result

    def display(self, view_mode: str = "all", notable_count: int = 5) -> str:
        """Displays the tournament results in the console."""
        if self.result is None:
            raise RuntimeError("Experiment has not been run yet. Call run() first.")
        return display_results(self.result, view_mode=view_mode, notable_count=notable_count)

