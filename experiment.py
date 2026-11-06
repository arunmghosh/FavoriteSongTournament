"""Experiment harness for Favorite Song Tournament.

Constructs, configures, and executes the complete tournament experiment pipeline:
- Ingestion of playlist data (cached dataset, custom file, or synthetic)
- Data type configuration (names only, absolute, relative)
- Playlist subsetting (8 to dataset size) and internal shuffling
- Setting up tournament parameters (format, match size, criteria)
- Running tournament algorithms (Swiss, Sorting with DQN or manual, Hybrid)
- Computing statistical metrics (correlations, feature influence)
- Producing comprehensive TournamentResult objects for display and inspection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from algorithms import (
    HybridTournament,
    JudgeCallback,
    SeparatorInsertCallback,
    SortingTournament,
    SwissTournament,
)
from data_loader import (
    Playlist,
    Song,
    generate_synthetic_playlist,
    list_cached_datasets,
    load_cached_dataset,
    load_dataset,
    transform_absolute_to_relative,
    transform_relative_to_absolute,
)
from results import (
    REPORTS_DIR,
    TournamentResult,
    calculate_feature_correlations,
    display_results,
    export_report,
    generate_feature_insights,
)


class TournamentExperiment:
    """Orchestrates an end-to-end Favorite Song Tournament experiment."""

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        cached_name: Optional[str] = None,
        playlist_size: Optional[int] = None,
        tournament_format: str = "swiss",
        hybrid_sub_format: str = "swiss",
        match_size: int = 2,
        data_type: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        judge_callback: Optional[JudgeCallback] = None,
        separator_insert_callback: Optional[SeparatorInsertCallback] = None,
        prompt_automation_callback: Optional[Callable[[], bool]] = None,
        automate_subsequent_rounds: bool = True,
        random_seed: Optional[int] = 42,
        max_rounds: Optional[int] = None,
        on_round1_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self.filepath = filepath
        self.cached_name = cached_name
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

        self.data_type = data_type.lower() if data_type else None
        self.feature_names = feature_names
        self.judge_callback = judge_callback
        self.separator_insert_callback = separator_insert_callback
        self.prompt_automation_callback = prompt_automation_callback
        self.automate_subsequent_rounds = automate_subsequent_rounds
        self.random_seed = random_seed
        self.max_rounds = max_rounds
        self.on_round1_complete = on_round1_complete

        self.playlist: Optional[Playlist] = None
        self.result: Optional[TournamentResult] = None

    def setup(self) -> Playlist:
        """Loads and prepares the playlist data for the tournament."""
        base_playlist: Playlist

        if self.cached_name is not None:
            base_playlist = load_cached_dataset(self.cached_name)
        elif self.filepath is not None and Path(self.filepath).exists():
            dt = self.data_type or "relative"
            base_playlist = load_dataset(
                filepath=self.filepath,
                data_type=dt,
                feature_columns=self.feature_names,
                validate_size=True,
            )
        else:
            dt = self.data_type or "relative"
            size = self.playlist_size or 16
            base_playlist = generate_synthetic_playlist(
                size=size,
                feature_names=self.feature_names,
                data_type=dt,
                random_seed=self.random_seed,
            )

        # Handle user requested data_type conversion if specified
        if self.data_type == "names_only":
            base_playlist = base_playlist.to_names_only()
        elif self.data_type == "absolute" and base_playlist.data_type == "relative":
            base_playlist = transform_relative_to_absolute(base_playlist)
        elif self.data_type == "relative" and base_playlist.data_type == "absolute":
            base_playlist = transform_absolute_to_relative(base_playlist)

        # Hybrid tournament requires relative feature data (README 3c)
        if self.tournament_format == "hybrid":
            if base_playlist.data_type == "names_only":
                raise ValueError("Hybrid tournament requires relative feature data; cannot run with names only.")
            if base_playlist.data_type == "absolute":
                base_playlist = transform_absolute_to_relative(base_playlist)

        # Handle playlist size subsetting (8 - size of dataset) (README 2b)
        total_songs = len(base_playlist)
        if self.playlist_size is not None and self.playlist_size < total_songs:
            if self.playlist_size < 8:
                raise ValueError(f"Playlist size ({self.playlist_size}) must be at least 8.")
            base_playlist = base_playlist.get_random_subset(self.playlist_size, random_seed=self.random_seed)

        # Shuffle internally before the tournament runs without modifying cached data (README 2b)
        self.playlist = base_playlist.shuffled(random_seed=self.random_seed)
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
                separator_insert_callback=self.separator_insert_callback,
                prompt_automation_callback=self.prompt_automation_callback,
                automate_subsequent_rounds=self.automate_subsequent_rounds,
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
                separator_insert_callback=self.separator_insert_callback,
                prompt_automation_callback=self.prompt_automation_callback,
                automate_subsequent_rounds=self.automate_subsequent_rounds,
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

    def export(
        self,
        file_format: str = "txt",
        output_dir: Union[str, Path] = REPORTS_DIR,
        filename: Optional[str] = None,
        view_mode: str = "all",
        notable_count: int = 5,
    ) -> Union[Path, Dict[str, Path]]:
        """Exports tournament results to .txt, .md, or .pdf in the reports subfolder."""
        if self.result is None:
            raise RuntimeError("Experiment has not been run yet. Call run() first.")
        return export_report(
            result=self.result,
            file_format=file_format,
            output_dir=output_dir,
            filename=filename,
            view_mode=view_mode,
            notable_count=notable_count,
        )
