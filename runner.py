"""Console runner and interactive driver for Favorite Song Tournament (Phase 1).

Flow:
1. Prompts for playlist size (validated 8-128).
2. Prompts for data source (default 'favorite_song_data.xlsx' or synthetic).
3. Prompts for tournament format:
   - Swiss (No data needed)
   - Sorting (Absolute feature data, DQN learns from Round 1)
   - Hybrid (Relative feature data, eliminates strictly dominated songs)
4. Prompts for criteria/features (suggestions or custom).
5. Prompts for match size (2-8).
6. Runs tournament:
   - Prints each manual match to the console for the user to rank.
   - For Sorting: notifies user when DQN takes over and completes automated matches.
7. Prompts user to select output view (winner only, notable top N, all).
8. Prints complete tournament results, rankings table, and music taste insights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from data_loader import (
    SUGGESTED_FEATURES,
    Song,
    load_dataset,
)
from experiment import TournamentExperiment
from results import display_results


def print_banner() -> None:
    """Displays the application welcome banner."""
    print("=" * 80)
    print(" " * 24 + "🎵 FAVORITE SONG TOURNAMENT 🎵")
    print(" " * 26 + "Phase 1: Console Prototype")
    print("=" * 80)
    print("Discover your musical taste and crown your favorite song of the moment!\n")


def prompt_int(
    prompt_text: str,
    min_val: int,
    max_val: int,
    default_val: Optional[int] = None,
) -> int:
    """Prompts the user for an integer within [min_val, max_val]."""
    default_str = f" [Default: {default_val}]" if default_val is not None else ""
    while True:
        try:
            raw = input(f"{prompt_text}{default_str}: ").strip()
            if not raw and default_val is not None:
                return default_val
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  ⚠️  Please enter an integer between {min_val} and {max_val}.")
        except (ValueError, EOFError):
            if default_val is not None:
                return default_val
            print(f"  ⚠️  Invalid input. Please enter a valid number.")


def prompt_choice(
    prompt_text: str,
    options: List[str],
    default_index: int = 0,
) -> int:
    """Presents a list of options and prompts the user to select one."""
    print(prompt_text)
    for idx, opt in enumerate(options, 1):
        def_tag = " (Default)" if idx - 1 == default_index else ""
        print(f"  [{idx}] {opt}{def_tag}")
    while True:
        try:
            raw = input(f"Select option (1-{len(options)}) [Default: {default_index + 1}]: ").strip()
            if not raw:
                return default_index
            choice = int(raw) - 1
            if 0 <= choice < len(options):
                return choice
            print(f"  ⚠️  Please enter a number between 1 and {len(options)}.")
        except (ValueError, EOFError):
            return default_index


def console_judge_callback(songs: List[Song], match_id: int, round_num: int) -> List[Song]:
    """Interactive console prompt for a human user to judge/rank a match."""
    print("\n" + "-" * 70)
    print(f"🥊 MATCH #{match_id} (Round {round_num}) — Rank the following songs from best to worst:")
    print("-" * 70)
    for idx, s in enumerate(songs, 1):
        feat_str = ", ".join(f"{k}: {v:.1f}" for k, v in s.features.items())
        feats_info = f" ({feat_str})" if feat_str else ""
        print(f"  [{idx}] \"{s.title}\" by {s.artist}{feats_info}")
    print("-" * 70)

    n = len(songs)
    prompt_hint = " ".join(str(i) for i in range(1, n + 1))
    while True:
        try:
            raw = input(
                f"Enter your ranking in order (e.g. '{prompt_hint}' or '{' '.join(reversed(prompt_hint.split()))}'): "
            ).strip()
            if not raw:
                # Default: keep current order
                return songs

            parts = raw.replace(",", " ").split()
            if len(parts) != n:
                print(f"  ⚠️  Please provide ranks for all {n} songs separated by spaces.")
                continue

            indices = [int(p) - 1 for p in parts]
            if sorted(indices) != list(range(n)):
                print(f"  ⚠️  Please include each number from 1 to {n} exactly once.")
                continue

            ranked_songs = [songs[i] for i in indices]
            return ranked_songs
        except (ValueError, EOFError):
            print("  ⚠️  Invalid input format. Please enter numbers separated by spaces.")


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for Phase 1 Favorite Song Tournament."""
    parser = argparse.ArgumentParser(description="Favorite Song Tournament - Phase 1 Console Application")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration without interactive prompts")
    parser.add_argument("--file", type=str, default="favorite_song_data.xlsx", help="Dataset filepath")
    parser.add_argument("--size", type=int, default=16, help="Playlist size (8-128)")
    parser.add_argument("--format", type=str, choices=["swiss", "sorting", "hybrid"], default="swiss", help="Tournament format")
    parser.add_argument("--match-size", type=int, default=2, help="Match size (2-8)")
    parser.add_argument("--view", type=str, choices=["all", "notable", "winner"], default="all", help="Output view")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    print_banner()

    if args.demo:
        print("🤖 Running in automated DEMO mode...")
        filepath = args.file if Path(args.file).exists() else None
        exp = TournamentExperiment(
            filepath=filepath,
            playlist_size=args.size,
            tournament_format=args.format,
            match_size=args.match_size,
            random_seed=args.seed,
            on_round1_complete=lambda: print("\n⚡ Round 1 finished! DQN trained on decisions. Automating remaining rounds..."),
        )
        result = exp.run()
        print("\n✅ All automated match results have been decided!\n")
        display_results(result, view_mode=args.view)
        return

    # Interactive Mode
    # Step 1: Playlist size
    playlist_size = prompt_int("Enter playlist size (8-128)", min_val=8, max_val=128, default_val=args.size)

    # Step 2: Data Source
    default_file = "favorite_song_data.xlsx"
    file_exists = Path(default_file).exists()
    data_source_options = [
        f"Use existing dataset file ('{default_file}')" if file_exists else "Load from custom file path",
        "Generate synthetic playlist (for quick testing)",
    ]
    if file_exists:
        data_source_options.append("Specify a different file path")

    src_choice = prompt_choice("Choose data source:", data_source_options, default_index=0)

    dataset_path: Optional[str] = None
    if src_choice == 0 and file_exists:
        dataset_path = default_file
    elif (src_choice == 0 and not file_exists) or (file_exists and src_choice == 2):
        while True:
            p_in = input("Enter path to .xlsx or .csv dataset: ").strip()
            if Path(p_in).exists():
                dataset_path = p_in
                break
            print(f"  ⚠️  File '{p_in}' not found. Please try again.")

    # Step 3: Tournament Format
    format_options = [
        "Swiss (No prior data needed - pools songs by tournament score)",
        "Sorting (Absolute feature data - user judges Round 1, DQN learns & automates)",
        "Hybrid (Relative feature data - eliminates strictly dominated songs first)",
    ]
    fmt_idx = prompt_choice("Select tournament format:", format_options, default_index=0)
    tournament_format = ["swiss", "sorting", "hybrid"][fmt_idx]

    hybrid_sub_format = "swiss"
    if tournament_format == "hybrid":
        sub_choice = prompt_choice(
            "Select sub-tournament format for songs that survive dominance filter:",
            ["Swiss Tournament", "Sorting Tournament (DQN automated)"],
            default_index=0,
        )
        hybrid_sub_format = "swiss" if sub_choice == 0 else "sorting"

    # Step 4: Criteria selection
    print("\nFeature Criteria:")
    print("Suggested criteria:", ", ".join(SUGGESTED_FEATURES[:4]))
    crit_choice = prompt_choice(
        "Criteria selection:",
        [
            "Use default / dataset criteria (Melody, Catchiness, Lyricism, Message)",
            "Use custom criteria list",
        ],
        default_index=0,
    )
    feature_names: Optional[List[str]] = None
    if crit_choice == 1:
        custom_input = input("Enter feature names separated by commas: ").strip()
        if custom_input:
            feature_names = [f.strip() for f in custom_input.split(",") if f.strip()]

    # Step 5: Match Size
    match_size = prompt_int("Enter match size (2-8 songs per match)", min_val=2, max_val=8, default_val=2)

    # Step 6: Execute tournament
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING TOURNAMENT...")
    print("=" * 80)

    def on_round1_done() -> None:
        print("\n" + "*" * 70)
        print("🧠 ROUND 1 COMPLETE!")
        print("   The Deep Q-Network (DQN) has learned your preferences from Round 1.")
        print("   Automating subsequent match results...")
        print("*" * 70)

    exp = TournamentExperiment(
        filepath=dataset_path,
        playlist_size=playlist_size,
        tournament_format=tournament_format,
        hybrid_sub_format=hybrid_sub_format,
        match_size=match_size,
        feature_names=feature_names,
        judge_callback=console_judge_callback,
        random_seed=args.seed,
        on_round1_complete=on_round1_done,
    )

    playlist = exp.setup()
    print(f"Loaded playlist with {len(playlist)} songs.")
    if tournament_format == "hybrid":
        survivors, dominated = playlist.filter_strictly_dominated()
        print(f"Dominance analysis: {len(dominated)} songs strictly dominated, {len(survivors)} songs advancing.")

    print(f"Beginning {tournament_format.capitalize()} Tournament with match size {match_size}...\n")

    result = exp.run()

    if result.automated_matches > 0:
        print(f"\n✅ All {result.automated_matches} automated match results have been decided!")

    # Step 7: View selection
    view_options = [
        "View complete tournament rankings (All songs)",
        "View notable songs only (Top 5)",
        "View winner only",
    ]
    view_idx = prompt_choice("\nHow would you like to view the results?", view_options, default_index=0)
    view_mode = ["all", "notable", "winner"][view_idx]

    # Step 8: Display results
    exp.display(view_mode=view_mode)


if __name__ == "__main__":
    main()

