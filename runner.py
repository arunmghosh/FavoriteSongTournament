"""Console runner and interactive driver for Favorite Song Tournament (Phase 1).

Follows README specification:
1. Choose between using an existing (cached) dataset from `data/` or creating a new one from file.
2. Select data mode: names only, absolute feature data, or relative feature data.
   - When importing or caching a dataset, the user explicitly specifies what kind it is.
3. Step-by-step guidance for creating new feature data (absolute rubric or relative iterative upgrades).
4. Option to cache new/converted datasets to `data/` with descriptive notes.
5. Select playlist size (8 to dataset size); random subsetting and internal shuffling.
6. Select tournament format (Swiss, Sorting, Hybrid) and match size (2-8).
7. For Sorting: separator insertion prompt UX, followed by automation choice (DQN vs manual).
8. Results output selection: full rankings, top 5 (notable), or winner only, with taste correlations.
9. Enter 'Q' at any point to quit the current run cleanly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from algorithms import SeparatorInsertCallback
from data_loader import (
    SUGGESTED_FEATURES,
    Playlist,
    Song,
    create_absolute_feature_data,
    create_relative_feature_data,
    generate_synthetic_playlist,
    list_cached_datasets,
    load_cached_dataset,
    load_dataset,
    save_dataset_to_cache,
    transform_absolute_to_relative,
    transform_relative_to_absolute,
)
from experiment import TournamentExperiment
from results import display_results


class UserQuitTournament(Exception):
    """Raised when the user enters 'Q' or 'q' at any prompt to quit the current run."""
    pass


def safe_input(prompt_text: str = "") -> str:
    """Reads input from the console and raises UserQuitTournament if 'Q' or 'q' is entered."""
    try:
        raw = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        raise UserQuitTournament("User requested exit.")
    if raw.lower() in ("q", "quit"):
        raise UserQuitTournament("User requested exit.")
    return raw


def print_banner() -> None:
    """Displays the application welcome banner."""
    print("=" * 80)
    print(" " * 24 + "🎵 FAVORITE SONG TOURNAMENT 🎵")
    print(" " * 26 + "Phase 1: Console Prototype")
    print("=" * 80)
    print("Discover your musical taste and crown your favorite song of the moment!")
    print("Tip: Enter 'Q' at any prompt to quit the current tournament run.\n")


def prompt_int(
    prompt_text: str,
    min_val: int,
    max_val: int,
    default_val: Optional[int] = None,
) -> int:
    """Prompts the user for an integer within [min_val, max_val] or 'Q' to quit."""
    default_str = f" [Default: {default_val}]" if default_val is not None else ""
    while True:
        try:
            raw = safe_input(f"{prompt_text}{default_str}: ")
            if not raw and default_val is not None:
                return default_val
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  ⚠️  Please enter an integer between {min_val} and {max_val} (or 'Q' to quit).")
        except ValueError:
            if default_val is not None and not raw:
                return default_val
            print(f"  ⚠️  Invalid input. Please enter a valid number (or 'Q' to quit).")


def prompt_choice(
    prompt_text: str,
    options: List[str],
    default_index: int = 0,
) -> int:
    """Presents a list of options and prompts the user to select one or enter 'Q' to quit."""
    print(prompt_text)
    for idx, opt in enumerate(options, 1):
        def_tag = " (Default)" if idx - 1 == default_index else ""
        print(f"  [{idx}] {opt}{def_tag}")
    while True:
        try:
            raw = safe_input(f"Select option (1-{len(options)}) [Default: {default_index + 1}]: ")
            if not raw:
                return default_index
            choice = int(raw) - 1
            if 0 <= choice < len(options):
                return choice
            print(f"  ⚠️  Please enter a number between 1 and {len(options)} (or 'Q' to quit).")
        except ValueError:
            print(f"  ⚠️  Invalid choice. Please enter a number from 1 to {len(options)} (or 'Q' to quit).")


def prompt_yes_no(prompt_text: str, default_yes: bool = True) -> bool:
    """Prompts the user for a yes/no question or 'Q' to quit."""
    def_hint = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = safe_input(f"{prompt_text} {def_hint}: ").lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  ⚠️  Please enter 'y' or 'n' (or 'Q' to quit).")


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
        raw = safe_input(
            f"Enter your ranking in order (e.g. '{prompt_hint}' or 'Q' to quit): "
        )
        if not raw:
            return songs

        parts = raw.replace(",", " ").split()
        if len(parts) != n:
            print(f"  ⚠️  Please provide ranks for all {n} songs separated by spaces (or 'Q' to quit).")
            continue

        try:
            indices = [int(p) - 1 for p in parts]
            if sorted(indices) != list(range(n)):
                print(f"  ⚠️  Please include each number from 1 to {n} exactly once.")
                continue

            ranked_songs = [songs[i] for i in indices]
            return ranked_songs
        except ValueError:
            print("  ⚠️  Invalid input format. Please enter numbers separated by spaces (or 'Q' to quit).")


def console_separator_insert_callback(
    candidate: Song,
    ordered_separators: List[Song],
    match_id: int,
    round_num: int,
) -> int:
    """Prompt asking user where to insert candidate among ranked separators (README 3b)."""
    s_count = len(ordered_separators)
    print("\n" + "-" * 70)
    print(f"🥊 MATCH #{match_id} (Round {round_num}) — Place the new song among the separators:")
    print(f"  Candidate: \"{candidate.title}\" by {candidate.artist}")
    print("-" * 70)
    print(f"  [1] Above \"{ordered_separators[0].title}\" (1st place)")
    for i in range(1, s_count):
        print(f"  [{i + 1}] Between \"{ordered_separators[i - 1].title}\" and \"{ordered_separators[i].title}\"")
    print(f"  [{s_count + 1}] Below \"{ordered_separators[-1].title}\" (last place)")
    print("-" * 70)

    choice = prompt_int(
        f"Where would you place \"{candidate.title}\"?",
        min_val=1,
        max_val=s_count + 1,
        default_val=1,
    )
    # Convert 1-based slot to 0-based insert index
    return choice - 1


# ==============================================================================
# INTERACTIVE DATASET CREATION / CONFIGURATION WORKFLOW
# ==============================================================================

def configure_dataset_interactive() -> Playlist:
    """Guides the user through selecting or creating, scoring, and caching a dataset."""
    cached_list = list_cached_datasets()

    # Step 1: Use cached or create new dataset
    source_options = []
    if cached_list:
        source_options.append("Choose an existing (cached) dataset from the data folder")
    source_options.append("Import a new dataset from a spreadsheet file (.xlsx or .csv)")
    source_options.append("Generate a synthetic dataset (for testing)")

    src_idx = prompt_choice("\nHow would you like to provide the playlist songs?", source_options, default_index=0)
    chosen_mode = source_options[src_idx]

    playlist: Playlist

    if "cached" in chosen_mode:
        cached_menu = [
            f"{d['name']} ({d['data_type']}, {d['song_count']} songs) — {d.get('note', '')}"
            for d in cached_list
        ]
        d_idx = prompt_choice("\nSelect a cached dataset:", cached_menu, default_index=0)
        selected_record = cached_list[d_idx]
        playlist = load_cached_dataset(selected_record["name"])
        print(f"\n✅ Loaded '{playlist.name}' ({len(playlist)} songs, {playlist.data_type} data).")

    elif "Import" in chosen_mode:
        while True:
            file_input = safe_input("\nEnter path to spreadsheet (.xlsx or .csv) containing songs (8-128 rows): ")
            path = Path(file_input)
            if not path.exists():
                print(f"  ⚠️  File '{file_input}' not found. Please try again.")
                continue

            # Explicitly ask user to specify what kind of dataset is being imported
            kind_options = [
                "Just song names (titles and artists only)",
                "Absolute feature data (independent rubric scores)",
                "Relative feature data (comparative tier ratings)",
            ]
            k_idx = prompt_choice("\nSpecify the kind of dataset you are importing:", kind_options, default_index=0)
            imported_kind = ["names_only", "absolute", "relative"][k_idx]

            try:
                playlist = load_dataset(path, data_type=imported_kind, validate_size=True)
                if imported_kind != "names_only" and playlist.data_type == "names_only":
                    print("  ⚠️  Note: No numeric feature columns found in spreadsheet. Imported as 'names_only'.")
                else:
                    playlist.data_type = imported_kind

                print(f"\n✅ Successfully imported '{playlist.name}' with {len(playlist)} songs as {playlist.data_type.upper()}.")
                break
            except Exception as e:
                print(f"  ⚠️  Failed to load spreadsheet: {e}. Please try another file.")
    else:
        synth_size = prompt_int("\nEnter number of songs for synthetic playlist (8-128)", min_val=8, max_val=128, default_val=16)
        synth_kind_opts = [
            "Just song names",
            "Absolute feature data",
            "Relative feature data",
        ]
        sk_idx = prompt_choice("Specify dataset kind for synthetic playlist:", synth_kind_opts, default_index=2)
        synth_kind = ["names_only", "absolute", "relative"][sk_idx]
        playlist = generate_synthetic_playlist(size=synth_size, data_type=synth_kind)
        print(f"\n✅ Generated synthetic playlist with {len(playlist)} songs ({synth_kind.upper()}).")

    # Step 2: Prompt for data mode / modifications (README 2b)
    has_features = len(playlist.feature_names) > 0 and playlist.data_type != "names_only"
    print(f"\nCurrent dataset data type: {playlist.data_type.upper()}")

    mode_options = [
        "Use just song names (title and artist, no feature data)",
        "Use absolute feature data (independent rubric scores)",
        "Use relative feature data (comparative tier ratings)",
    ]
    cur_def = 0 if not has_features else (1 if playlist.data_type == "absolute" else 2)
    mode_idx = prompt_choice("\nHow do you want to configure the feature data for this tournament?", mode_options, default_index=cur_def)

    target_mode = ["names_only", "absolute", "relative"][mode_idx]

    # Handle conversion or feature creation
    if target_mode == "names_only":
        if playlist.data_type != "names_only":
            playlist = playlist.to_names_only()
            print("✂️  Stripped feature scores; using song titles and artists only.")

    elif target_mode == "absolute":
        if playlist.data_type == "absolute" and has_features:
            print("✅ Dataset already contains absolute feature data.")
        elif playlist.data_type == "relative" and has_features:
            if prompt_yes_no("Convert existing relative feature scores to absolute scale (1-10)?", default_yes=True):
                playlist = transform_relative_to_absolute(playlist)
                print("🔄 Converted relative data to absolute rubric scale.")
            else:
                playlist = create_absolute_data_interactive(playlist)
        else:
            print("\n📝 Creating absolute feature data step-by-step...")
            playlist = create_absolute_data_interactive(playlist)

    elif target_mode == "relative":
        if playlist.data_type == "relative" and has_features:
            print("✅ Dataset already contains relative feature data.")
        elif playlist.data_type == "absolute" and has_features:
            if prompt_yes_no("Convert existing absolute feature scores to relative tier ratings?", default_yes=True):
                playlist = transform_absolute_to_relative(playlist)
                print("🔄 Converted absolute data to relative geometric tiers.")
            else:
                playlist = create_relative_data_interactive(playlist)
        else:
            print("\n📝 Creating relative feature data step-by-step...")
            playlist = create_relative_data_interactive(playlist)

    # Step 3: Prompt to cache new/modified dataset, explicitly specifying kind (README 2b)
    if prompt_yes_no(f"\nWould you like to cache '{playlist.name}' to the data folder for future tournaments?", default_yes=False):
        cache_name = safe_input(f"Enter dataset name [Default: {playlist.name}]: ") or playlist.name

        # User explicitly specifies kind when caching
        cache_kind_options = [
            "Just song names (no feature scores)",
            "Absolute feature data",
            "Relative feature data",
        ]
        cur_kind_def = 0 if playlist.data_type == "names_only" else (1 if playlist.data_type == "absolute" else 2)
        ck_idx = prompt_choice(f"Specify the kind of data to cache '{cache_name}' as:", cache_kind_options, default_index=cur_kind_def)
        chosen_cache_kind = ["names_only", "absolute", "relative"][ck_idx]

        note = safe_input("Enter a short descriptive note for this dataset: ")
        saved_path = save_dataset_to_cache(playlist, name=cache_name, note=note, data_type=chosen_cache_kind)
        print(f"💾 Dataset successfully cached as {chosen_cache_kind.upper()} to '{saved_path}' with metadata!")

    return playlist


def create_absolute_data_interactive(playlist: Playlist) -> Playlist:
    """Guides user through scoring absolute feature data step-by-step."""
    features = select_criteria_interactive()
    rubrics: Dict[str, Dict[str, Any]] = {}

    print("\nDefine Rubric for each feature (min, max, description):")
    for f in features:
        print(f"  • Feature: '{f}'")
        rubrics[f] = {"min": 1.0, "max": 10.0, "description": f"Rating for {f} on scale 1-10"}

    batch_size = prompt_int("How many songs would you like to score at a time?", min_val=1, max_val=len(playlist), default_val=1)

    real_songs = list(playlist.real_songs)
    new_songs: List[Song] = []

    print("\nEnter scores for each song using your rubric (1-10) or 'Q' to quit:")
    for i in range(0, len(real_songs), batch_size):
        batch = real_songs[i: i + batch_size]
        for s in batch:
            print(f"\nSong: \"{s.title}\" by {s.artist}")
            song_feats: Dict[str, float] = {}
            for f in features:
                val = prompt_int(f"  Score for '{f}' (1-10)", min_val=1, max_val=10, default_val=5)
                song_feats[f] = float(val)
            s_copy = s.copy()
            s_copy.features = song_feats
            new_songs.append(s_copy)

    abs_playlist = Playlist(songs=new_songs, data_type="absolute", feature_names=features, name=f"{playlist.name}_scored")
    return abs_playlist


def create_relative_data_interactive(playlist: Playlist) -> Playlist:
    """Guides user through scoring relative feature data iteratively (README 2b, 3c)."""
    features = select_criteria_interactive()
    batch_size = prompt_int("Choose batch size for relative upgrade comparisons (2-8)", min_val=2, max_val=8, default_val=6)

    def interactive_upgrade_judge(batch: List[Song], feat_name: str, current_tier: int) -> List[Song]:
        max_upgrades = max(1, len(batch) // 2)
        print("\n" + "-" * 70)
        print(f"Relative Upgrade: '{feat_name}' (Tier {current_tier} -> {current_tier + 1})")
        print(f"Select AT MOST {max_upgrades} of the following songs to promote (or 'Q' to quit):")
        for idx, s in enumerate(batch, 1):
            print(f"  [{idx}] \"{s.title}\" by {s.artist}")
        print("-" * 70)
        raw = safe_input("Enter the numbers of promoted songs separated by spaces (e.g. '1 2'): ")
        if not raw:
            return batch[:max_upgrades]
        try:
            choices = [int(x) - 1 for x in raw.replace(",", " ").split() if x.isdigit()]
            selected = [batch[c] for c in choices if 0 <= c < len(batch)]
            return selected[:max_upgrades]
        except Exception:
            return batch[:max_upgrades]

    rel_playlist = create_relative_feature_data(
        playlist=playlist,
        feature_names=features,
        batch_size=batch_size,
        judge_batch_callback=interactive_upgrade_judge,
    )
    return rel_playlist


def select_criteria_interactive() -> List[str]:
    """Allows user to select criteria from suggestions or enter custom ones."""
    crit_options = [
        "Use suggested standard criteria (Melody, Catchiness, Lyricism, Message)",
        "Enter custom criteria names",
    ]
    c_idx = prompt_choice("\nSelect feature criteria to evaluate:", crit_options, default_index=0)
    if c_idx == 0:
        return ["Melody", "Catchiness", "Lyricism", "Message"]
    else:
        while True:
            raw = safe_input("Enter feature names separated by commas (e.g., Melody, Rhythm, Vocals): ")
            feats = [f.strip() for f in raw.split(",") if f.strip()]
            if feats:
                return feats
            print("  ⚠️  Please enter at least one feature name.")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for Phase 1 Favorite Song Tournament."""
    parser = argparse.ArgumentParser(description="Favorite Song Tournament - Phase 1 Console Application")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration without interactive prompts")
    parser.add_argument("--cached", type=str, help="Name of cached dataset to load")
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
            cached_name=args.cached,
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

    # Interactive flow enclosed in UserQuitTournament handler
    try:
        # Step 1: Configure dataset (cached, new, or synthetic)
        playlist = configure_dataset_interactive()

        # Step 2: Choose playlist size (8 - size of dataset) (README 2b)
        total_available = len(playlist)
        print(f"\nTotal songs in selected dataset: {total_available}")
        chosen_size = prompt_int(
            f"Choose tournament playlist size (8-{total_available})",
            min_val=8,
            max_val=total_available,
            default_val=total_available,
        )
        if chosen_size < total_available:
            playlist = playlist.get_random_subset(chosen_size)
            print(f"🎲 Selected random subset of {chosen_size} songs.")

        # Shuffle internally before tournament runs (README 2b)
        playlist = playlist.shuffled()

        # Step 3: Choose Tournament format compatible with data (README 2b)
        format_options = [
            "Swiss (Ideal for names only, but compatible with any data)",
            "Sorting (Multi-pivot quicksort with separator insertion, DQN automation)",
        ]
        if playlist.data_type == "relative":
            format_options.append("Hybrid (Filters strictly dominated songs, then runs Swiss/Sorting)")

        fmt_idx = prompt_choice("\nSelect tournament format:", format_options, default_index=0)
        tournament_format = ["swiss", "sorting", "hybrid"][fmt_idx]

        hybrid_sub_format = "swiss"
        if tournament_format == "hybrid":
            sub_choice = prompt_choice(
                "Select sub-tournament format for surviving songs:",
                ["Swiss Tournament", "Sorting Tournament"],
                default_index=0,
            )
            hybrid_sub_format = "swiss" if sub_choice == 0 else "sorting"

        # Step 4: Choose match size (2-8)
        match_size = prompt_int("\nEnter match size (2-8 songs per match)", min_val=2, max_val=8, default_val=2)

        # Step 5: Automation prompt callback for Sorting (README 2b, 3b)
        has_features = playlist.data_type != "names_only" and len(playlist.feature_names) > 0

        def prompt_automation() -> bool:
            if not has_features:
                return False
            print("\n" + "*" * 70)
            print("🧠 ROUND 1 COMPLETE!")
            print("   The Deep Q-Network (DQN) has learned your preferences from Round 1 decisions.")
            print("*" * 70)
            return prompt_yes_no("Would you like to automate the remaining tournament rounds using the DQN?", default_yes=True)

        print("\n" + "=" * 80)
        print(f"🚀 STARTING {tournament_format.upper()} TOURNAMENT...")
        print(f"   Songs: {len(playlist)} | Match Size: {match_size} | Data Type: {playlist.data_type}")
        print("=" * 80)

        exp = TournamentExperiment(
            playlist_size=chosen_size,
            tournament_format=tournament_format,
            hybrid_sub_format=hybrid_sub_format,
            match_size=match_size,
            data_type=playlist.data_type,
            judge_callback=console_judge_callback,
            separator_insert_callback=console_separator_insert_callback,
            prompt_automation_callback=prompt_automation,
            automate_subsequent_rounds=True,
        )
        exp.playlist = playlist

        result = exp.run()

        if result.automated_matches > 0:
            print(f"\n✅ All {result.automated_matches} automated match results have been decided!")

        # Step 6: View selection (README 2b, Phase 3: "one, top five, all")
        view_options = [
            "View complete tournament rankings (All songs)",
            "View notable songs only (Top 5)",
            "View winner only (Top 1)",
        ]
        view_idx = prompt_choice("\nHow would you like to view the results?", view_options, default_index=0)
        view_mode = ["all", "notable", "winner"][view_idx]

        # Step 7: Display results
        exp.display(view_mode=view_mode, notable_count=5)

    except UserQuitTournament:
        print("\n\n👋 Tournament run cancelled by user. See you next time!\n")
        return


if __name__ == "__main__":
    main()
