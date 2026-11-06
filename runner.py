"""Console runner and interactive driver for Favorite Song Tournament (Phase 2).

Follows README specification:
1. Upon launching, users are prompted if they want to:
   - Draw from the existing cache
   - Import a new dataset
   - Manage the cache (view or delete cached datasets)
2. Select data mode: names only, absolute feature data, or relative feature data.
   - When importing or caching a dataset, the user explicitly specifies what kind it is.
3. Step-by-step guidance for creating new feature data (absolute rubric or relative iterative upgrades).
4. Option to cache new/converted datasets to `data/` with descriptive notes.
5. Select playlist size (8 to dataset size); random subsetting and internal shuffling.
6. Select tournament format (Swiss, Sorting, Hybrid) and match size (2-8).
7. For Sorting: separator insertion prompt UX, followed by automation choice (DQN vs manual).
8. Results output selection: full rankings, top 5 (notable), or winner only, with taste correlations.
9. File output: exports reports to the `reports/` subfolder in .txt, .md, or .pdf format.
10. Restart & Quit: Type 'R' into any prompt to restart setup, or 'Q' to quit the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from algorithms import SeparatorInsertCallback
from data_loader import (
    SUGGESTED_FEATURES,
    Playlist,
    Song,
    create_absolute_feature_data,
    create_relative_feature_data,
    delete_cached_dataset,
    generate_synthetic_playlist,
    list_cached_datasets,
    load_cached_dataset,
    load_dataset,
    save_dataset_to_cache,
    transform_absolute_to_relative,
    transform_relative_to_absolute,
)
from experiment import TournamentExperiment
from results import (
    REPORTS_DIR,
    display_results,
    export_report,
)


class UserQuitTournament(Exception):
    """Raised when the user enters 'Q' or 'q' at any prompt to quit the current run."""
    pass


class UserRestartTournament(Exception):
    """Raised when the user enters 'R' or 'r' at any prompt to restart the tournament setup."""
    pass


def safe_input(prompt_text: str = "") -> str:
    """Reads input from the console.

    Raises:
        UserQuitTournament: If user enters 'Q' or 'q'.
        UserRestartTournament: If user enters 'R' or 'r'.
    """
    try:
        raw = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        raise UserQuitTournament("User requested exit.")
    if raw.lower() in ("q", "quit"):
        raise UserQuitTournament("User requested exit.")
    if raw.lower() in ("r", "restart"):
        raise UserRestartTournament("User requested restart.")
    return raw


def print_banner() -> None:
    """Displays the application welcome banner."""
    print("=" * 80)
    print(" " * 24 + "🎵 FAVORITE SONG TOURNAMENT 🎵")
    print(" " * 25 + "Phase 2: File Output & Reports")
    print("=" * 80)
    print("Discover your musical taste and crown your favorite song of the moment!")
    print("Tip: Enter 'Q' at any prompt to quit, or 'R' to restart setup.\n")


def prompt_int(
    prompt_text: str,
    min_val: int,
    max_val: int,
    default_val: Optional[int] = None,
) -> int:
    """Prompts the user for an integer within [min_val, max_val], with 'Q' and 'R' support."""
    default_str = f" [Default: {default_val}]" if default_val is not None else ""
    while True:
        try:
            raw = safe_input(f"{prompt_text}{default_str}: ")
            if not raw and default_val is not None:
                return default_val
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  ⚠️  Please enter an integer between {min_val} and {max_val} (or 'Q' to quit, 'R' to restart).")
        except ValueError:
            if default_val is not None and not raw:
                return default_val
            print(f"  ⚠️  Invalid input. Please enter a valid number (or 'Q' to quit, 'R' to restart).")


def prompt_choice(
    prompt_text: str,
    options: List[str],
    default_index: int = 0,
) -> int:
    """Presents a list of options and prompts the user to select one, with 'Q' and 'R' support."""
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
            print(f"  ⚠️  Please enter a number between 1 and {len(options)} (or 'Q' to quit, 'R' to restart).")
        except ValueError:
            print(f"  ⚠️  Invalid choice. Please enter a number from 1 to {len(options)} (or 'Q' to quit, 'R' to restart).")


def prompt_yes_no(prompt_text: str, default_yes: bool = True) -> bool:
    """Prompts the user for a yes/no question, with 'Q' and 'R' support."""
    def_hint = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = safe_input(f"{prompt_text} {def_hint}: ").lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  ⚠️  Please enter 'y' or 'n' (or 'Q' to quit, 'R' to restart).")


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
            f"Enter your ranking in order (e.g. '{prompt_hint}', 'R' to restart, 'Q' to quit): "
        )
        if not raw:
            return songs

        parts = raw.replace(",", " ").split()
        if len(parts) != n:
            print(f"  ⚠️  Please provide ranks for all {n} songs separated by spaces.")
            continue

        try:
            indices = [int(p) - 1 for p in parts]
            if sorted(indices) != list(range(n)):
                print(f"  ⚠️  Please include each number from 1 to {n} exactly once.")
                continue

            ranked_songs = [songs[i] for i in indices]
            return ranked_songs
        except ValueError:
            print("  ⚠️  Invalid input format. Please enter numbers separated by spaces.")


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
    return choice - 1


# ==============================================================================
# CACHE MANAGEMENT WORKFLOW (README Phase 2)
# ==============================================================================

def manage_cache_interactive() -> None:
    """Allows user to inspect the full cache, view dataset contents and indices, and delete datasets (README Phase 2)."""
    while True:
        cached_list = list_cached_datasets()
        print("\n" + "=" * 80)
        print("🗄️  CACHE MANAGEMENT — FULL CACHE DIRECTORY")
        print("=" * 80)

        if not cached_list:
            print("  (Cache folder is currently empty)")
            print("-" * 80)
            safe_input("Press Enter to return to main menu...")
            return

        print(f"Total Cached Datasets: {len(cached_list)}\n")
        print(f"{'Idx':<5} | {'Dataset Name':<24} | {'Data Kind':<12} | {'Songs':<6} | {'Features'}")
        print("-" * 80)
        for idx, d in enumerate(cached_list, 1):
            feats = ", ".join(d.get("feature_names", [])) if d.get("feature_names") else "None (names only)"
            if len(feats) > 28:
                feats = feats[:25] + "..."
            kind_str = d.get("data_type", "unknown").upper()
            songs_str = str(d.get("song_count", "?"))
            print(f"[{idx:<2}] | {d['name']:<24} | {kind_str:<12} | {songs_str:<6} | {feats}")
            print(f"      • File: {d.get('filename', '')}  |  Note: {d.get('note', 'No note')}")
            print("-" * 80)

        actions = [
            "View full contents of a cached dataset (all songs, song indices, and feature ratings)",
            "Delete a dataset from the cache",
            "Return to main menu",
        ]
        act_idx = prompt_choice("\nChoose an action:", actions, default_index=0)
        if act_idx == 2:
            return

        if act_idx == 0:
            target_idx = prompt_int(
                f"\nEnter dataset index to view in full (1-{len(cached_list)})",
                min_val=1,
                max_val=len(cached_list),
                default_val=1,
            ) - 1
            target = cached_list[target_idx]
            target_name = target["name"]
            try:
                loaded = load_cached_dataset(target_name)
                print("\n" + "=" * 80)
                print(f"📄 FULL DATASET INSPECTION: [{target_idx + 1}] {loaded.name}")
                print("=" * 80)
                print(f"  • Dataset Index:    {target_idx + 1}")
                print(f"  • Data Type:        {loaded.data_type.upper()}")
                print(f"  • Total Songs:      {len(loaded)}")
                feat_str = ", ".join(loaded.feature_names) if loaded.feature_names else "None (names only)"
                print(f"  • Features:         {feat_str}")
                print(f"  • Source File:      {target.get('filename', '')}")
                print(f"  • Note:             {target.get('note', loaded.note or 'No note')}")
                print("-" * 80)

                feat_cols = loaded.feature_names
                feat_header = "".join(f"  {f[:10]:>10}" for f in feat_cols) if feat_cols else ""
                print(f"{'Idx':<5} {'Song Title':<30} {'Artist':<22}{feat_header}")
                print("-" * 80)
                for s_idx, song in enumerate(loaded.real_songs, 1):
                    f_vals = "".join(f"  {song.features.get(f, 0.0):>10.1f}" for f in feat_cols) if feat_cols else ""
                    title_trunc = song.title[:28] if len(song.title) > 28 else song.title
                    artist_trunc = song.artist[:20] if len(song.artist) > 20 else song.artist
                    print(f"{s_idx:<5} {title_trunc:<30} {artist_trunc:<22}{f_vals}")
                print("-" * 80)
            except Exception as e:
                print(f"  ⚠️  Could not load full dataset details: {e}")

            safe_input("\nPress Enter to return to Cache Management...")

        elif act_idx == 1:
            del_idx = prompt_int(
                f"\nEnter dataset index to delete (1-{len(cached_list)})",
                min_val=1,
                max_val=len(cached_list),
            ) - 1
            target = cached_list[del_idx]
            target_name = target["name"]

            if prompt_yes_no(f"⚠️  Are you sure you want to permanently delete '[{del_idx + 1}] {target_name}'?", default_yes=False):
                success = delete_cached_dataset(target_name)
                if success:
                    print(f"🗑️  Successfully deleted '{target_name}' from the cache.")
                else:
                    print(f"⚠️  Could not delete '{target_name}'.")


# ==============================================================================
# INTERACTIVE DATASET CREATION / CONFIGURATION WORKFLOW
# ==============================================================================

def configure_dataset_interactive() -> Playlist:
    """Initial interactive prompt: import, draw from cache, or manage cache (README Phase 2)."""
    while True:
        cached_list = list_cached_datasets()
        has_cache = len(cached_list) > 0

        # README Phase 2: "Upon launching the program, users are prompted if they want to
        # import a dataset, draw from the existing cache, or manage the cache."
        source_options = [
            "Draw from the existing cache (select an existing dataset)" if has_cache else "Draw from existing cache (empty)",
            "Import a new dataset (from spreadsheet file .xlsx or .csv)",
            "Manage the cache (view details or delete cached datasets)",
            "Generate a synthetic dataset (for quick testing)",
        ]

        def_idx = 0 if has_cache else 1
        src_idx = prompt_choice("\nHow would you like to proceed?", source_options, default_index=def_idx)
        chosen_mode = source_options[src_idx]

        if "Manage the cache" in chosen_mode:
            manage_cache_interactive()
            continue

        playlist: Playlist

        if "Draw from the existing cache" in chosen_mode:
            if not has_cache:
                print("  ⚠️  Cache is empty. Please import a dataset or generate a synthetic one.")
                continue
            cached_menu = [
                f"{d['name']} ({d['data_type']}, {d['song_count']} songs) — {d.get('note', '')}"
                for d in cached_list
            ]
            d_idx = prompt_choice("\nSelect a cached dataset:", cached_menu, default_index=0)
            selected_record = cached_list[d_idx]
            playlist = load_cached_dataset(selected_record["name"])
            print(f"\n✅ Loaded '{playlist.name}' ({len(playlist)} songs, {playlist.data_type} data).")

        elif "Import a new dataset" in chosen_mode:
            while True:
                file_input = safe_input("\nEnter path to spreadsheet (.xlsx or .csv) containing songs (8-128 rows): ")
                path = Path(file_input)
                if not path.exists():
                    print(f"  ⚠️  File '{file_input}' not found. Please try again.")
                    continue

                # User explicitly specifies what kind of dataset is being imported
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
        if has_features:
            print(f"Features in dataset: {', '.join(playlist.feature_names)}")

        if has_features:
            config_options = [
                f"Use existing feature data as is ({playlist.data_type.upper()} with: {', '.join(playlist.feature_names)})",
                "Use only song names (strip features down to title and artist only)",
                "Define NEW custom criteria for these songs (walk through entering new data)",
            ]
            if playlist.data_type == "relative":
                config_options.append("Convert existing relative feature scores to absolute scale (1-10)")
            elif playlist.data_type == "absolute":
                config_options.append("Convert existing absolute feature scores to relative geometric tiers")

            cfg_choice = prompt_choice("\nHow do you want to configure the feature data for this tournament?", config_options, default_index=0)
            chosen_cfg = config_options[cfg_choice]

            if "Use existing feature data as is" in chosen_cfg:
                print(f"✅ Keeping existing {playlist.data_type.upper()} feature data.")

            elif "Convert existing" in chosen_cfg:
                if playlist.data_type == "relative":
                    playlist = transform_relative_to_absolute(playlist)
                    print("🔄 Converted relative data to absolute rubric scale.")
                else:
                    playlist = transform_absolute_to_relative(playlist)
                    print("🔄 Converted absolute data to relative geometric tiers.")

            elif "Define NEW custom criteria" in chosen_cfg:
                type_choice = prompt_choice(
                    "\nWhat type of feature data would you like to create for these songs?",
                    [
                        "Absolute feature data (independent rubric scores)",
                        "Relative feature data (comparative tier ratings)",
                    ],
                    default_index=0,
                )
                custom_feats = select_criteria_interactive()
                if type_choice == 0:
                    playlist = create_absolute_data_interactive(playlist, features=custom_feats)
                else:
                    playlist = create_relative_data_interactive(playlist, features=custom_feats)

            elif "Use only song names" in chosen_cfg:
                playlist = playlist.to_names_only()
                print("✂️  Stripped feature scores; using song titles and artists only.")

        # If playlist is names_only (either imported as names_only, or stripped above):
        if playlist.data_type == "names_only":
            print(f"\nThis dataset currently contains {len(playlist)} songs with names and artists only (no feature scores).")
            # README 2b: "If a dataset has only song titles and artist labels, the user is asked if they want to
            # create additional feature data or use it as is. If they want to create additional data, they specify
            # the criteria they want, and whether they want to create absolute or relative feature data."
            names_options = [
                "Use song names as is (no feature data — runs Swiss tournament with manual calls)",
                "Create additional feature data (define criteria and enter feature scores)",
            ]
            names_choice = prompt_choice("How would you like to proceed with this song list?", names_options, default_index=0)
            if names_choice == 1:
                type_choice = prompt_choice(
                    "\nWhat type of feature data would you like to create?",
                    [
                        "Absolute feature data (independent rubric scores per song)",
                        "Relative feature data (comparative tier ratings promoted in batches)",
                    ],
                    default_index=0,
                )
                custom_feats = select_criteria_interactive()
                if type_choice == 0:
                    print("\n📝 Step-by-step guidance: Creating absolute feature data...")
                    playlist = create_absolute_data_interactive(playlist, features=custom_feats)
                else:
                    print("\n📝 Step-by-step guidance: Creating relative feature data...")
                    playlist = create_relative_data_interactive(playlist, features=custom_feats)

        # Step 3: Prompt to cache new/modified dataset, explicitly specifying kind (README 2b)
        if prompt_yes_no(f"\nWould you like to cache '{playlist.name}' to the data folder for future tournaments?", default_yes=False):
            cache_name = safe_input(f"Enter dataset name [Default: {playlist.name}]: ") or playlist.name

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


def create_absolute_data_interactive(
    playlist: Playlist,
    features: Optional[List[str]] = None,
) -> Playlist:
    """Guides user through scoring absolute feature data step-by-step."""
    if features is None:
        features = select_criteria_interactive()

    rubrics: Dict[str, Dict[str, Any]] = {}
    print("\n" + "=" * 70)
    print("📋 ABSOLUTE DATA RUBRIC CONFIGURATION")
    print("=" * 70)
    rubric_mode = prompt_choice(
        "Choose rubric scale:",
        [
            "Standard 1-10 rating scale (Default)",
            "Define custom minimum and maximum rubric scale for each feature",
        ],
        default_index=0,
    )

    for f in features:
        if rubric_mode == 1:
            print(f"\nDefining rubric for '{f}':")
            f_min = float(prompt_int(f"  Minimum score for '{f}'", min_val=0, max_val=100, default_val=1))
            f_max = float(prompt_int(f"  Maximum score for '{f}'", min_val=int(f_min) + 1, max_val=1000, default_val=10))
            f_desc = safe_input(f"  Brief description / rubric guidance for '{f}' [optional]: ")
            rubrics[f] = {"min": f_min, "max": f_max, "description": f_desc or f"Rating for {f} on scale {f_min}-{f_max}"}
        else:
            rubrics[f] = {"min": 1.0, "max": 10.0, "description": f"Rating for {f} on standard scale 1-10"}

    batch_size = prompt_int(
        f"\nHow many songs would you like to score at a time? (1-{len(playlist)})",
        min_val=1,
        max_val=len(playlist),
        default_val=1,
    )

    real_songs = list(playlist.real_songs)
    new_songs: List[Song] = []
    raw_scores: Dict[str, List[float]] = {f: [] for f in features}

    print("\n" + "=" * 70)
    print(f"📝 SCORING SONGS ({len(real_songs)} songs total)")
    print("   Enter scores for each feature, or 'Q' to quit, 'R' to restart setup:")
    print("=" * 70)

    for i in range(0, len(real_songs), batch_size):
        batch = real_songs[i: i + batch_size]
        for s_idx, s in enumerate(batch, i + 1):
            print(f"\n[{s_idx}/{len(real_songs)}] Song: \"{s.title}\" by {s.artist}")
            song_feats: Dict[str, float] = {}
            for f in features:
                r_min = int(rubrics[f]["min"])
                r_max = int(rubrics[f]["max"])
                def_val = (r_min + r_max) // 2
                val = prompt_int(f"  Score for '{f}' ({r_min}-{r_max})", min_val=r_min, max_val=r_max, default_val=def_val)
                song_feats[f] = float(val)
                raw_scores[f].append(float(val))
            s_copy = s.copy()
            s_copy.features = song_feats
            new_songs.append(s_copy)

    # Silently standardize / normalize each feature to [1.0, 10.0] as specified in README 2b
    for f in features:
        f_min = min(raw_scores[f]) if raw_scores[f] else 1.0
        f_max = max(raw_scores[f]) if raw_scores[f] else 10.0
        for s in new_songs:
            val = s.features[f]
            if f_max > f_min:
                norm_val = 1.0 + ((val - f_min) / (f_max - f_min)) * 9.0
            else:
                norm_val = 5.0
            s.features[f] = round(float(norm_val), 2)

    abs_playlist = Playlist(
        songs=new_songs,
        data_type="absolute",
        feature_names=features,
        name=f"{playlist.name}_absolute",
    )
    print(f"\n✅ Created absolute feature data for {len(abs_playlist)} songs with features: {', '.join(features)}.")
    return abs_playlist


def create_relative_data_interactive(
    playlist: Playlist,
    features: Optional[List[str]] = None,
) -> Playlist:
    """Guides user through scoring relative feature data iteratively (README 2b, 3c)."""
    if features is None:
        features = select_criteria_interactive()

    batch_size = prompt_int(
        "\nChoose batch size for relative upgrade comparisons (2-8 songs per batch)",
        min_val=2,
        max_val=8,
        default_val=min(6, len(playlist)),
    )

    print("\n" + "=" * 70)
    print("🏆 RELATIVE FEATURE GRADING (GEOMETRIC TIERS)")
    print("   All songs start at Tier 1. At most half of the songs in each batch")
    print("   will be promoted to the next tier until a sole top song remains.")
    print("=" * 70)

    def interactive_upgrade_judge(batch: List[Song], feat_name: str, current_tier: int) -> List[Song]:
        max_upgrades = max(1, len(batch) // 2)
        print("\n" + "-" * 70)
        print(f"🎵 Feature: '{feat_name}' | Upgrading Tier {current_tier} -> Tier {current_tier + 1}")
        print(f"Select AT MOST {max_upgrades} of the following songs to promote:")
        for idx, s in enumerate(batch, 1):
            print(f"  [{idx}] \"{s.title}\" by {s.artist}")
        print("-" * 70)
        raw = safe_input(f"Enter song numbers to promote separated by spaces (e.g. '1 2', max {max_upgrades}): ")
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
    rel_playlist.name = f"{playlist.name}_relative"
    print(f"\n✅ Created relative feature rankings for {len(rel_playlist)} songs with features: {', '.join(features)}.")
    return rel_playlist


def select_criteria_interactive(current_features: Optional[List[str]] = None) -> List[str]:
    """Allows user to select criteria from suggestions, existing ones, or enter custom ones."""
    crit_options: List[str] = []
    if current_features:
        crit_options.append(f"Keep existing criteria ({', '.join(current_features)})")
    crit_options.append("Use suggested standard criteria (Melody, Catchiness, Lyricism, Message)")
    crit_options.append("Enter custom criteria names")

    c_idx = prompt_choice("\nSelect feature criteria to evaluate:", crit_options, default_index=0)
    chosen_label = crit_options[c_idx]

    if "Keep existing" in chosen_label and current_features:
        return list(current_features)
    elif "Use suggested" in chosen_label:
        return ["Melody", "Catchiness", "Lyricism", "Message"]
    else:
        while True:
            raw = safe_input("Enter custom feature names separated by commas (e.g., Melody, Rhythm, Vocals): ")
            feats = [f.strip() for f in raw.split(",") if f.strip()]
            if feats:
                print(f"✅ Selected custom criteria: {', '.join(feats)}")
                return feats
            print("  ⚠️  Please enter at least one feature name.")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for Favorite Song Tournament."""
    parser = argparse.ArgumentParser(description="Favorite Song Tournament - Console Application")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration without interactive prompts")
    parser.add_argument("--cached", type=str, help="Name of cached dataset to load")
    parser.add_argument("--file", type=str, default="favorite_song_data.xlsx", help="Dataset filepath")
    parser.add_argument("--size", type=int, default=16, help="Playlist size (8-128)")
    parser.add_argument("--format", type=str, choices=["swiss", "sorting", "hybrid"], default="swiss", help="Tournament format")
    parser.add_argument("--match-size", type=int, default=2, help="Match size (2-8)")
    parser.add_argument("--view", type=str, choices=["all", "notable", "winner"], default="all", help="Output view")
    parser.add_argument("--export", type=str, choices=["txt", "md", "pdf", "all"], help="Export report to reports/ folder")
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

        if args.export:
            exp_out = exp.export(file_format=args.export, view_mode=args.view, notable_count=5)
            print(f"\n💾 Report exported in demo mode: {exp_out}")
        return

    # Interactive flow in restart loop (README Phase 2: 'R' to restart)
    while True:
        try:
            # Step 1: Configure dataset (cached, new, or manage cache)
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

            # Step 8: Export report to reports/ subfolder (README Phase 2)
            export_options = [
                "Markdown report (.md)",
                "Plain text report (.txt)",
                "PDF Document (.pdf)",
                "Export all formats (.md, .txt, .pdf)",
                "Do not export to file",
            ]
            exp_choice = prompt_choice("\nWould you like to export your tournament results to the 'reports' folder?", export_options, default_index=0)
            if exp_choice < 4:
                fmt_map = ["md", "txt", "pdf", "all"]
                chosen_fmt = fmt_map[exp_choice]
                exported = exp.export(file_format=chosen_fmt, view_mode=view_mode, notable_count=5)
                if isinstance(exported, dict):
                    print("\n💾 Tournament reports successfully exported to 'reports/':")
                    for k, p in exported.items():
                        print(f"   • {k.upper()}: {p}")
                else:
                    print(f"\n💾 Tournament report successfully exported to '{exported}'!")

            break  # Finished full tournament run successfully!

        except UserRestartTournament:
            print("\n" + "=" * 80)
            print("🔄 RESTARTING TOURNAMENT SETUP...")
            print("=" * 80 + "\n")
            continue

        except UserQuitTournament:
            print("\n\n👋 Tournament run cancelled by user. See you next time!\n")
            break


if __name__ == "__main__":
    main()
