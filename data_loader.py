"""Data loader and dataset caching module for Favorite Song Tournament.

Features:
- Loading and caching datasets in the `data/` directory.
- Metadata tracking (data type: names_only, absolute, relative, along with descriptive notes).
- Interactive and programmatic feature creation (absolute rubrics and iterative relative tiers).
- Dataset transformations (relative <-> absolute, stripping to names only).
- Playlist subsetting (8 to dataset size) and internal shuffling.
- Strict Pareto dominance filtering and dummy padding.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
METADATA_FILE = DATA_DIR / "metadata.json"

SUGGESTED_FEATURES: List[str] = [
    "Melody",
    "Catchiness",
    "Lyricism",
    "Message",
    "Production",
    "Vocal Performance",
    "Rhythm/Groove",
    "Emotional Impact",
]


@dataclass
class Song:
    """Represents a song within the tournament."""

    id: int
    title: str
    artist: str
    features: Dict[str, float] = field(default_factory=dict)
    is_dummy: bool = False

    def __post_init__(self) -> None:
        self.features = {k: float(v) for k, v in self.features.items()}

    @property
    def display_name(self) -> str:
        """Formatted display name."""
        if self.is_dummy:
            return f"[Dummy] {self.title}"
        return f'"{self.title}" by {self.artist}'

    def __str__(self) -> str:
        return self.display_name

    def copy(self) -> Song:
        """Create a deep copy of the song."""
        return Song(
            id=self.id,
            title=self.title,
            artist=self.artist,
            features=dict(self.features),
            is_dummy=self.is_dummy,
        )


class Playlist:
    """Represents a collection of songs and their feature metadata."""

    def __init__(
        self,
        songs: List[Song],
        data_type: str = "relative",
        feature_names: Optional[List[str]] = None,
        name: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        self.songs: List[Song] = [s.copy() for s in songs]
        self.data_type: str = data_type.lower()
        if self.data_type not in ("names_only", "relative", "absolute"):
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'names_only', 'relative', or 'absolute'.")

        self.name = name or "Untitled Playlist"
        self.note = note or ""

        if self.data_type == "names_only":
            self.feature_names: List[str] = []
            for s in self.songs:
                s.features = {}
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        elif self.songs and self.songs[0].features:
            self.feature_names = list(self.songs[0].features.keys())
        else:
            self.feature_names = []

    def __len__(self) -> int:
        return len(self.real_songs)

    @property
    def total_count(self) -> int:
        """Total number of songs, including dummies."""
        return len(self.songs)

    @property
    def real_songs(self) -> List[Song]:
        """List of non-dummy songs."""
        return [s for s in self.songs if not s.is_dummy]

    @property
    def dummy_songs(self) -> List[Song]:
        """List of dummy songs."""
        return [s for s in self.songs if s.is_dummy]

    def __getitem__(self, index: int) -> Song:
        return self.real_songs[index]

    def __iter__(self):
        return iter(self.real_songs)

    def validate(self, min_size: int = 8, max_size: int = 128) -> None:
        """Validates that playlist size is within allowed range (8-128) and data is clean."""
        real_count = len(self.real_songs)
        if real_count < min_size or real_count > max_size:
            raise ValueError(
                f"Playlist size ({real_count}) must be between {min_size} and {max_size} songs."
            )

        for s in self.real_songs:
            if not s.title or not s.artist:
                raise ValueError(f"Song ID {s.id} is missing a title or artist.")
            if self.data_type != "names_only":
                for f in self.feature_names:
                    if f not in s.features or np.isnan(s.features[f]):
                        raise ValueError(f"Song '{s.title}' is missing a valid score for feature '{f}'.")

    def to_names_only(self) -> Playlist:
        """Returns a copy of the playlist using only song names (title and artist)."""
        stripped_songs = [
            Song(id=s.id, title=s.title, artist=s.artist, features={}, is_dummy=s.is_dummy)
            for s in self.songs
        ]
        return Playlist(
            songs=stripped_songs,
            data_type="names_only",
            feature_names=[],
            name=f"{self.name} (Names Only)",
            note=self.note,
        )

    def get_random_subset(self, size: int, random_seed: Optional[int] = None) -> Playlist:
        """Returns a random subset of the playlist without modifying the original."""
        real = list(self.real_songs)
        if size < 8 or size > len(real):
            raise ValueError(f"Subset size ({size}) must be between 8 and dataset size ({len(real)}).")

        rng = random.Random(random_seed)
        sampled = rng.sample(real, size)

        # Re-index ids 1..size for cleanliness
        new_songs = [
            Song(id=idx + 1, title=s.title, artist=s.artist, features=dict(s.features))
            for idx, s in enumerate(sampled)
        ]
        return Playlist(
            songs=new_songs,
            data_type=self.data_type,
            feature_names=self.feature_names,
            name=f"{self.name} (Subset {size})",
            note=self.note,
        )

    def shuffled(self, random_seed: Optional[int] = None) -> Playlist:
        """Returns a copy of the playlist with songs shuffled internally."""
        real = [s.copy() for s in self.real_songs]
        rng = random.Random(random_seed)
        rng.shuffle(real)
        return Playlist(
            songs=real,
            data_type=self.data_type,
            feature_names=self.feature_names,
            name=self.name,
            note=self.note,
        )

    def pad_with_dummies(self, match_size: int) -> Tuple[Playlist, int]:
        """Pads the playlist with dummy songs to make total count a multiple of match_size."""
        if match_size < 2:
            raise ValueError(f"Match size ({match_size}) must be at least 2.")

        real_songs = [s.copy() for s in self.real_songs]
        n = len(real_songs)
        remainder = n % match_size
        dummies_needed = (match_size - remainder) % match_size

        padded_songs = list(real_songs)
        dummy_features = {f: 0.0 for f in self.feature_names}
        for i in range(dummies_needed):
            dummy = Song(
                id=-(i + 1),
                title=f"Dummy_{i + 1}",
                artist="Dummy",
                features=dict(dummy_features),
                is_dummy=True,
            )
            padded_songs.append(dummy)

        padded_playlist = Playlist(
            songs=padded_songs,
            data_type=self.data_type,
            feature_names=self.feature_names,
            name=self.name,
            note=self.note,
        )
        return padded_playlist, dummies_needed

    def filter_strictly_dominated(self) -> Tuple[Playlist, List[Song]]:
        """Identifies and eliminates songs that are strictly dominated by another song."""
        if not self.feature_names or self.data_type == "names_only":
            return self.copy(), []

        survivors: List[Song] = []
        dominated: List[Song] = []

        for candidate in self.real_songs:
            is_dominated = False
            for opponent in self.real_songs:
                if candidate.id == opponent.id:
                    continue
                if all(
                    opponent.features[f] > candidate.features[f]
                    for f in self.feature_names
                ):
                    is_dominated = True
                    break

            if is_dominated:
                dominated.append(candidate.copy())
            else:
                survivors.append(candidate.copy())

        survivor_playlist = Playlist(
            songs=survivors,
            data_type=self.data_type,
            feature_names=self.feature_names,
            name=f"{self.name} (Survivors)",
            note=self.note,
        )
        return survivor_playlist, dominated

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the real songs into a pandas DataFrame."""
        rows = []
        for s in self.real_songs:
            row: Dict[str, Any] = {"Song": s.title, "Artist": s.artist}
            if self.data_type != "names_only":
                row.update(s.features)
            rows.append(row)
        return pd.DataFrame(rows)

    def copy(self) -> Playlist:
        """Create a deep copy of the playlist."""
        return Playlist(
            songs=self.songs,
            data_type=self.data_type,
            feature_names=self.feature_names,
            name=self.name,
            note=self.note,
        )


# ==============================================================================
# DATASET CACHING FUNCTIONS
# ==============================================================================

def ensure_cache_dir(data_dir: Union[str, Path] = DATA_DIR) -> Path:
    """Ensures the cache directory exists and initializes default metadata."""
    dir_path = Path(data_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    meta_file = dir_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    root_data = Path("favorite_song_data.xlsx")
    target_data = dir_path / "favorite_song_data.xlsx"
    if root_data.exists() and not target_data.exists():
        import shutil
        shutil.copy2(root_data, target_data)

    if target_data.exists() and "favorite_song_data" not in metadata:
        metadata["favorite_song_data"] = {
            "name": "favorite_song_data",
            "filename": "favorite_song_data.xlsx",
            "filepath": str(target_data),
            "data_type": "relative",
            "feature_names": ["Melody", "Catchiness", "Lyricism", "Message"],
            "song_count": 128,
            "note": "Original benchmark dataset with relative feature scores (1 to 8).",
            "created_at": datetime.now().isoformat(),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    elif not meta_file.exists():
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return dir_path


def list_cached_datasets(data_dir: Union[str, Path] = DATA_DIR) -> List[Dict[str, Any]]:
    """Returns a list of all cached datasets in the data folder with their metadata."""
    dir_path = ensure_cache_dir(data_dir)
    meta_file = dir_path / "metadata.json"

    metadata: Dict[str, Any] = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # Discover any spreadsheet files in data_dir that may not be in metadata
    cached_list: List[Dict[str, Any]] = []
    seen_files = set()

    # Add existing metadata records
    for key, info in metadata.items():
        fpath = Path(info.get("filepath", dir_path / info.get("filename", "")))
        if fpath.exists():
            cached_list.append(info)
            seen_files.add(fpath.resolve())

    # Add any untracked .xlsx or .csv files in data_dir
    for f in dir_path.iterdir():
        if f.suffix.lower() in (".xlsx", ".xls", ".csv") and f.resolve() not in seen_files:
            try:
                p = load_dataset(f, validate_size=False)
                entry = {
                    "name": f.stem,
                    "filename": f.name,
                    "filepath": str(f),
                    "data_type": p.data_type,
                    "feature_names": p.feature_names,
                    "song_count": len(p),
                    "note": "Auto-discovered dataset in data folder.",
                    "created_at": datetime.now().isoformat(),
                }
                cached_list.append(entry)
            except Exception:
                continue

    return cached_list


def save_dataset_to_cache(
    playlist: Playlist,
    name: str,
    note: str = "",
    data_type: Optional[str] = None,
    file_format: str = "xlsx",
    data_dir: Union[str, Path] = DATA_DIR,
) -> Path:
    """Saves a playlist to the cache data directory with descriptive metadata.

    Args:
        playlist: The Playlist to save.
        name: Name for the dataset.
        note: Descriptive note about the dataset.
        data_type: Optional explicit data kind ('names_only', 'absolute', 'relative').
        file_format: 'xlsx' or 'csv'.
        data_dir: Path to data directory.

    Returns:
        Path to saved file.
    """
    dir_path = ensure_cache_dir(data_dir)

    target_playlist = playlist.copy()
    if data_type is not None:
        dt = data_type.lower()
        if dt == "names_only":
            target_playlist = target_playlist.to_names_only()
        elif dt == "absolute" and target_playlist.data_type == "relative":
            target_playlist = transform_relative_to_absolute(target_playlist)
        elif dt == "relative" and target_playlist.data_type == "absolute":
            target_playlist = transform_absolute_to_relative(target_playlist)
        elif dt in ("names_only", "absolute", "relative"):
            target_playlist.data_type = dt

    # Sanitize name
    clean_name = "".join(c for c in name if c.isalnum() or c in ("_", "-", " ")).strip()
    clean_name = clean_name.replace(" ", "_")
    if not clean_name:
        clean_name = f"dataset_{int(datetime.now().timestamp())}"

    filename = f"{clean_name}.{file_format.lstrip('.')}"
    file_path = dir_path / filename

    df = target_playlist.to_dataframe()
    if file_format.lower() in ("xlsx", "xls"):
        df.to_excel(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

    # Update metadata.json
    meta_file = dir_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    metadata[clean_name] = {
        "name": clean_name,
        "filename": filename,
        "filepath": str(file_path),
        "data_type": target_playlist.data_type,
        "feature_names": target_playlist.feature_names,
        "song_count": len(target_playlist),
        "note": note or f"Cached playlist with {len(target_playlist)} songs ({target_playlist.data_type}).",
        "created_at": datetime.now().isoformat(),
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return file_path


def load_cached_dataset(
    dataset_name: str,
    data_dir: Union[str, Path] = DATA_DIR,
) -> Playlist:
    """Loads a cached dataset by name or filename from the data directory."""
    dir_path = ensure_cache_dir(data_dir)
    meta_file = dir_path / "metadata.json"

    meta_info: Optional[Dict[str, Any]] = None
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            meta_info = metadata.get(dataset_name)
        except Exception:
            pass

    target_path = None
    if meta_info and "filepath" in meta_info and Path(meta_info["filepath"]).exists():
        target_path = Path(meta_info["filepath"])
    elif (dir_path / f"{dataset_name}.xlsx").exists():
        target_path = dir_path / f"{dataset_name}.xlsx"
    elif (dir_path / f"{dataset_name}.csv").exists():
        target_path = dir_path / f"{dataset_name}.csv"
    elif (dir_path / dataset_name).exists():
        target_path = dir_path / dataset_name
    elif Path(dataset_name).exists():
        target_path = Path(dataset_name)
    elif Path(f"{dataset_name}.xlsx").exists():
        target_path = Path(f"{dataset_name}.xlsx")
    else:
        raise FileNotFoundError(f"Cached dataset '{dataset_name}' not found in {dir_path}.")

    data_type = meta_info.get("data_type", "relative") if meta_info else "relative"
    feat_cols = meta_info.get("feature_names") if meta_info else None
    note = meta_info.get("note", "") if meta_info else ""

    playlist = load_dataset(
        target_path,
        data_type=data_type,
        feature_columns=feat_cols,
        validate_size=True,
    )
    playlist.name = dataset_name
    playlist.note = note
    return playlist


# ==============================================================================
# DATASET LOADING & CREATION
# ==============================================================================

def load_dataset(
    filepath: Union[str, Path],
    data_type: str = "relative",
    feature_columns: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
    validate_size: bool = True,
) -> Playlist:
    """Loads a playlist from an Excel (.xlsx) or CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use .xlsx or .csv.")

    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip()
        if c_clean.lower() in ("song", "title", "track", "track_name"):
            col_map[col] = "Song"
        elif c_clean.lower() in ("artist", "artist_name", "performer"):
            col_map[col] = "Artist"
    df = df.rename(columns=col_map)

    if "Song" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Song"})
    if "Artist" not in df.columns:
        if len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: "Artist"})
        else:
            df["Artist"] = "Unknown"

    numeric_cols = [
        c for c in df.columns
        if c not in ("Song", "Artist") and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Detect if dataset is names only
    effective_data_type = data_type.lower()
    if not numeric_cols:
        effective_data_type = "names_only"
        features_to_use = []
    elif effective_data_type == "names_only":
        features_to_use = []
    elif feature_columns is not None:
        missing = [f for f in feature_columns if f not in df.columns]
        if missing:
            raise ValueError(f"Specified feature columns not found in file: {missing}")
        features_to_use = list(feature_columns)
    else:
        features_to_use = numeric_cols

    if max_songs is not None and max_songs > 0:
        df = df.iloc[:max_songs].copy()

    df = df.dropna(subset=["Song"])

    for f in features_to_use:
        if df[f].isnull().any():
            raise ValueError(f"Feature column '{f}' contains missing (NaN) values.")

    songs: List[Song] = []
    for idx, row in df.iterrows():
        song_features = {f: float(row[f]) for f in features_to_use} if features_to_use else {}
        songs.append(
            Song(
                id=int(idx) + 1,
                title=str(row["Song"]).strip(),
                artist=str(row["Artist"]).strip(),
                features=song_features,
            )
        )

    playlist = Playlist(
        songs=songs,
        data_type=effective_data_type,
        feature_names=features_to_use,
        name=path.stem,
    )
    if validate_size:
        playlist.validate(min_size=8, max_size=128)
    return playlist


def create_absolute_feature_data(
    playlist: Playlist,
    feature_names: List[str],
    rubrics: Dict[str, Dict[str, Any]],
    scoring_callback: Callable[[Song, str, Dict[str, Any]], float],
) -> Playlist:
    """Interactively or programmatically constructs absolute feature data for a playlist.

    For each feature, the user/caller assigns a score to each song using the rubric.
    The scores are then silently normalized to a standardized scale [1.0, 10.0].

    Args:
        playlist: Input playlist (e.g. names only).
        feature_names: List of features to score.
        rubrics: Dict of feature_name -> {"min": float, "max": float, "description": str}.
        scoring_callback: Callable(song, feature_name, rubric) -> raw_score.

    Returns:
        New Playlist with standardized absolute feature scores.
    """
    new_songs: List[Song] = [s.copy() for s in playlist.real_songs]
    raw_scores: Dict[str, List[float]] = {f: [] for f in feature_names}

    for s in new_songs:
        for f in feature_names:
            rubric = rubrics.get(f, {"min": 1.0, "max": 10.0, "description": "Standard scale 1-10"})
            raw_val = scoring_callback(s, f, rubric)
            s.features[f] = float(raw_val)
            raw_scores[f].append(float(raw_val))

    # Silently standardize / normalize each feature to [1.0, 10.0]
    for f in feature_names:
        f_min = min(raw_scores[f])
        f_max = max(raw_scores[f])
        for s in new_songs:
            val = s.features[f]
            if f_max > f_min:
                norm_val = 1.0 + ((val - f_min) / (f_max - f_min)) * 9.0
            else:
                norm_val = 5.0
            s.features[f] = round(float(norm_val), 2)

    return Playlist(
        songs=new_songs,
        data_type="absolute",
        feature_names=feature_names,
        name=f"{playlist.name} (Absolute)",
    )


def create_relative_feature_data(
    playlist: Playlist,
    feature_names: List[str],
    batch_size: int = 6,
    judge_batch_callback: Optional[Callable[[List[Song], str, int], List[Song]]] = None,
) -> Playlist:
    """Constructs relative feature data iteratively using batch upgrade decisions (README 2b, 3c).

    Algorithm:
    1. For each feature, all songs start with score 1.
    2. At most half of the songs are upgraded to 2.
    3. At most half of those are upgraded to 3, and so on until one song has the unique high score.
    """
    real_songs = [s.copy() for s in playlist.real_songs]
    n = len(real_songs)
    if n == 0:
        return playlist.copy()

    feature_scores: Dict[int, Dict[str, float]] = {s.id: {} for s in real_songs}

    for f in feature_names:
        scores = interactive_relative_grading(
            songs=real_songs,
            feature_name=f,
            batch_size=batch_size,
            judge_batch_callback=judge_batch_callback,
        )
        for s in real_songs:
            feature_scores[s.id][f] = float(scores.get(s.id, 1))

    new_songs: List[Song] = []
    for s in real_songs:
        new_songs.append(
            Song(
                id=s.id,
                title=s.title,
                artist=s.artist,
                features=feature_scores[s.id],
            )
        )

    return Playlist(
        songs=new_songs,
        data_type="relative",
        feature_names=feature_names,
        name=f"{playlist.name} (Relative)",
    )


def transform_relative_to_absolute(
    playlist: Playlist,
    scale_min: float = 1.0,
    scale_max: float = 10.0,
) -> Playlist:
    """Transforms relative tier scores (1..K) into an absolute scale [scale_min, scale_max]."""
    if playlist.data_type == "names_only" or not playlist.feature_names:
        return playlist.copy()

    new_songs: List[Song] = []
    feature_names = playlist.feature_names

    max_tiers: Dict[str, float] = {}
    for f in feature_names:
        scores = [s.features[f] for s in playlist.real_songs if f in s.features]
        max_tiers[f] = max(scores) if scores else 1.0

    for s in playlist.real_songs:
        abs_features: Dict[str, float] = {}
        for f in feature_names:
            rel_val = s.features.get(f, 1.0)
            max_t = max_tiers[f]
            if max_t > 1.0:
                norm = (rel_val - 1.0) / (max_t - 1.0)
                abs_val = scale_min + norm * (scale_max - scale_min)
            else:
                abs_val = scale_max
            abs_features[f] = round(float(abs_val), 2)

        new_songs.append(
            Song(
                id=s.id,
                title=s.title,
                artist=s.artist,
                features=abs_features,
                is_dummy=s.is_dummy,
            )
        )

    return Playlist(
        songs=new_songs,
        data_type="absolute",
        feature_names=feature_names,
        name=f"{playlist.name} (Absolute)",
    )


def transform_absolute_to_relative(playlist: Playlist) -> Playlist:
    """Transforms absolute scores into geometric relative tiers as specified in README 3c."""
    real_songs = playlist.real_songs
    n = len(real_songs)
    if n == 0 or playlist.data_type == "names_only" or not playlist.feature_names:
        return playlist.copy()

    feature_names = playlist.feature_names
    new_features_by_song_id: Dict[int, Dict[str, float]] = {s.id: {} for s in real_songs}

    for f in feature_names:
        sorted_songs = sorted(
            real_songs,
            key=lambda s: (s.features.get(f, 0.0), -s.id),
            reverse=True,
        )

        scores: Dict[int, int] = {s.id: 1 for s in sorted_songs}
        current_eligible = list(sorted_songs)
        current_tier = 1

        while len(current_eligible) > 1:
            next_tier = current_tier + 1
            num_to_upgrade = max(1, len(current_eligible) // 2)

            upgraded = current_eligible[:num_to_upgrade]
            for s in upgraded:
                scores[s.id] = next_tier

            current_eligible = upgraded
            current_tier = next_tier

        for s in real_songs:
            new_features_by_song_id[s.id][f] = float(scores[s.id])

    new_songs: List[Song] = []
    for s in real_songs:
        new_songs.append(
            Song(
                id=s.id,
                title=s.title,
                artist=s.artist,
                features=new_features_by_song_id[s.id],
                is_dummy=s.is_dummy,
            )
        )

    return Playlist(
        songs=new_songs,
        data_type="relative",
        feature_names=feature_names,
        name=f"{playlist.name} (Relative)",
    )


def interactive_relative_grading(
    songs: List[Song],
    feature_name: str,
    batch_size: int = 6,
    judge_batch_callback: Optional[Callable[[List[Song], str, int], List[Song]]] = None,
) -> Dict[int, int]:
    """Generates relative feature scores iteratively via batch upgrade decisions."""
    n = len(songs)
    if n == 0:
        return {}
    if n == 1:
        return {songs[0].id: 1}

    scores: Dict[int, int] = {s.id: 1 for s in songs}
    current_tier = 1

    while True:
        eligible = [s for s in songs if scores[s.id] == current_tier]
        if len(eligible) <= 1:
            break

        next_tier = current_tier + 1
        batches = [
            eligible[i: i + batch_size]
            for i in range(0, len(eligible), batch_size)
        ]

        upgraded_this_round: List[Song] = []
        for batch in batches:
            max_upgrades = max(1, len(batch) // 2)
            if judge_batch_callback is not None:
                selected = judge_batch_callback(batch, feature_name, current_tier)
                selected = selected[:max_upgrades]
            else:
                selected = batch[:max_upgrades]

            for s in selected:
                scores[s.id] = next_tier
                upgraded_this_round.append(s)

        if not upgraded_this_round:
            scores[eligible[0].id] = next_tier
        elif len(upgraded_this_round) == len(eligible):
            to_revert = upgraded_this_round[len(upgraded_this_round) // 2:]
            for s in to_revert:
                scores[s.id] = current_tier

        current_tier = next_tier

    return scores


def generate_synthetic_playlist(
    size: int = 16,
    feature_names: Optional[List[str]] = None,
    data_type: str = "relative",
    random_seed: Optional[int] = 42,
) -> Playlist:
    """Generates a synthetic playlist with reproducible data for testing."""
    if size < 8 or size > 128:
        raise ValueError("Synthetic playlist size must be between 8 and 128.")

    rng = np.random.default_rng(random_seed)
    features = feature_names or ["Melody", "Catchiness", "Lyricism", "Message"]

    titles = [
        "Blinding Lights", "Levitating", "As It Was", "Cruel Summer",
        "Stay", "Watermelon Sugar", "Bad Guy", "Circles",
        "Save Your Tears", "Peaches", "Good 4 U", "Kiss Me More",
        "Heat Waves", "Industry Baby", "Easy On Me", "Montero",
        "Drivers License", "Deja Vu", "Solar Power", "Shivers",
        "Ghost", "Cold Heart", "Happier Than Ever", "Butter",
        "Dynamite", "Positions", "Vampire", "Anti-Hero",
        "Flowers", "Kill Bill", "Karma", "Calm Down",
    ]

    artists = [
        "The Weeknd", "Dua Lipa", "Harry Styles", "Taylor Swift",
        "The Kid LAROI", "Harry Styles", "Billie Eilish", "Post Malone",
        "The Weeknd", "Justin Bieber", "Olivia Rodrigo", "Doja Cat",
        "Glass Animals", "Lil Nas X", "Adele", "Lil Nas X",
        "Olivia Rodrigo", "Olivia Rodrigo", "Lorde", "Ed Sheeran",
        "Justin Bieber", "Elton John", "Billie Eilish", "BTS",
        "BTS", "Ariana Grande", "Olivia Rodrigo", "Taylor Swift",
        "Miley Cyrus", "SZA", "Taylor Swift", "Rema",
    ]

    songs: List[Song] = []
    for i in range(size):
        title = titles[i % len(titles)] if i < len(titles) else f"Song_{i + 1}"
        artist = artists[i % len(artists)] if i < len(artists) else f"Artist_{i + 1}"
        feat_dict = {f: float(rng.uniform(1.0, 10.0)) for f in features}
        songs.append(Song(id=i + 1, title=title, artist=artist, features=feat_dict))

    base_playlist = Playlist(songs=songs, data_type="absolute", feature_names=features)

    if data_type == "names_only":
        return base_playlist.to_names_only()
    elif data_type == "relative":
        return transform_absolute_to_relative(base_playlist)
    return base_playlist
