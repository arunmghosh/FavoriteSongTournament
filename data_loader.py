"""Data loader module for Favorite Song Tournament.

Handles loading playlists from file (Excel/CSV), validating playlist constraints,
identifying strictly dominated songs, and transforming feature scores between
relative and absolute scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        # Ensure all feature values are floats
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
    ) -> None:
        self.songs: List[Song] = [s.copy() for s in songs]
        self.data_type: str = data_type.lower()
        if self.data_type not in ("relative", "absolute"):
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'relative' or 'absolute'.")

        if feature_names is not None:
            self.feature_names: List[str] = list(feature_names)
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

        # Check for missing song names or invalid features
        for s in self.real_songs:
            if not s.title or not s.artist:
                raise ValueError(f"Song ID {s.id} is missing a title or artist.")
            for f in self.feature_names:
                if f not in s.features or np.isnan(s.features[f]):
                    raise ValueError(f"Song '{s.title}' is missing a valid score for feature '{f}'.")

    def pad_with_dummies(self, match_size: int) -> Tuple[Playlist, int]:
        """Pads the playlist with dummy songs to make total count a multiple of match_size.

        Returns a new padded Playlist and the number of dummies added.
        """
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
        )
        return padded_playlist, dummies_needed

    def filter_strictly_dominated(self) -> Tuple[Playlist, List[Song]]:
        """Identifies and eliminates songs that are strictly dominated by another song.

        A song B is strictly dominated by song A if for ALL features f:
            A.features[f] > B.features[f]

        Returns:
            Tuple of (new Playlist with surviving songs, list of eliminated songs).
        """
        if not self.feature_names:
            return Playlist(self.real_songs, self.data_type, self.feature_names), []

        survivors: List[Song] = []
        dominated: List[Song] = []

        for candidate in self.real_songs:
            is_dominated = False
            for opponent in self.real_songs:
                if candidate.id == opponent.id:
                    continue
                # Strictly dominated means strictly lower for EVERY feature
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
        )
        return survivor_playlist, dominated

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the real songs into a pandas DataFrame."""
        rows = []
        for s in self.real_songs:
            row: Dict[str, Any] = {"Song": s.title, "Artist": s.artist}
            row.update(s.features)
            rows.append(row)
        return pd.DataFrame(rows)

    def copy(self) -> Playlist:
        """Create a deep copy of the playlist."""
        return Playlist(
            songs=self.songs,
            data_type=self.data_type,
            feature_names=self.feature_names,
        )


def load_dataset(
    filepath: Union[str, Path],
    data_type: str = "relative",
    feature_columns: Optional[List[str]] = None,
    max_songs: Optional[int] = None,
    validate_size: bool = True,
) -> Playlist:
    """Loads a playlist from an Excel (.xlsx) or CSV file.

    Args:
        filepath: Path to dataset file.
        data_type: 'relative' or 'absolute'.
        feature_columns: Specific feature column names to load (default: all numeric columns).
        max_songs: Optional limit on songs to load (e.g. to test with 16 songs).
        validate_size: Whether to enforce the 8-128 song range check.

    Returns:
        Playlist object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use .xlsx or .csv.")

    # Standardize column names
    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip()
        if c_clean.lower() in ("song", "title", "track", "track_name"):
            col_map[col] = "Song"
        elif c_clean.lower() in ("artist", "artist_name", "performer"):
            col_map[col] = "Artist"
    df = df.rename(columns=col_map)

    if "Song" not in df.columns:
        # Fall back to first column as song title
        df = df.rename(columns={df.columns[0]: "Song"})
    if "Artist" not in df.columns:
        # Fall back to second column as artist if available, else 'Unknown'
        if len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: "Artist"})
        else:
            df["Artist"] = "Unknown"

    # Identify feature columns
    if feature_columns is not None:
        missing = [f for f in feature_columns if f not in df.columns]
        if missing:
            raise ValueError(f"Specified feature columns not found in file: {missing}")
        features_to_use = list(feature_columns)
    else:
        features_to_use = [
            c for c in df.columns
            if c not in ("Song", "Artist") and pd.api.types.is_numeric_dtype(df[c])
        ]

    if max_songs is not None and max_songs > 0:
        df = df.iloc[:max_songs].copy()

    # Drop rows where Song is NaN
    df = df.dropna(subset=["Song"])

    # Ensure no missing values in feature columns
    for f in features_to_use:
        if df[f].isnull().any():
            raise ValueError(f"Feature column '{f}' contains missing (NaN) values.")

    songs: List[Song] = []
    for idx, row in df.iterrows():
        song_features = {f: float(row[f]) for f in features_to_use}
        songs.append(
            Song(
                id=int(idx) + 1,
                title=str(row["Song"]).strip(),
                artist=str(row["Artist"]).strip(),
                features=song_features,
            )
        )

    playlist = Playlist(songs=songs, data_type=data_type, feature_names=features_to_use)
    if validate_size:
        playlist.validate(min_size=8, max_size=128)
    return playlist


def transform_relative_to_absolute(
    playlist: Playlist,
    scale_min: float = 1.0,
    scale_max: float = 10.0,
) -> Playlist:
    """Transforms relative tier scores (1..K) into an absolute scale [scale_min, scale_max].

    Args:
        playlist: Playlist with relative feature data.
        scale_min: Minimum absolute scale value (default 1.0).
        scale_max: Maximum absolute scale value (default 10.0).

    Returns:
        New Playlist with absolute feature scores.
    """
    new_songs: List[Song] = []
    feature_names = playlist.feature_names

    # Determine max tier for each feature
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
                # Linear mapping from [1, max_t] -> [scale_min, scale_max]
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

    return Playlist(songs=new_songs, data_type="absolute", feature_names=feature_names)


def transform_absolute_to_relative(playlist: Playlist) -> Playlist:
    """Transforms absolute scores into geometric relative tiers as specified in README 3c:

    Algorithm:
    1. Start each song with a score of 1 for the feature.
    2. At most half of the songs are upgraded to 2.
    3. At most half of the songs with score 2 are upgraded to score 3, and so on until
       one song has the unique high score for that feature.

    For N songs:
    - Tier 1: bottom half
    - Tier 2: top half (except top quarter)
    - ...
    - Tier K: unique top 1 song

    Returns:
        New Playlist with relative feature data.
    """
    real_songs = playlist.real_songs
    n = len(real_songs)
    if n == 0:
        return playlist.copy()

    feature_names = playlist.feature_names
    new_features_by_song_id: Dict[int, Dict[str, float]] = {s.id: {} for s in real_songs}

    for f in feature_names:
        # Sort songs by absolute feature score descending
        # Use song id as tie-breaker for deterministic grading
        sorted_songs = sorted(
            real_songs,
            key=lambda s: (s.features.get(f, 0.0), -s.id),
            reverse=True,
        )

        # Base score 1 for all songs
        scores: Dict[int, int] = {s.id: 1 for s in sorted_songs}

        # Progressively upgrade top half
        current_eligible = list(sorted_songs)
        current_tier = 1

        while len(current_eligible) > 1:
            next_tier = current_tier + 1
            # "At most half of the songs are upgraded"
            # When eligible count is 2, half is 1, so exactly 1 song reaches top tier
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

    return Playlist(songs=new_songs, data_type="relative", feature_names=feature_names)


def interactive_relative_grading(
    songs: List[Song],
    feature_name: str,
    batch_size: int = 6,
    judge_batch_callback: Optional[Callable[[List[Song], str, int], List[Song]]] = None,
) -> Dict[int, int]:
    """Generates relative feature scores iteratively via batch upgrade decisions (README 3c).

    In each round, the songs currently at tier T are divided into batches.
    For each batch, the judge selects at most half to upgrade to tier T+1.
    Continues until exactly 1 song reaches the unique high score.

    Args:
        songs: List of songs to grade.
        feature_name: Name of the feature being judged.
        batch_size: Number of songs per comparison batch.
        judge_batch_callback: Function taking (batch, feature_name, current_tier) and
            returning the subset of songs chosen for upgrade (at most len(batch) // 2).

    Returns:
        Mapping of song.id -> relative tier score.
    """
    n = len(songs)
    if n == 0:
        return {}
    if n == 1:
        return {songs[0].id: 1}

    scores: Dict[int, int] = {s.id: 1 for s in songs}
    current_tier = 1

    while True:
        # Eligible songs are those that reached current_tier
        eligible = [s for s in songs if scores[s.id] == current_tier]
        if len(eligible) <= 1:
            break

        next_tier = current_tier + 1

        # Break eligible songs into batches
        batches = [
            eligible[i: i + batch_size]
            for i in range(0, len(eligible), batch_size)
        ]

        upgraded_this_round: List[Song] = []
        for batch in batches:
            max_upgrades = max(1, len(batch) // 2)
            if judge_batch_callback is not None:
                selected = judge_batch_callback(batch, feature_name, current_tier)
                # Enforce at most half
                selected = selected[:max_upgrades]
            else:
                # Default heuristic: top half based on current feature score or order
                selected = batch[:max_upgrades]

            for s in selected:
                scores[s.id] = next_tier
                upgraded_this_round.append(s)

        # Safety check: if no one upgraded or everyone upgraded, force progress
        if not upgraded_this_round:
            scores[eligible[0].id] = next_tier
        elif len(upgraded_this_round) == len(eligible):
            # Downgrade half to prevent infinite loop
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
    """Generates a synthetic playlist with reproducible feature data for testing."""
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
        # Generate random absolute base scores between 1.0 and 10.0
        feat_dict = {f: float(rng.uniform(1.0, 10.0)) for f in features}
        songs.append(Song(id=i + 1, title=title, artist=artist, features=feat_dict))

    base_playlist = Playlist(songs=songs, data_type="absolute", feature_names=features)

    if data_type == "relative":
        return transform_absolute_to_relative(base_playlist)
    return base_playlist

