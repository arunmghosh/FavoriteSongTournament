"""Tournament algorithms for Favorite Song Tournament.

Implements:
1. Swiss Tournament (pools by score, strength of schedule tie-breakers, dummy distribution <= 1 per match)
2. Sorting Tournament (multi-pivot pool sorting, separator ranking caching with insertion UX, DQN learning or manual completion)
3. Hybrid Tournament (relative feature data, strictly dominated song dropout, sub-tournament)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import Playlist, Song


@dataclass
class MatchRecord:
    """Records the details and outcome of a single match."""

    match_id: int
    round_num: int
    songs_in_match: List[Song]
    ranked_outcome: List[Song]  # Ordered 1st place to last place
    is_automated: bool = False
    details: str = ""


# Judge callback signatures
JudgeCallback = Callable[[List[Song], int, int], List[Song]]
# Separator insertion callback: (candidate, ordered_separators, match_id, round_num) -> insert_index (0..len(separators))
SeparatorInsertCallback = Callable[[Song, List[Song], int, int], int]


class QNetwork(nn.Module):
    """Deep Q-Network estimating the latent preference / utility score of a song from its features."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Agent that learns user preferences from manual match comparisons using a Deep Q-Network."""

    def __init__(
        self,
        feature_names: List[str],
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.feature_names = list(feature_names)
        self.dim = len(self.feature_names)
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.model = QNetwork(max(1, self.dim))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MarginRankingLoss(margin=0.5)

        # Experience replay buffer: stores (winner_feature_vec, loser_feature_vec)
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []

    def song_to_tensor(self, song: Song) -> torch.Tensor:
        """Converts song feature dict to torch tensor."""
        vec = [float(song.features.get(f, 0.0)) for f in self.feature_names]
        if not vec:
            vec = [0.0]
        return torch.tensor(vec, dtype=torch.float32)

    def record_match_ranking(self, ranked_songs: List[Song]) -> None:
        """Extracts pairwise preferences from a ranked match and stores in replay buffer."""
        if not self.feature_names:
            return
        real_songs = [s for s in ranked_songs if not s.is_dummy]
        n = len(real_songs)
        for i in range(n):
            for j in range(i + 1, n):
                winner_vec = np.array([real_songs[i].features.get(f, 0.0) for f in self.feature_names], dtype=np.float32)
                loser_vec = np.array([real_songs[j].features.get(f, 0.0) for f in self.feature_names], dtype=np.float32)
                self.replay_buffer.append((winner_vec, loser_vec))

    def train_on_buffer(self, epochs: int = 60, batch_size: int = 16) -> float:
        """Trains the DQN on all collected pairwise preferences."""
        if not self.replay_buffer or not self.feature_names:
            return 0.0

        self.model.train()
        total_loss = 0.0
        n_samples = len(self.replay_buffer)

        winners = torch.tensor(np.array([item[0] for item in self.replay_buffer]), dtype=torch.float32)
        losers = torch.tensor(np.array([item[1] for item in self.replay_buffer]), dtype=torch.float32)
        targets = torch.ones(n_samples, dtype=torch.float32)

        for _ in range(epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start: start + batch_size]
                b_win = winners[batch_idx]
                b_lose = losers[batch_idx]
                b_tgt = targets[batch_idx]

                self.optimizer.zero_grad()
                q_win = self.model(b_win).squeeze(-1)
                q_lose = self.model(b_lose).squeeze(-1)
                loss = self.criterion(q_win, q_lose, b_tgt)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / (epochs * max(1, n_samples // batch_size))

    def predict_score(self, song: Song) -> float:
        """Predicts latent Q-value for a song."""
        if song.is_dummy:
            return -1e9
        if not self.feature_names:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            t = self.song_to_tensor(song).unsqueeze(0)
            return float(self.model(t).item())

    def rank_songs(self, songs: List[Song]) -> List[Song]:
        """Ranks a list of songs descending by predicted Q-value."""
        real_songs = [s for s in songs if not s.is_dummy]
        dummies = [s for s in songs if s.is_dummy]

        scored = [(s, self.predict_score(s)) for s in real_songs]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked_real = [s for s, _ in scored]
        return ranked_real + dummies


class SwissTournament:
    """Executes a Swiss-system tournament.

    Key properties (README 3a, Section 4):
    - Songs start with score 0.
    - Each match has size M (2-8).
    - Ranked 1 to M: 1st place gets M-1 pts, 2nd gets M-2 pts, ..., last gets 0 pts.
    - If playlist size is not divisible by match size, pad with dummy songs.
    - Dummy distribution constraint: At most one dummy song in a match (Section 4).
    - Dummies automatically lose (0 points) and are never shown to the user.
    - Ends when 1 song has a unique winning score at the conclusion of a round.
    - Tie-breakers: final scores, then strength of schedule (Buchholz), then random.
    """

    def __init__(
        self,
        playlist: Playlist,
        match_size: int = 2,
        judge_callback: Optional[JudgeCallback] = None,
        max_rounds: Optional[int] = None,
        random_seed: Optional[int] = 42,
    ) -> None:
        if match_size < 2 or match_size > 8:
            raise ValueError(f"Match size {match_size} must be between 2 and 8.")

        self.playlist = playlist.copy()
        self.match_size = match_size
        self.judge_callback = judge_callback or self._default_heuristic_judge
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

        n_real = len(self.playlist.real_songs)
        default_max = max(10, 2 * int(np.ceil(np.log2(max(2, n_real)))))
        self.max_rounds = max_rounds if max_rounds is not None else default_max

        # Dummy count needed per round
        k_matches = int(np.ceil(n_real / self.match_size))
        self.dummies_count = (k_matches * self.match_size) - n_real

        # State tracking
        self.scores: Dict[int, float] = {s.id: 0.0 for s in self.playlist.real_songs}
        self.opponents: Dict[int, List[int]] = {s.id: [] for s in self.playlist.real_songs}
        self.match_history: List[MatchRecord] = []
        self.round_num = 0
        self.match_counter = 0

    @property
    def padded_playlist(self) -> Playlist:
        padded, _ = self.playlist.pad_with_dummies(self.match_size)
        return padded

    def _default_heuristic_judge(self, songs: List[Song], match_id: int, round_num: int) -> List[Song]:
        """Default fallback judge based on average feature score or random seed."""
        real = [s for s in songs if not s.is_dummy]
        dummies = [s for s in songs if s.is_dummy]

        def song_key(s: Song) -> float:
            if s.features:
                return float(np.mean(list(s.features.values())))
            return float(self.rng.random())

        sorted_real = sorted(real, key=song_key, reverse=True)
        return sorted_real + dummies

    def calculate_strength_of_schedule(self, song_id: int) -> float:
        """Buchholz system: sum of scores of all real opponents faced."""
        opp_ids = self.opponents.get(song_id, [])
        return sum(self.scores.get(opp_id, 0.0) for opp_id in opp_ids if opp_id > 0)

    def run_round(self) -> None:
        """Executes a single round of Swiss tournament matches with at most 1 dummy per match."""
        self.round_num += 1

        real_songs = list(self.playlist.real_songs)
        self.rng.shuffle(real_songs)

        # Sort real songs by current score descending, with SOS as secondary
        real_songs.sort(
            key=lambda s: (
                self.scores[s.id],
                self.calculate_strength_of_schedule(s.id),
            ),
            reverse=True,
        )

        n = len(real_songs)
        m = self.match_size
        k_matches = int(np.ceil(n / m))
        dummies_needed = (k_matches * m) - n

        # Distribute real songs across k_matches.
        # Section 4: "There should also be at most one dummy song in a match"
        # If dummies_needed <= k_matches:
        # The bottom `dummies_needed` matches have m - 1 real songs and 1 dummy.
        # The top `k_matches - dummies_needed` matches have m real songs and 0 dummies.
        match_groups: List[List[Song]] = []
        real_idx = 0

        # Number of matches with full real songs vs. matches with 1 dummy
        num_full_matches = k_matches - min(dummies_needed, k_matches)
        dummy_counter = 0

        for match_idx in range(k_matches):
            if match_idx < num_full_matches:
                # Full match of m real songs
                group = real_songs[real_idx: real_idx + m]
                real_idx += m
            else:
                # Match with m - 1 real songs and 1 dummy
                real_count = max(1, m - 1)
                group = real_songs[real_idx: real_idx + real_count]
                real_idx += real_count
                dummy_counter += 1
                dummy = Song(
                    id=-dummy_counter,
                    title=f"Dummy_{dummy_counter}",
                    artist="Dummy",
                    features={},
                    is_dummy=True,
                )
                group.append(dummy)

            match_groups.append(group)

        # Execute each match
        for match_songs in match_groups:
            real_in_match = [s for s in match_songs if not s.is_dummy]
            dummies_in_match = [s for s in match_songs if s.is_dummy]

            self.match_counter += 1

            if len(real_in_match) > 1:
                ranked_real = self.judge_callback(real_in_match, self.match_counter, self.round_num)
                if set(s.id for s in ranked_real) != set(s.id for s in real_in_match):
                    ranked_real = real_in_match
            else:
                ranked_real = real_in_match

            # Dummies always placed last (gets 0 points, never shown to user)
            full_ranking = ranked_real + dummies_in_match

            # Assign points:
            # 1st place gets M-1 pts, 2nd gets M-2 pts, ..., dummy gets 0 pts
            match_len = len(full_ranking)
            for rank_idx, s in enumerate(full_ranking):
                if not s.is_dummy:
                    points = (match_len - 1) - rank_idx
                    self.scores[s.id] += points

            # Record opponent history for real songs
            for s1 in real_in_match:
                for s2 in real_in_match:
                    if s1.id != s2.id:
                        self.opponents[s1.id].append(s2.id)

            record = MatchRecord(
                match_id=self.match_counter,
                round_num=self.round_num,
                songs_in_match=real_in_match,
                ranked_outcome=full_ranking,
                is_automated=False,
            )
            self.match_history.append(record)

    def has_unique_winner(self) -> bool:
        """Checks if at least one song has a unique winning score."""
        real_songs = self.playlist.real_songs
        if len(real_songs) <= 1:
            return True

        sorted_by_score = sorted(real_songs, key=lambda s: self.scores[s.id], reverse=True)
        return self.scores[sorted_by_score[0].id] > self.scores[sorted_by_score[1].id]

    def run(self) -> Tuple[List[Dict[str, Any]], Song]:
        """Runs the Swiss tournament until completion."""
        while self.round_num < self.max_rounds:
            self.run_round()
            if self.has_unique_winner():
                break

        real_songs = list(self.playlist.real_songs)
        tie_break_seeds = {s.id: self.rng.random() for s in real_songs}

        real_songs.sort(
            key=lambda s: (
                self.scores[s.id],
                self.calculate_strength_of_schedule(s.id),
                tie_break_seeds[s.id],
            ),
            reverse=True,
        )

        rankings: List[Dict[str, Any]] = []
        for idx, s in enumerate(real_songs):
            rankings.append({
                "rank": idx + 1,
                "song": s,
                "score": self.scores[s.id],
                "sos": self.calculate_strength_of_schedule(s.id),
                "features": dict(s.features),
            })

        winner = real_songs[0]
        return rankings, winner


class SortingTournament:
    """Executes a Sorting tournament with multi-pivot pool sorting and DQN learning or manual completion.

    Key properties (README 3b):
    1. Assign each song a score of 0.
    2. Choose (match size - 1) songs at random as separators.
    3. Round 1: Each match is the separators plus one other song.
       User ranks the separators once, then for each new song, they simply specify where to insert it!
    4. Songs are sorted into pools based on how many separators they each beat (0 to M-1).
    5. The final rankings of the separators are known based on pool boundaries.
    6. If feature data is provided, user can automate subsequent rounds with DQN or complete manually.
    7. Tournament ends when there are no more pools of size > 1.
    """

    def __init__(
        self,
        playlist: Playlist,
        match_size: int = 2,
        user_judge_callback: Optional[JudgeCallback] = None,
        separator_insert_callback: Optional[SeparatorInsertCallback] = None,
        prompt_automation_callback: Optional[Callable[[], bool]] = None,
        automate_subsequent_rounds: bool = True,
        random_seed: Optional[int] = 42,
    ) -> None:
        if match_size < 2 or match_size > 8:
            raise ValueError(f"Match size {match_size} must be between 2 and 8.")

        self.playlist = playlist.copy()
        self.match_size = match_size
        self.user_judge_callback = user_judge_callback
        self.separator_insert_callback = separator_insert_callback
        self.prompt_automation_callback = prompt_automation_callback
        self.automate_subsequent_rounds = automate_subsequent_rounds
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

        self.has_features = self.playlist.data_type != "names_only" and len(self.playlist.feature_names) > 0

        self.dqn = DQNAgent(
            feature_names=self.playlist.feature_names if self.has_features else [],
            random_seed=random_seed,
        )

        self.match_history: List[MatchRecord] = []
        self.match_counter = 0
        self.round_num = 1
        self.automated_matches_count = 0
        self.manual_matches_count = 0

    def _judge_match(
        self,
        songs: List[Song],
        is_automated: bool,
    ) -> List[Song]:
        """Judges a match using either the user callback or the trained DQN."""
        self.match_counter += 1

        if is_automated and self.has_features:
            ranked = self.dqn.rank_songs(songs)
            self.automated_matches_count += 1
        else:
            if self.user_judge_callback is not None:
                ranked = self.user_judge_callback(songs, self.match_counter, self.round_num)
            else:
                ranked = sorted(
                    songs,
                    key=lambda s: sum(s.features.values()) if s.features else self.rng.random(),
                    reverse=True,
                )
            self.manual_matches_count += 1
            if self.has_features:
                self.dqn.record_match_ranking(ranked)

        record = MatchRecord(
            match_id=self.match_counter,
            round_num=self.round_num,
            songs_in_match=songs,
            ranked_outcome=ranked,
            is_automated=is_automated,
        )
        self.match_history.append(record)
        return ranked

    def run(
        self,
        on_round1_complete: Optional[Callable[[], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], Song]:
        """Runs the Sorting tournament."""
        all_songs = list(self.playlist.real_songs)
        n = len(all_songs)
        if n <= 1:
            rankings = [{"rank": 1, "song": all_songs[0], "score": 1.0, "sos": 0.0, "features": all_songs[0].features}]
            return rankings, all_songs[0]

        # Step 2: Choose (match size - 1) songs at random as separators
        num_separators = min(self.match_size - 1, n - 1)
        shuffled = list(all_songs)
        self.rng.shuffle(shuffled)
        separators = shuffled[:num_separators]
        remaining = shuffled[num_separators:]

        # Step 3: Establish ranking among separators once (README 3b)
        # "To prevent the user from having to re-rank the separators in every match,
        # save the rankings of the separators from the first match of a round and just ask
        # the user where they would insert the new song."
        self.round_num = 1
        if len(separators) > 1:
            ordered_separators = self._judge_match(separators, is_automated=False)
        else:
            ordered_separators = list(separators)

        # sep_asc[0] is lowest separator, sep_asc[-1] is highest separator
        sep_asc = list(reversed(ordered_separators))

        # Pools: pool 0 (beat 0 seps) up to pool S (beat all S seps)
        pools: Dict[int, List[Song]] = {k: [] for k in range(num_separators + 1)}

        for candidate in remaining:
            self.match_counter += 1
            if self.separator_insert_callback is not None:
                # Ask user where to insert candidate among ordered separators
                # insert_index: 0 means top/1st place, num_separators means bottom/last place
                insert_idx = self.separator_insert_callback(
                    candidate,
                    ordered_separators,
                    self.match_counter,
                    self.round_num,
                )
                insert_idx = max(0, min(num_separators, insert_idx))
                match_ranked = (
                    ordered_separators[:insert_idx]
                    + [candidate]
                    + ordered_separators[insert_idx:]
                )
                self.manual_matches_count += 1
                if self.has_features:
                    self.dqn.record_match_ranking(match_ranked)

                record = MatchRecord(
                    match_id=self.match_counter,
                    round_num=self.round_num,
                    songs_in_match=ordered_separators + [candidate],
                    ranked_outcome=match_ranked,
                    is_automated=False,
                    details=f"Inserted at index {insert_idx}",
                )
                self.match_history.append(record)
            else:
                # Fallback to full match judging
                match_songs = ordered_separators + [candidate]
                # Note: decrement counter by 1 since _judge_match increments it
                self.match_counter -= 1
                match_ranked = self._judge_match(match_songs, is_automated=False)
                insert_idx = match_ranked.index(candidate)

            # Number of separators beat = separators that ranked below candidate
            separators_beat = num_separators - insert_idx
            pools[separators_beat].append(candidate)

        # Train DQN if feature data is present
        if self.has_features:
            self.dqn.train_on_buffer(epochs=80)

        # Automation prompt option (README 2b, 3b)
        should_automate = self.automate_subsequent_rounds and self.has_features
        if self.prompt_automation_callback is not None and self.has_features:
            should_automate = self.prompt_automation_callback()

        if on_round1_complete is not None:
            on_round1_complete()

        # Step 6: Recursively sort each pool
        self.round_num = 2

        def sort_pool(pool: List[Song]) -> List[Song]:
            p_len = len(pool)
            if p_len <= 1:
                return pool

            if p_len <= self.match_size:
                return self._judge_match(pool, is_automated=should_automate)

            s_count = min(self.match_size - 1, p_len - 1)
            p_shuffled = list(pool)
            self.rng.shuffle(p_shuffled)
            sub_seps = p_shuffled[:s_count]
            sub_rem = p_shuffled[s_count:]

            if len(sub_seps) > 1:
                sub_seps = self._judge_match(sub_seps, is_automated=should_automate)

            sub_sep_asc = list(reversed(sub_seps))
            sub_pools: Dict[int, List[Song]] = {k: [] for k in range(s_count + 1)}

            for cand in sub_rem:
                m_songs = sub_seps + [cand]
                m_ranked = self._judge_match(m_songs, is_automated=should_automate)
                cand_idx = m_ranked.index(cand)
                beat_count = sum(1 for sep in sub_seps if m_ranked.index(sep) > cand_idx)
                sub_pools[beat_count].append(cand)

            res: List[Song] = []
            for k in range(s_count, -1, -1):
                res.extend(sort_pool(sub_pools[k]))
                if k > 0:
                    res.append(sub_sep_asc[k - 1])
            return res

        final_sorted: List[Song] = []
        for k in range(num_separators, -1, -1):
            sorted_k = sort_pool(pools[k])
            final_sorted.extend(sorted_k)
            if k > 0:
                final_sorted.append(sep_asc[k - 1])

        total_songs = len(final_sorted)
        rankings: List[Dict[str, Any]] = []
        for idx, s in enumerate(final_sorted):
            rank = idx + 1
            score = float(total_songs - idx)
            rankings.append({
                "rank": rank,
                "song": s,
                "score": score,
                "sos": 0.0,
                "features": dict(s.features),
            })

        winner = final_sorted[0]
        return rankings, winner


class HybridTournament:
    """Executes a Hybrid tournament.

    Key properties (README 3c):
    1. Uses relative feature data.
    2. Identifies and eliminates strictly dominated songs.
    3. Dominated songs are not included in the tournament or final rankings.
    4. Runs a smaller Swiss or Sorting tournament on the remaining songs.
    """

    def __init__(
        self,
        playlist: Playlist,
        sub_format: str = "swiss",
        match_size: int = 2,
        judge_callback: Optional[JudgeCallback] = None,
        separator_insert_callback: Optional[SeparatorInsertCallback] = None,
        prompt_automation_callback: Optional[Callable[[], bool]] = None,
        automate_subsequent_rounds: bool = True,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.sub_format = sub_format.lower()
        if self.sub_format not in ("swiss", "sorting"):
            raise ValueError(f"Sub-format '{sub_format}' must be 'swiss' or 'sorting'.")

        self.match_size = match_size
        self.judge_callback = judge_callback
        self.random_seed = random_seed

        # Eliminate strictly dominated songs
        self.survivor_playlist, self.eliminated_songs = playlist.filter_strictly_dominated()

        if self.sub_format == "swiss":
            self.engine = SwissTournament(
                playlist=self.survivor_playlist,
                match_size=match_size,
                judge_callback=judge_callback,
                random_seed=random_seed,
            )
        else:
            self.engine = SortingTournament(
                playlist=self.survivor_playlist,
                match_size=match_size,
                user_judge_callback=judge_callback,
                separator_insert_callback=separator_insert_callback,
                prompt_automation_callback=prompt_automation_callback,
                automate_subsequent_rounds=automate_subsequent_rounds,
                random_seed=random_seed,
            )

    @property
    def match_history(self) -> List[MatchRecord]:
        return self.engine.match_history

    def run(
        self,
        on_round1_complete: Optional[Callable[[], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], Song, List[Song]]:
        """Runs the Hybrid tournament."""
        if isinstance(self.engine, SortingTournament):
            rankings, winner = self.engine.run(on_round1_complete=on_round1_complete)
        else:
            rankings, winner = self.engine.run()

        return rankings, winner, self.eliminated_songs
