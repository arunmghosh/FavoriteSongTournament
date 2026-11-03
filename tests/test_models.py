"""Tests for tournament algorithms and models in Favorite Song Tournament.

Tests:
1. Swiss Tournament logic, dummy padding, pairing, scoring, and tie-breaking.
2. Deep Q-Network (DQN) agent learning, scoring, and ranking.
3. Sorting Tournament separator partitioning, pool hierarchy preservation, and DQN automation.
4. Hybrid Tournament dominance dropout and sub-tournament transition.
"""

import unittest
from typing import List

from algorithms import (
    DQNAgent,
    HybridTournament,
    SortingTournament,
    SwissTournament,
)
from data_loader import (
    Playlist,
    Song,
    generate_synthetic_playlist,
    transform_relative_to_absolute,
)


class TestTournamentModels(unittest.TestCase):
    """Tournament model and algorithm logic tests."""

    def test_swiss_tournament_dummy_padding_and_scoring(self) -> None:
        """Verify Swiss tournament correctly pads with dummies, shields them from judge, and scores matches."""
        # 9 songs with match size 2 -> requires 1 dummy (total 10)
        songs = [
            Song(id=i + 1, title=f"Song_{i + 1}", artist="Artist", features={"F": float(i + 1)})
            for i in range(9)
        ]
        playlist = Playlist(songs=songs, data_type="relative", feature_names=["F"])

        dummy_seen_by_judge = False

        def tracking_judge(match_songs: List[Song], match_id: int, round_num: int) -> List[Song]:
            nonlocal dummy_seen_by_judge
            for s in match_songs:
                if s.is_dummy:
                    dummy_seen_by_judge = True
            # Rank higher feature first
            return sorted(match_songs, key=lambda s: s.features["F"], reverse=True)

        swiss = SwissTournament(playlist, match_size=2, judge_callback=tracking_judge, random_seed=42)

        self.assertEqual(swiss.dummies_count, 1)
        self.assertEqual(swiss.padded_playlist.total_count, 10)

        rankings, winner = swiss.run()

        # Verify dummy was never presented to the judge
        self.assertFalse(dummy_seen_by_judge, "Dummy song should never be presented to the judge callback.")

        # Exactly 9 real songs in final ranking
        self.assertEqual(len(rankings), 9)

        # Winner should be the highest-scoring song
        self.assertIsInstance(winner, Song)
        self.assertFalse(winner.is_dummy)
        self.assertEqual(rankings[0]["song"].id, winner.id)

    def test_dummy_distribution_at_most_one_per_match(self) -> None:
        """Verify Section 4 rule: at most one dummy per match (e.g. 13 songs, match size 3)."""
        songs = [
            Song(id=i + 1, title=f"Song_{i + 1}", artist="Artist", features={"F": float(i + 1)})
            for i in range(13)
        ]
        playlist = Playlist(songs=songs, data_type="relative", feature_names=["F"])
        swiss = SwissTournament(playlist, match_size=3, random_seed=42)
        swiss.run_round()

        # Total matches in round: ceil(13 / 3) = 5
        self.assertEqual(len(swiss.match_history), 5)

        dummy_counts = [sum(1 for s in m.ranked_outcome if s.is_dummy) for m in swiss.match_history]
        real_counts = [len(m.songs_in_match) for m in swiss.match_history]

        # No match should have > 1 dummy
        self.assertLessEqual(max(dummy_counts), 1)
        # Exactly 2 matches with 1 dummy, 3 matches with 0 dummies
        self.assertEqual(dummy_counts.count(1), 2)
        self.assertEqual(dummy_counts.count(0), 3)
        # Real songs per match should be [3, 3, 3, 2, 2]
        self.assertEqual(real_counts, [3, 3, 3, 2, 2])

    def test_sorting_tournament_separator_insertion_ux(self) -> None:
        """Verify Sorting tournament separator insertion callback correctly places candidates into pools."""
        playlist = generate_synthetic_playlist(size=12, data_type="absolute", random_seed=42)

        def synthetic_insert_cb(candidate: Song, ordered_seps: List[Song], mid: int, rnd: int) -> int:
            # Place candidate based on sum of features compared to separators
            c_score = sum(candidate.features.values())
            for idx, sep in enumerate(ordered_seps):
                if c_score > sum(sep.features.values()):
                    return idx
            return len(ordered_seps)

        sorting = SortingTournament(
            playlist=playlist,
            match_size=3,
            separator_insert_callback=synthetic_insert_cb,
            random_seed=42,
        )
        rankings, winner = sorting.run()

        self.assertEqual(len(rankings), 12)
        self.assertGreater(sorting.manual_matches_count, 0)
        self.assertGreater(sorting.automated_matches_count, 0)

    def test_sorting_tournament_manual_completion_mode(self) -> None:
        """Verify Sorting tournament when user chooses to manually complete all rounds."""
        playlist = generate_synthetic_playlist(size=9, data_type="names_only", random_seed=42)

        sorting = SortingTournament(
            playlist=playlist,
            match_size=3,
            automate_subsequent_rounds=False,
            random_seed=42,
        )
        rankings, winner = sorting.run()

        self.assertEqual(len(rankings), 9)
        self.assertEqual(sorting.automated_matches_count, 0)
        self.assertGreater(sorting.manual_matches_count, 0)

    def test_swiss_tournament_match_points(self) -> None:
        """Verify that in a match of size M, rank r gets M - r points."""
        songs = [
            Song(id=1, title="A", artist="X", features={"F": 10.0}),
            Song(id=2, title="B", artist="X", features={"F": 8.0}),
            Song(id=3, title="C", artist="X", features={"F": 5.0}),
            Song(id=4, title="D", artist="X", features={"F": 2.0}),
        ]
        playlist = Playlist(songs=songs, data_type="relative", feature_names=["F"])

        def deterministic_judge(m_songs: List[Song], mid: int, rnd: int) -> List[Song]:
            return sorted(m_songs, key=lambda s: s.features["F"], reverse=True)

        swiss = SwissTournament(playlist, match_size=4, judge_callback=deterministic_judge)
        swiss.run_round()

        # For match size 4:
        # A (rank 1) -> 3 pts
        # B (rank 2) -> 2 pts
        # C (rank 3) -> 1 pt
        # D (rank 4) -> 0 pts
        self.assertEqual(swiss.scores[1], 3.0)
        self.assertEqual(swiss.scores[2], 2.0)
        self.assertEqual(swiss.scores[3], 1.0)
        self.assertEqual(swiss.scores[4], 0.0)

    def test_dqn_agent_learning_and_ranking(self) -> None:
        """Verify that the DQN agent learns pairwise preferences and ranks higher-utility songs first."""
        agent = DQNAgent(feature_names=["Catchiness", "Melody"], random_seed=42)

        s_high = Song(1, "High", "A", features={"Catchiness": 10.0, "Melody": 10.0})
        s_mid = Song(2, "Mid", "A", features={"Catchiness": 5.0, "Melody": 5.0})
        s_low = Song(3, "Low", "A", features={"Catchiness": 1.0, "Melody": 1.0})

        # Record consistent preferences: High > Mid > Low
        for _ in range(10):
            agent.record_match_ranking([s_high, s_mid, s_low])

        initial_loss = agent.train_on_buffer(epochs=10)
        final_loss = agent.train_on_buffer(epochs=50)

        # Predict score
        q_high = agent.predict_score(s_high)
        q_mid = agent.predict_score(s_mid)
        q_low = agent.predict_score(s_low)

        self.assertGreater(q_high, q_mid)
        self.assertGreater(q_mid, q_low)

        # Ranked list should place High first, Low last
        ranked = agent.rank_songs([s_mid, s_low, s_high])
        self.assertEqual([s.id for s in ranked], [1, 2, 3])

    def test_sorting_tournament_pool_hierarchy(self) -> None:
        """Verify Sorting tournament multi-pivot pool sorting and hierarchy guarantees."""
        playlist = generate_synthetic_playlist(size=12, data_type="absolute", random_seed=42)

        def synthetic_user_judge(songs: List[Song], mid: int, rnd: int) -> List[Song]:
            # Always prefer higher sum of features
            return sorted(songs, key=lambda s: sum(s.features.values()), reverse=True)

        sorting = SortingTournament(
            playlist=playlist,
            match_size=3,
            user_judge_callback=synthetic_user_judge,
            random_seed=42,
        )

        rankings, winner = sorting.run()

        # All 12 songs must be ranked
        self.assertEqual(len(rankings), 12)
        ranked_ids = [r["song"].id for r in rankings]
        self.assertEqual(len(set(ranked_ids)), 12)

        # Scores should be strictly descending
        scores = [r["score"] for r in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check that manual matches occurred in Round 1 and automated in Round 2
        self.assertGreater(sorting.manual_matches_count, 0)
        self.assertGreater(sorting.automated_matches_count, 0)

    def test_hybrid_tournament_dominance_and_sub_tournament(self) -> None:
        """Verify Hybrid tournament eliminates strictly dominated songs and runs sub-tournament."""
        # s2 is strictly dominated by s1
        s1 = Song(1, "Strong", "A", features={"M": 8.0, "C": 8.0})
        s2 = Song(2, "Weak", "A", features={"M": 2.0, "C": 2.0})
        # Others with distinct tradeoffs to form a valid tournament
        other_songs = [
            Song(i, f"Song_{i}", "A", features={"M": float(i), "C": float(10 - i)})
            for i in range(3, 11)
        ]
        playlist = Playlist([s1, s2] + other_songs, data_type="relative", feature_names=["M", "C"])

        hybrid = HybridTournament(playlist, sub_format="swiss", match_size=2, random_seed=42)
        rankings, winner, eliminated = hybrid.run()

        eliminated_ids = [s.id for s in eliminated]
        self.assertIn(2, eliminated_ids, "Song 2 should be eliminated as strictly dominated.")

        ranked_ids = [r["song"].id for r in rankings]
        self.assertNotIn(2, ranked_ids, "Eliminated song should not appear in final rankings.")
        self.assertEqual(len(rankings), len(playlist) - len(eliminated))


if __name__ == "__main__":
    unittest.main()

