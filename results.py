"""Results presentation and statistical analysis module for Favorite Song Tournament.

Calculates:
- Pearson correlation (r) and Spearman rank correlation (rho) between feature scores and tournament outcomes.
- Generates descriptive commentary explaining the user's music taste.
- Displays cleanly formatted console outputs (Winner banner, rankings table, and taste insights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from data_loader import Song


@dataclass
class TournamentResult:
    """Encapsulates full tournament execution results and analytical insights."""

    tournament_format: str
    match_size: int
    total_rounds: int
    total_matches: int
    manual_matches: int
    automated_matches: int
    rankings: List[Dict[str, Any]]
    winner: Song
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    eliminated_songs: List[Song] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


def calculate_feature_correlations(
    rankings: List[Dict[str, Any]],
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculates Pearson (r) and Spearman (rho) correlations between feature scores and tournament outcome.

    The tournament outcome metric used is inverted rank (N - rank + 1), where higher is better.
    """
    if not feature_names or len(rankings) < 2:
        return {}

    n = len(rankings)
    outcomes = np.array([float(n - r["rank"] + 1) for r in rankings], dtype=np.float64)
    correlations: Dict[str, Dict[str, float]] = {}

    for f in feature_names:
        feature_vals = np.array(
            [float(r.get("features", {}).get(f, r["song"].features.get(f, 0.0))) for r in rankings],
            dtype=np.float64,
        )

        if np.std(feature_vals) == 0.0 or np.std(outcomes) == 0.0:
            correlations[f] = {
                "pearson_r": 0.0,
                "pearson_p": 1.0,
                "spearman_rho": 0.0,
                "spearman_p": 1.0,
            }
            continue

        try:
            p_res = stats.pearsonr(feature_vals, outcomes)
            pearson_r = float(p_res.statistic)
            pearson_p = float(p_res.pvalue)
        except Exception:
            pearson_r, pearson_p = 0.0, 1.0

        try:
            s_res = stats.spearmanr(feature_vals, outcomes)
            spearman_rho = float(s_res.statistic)
            spearman_p = float(s_res.pvalue)
        except Exception:
            spearman_rho, spearman_p = 0.0, 1.0

        pearson_r = 0.0 if np.isnan(pearson_r) else pearson_r
        spearman_rho = 0.0 if np.isnan(spearman_rho) else spearman_rho

        correlations[f] = {
            "pearson_r": round(pearson_r, 3),
            "pearson_p": round(pearson_p, 4),
            "spearman_rho": round(spearman_rho, 3),
            "spearman_p": round(spearman_p, 4),
        }

    return correlations


def generate_feature_insights(correlations: Dict[str, Dict[str, float]]) -> List[str]:
    """Generates descriptive, human-readable narrative comments on how each feature contributed to rankings."""
    if not correlations:
        return ["No feature data provided for this tournament (names only). Feature correlation analysis omitted."]

    insights: List[str] = []

    sorted_features = sorted(
        correlations.items(),
        key=lambda item: item[1]["pearson_r"],
        reverse=True,
    )

    top_feature, top_stats = sorted_features[0]
    top_r = top_stats["pearson_r"]

    if top_r > 0.2:
        insights.append(
            f"Primary Driver: '{top_feature}' demonstrated the strongest positive correlation "
            f"(r = {top_r:+.2f}, ρ = {top_stats['spearman_rho']:+.2f}) with your tournament rankings, "
            f"indicating it is the single most dominant factor defining your music preference in this playlist."
        )

    for feat, metrics in sorted_features:
        r = metrics["pearson_r"]
        rho = metrics["spearman_rho"]

        if r >= 0.65:
            level = "Strong Positive"
            comment = f"Songs excelling in '{feat}' consistently dominated matches and reached top tournament ranks."
        elif r >= 0.35:
            level = "Moderate Positive"
            comment = f"'{feat}' meaningfully boosted songs toward higher rankings, acting as an important secondary factor."
        elif r >= 0.10:
            level = "Mild Positive"
            comment = f"'{feat}' had a subtle positive impact, but other musical elements frequently took precedence."
        elif r > -0.10:
            level = "Neutral / Negligible"
            comment = f"'{feat}' showed minimal relationship with match outcomes; you valued songs regardless of this attribute."
        elif r > -0.35:
            level = "Mild Negative"
            comment = f"Songs with lower '{feat}' scores occasionally edged out higher-scoring songs in your decisions."
        else:
            level = "Noticeable Inverse"
            comment = f"There was an inverse correlation with '{feat}', suggesting high scores in this area did not align with your tastes."

        insights.append(f"• {feat:<15} [r = {r:+.2f}, ρ = {rho:+.2f} | {level}]: {comment}")

    return insights


def display_results(
    result: TournamentResult,
    view_mode: str = "all",
    notable_count: int = 5,
    print_output: bool = True,
) -> str:
    """Formats and displays the complete tournament results in the console."""
    lines: List[str] = []

    def p(line: str = "") -> None:
        lines.append(line)

    p()
    p("=" * 80)
    p(" " * 26 + "🏆 TOURNAMENT WINNER 🏆")
    p(" " * max(0, (80 - len(result.winner.display_name)) // 2) + result.winner.display_name)
    p("=" * 80)
    p()

    # Tournament Summary Stats
    p("📋 TOURNAMENT EXECUTION SUMMARY")
    p("-" * 80)
    p(f"  • Format:               {result.tournament_format.capitalize()}")
    p(f"  • Match Size:           {result.match_size} songs per match")
    p(f"  • Total Rounds:         {result.total_rounds}")
    p(f"  • Total Matches:        {result.total_matches} "
      f"(Manual: {result.manual_matches}, Automated: {result.automated_matches})")
    p(f"  • Ranked Songs:         {len(result.rankings)}")
    if result.eliminated_songs:
        p(f"  • Strictly Dominated:   {len(result.eliminated_songs)} songs dropped prior to tournament")
    p("-" * 80)
    p()

    view_mode = view_mode.lower()
    if view_mode in ("winner", "one"):
        display_rankings = result.rankings[:1]
        ranking_title = "🏆 FINAL WINNER"
    elif view_mode in ("notable", "top five", "top 5"):
        display_rankings = result.rankings[:max(1, notable_count)]
        ranking_title = f"🌟 NOTABLE SONGS (TOP {len(display_rankings)})"
    else:
        display_rankings = result.rankings
        ranking_title = "📊 COMPLETE TOURNAMENT RANKINGS"

    p(ranking_title)
    p("-" * 80)

    feat_cols = result.feature_names
    feat_header = "".join(f"{f[:8]:>9}" for f in feat_cols) if feat_cols else ""
    header = f"{'Rank':<6}{'Song Title':<30}{'Artist':<22}{'Score':>7}{feat_header}"
    p(header)
    p("-" * 80)

    for item in display_rankings:
        rank = item["rank"]
        song: Song = item["song"]
        score = item["score"]
        title_disp = (song.title[:27] + "...") if len(song.title) > 30 else song.title
        artist_disp = (song.artist[:19] + "...") if len(song.artist) > 22 else song.artist

        feats_str = "".join(f"{item.get('features', {}).get(f, 0.0):>9.1f}" for f in feat_cols) if feat_cols else ""
        row = f"{rank:<6}{title_disp:<30}{artist_disp:<22}{score:>7.1f}{feats_str}"
        p(row)

    p("-" * 80)
    p()

    # Music Taste Insights
    if result.correlations:
        p("🧠 MUSIC TASTE INSIGHTS & FEATURE CORRELATIONS")
        p("-" * 80)
        p(f"{'Feature':<18}{'Pearson (r)':<15}{'Spearman (ρ)':<16}{'Contribution Level':<20}")
        p("-" * 80)

        for feat, metrics in result.correlations.items():
            r = metrics["pearson_r"]
            rho = metrics["spearman_rho"]
            if abs(r) >= 0.65:
                lvl = "Strong Influence"
            elif abs(r) >= 0.35:
                lvl = "Moderate Influence"
            elif abs(r) >= 0.10:
                lvl = "Mild Influence"
            else:
                lvl = "Negligible"
            p(f"{feat:<18}{r:>+8.2f}       {rho:>+8.2f}        {lvl:<20}")

        p("-" * 80)
        p()
        p("📝 PREFERENCE ANALYSIS & COMMENTARY:")
        for comment in result.insights:
            p(f"  {comment}")
        p("=" * 80)
        p()
    else:
        p("ℹ️  Feature Analysis: No song feature data was provided for this tournament (names only).")
        p("=" * 80)
        p()

    output_str = "\n".join(lines)
    if print_output:
        print(output_str)
    return output_str
