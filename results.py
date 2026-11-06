"""Results presentation and statistical analysis module for Favorite Song Tournament.

Calculates:
- Pearson correlation (r) and Spearman rank correlation (rho) between feature scores and tournament outcomes.
- Generates descriptive commentary explaining the user's music taste.
- Displays cleanly formatted console outputs (Winner banner, rankings table, and taste insights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


REPORTS_DIR = Path("reports")


def generate_markdown_report(
    result: TournamentResult,
    view_mode: str = "all",
    notable_count: int = 5,
) -> str:
    """Generates a GitHub-Flavored Markdown report of the tournament results."""
    lines: List[str] = []

    def p(line: str = "") -> None:
        lines.append(line)

    p(f"# 🎵 Favorite Song Tournament - Final Results")
    p()
    p("## 🏆 Tournament Winner")
    p(f"> **\"{result.winner.title}\"** by *{result.winner.artist}*")
    p()
    p("## 📋 Tournament Execution Summary")
    p(f"- **Format:** {result.tournament_format.capitalize()}")
    p(f"- **Match Size:** {result.match_size} songs per match")
    p(f"- **Total Rounds:** {result.total_rounds}")
    p(f"- **Total Matches:** {result.total_matches} (Manual: {result.manual_matches}, Automated: {result.automated_matches})")
    p(f"- **Ranked Songs:** {len(result.rankings)}")
    if result.eliminated_songs:
        p(f"- **Strictly Dominated:** {len(result.eliminated_songs)} songs eliminated prior to tournament")
    p()

    view_mode = view_mode.lower()
    if view_mode in ("winner", "one"):
        display_rankings = result.rankings[:1]
        ranking_title = "## 🏆 Final Winner"
    elif view_mode in ("notable", "top five", "top 5"):
        display_rankings = result.rankings[:max(1, notable_count)]
        ranking_title = f"## 🌟 Top {len(display_rankings)} Ranked Songs"
    else:
        display_rankings = result.rankings
        ranking_title = "## 📊 Complete Tournament Rankings"

    p(ranking_title)
    p()

    feat_cols = result.feature_names
    feat_header = "".join(f" | {f}" for f in feat_cols) if feat_cols else ""
    feat_align = "".join(" | ---:" for _ in feat_cols) if feat_cols else ""
    p(f"| Rank | Song Title | Artist | Tournament Score{feat_header} |")
    p(f"| ---: | :--- | :--- | ---:{feat_align} |")

    for item in display_rankings:
        rank = item["rank"]
        song: Song = item["song"]
        score = item["score"]
        feats_str = "".join(f" | {item.get('features', {}).get(f, 0.0):.1f}" for f in feat_cols) if feat_cols else ""
        p(f"| {rank} | {song.title} | {song.artist} | {score:.1f}{feats_str} |")

    p()

    if result.correlations:
        p("## 🧠 Music Taste Insights & Feature Correlations")
        p()
        p("| Feature | Pearson (r) | Spearman (ρ) | Contribution Level |")
        p("| :--- | ---: | ---: | :--- |")
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
            p(f"| **{feat}** | {r:+.2f} | {rho:+.2f} | {lvl} |")
        p()
        p("### 📝 Preference Analysis & Commentary")
        for comment in result.insights:
            p(f"- {comment}")
        p()
    else:
        p("## ℹ️ Feature Analysis")
        p("No song feature data was provided for this tournament (names only).")
        p()

    return "\n".join(lines)


def generate_text_report(
    result: TournamentResult,
    view_mode: str = "all",
    notable_count: int = 5,
) -> str:
    """Generates a plain-text report of the tournament results."""
    return display_results(result, view_mode=view_mode, notable_count=notable_count, print_output=False)


def generate_pdf_report(
    result: TournamentResult,
    output_path: Union[str, Path],
    view_mode: str = "all",
    notable_count: int = 5,
) -> Path:
    """Generates a styled PDF report of the tournament results using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1DB954"),
        alignment=1,
    )
    winner_box_style = ParagraphStyle(
        "WinnerStyle",
        parent=styles["Normal"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#111111"),
        alignment=1,
    )
    heading2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#191414"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#333333"),
    )
    table_cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("🎵 FAVORITE SONG TOURNAMENT", title_style))
    elements.append(Spacer(1, 8))

    # Winner Callout Box
    winner_html = f"<b>🏆 TOURNAMENT WINNER</b><br/><font size='16'><b>\"{result.winner.title}\"</b></font><br/>by {result.winner.artist}"
    winner_table = Table([[Paragraph(winner_html, winner_box_style)]], colWidths=[540])
    winner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F8F0")),
        ("BORDER", (0, 0), (-1, -1), 1.5, colors.HexColor("#1DB954")),
        ("PADDING", (0, 0), (-1, -1), 12),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(winner_table)
    elements.append(Spacer(1, 12))

    # Execution Summary
    elements.append(Paragraph("📋 Execution Summary", heading2_style))
    summary_data = [
        [Paragraph(f"<b>Format:</b> {result.tournament_format.capitalize()}", body_style),
         Paragraph(f"<b>Match Size:</b> {result.match_size} songs", body_style)],
        [Paragraph(f"<b>Rounds:</b> {result.total_rounds}", body_style),
         Paragraph(f"<b>Total Matches:</b> {result.total_matches} (Manual: {result.manual_matches}, Automated: {result.automated_matches})", body_style)],
        [Paragraph(f"<b>Ranked Songs:</b> {len(result.rankings)}", body_style),
         Paragraph(f"<b>Dominated Songs Dropped:</b> {len(result.eliminated_songs)}", body_style)],
    ]
    summary_table = Table(summary_data, colWidths=[270, 270])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9F9F9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Rankings Table
    view_mode = view_mode.lower()
    if view_mode in ("winner", "one"):
        display_rankings = result.rankings[:1]
        ranking_title = "🏆 Final Winner"
    elif view_mode in ("notable", "top five", "top 5"):
        display_rankings = result.rankings[:max(1, notable_count)]
        ranking_title = f"🌟 Top {len(display_rankings)} Ranked Songs"
    else:
        display_rankings = result.rankings
        ranking_title = "📊 Tournament Rankings"

    elements.append(Paragraph(ranking_title, heading2_style))

    feat_cols = result.feature_names
    table_headers = ["Rank", "Song Title", "Artist", "Score"] + feat_cols
    col_widths = [36, 170, 130, 44] + [40] * len(feat_cols)

    table_rows = [[Paragraph(f"<b>{h}</b>", table_cell_style) for h in table_headers]]
    for item in display_rankings:
        rank = str(item["rank"])
        song = item["song"]
        score = f"{item['score']:.1f}"
        feat_vals = [f"{item.get('features', {}).get(f, 0.0):.1f}" for f in feat_cols]
        row_cells = [rank, song.title, song.artist, score] + feat_vals
        table_rows.append([Paragraph(c, table_cell_style) for c in row_cells])

    ranking_pdf_table = Table(table_rows, colWidths=col_widths)
    ranking_pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E3A59")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D0D0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(ranking_pdf_table)
    elements.append(Spacer(1, 14))

    # Insights & Correlations
    if result.correlations:
        elements.append(Paragraph("🧠 Music Taste Insights & Feature Correlations", heading2_style))
        corr_headers = ["Feature", "Pearson (r)", "Spearman (ρ)", "Contribution Level"]
        corr_rows = [[Paragraph(f"<b>{h}</b>", table_cell_style) for h in corr_headers]]
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
            corr_rows.append([
                Paragraph(feat, table_cell_style),
                Paragraph(f"{r:+.2f}", table_cell_style),
                Paragraph(f"{rho:+.2f}", table_cell_style),
                Paragraph(lvl, table_cell_style),
            ])
        corr_table = Table(corr_rows, colWidths=[135, 135, 135, 135])
        corr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D0D0")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (1, 0), (2, -1), "CENTER"),
        ]))
        elements.append(corr_table)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Preference Analysis & Commentary:</b>", body_style))
        for comment in result.insights:
            elements.append(Paragraph(f"• {comment}", body_style))
            elements.append(Spacer(1, 3))

    doc.build(elements)
    return out_file


def export_report(
    result: TournamentResult,
    file_format: str = "txt",
    output_dir: Union[str, Path] = REPORTS_DIR,
    filename: Optional[str] = None,
    view_mode: str = "all",
    notable_count: int = 5,
) -> Union[Path, Dict[str, Path]]:
    """Exports the tournament results to a file in the reports subfolder.

    Args:
        result: The TournamentResult object.
        file_format: 'txt', 'md', 'pdf', or 'all'.
        output_dir: Target folder (default: 'reports').
        filename: Optional base filename without extension.
        view_mode: 'all', 'notable', or 'winner'.
        notable_count: Count for notable songs.

    Returns:
        Path to generated file (or Dict[str, Path] if 'all').
    """
    from datetime import datetime
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename or f"tournament_{result.tournament_format}_{timestamp}"

    fmt = file_format.lower().lstrip(".")

    if fmt == "all":
        paths: Dict[str, Path] = {}
        # txt
        txt_path = out_dir / f"{base_name}.txt"
        txt_path.write_text(generate_text_report(result, view_mode, notable_count), encoding="utf-8")
        paths["txt"] = txt_path
        # md
        md_path = out_dir / f"{base_name}.md"
        md_path.write_text(generate_markdown_report(result, view_mode, notable_count), encoding="utf-8")
        paths["md"] = md_path
        # pdf
        pdf_path = out_dir / f"{base_name}.pdf"
        generate_pdf_report(result, pdf_path, view_mode, notable_count)
        paths["pdf"] = pdf_path
        return paths

    elif fmt == "md":
        md_path = out_dir / f"{base_name}.md"
        md_path.write_text(generate_markdown_report(result, view_mode, notable_count), encoding="utf-8")
        return md_path

    elif fmt == "pdf":
        pdf_path = out_dir / f"{base_name}.pdf"
        generate_pdf_report(result, pdf_path, view_mode, notable_count)
        return pdf_path

    else:  # default 'txt'
        txt_path = out_dir / f"{base_name}.txt"
        txt_path.write_text(generate_text_report(result, view_mode, notable_count), encoding="utf-8")
        return txt_path
