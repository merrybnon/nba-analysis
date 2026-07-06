"""Distribution plots for player game-log stats."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from nba_analysis.data import get_game_logs, normalize_season


def player_stat_dist(
    player: str,
    seasons: int | str | list[int | str],
    stat: str = "PTS",
    season_type: str = "Regular Season",
    binwidth: float | None = None,
    kde: bool = True,
    hue_by_season: bool = False,
    ax: plt.Axes | None = None,
    refresh: bool = False,
) -> plt.Axes:
    """Plot the per-game distribution of a stat for a player.

    Parameters
    ----------
    player : full player name, e.g. 'Nikola Jokic'
    seasons : ending year (2025 = the 2024-25 season), an API string
        like '2024-25', or a list such as [2023, 2024, 2025]
    stat : any numeric game-log column ('PTS', 'AST', 'REB', ...)
    season_type : 'Regular Season' or 'Playoffs'
    binwidth : histogram bin width (defaults to seaborn's choice)
    kde : overlay a kernel density estimate
    hue_by_season : color the histogram by season instead of pooling
    ax : optionally draw into an existing matplotlib Axes
    refresh : re-download data instead of using the cache

    Returns the matplotlib Axes so callers can tweak further.
    """
    df = get_game_logs(player, seasons, season_type=season_type, refresh=refresh)

    if stat not in df.columns:
        raise ValueError(f"Unknown stat {stat!r}; available: {sorted(df.columns)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=stat,
        hue="SEASON" if hue_by_season else None,
        binwidth=binwidth,
        kde=kde,
        edgecolor="white",
        ax=ax,
    )

    season_list = seasons if isinstance(seasons, list) else [seasons]
    season_label = ", ".join(normalize_season(s) for s in season_list)
    n_games = len(df)
    mean_val = df[stat].mean()

    ax.set_title(
        f"{player} — {stat} per game\n"
        f"{season_type}, {season_label} ({n_games} games, mean {mean_val:.1f})"
    )
    ax.set_xlabel(f"{stat} per game")
    ax.set_ylabel("Games")

    if not hue_by_season:
        ax.axvline(mean_val, color="crimson", linestyle="--", linewidth=1, alpha=0.8)

    return ax
