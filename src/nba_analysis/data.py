"""Fetch and cache NBA player game logs via nba_api.

The cache lives in <project root>/data/cache/ as one parquet file per
(player, season, season type). Completed seasons never change, so cached
files are reused indefinitely; pass refresh=True to re-download (do this
for the in-progress season to pick up new games).

Seasons are identified by their ending year: 2025 means the 2024-25
season. API-native strings like "2024-25" are also accepted as-is.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# src/nba_analysis/data.py -> project root is two parents up from this file's dir
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Be polite to stats.nba.com: minimum gap between consecutive requests.
_REQUEST_DELAY_SECONDS = 0.6
_last_request_time = 0.0


def _throttle() -> None:
    """Sleep just enough to keep network requests _REQUEST_DELAY_SECONDS apart."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _REQUEST_DELAY_SECONDS:
        time.sleep(_REQUEST_DELAY_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def get_player_id(name: str) -> int:
    """Resolve a full player name (e.g. 'Nikola Jokic') to an NBA player ID.

    Matching is case-insensitive and diacritic-tolerant on nba_api's side,
    so 'nikola jokic' works even though the official name is 'Nikola Jokić'.
    """
    matches = players.find_players_by_full_name(name)
    if not matches:
        raise ValueError(f"No player found matching {name!r}")
    if len(matches) > 1:
        options = ", ".join(p["full_name"] for p in matches)
        raise ValueError(f"Ambiguous name {name!r}; matches: {options}")
    return matches[0]["id"]


def normalize_season(season: int | str) -> str:
    """Convert a season's ending year to the NBA API format.

    2025 -> '2024-25' (the season that ends in 2025). Strings are assumed
    to already be API-formatted and pass through unchanged.
    """
    if isinstance(season, int):
        return f"{season - 1}-{str(season)[-2:]}"
    return season


def _cache_path(player_id: int, season: str, season_type: str) -> Path:
    tag = season_type.replace(" ", "")
    return CACHE_DIR / f"{player_id}_{season}_{tag}.parquet"


def get_game_log(
    player: str | int,
    season: int | str,
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return one season of game logs for a player, using the local cache.

    Parameters
    ----------
    player : full name (str) or NBA player ID (int)
    season : ending year (2025 = the 2024-25 season) or API string '2024-25'
    season_type : 'Regular Season', 'Playoffs', or 'Pre Season'
    refresh : force a re-download even if a cached copy exists
    """
    player_id = get_player_id(player) if isinstance(player, str) else player
    season = normalize_season(season)
    path = _cache_path(player_id, season, season_type)

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    _throttle()
    log = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = log.get_data_frames()[0]

    # Tidy up: proper dtypes and chronological order.
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def get_game_logs(
    player: str | int,
    seasons: int | str | list[int | str],
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return game logs for one or more seasons, concatenated chronologically.

    `seasons` may be a single season (2025 or '2024-25') or a list of them.
    """
    if isinstance(seasons, (int, str)):
        seasons = [seasons]

    frames = [
        get_game_log(player, season, season_type, refresh=refresh)
        for season in seasons
    ]
    return pd.concat(frames, ignore_index=True)
