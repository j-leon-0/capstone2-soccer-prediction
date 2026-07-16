"""Fetch club Elo ratings from the ClubElo API and build match-level Elo features.

ClubElo (http://clubelo.com) publishes a free daily Elo rating for every club.
Two endpoints are used:

* ``http://api.clubelo.com/<ClubSlug>`` returns the *full* dated history for one
  club as CSV, with one row per rating span (``From``/``To`` inclusive dates).
  The club slug is the ClubElo display name with spaces removed (e.g.
  ``Man City`` -> ``ManCity``).

For every match we look up each team's Elo on the match date (the row whose
``From <= date <= To``). Because ClubElo updates a club's rating the day *after*
a match, the rating valid on kickoff day is strictly pre-match — no leakage.

Outputs
-------
* ``data/raw/external/clubelo/<slug>.csv`` cached raw per-club history.
* ``data/processed/external/epl_elo.csv`` one row per match with
  ``home_elo``, ``away_elo`` and derived fields.

Re-run safe: cached club files are reused unless ``--refresh`` is passed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.error

MATCHES_PATH = Path("data/processed/epl_combined_cleaned.csv")
CACHE_DIR = Path("data/raw/external/clubelo")
OUTPUT_PATH = Path("data/processed/external/epl_elo.csv")

API_BASE = "http://api.clubelo.com"

# Map our dataset team names -> ClubElo API club slug (display name w/o spaces).
# Most teams are just the name with spaces removed; the exceptions are listed.
NAME_OVERRIDES = {
    "Man City": "ManCity",
    "Man United": "ManUnited",
    "Nott'm Forest": "Forest",
    "Sheffield United": "SheffieldUnited",
    "West Brom": "WestBrom",
    "West Ham": "WestHam",
    "Crystal Palace": "CrystalPalace",
    "Aston Villa": "AstonVilla",
}


def team_to_slug(team: str) -> str:
    if team in NAME_OVERRIDES:
        return NAME_OVERRIDES[team]
    return team.replace(" ", "").replace("'", "")


def fetch_club(slug: str, refresh: bool = False, pause: float = 0.5) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{slug}.csv"
    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file)

    url = f"{API_BASE}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-soccer/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(raw)
    time.sleep(pause)
    return pd.read_csv(cache_file)


def build_lookup(history: pd.DataFrame) -> pd.DataFrame:
    """Clean a per-club history into a date-indexed, sorted Elo table."""
    hist = history.copy()
    hist = hist[pd.to_numeric(hist["Elo"], errors="coerce").notna()]
    hist["Elo"] = hist["Elo"].astype(float)
    hist["From"] = pd.to_datetime(hist["From"], errors="coerce")
    hist["To"] = pd.to_datetime(hist["To"], errors="coerce")
    hist = hist.dropna(subset=["From", "To"]).sort_values("From").reset_index(drop=True)
    return hist[["From", "To", "Elo"]]


def elo_on_date(lookup: pd.DataFrame, date: pd.Timestamp) -> float:
    if lookup.empty or pd.isna(date):
        return np.nan
    mask = (lookup["From"] <= date) & (lookup["To"] >= date)
    hit = lookup.loc[mask, "Elo"]
    if len(hit):
        return float(hit.iloc[-1])
    # Fall back to the most recent rating strictly before the date (e.g. gaps).
    prior = lookup.loc[lookup["To"] < date, "Elo"]
    if len(prior):
        return float(prior.iloc[-1])
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Ignore cached club files.")
    args = parser.parse_args()

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))

    lookups: dict[str, pd.DataFrame] = {}
    for team in teams:
        slug = team_to_slug(team)
        try:
            history = fetch_club(slug, refresh=args.refresh)
            lookups[team] = build_lookup(history)
            print(f"  {team:18s} -> {slug:18s} {len(lookups[team]):5d} spans")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"  {team:18s} -> {slug:18s} FAILED: {exc}")
            lookups[team] = pd.DataFrame(columns=["From", "To", "Elo"])

    home_elo = []
    away_elo = []
    for _, row in matches.iterrows():
        home_elo.append(elo_on_date(lookups[row["HomeTeam"]], row["Date"]))
        away_elo.append(elo_on_date(lookups[row["AwayTeam"]], row["Date"]))

    out = matches[["Season", "Date", "HomeTeam", "AwayTeam"]].copy()
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    # Elo win-expectancy for the home side (standard 400-scale logistic).
    out["elo_expected_home"] = 1.0 / (1.0 + 10 ** (-out["elo_diff"] / 400.0))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    missing = out[["home_elo", "away_elo"]].isna().any(axis=1).sum()
    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH}")
    print(f"Rows with a missing Elo: {missing}")
    print(out[["home_elo", "away_elo", "elo_diff", "elo_expected_home"]].describe().round(2))


if __name__ == "__main__":
    main()
