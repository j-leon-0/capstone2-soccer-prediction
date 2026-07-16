"""Build true fixture-congestion features across ALL competitions.

The base pipeline's ``days_since_last_match`` counts only Premier League games, so a
club's midweek FA Cup / EFL Cup / Champions League / Europa / Conference fixtures are
invisible — which *overstates* rest for the busiest clubs. This script unions every
PL match with the scraped cup/European matches (``data/raw/external/fbref_fixtures/``)
to compute each team's real match calendar, then derives pre-match congestion features.

Output: ``data/processed/external/epl_congestion_features.csv`` keyed by
``(Season, HomeTeam, AwayTeam)`` with, for home and away and their diff:

* ``rest_days_all``          days since the team's previous match in ANY competition
* ``matches_last_14_all``    number of matches played in the 14 days before kickoff
* ``played_euro_last_5``     1 if the team had a European tie in the 5 days before
* ``rest_days_gain``         PL-only rest minus true rest (how much the PL-only
                             feature over-counts because of hidden midweek games)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MATCHES_PATH = Path("data/processed/epl_combined_cleaned.csv")
FIXTURES_DIR = Path("data/raw/external/fbref_fixtures")
OUTPUT_PATH = Path("data/processed/external/epl_congestion_features.csv")

EURO_COMPS = {"UCL", "UEL", "UECL"}
MAX_REST_CAP = 60  # cap season-boundary gaps

FBREF_TO_OURS = {
    "Aston-Villa": "Aston Villa", "Brighton-and-Hove-Albion": "Brighton",
    "Cardiff-City": "Cardiff", "Crystal-Palace": "Crystal Palace",
    "Huddersfield-Town": "Huddersfield", "Hull-City": "Hull",
    "Ipswich-Town": "Ipswich", "Leeds-United": "Leeds", "Leicester-City": "Leicester",
    "Luton-Town": "Luton", "Manchester-City": "Man City", "Manchester-United": "Man United",
    "Newcastle-United": "Newcastle", "Norwich-City": "Norwich",
    "Nottingham-Forest": "Nott'm Forest", "Sheffield-United": "Sheffield United",
    "Stoke-City": "Stoke", "Swansea-City": "Swansea", "Tottenham-Hotspur": "Tottenham",
    "West-Bromwich-Albion": "West Brom", "West-Ham-United": "West Ham",
    "Wolverhampton-Wanderers": "Wolves",
}


def fbref_name(slug: str) -> str:
    return FBREF_TO_OURS.get(slug, slug)


def load_all_matches(pl: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (team, date, competition) across all competitions."""
    # Premier League legs (both teams), from our own canonical data.
    pl_long = pd.concat([
        pd.DataFrame({"Team": pl["HomeTeam"], "date": pl["Date"], "comp": "PL"}),
        pd.DataFrame({"Team": pl["AwayTeam"], "date": pl["Date"], "comp": "PL"}),
    ], ignore_index=True)

    # Cup / European legs from FBref (already filtered to PL-involving matches).
    rows = []
    for f in sorted(FIXTURES_DIR.glob("*.tsv")):
        comp = f.name.split("_")[0]
        d = pd.read_csv(f, sep="\t", header=None, names=["date", "h", "a"])
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        for side in ("h", "a"):
            rows.append(pd.DataFrame({"Team": d[side].map(fbref_name),
                                      "date": d["date"], "comp": comp}))
    cup_long = pd.concat(rows, ignore_index=True)

    allm = pd.concat([pl_long, cup_long], ignore_index=True)
    allm["date"] = pd.to_datetime(allm["date"])
    # A team can appear once per date per comp; drop exact dupes.
    allm = allm.drop_duplicates(subset=["Team", "date", "comp"]).sort_values(["Team", "date"])
    return allm


def team_congestion(cal: pd.DataFrame, team: str, date: pd.Timestamp) -> dict:
    """Congestion metrics for one team as of one match date (strictly prior games)."""
    g = cal[cal["Team"] == team]
    prior = g[g["date"] < date]
    if prior.empty:
        return {"rest_days_all": np.nan, "matches_last_14_all": 0, "played_euro_last_5": 0}
    last = prior["date"].max()
    rest = (date - last).days
    last14 = prior[prior["date"] >= date - pd.Timedelta(days=14)]
    euro5 = prior[(prior["date"] >= date - pd.Timedelta(days=5)) & (prior["comp"].isin(EURO_COMPS))]
    return {
        "rest_days_all": min(rest, MAX_REST_CAP),
        "matches_last_14_all": int(len(last14)),
        "played_euro_last_5": int(len(euro5) > 0),
    }


def main() -> None:
    pl = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
    cal = load_all_matches(pl)

    # Report how many PL-team names failed to map (would silently lose congestion).
    mapped = set(cal["Team"])
    our = set(pl["HomeTeam"]) | set(pl["AwayTeam"])
    missing = our - mapped
    if missing:
        print("WARNING: PL teams with no calendar entries:", sorted(missing))

    recs = []
    for _, m in pl.iterrows():
        h = team_congestion(cal, m["HomeTeam"], m["Date"])
        a = team_congestion(cal, m["AwayTeam"], m["Date"])
        recs.append({
            "Season": m["Season"], "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
            "home_rest_days_all": h["rest_days_all"], "away_rest_days_all": a["rest_days_all"],
            "home_matches_last_14_all": h["matches_last_14_all"],
            "away_matches_last_14_all": a["matches_last_14_all"],
            "home_played_euro_last_5": h["played_euro_last_5"],
            "away_played_euro_last_5": a["played_euro_last_5"],
        })
    out = pd.DataFrame(recs)
    out["diff_rest_days_all"] = out["home_rest_days_all"] - out["away_rest_days_all"]
    out["diff_matches_last_14_all"] = out["home_matches_last_14_all"] - out["away_matches_last_14_all"]
    out["diff_played_euro_last_5"] = out["home_played_euro_last_5"] - out["away_played_euro_last_5"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    n_euro = (out[["home_played_euro_last_5", "away_played_euro_last_5"]].sum().sum())
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    print(f"Team-matches within 5 days of a European tie: {int(n_euro)}")
    print("Congestion (matches in last 14 days) distribution, home side:")
    print(out["home_matches_last_14_all"].value_counts().sort_index())


if __name__ == "__main__":
    main()
