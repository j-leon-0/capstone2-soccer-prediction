"""Build leakage-free pre-match features from external sources (Elo + xG).

Inputs
------
* ``data/processed/epl_combined_cleaned.csv``     canonical match list (join spine)
* ``data/processed/external/epl_elo.csv``          ClubElo ratings (from fetch_clubelo.py)
* ``data/raw/external/understat/EPL_<season>.tsv``  per-match xG + Understat forecast

Output
------
* ``data/processed/external/epl_external_features.csv`` one row per match keyed by
  ``(Season, HomeTeam, AwayTeam)`` with Elo, xG-form and Understat-forecast columns.

Design notes
------------
* Every rolling/season xG stat is ``shift(1)`` before aggregation, so a match only
  ever sees strictly-prior matches -> no target leakage.
* xG form is built on a long "team-match" table (one row per team per match), the
  same shape the base pipeline uses, then pivoted back to home_/away_ columns.
* We also expose ``xg_performance`` (goals minus xG) form: teams that persistently
  out/under-score their xG are a well-known regression-to-mean signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MATCHES_PATH = Path("data/processed/epl_combined_cleaned.csv")
ELO_PATH = Path("data/processed/external/epl_elo.csv")
UNDERSTAT_DIR = Path("data/raw/external/understat")
SQUAD_VALUE_PATH = Path("data/processed/external/epl_squad_value.csv")
OUTPUT_PATH = Path("data/processed/external/epl_external_features.csv")

# Understat full club names -> our dataset's short names.
UNDERSTAT_TO_OURS = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

XG_WINDOWS = [5, 10]


def load_understat() -> pd.DataFrame:
    cols = ["date", "home", "away", "xg_h", "xg_a", "g_h", "g_a", "fc_w", "fc_d", "fc_l"]
    frames = []
    for f in sorted(UNDERSTAT_DIR.glob("EPL_*.tsv")):
        d = pd.read_csv(f, sep="\t", header=None, names=cols)
        d["Season"] = f.name.replace("EPL_", "").replace(".tsv", "")
        frames.append(d)
    us = pd.concat(frames, ignore_index=True)
    us["home"] = us["home"].replace(UNDERSTAT_TO_OURS)
    us["away"] = us["away"].replace(UNDERSTAT_TO_OURS)
    return us.rename(columns={"home": "HomeTeam", "away": "AwayTeam"})


def build_team_xg_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with xG for/against and pre-match rolling form."""
    home = matches[["Season", "Kickoff", "HomeTeam", "AwayTeam", "xg_h", "xg_a", "g_h", "g_a"]].copy()
    home = home.rename(columns={
        "HomeTeam": "Team", "AwayTeam": "Opp",
        "xg_h": "xg_for", "xg_a": "xg_against", "g_h": "g_for", "g_a": "g_against",
    })
    away = matches[["Season", "Kickoff", "AwayTeam", "HomeTeam", "xg_a", "xg_h", "g_a", "g_h"]].copy()
    away = away.rename(columns={
        "AwayTeam": "Team", "HomeTeam": "Opp",
        "xg_a": "xg_for", "xg_h": "xg_against", "g_a": "g_for", "g_h": "g_against",
    })
    tm = pd.concat([home, away], ignore_index=True)
    tm["xg_diff"] = tm["xg_for"] - tm["xg_against"]
    # xG over/under-performance: actual goals minus expected (finishing/variance signal).
    tm["xg_perf_for"] = tm["g_for"] - tm["xg_for"]
    tm["xg_perf_against"] = tm["g_against"] - tm["xg_against"]
    tm = tm.sort_values(["Team", "Kickoff"]).reset_index(drop=True)

    g = tm.groupby("Team", group_keys=False)
    for w in XG_WINDOWS:
        for col in ["xg_for", "xg_against", "xg_diff"]:
            tm[f"{col}_last_{w}"] = g[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    for col in ["xg_perf_for", "xg_perf_against"]:
        tm[f"{col}_last_10"] = g[col].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).mean()
        )

    # Season-to-date xG (strictly prior matches).
    sg = tm.groupby(["Season", "Team"], group_keys=False)
    tm["season_n"] = sg.cumcount()
    for col in ["xg_for", "xg_against", "xg_diff"]:
        csum = sg[col].cumsum() - tm[col]
        tm[f"season_{col}_pm"] = csum / tm["season_n"].replace(0, np.nan)

    feat_cols = (
        [f"{c}_last_{w}" for w in XG_WINDOWS for c in ["xg_for", "xg_against", "xg_diff"]]
        + ["xg_perf_for_last_10", "xg_perf_against_last_10"]
        + [f"season_{c}_pm" for c in ["xg_for", "xg_against", "xg_diff"]]
    )
    return tm[["Season", "Kickoff", "Team", "Opp"] + feat_cols], feat_cols


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
    matches["Kickoff"] = matches["Date"]

    us = load_understat()

    # Join xG onto the spine by (Season, HomeTeam, AwayTeam) -- unique per season.
    merged = matches.merge(
        us[["Season", "HomeTeam", "AwayTeam", "xg_h", "xg_a", "g_h", "g_a", "fc_w", "fc_d", "fc_l"]],
        on=["Season", "HomeTeam", "AwayTeam"], how="left",
    )
    miss = merged["xg_h"].isna().sum()
    print(f"Matches missing xG after join: {miss} / {len(merged)}")
    if miss:
        bad = merged[merged["xg_h"].isna()][["Season", "HomeTeam", "AwayTeam"]].head(20)
        print("Examples:\n", bad.to_string(index=False))

    team_xg, xg_feat_cols = build_team_xg_log(merged)

    home_xg = team_xg.rename(columns={"Team": "HomeTeam"}).add_prefix("home_xg_")
    home_xg = home_xg.rename(columns={
        "home_xg_Season": "Season", "home_xg_Kickoff": "Kickoff", "home_xg_HomeTeam": "HomeTeam"
    }).drop(columns=["home_xg_Opp"])
    away_xg = team_xg.rename(columns={"Team": "AwayTeam"}).add_prefix("away_xg_")
    away_xg = away_xg.rename(columns={
        "away_xg_Season": "Season", "away_xg_Kickoff": "Kickoff", "away_xg_AwayTeam": "AwayTeam"
    }).drop(columns=["away_xg_Opp"])

    out = merged[["Season", "Date", "HomeTeam", "AwayTeam"]].copy()
    out["Kickoff"] = merged["Kickoff"]
    out = out.merge(home_xg, on=["Season", "Kickoff", "HomeTeam"], how="left")
    out = out.merge(away_xg, on=["Season", "Kickoff", "AwayTeam"], how="left")

    # Home-minus-away xG differentials.
    for col in xg_feat_cols:
        out[f"diff_xg_{col}"] = out[f"home_xg_{col}"] - out[f"away_xg_{col}"]

    # Understat pre-match forecast (their own model probabilities).
    out["understat_home_prob"] = merged["fc_w"]
    out["understat_draw_prob"] = merged["fc_d"]
    out["understat_away_prob"] = merged["fc_a"] if "fc_a" in merged else merged["fc_l"]

    # Elo.
    elo = pd.read_csv(ELO_PATH, parse_dates=["Date"])
    out = out.merge(
        elo[["Season", "HomeTeam", "AwayTeam", "home_elo", "away_elo", "elo_diff", "elo_expected_home"]],
        on=["Season", "HomeTeam", "AwayTeam"], how="left",
    )

    # Season-start squad market value (Transfermarkt) -- fixed pre-season, no leakage.
    sv = pd.read_csv(SQUAD_VALUE_PATH)
    home_sv = sv.rename(columns={
        "Team": "HomeTeam", "squad_value_m": "home_squad_value_m",
        "value_rank": "home_value_rank", "value_share": "home_value_share"})
    away_sv = sv.rename(columns={
        "Team": "AwayTeam", "squad_value_m": "away_squad_value_m",
        "value_rank": "away_value_rank", "value_share": "away_value_share"})
    out = out.merge(home_sv, on=["Season", "HomeTeam"], how="left")
    out = out.merge(away_sv, on=["Season", "AwayTeam"], how="left")
    out["value_diff_m"] = out["home_squad_value_m"] - out["away_squad_value_m"]
    out["value_rank_diff"] = out["away_value_rank"] - out["home_value_rank"]  # +ve => home richer
    out["value_log_ratio"] = np.log(out["home_squad_value_m"] / out["away_squad_value_m"])

    out = out.drop(columns=["Kickoff"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows, {out.shape[1]} cols to {OUTPUT_PATH}")
    print("New feature columns:", [c for c in out.columns if c not in ("Season", "Date", "HomeTeam", "AwayTeam")][:8], "...")


if __name__ == "__main__":
    main()
