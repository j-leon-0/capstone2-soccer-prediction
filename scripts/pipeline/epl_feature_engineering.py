from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/epl_combined_cleaned.csv")
OUTPUT_PATH = Path("data/processed/epl_features.csv")

REST_GAP_THRESHOLD_DAYS = 120
LOW_HISTORY_MATCH_THRESHOLD = 10

TARGET_MAP = {"H": 0, "D": 1, "A": 2}

ODDS_PRECEDENCE = {
    "home": ["AvgCH", "AvgH", "B365CH", "B365H"],
    "draw": ["AvgCD", "AvgD", "B365CD", "B365D"],
    "away": ["AvgCA", "AvgA", "B365CA", "B365A"],
}

ROLLING_BASE_COLUMNS = [
    "points",
    "win",
    "draw",
    "loss",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "sot_for",
    "sot_against",
    "sot_diff",
    "corners_for",
    "corners_against",
    "corner_diff",
    "fouls_for",
    "fouls_against",
    "yellow_for",
    "yellow_against",
    "red_for",
    "red_against",
]

VENUE_VALUE_COLUMNS = [
    "points",
    "win",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
]

SEASON_TO_DATE_VALUE_COLUMNS = [
    "points",
    "win",
    "draw",
    "loss",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shot_diff",
    "sot_for",
    "sot_against",
    "sot_diff",
    "corners_for",
    "corners_against",
    "corner_diff",
    "fouls_for",
    "fouls_against",
    "yellow_for",
    "yellow_against",
    "red_for",
    "red_against",
]

PAIRED_METRICS = [
    "overall_points_last_5",
    "overall_points_last_10",
    "overall_win_last_5",
    "overall_goal_diff_last_5",
    "overall_goal_diff_last_10",
    "overall_goals_for_last_5",
    "overall_goals_against_last_5",
    "overall_shot_diff_last_5",
    "overall_sot_diff_last_5",
    "overall_corner_diff_last_5",
    "venue_points_last_5",
    "venue_goal_diff_last_5",
    "venue_sot_for_last_5",
    "venue_sot_against_last_5",
    "team_matches_played_before",
    "days_since_last_match",
    "long_gap_since_last_match",
    "low_history_flag",
    "season_points_per_match",
    "season_goal_diff_per_match",
    "season_goals_for_per_match",
    "season_goals_against_per_match",
    "season_shots_for_per_match",
    "season_shots_against_per_match",
    "season_sot_for_per_match",
    "season_sot_against_per_match",
    "season_points",
    "season_goal_diff",
    "venue_season_points_per_match",
    "venue_season_goal_diff_per_match",
    "venue_season_goals_for_per_match",
    "venue_season_goals_against_per_match",
    "venue_season_sot_for_per_match",
    "venue_season_sot_against_per_match",
]

TABLE_METRICS = [
    "table_matches_played",
    "table_points",
    "table_wins",
    "table_draws",
    "table_losses",
    "table_goals_for",
    "table_goals_against",
    "table_goal_diff",
    "table_points_per_match",
    "table_position",
]


def load_matches(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    matches = df.copy()

    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches["Time_For_Sort"] = matches["Time"].fillna("00:00")
    matches["Kickoff"] = pd.to_datetime(
        matches["Date"].dt.strftime("%Y-%m-%d") + " " + matches["Time_For_Sort"],
        errors="coerce",
    )

    matches = matches.sort_values(["Kickoff", "HomeTeam", "AwayTeam"]).reset_index(
        drop=True
    )
    matches["MatchID"] = np.arange(len(matches))

    matches["target_result"] = matches["FTR"]
    matches["target_result_encoded"] = matches["FTR"].map(TARGET_MAP)
    matches["target_home_win"] = (matches["FTR"] == "H").astype(int)
    matches["target_draw"] = (matches["FTR"] == "D").astype(int)
    matches["target_away_win"] = (matches["FTR"] == "A").astype(int)

    matches = add_market_features(matches)

    return matches


def first_available_numeric(matches: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.Series(np.nan, index=matches.index, dtype="float64")

    for col in columns:
        if col in matches.columns:
            values = values.combine_first(pd.to_numeric(matches[col], errors="coerce"))

    return values


def add_market_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()

    matches["market_home_odds"] = first_available_numeric(
        matches, ODDS_PRECEDENCE["home"]
    )
    matches["market_draw_odds"] = first_available_numeric(
        matches, ODDS_PRECEDENCE["draw"]
    )
    matches["market_away_odds"] = first_available_numeric(
        matches, ODDS_PRECEDENCE["away"]
    )

    for result in ["home", "draw", "away"]:
        odds_col = f"market_{result}_odds"
        implied_col = f"market_{result}_implied_prob"
        matches[implied_col] = np.where(
            matches[odds_col] > 0,
            1 / matches[odds_col],
            np.nan,
        )

    implied_cols = [
        "market_home_implied_prob",
        "market_draw_implied_prob",
        "market_away_implied_prob",
    ]
    matches["market_overround"] = matches[implied_cols].sum(axis=1, min_count=3)

    for result in ["home", "draw", "away"]:
        implied_col = f"market_{result}_implied_prob"
        norm_col = f"market_{result}_norm_prob"
        matches[norm_col] = matches[implied_col] / matches["market_overround"]

    return matches


def build_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    home_rows = matches[
        [
            "MatchID",
            "Season",
            "Date",
            "Kickoff",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "HS",
            "AS",
            "HST",
            "AST",
            "HF",
            "AF",
            "HC",
            "AC",
            "HY",
            "AY",
            "HR",
            "AR",
        ]
    ].copy()

    home_rows = home_rows.rename(
        columns={
            "HomeTeam": "Team",
            "AwayTeam": "Opponent",
            "FTHG": "goals_for",
            "FTAG": "goals_against",
            "HS": "shots_for",
            "AS": "shots_against",
            "HST": "sot_for",
            "AST": "sot_against",
            "HF": "fouls_for",
            "AF": "fouls_against",
            "HC": "corners_for",
            "AC": "corners_against",
            "HY": "yellow_for",
            "AY": "yellow_against",
            "HR": "red_for",
            "AR": "red_against",
        }
    )
    home_rows["Venue"] = "Home"
    home_rows["result"] = home_rows["FTR"].map({"H": "W", "D": "D", "A": "L"})

    away_rows = matches[
        [
            "MatchID",
            "Season",
            "Date",
            "Kickoff",
            "AwayTeam",
            "HomeTeam",
            "FTAG",
            "FTHG",
            "FTR",
            "AS",
            "HS",
            "AST",
            "HST",
            "AF",
            "HF",
            "AC",
            "HC",
            "AY",
            "HY",
            "AR",
            "HR",
        ]
    ].copy()

    away_rows = away_rows.rename(
        columns={
            "AwayTeam": "Team",
            "HomeTeam": "Opponent",
            "FTAG": "goals_for",
            "FTHG": "goals_against",
            "AS": "shots_for",
            "HS": "shots_against",
            "AST": "sot_for",
            "HST": "sot_against",
            "AF": "fouls_for",
            "HF": "fouls_against",
            "AC": "corners_for",
            "HC": "corners_against",
            "AY": "yellow_for",
            "HY": "yellow_against",
            "AR": "red_for",
            "HR": "red_against",
        }
    )
    away_rows["Venue"] = "Away"
    away_rows["result"] = away_rows["FTR"].map({"H": "L", "D": "D", "A": "W"})

    team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
    team_matches = team_matches.sort_values(["Team", "Kickoff", "MatchID"]).reset_index(
        drop=True
    )

    team_matches["points"] = team_matches["result"].map({"W": 3, "D": 1, "L": 0})
    team_matches["win"] = (team_matches["result"] == "W").astype(int)
    team_matches["draw"] = (team_matches["result"] == "D").astype(int)
    team_matches["loss"] = (team_matches["result"] == "L").astype(int)
    team_matches["goal_diff"] = (
        team_matches["goals_for"] - team_matches["goals_against"]
    )
    team_matches["shot_diff"] = (
        team_matches["shots_for"] - team_matches["shots_against"]
    )
    team_matches["sot_diff"] = team_matches["sot_for"] - team_matches["sot_against"]
    team_matches["corner_diff"] = (
        team_matches["corners_for"] - team_matches["corners_against"]
    )

    return team_matches


def add_rolling_features(
    data: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    data = data.sort_values(group_cols + ["Kickoff", "MatchID"]).copy()
    grouped = data.groupby(group_cols, group_keys=False)

    for window in windows:
        for col in value_cols:
            feature_name = f"{prefix}_{col}_last_{window}"
            data[feature_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )

    return data


def add_rest_features(team_features: pd.DataFrame) -> pd.DataFrame:
    team_features = team_features.sort_values(["Team", "Kickoff", "MatchID"]).copy()

    team_features["team_matches_played_before"] = team_features.groupby(
        "Team"
    ).cumcount()
    team_features["days_since_last_match"] = (
        team_features.groupby("Team")["Kickoff"].diff().dt.days
    )
    team_features["long_gap_since_last_match"] = (
        team_features["days_since_last_match"] > REST_GAP_THRESHOLD_DAYS
    ).astype(int)
    team_features.loc[
        team_features["days_since_last_match"] > REST_GAP_THRESHOLD_DAYS,
        "days_since_last_match",
    ] = np.nan
    team_features["low_history_flag"] = (
        team_features["team_matches_played_before"] < LOW_HISTORY_MATCH_THRESHOLD
    ).astype(int)

    return team_features


def add_season_to_date_features(team_features: pd.DataFrame) -> pd.DataFrame:
    team_features = team_features.sort_values(
        ["Season", "Team", "Kickoff", "MatchID"]
    ).copy()

    season_group = team_features.groupby(["Season", "Team"], group_keys=False)
    team_features["season_matches_played_before"] = season_group.cumcount()

    for col in SEASON_TO_DATE_VALUE_COLUMNS:
        total_col = f"season_{col}"
        per_match_col = f"season_{col}_per_match"
        team_features[total_col] = season_group[col].cumsum() - team_features[col]
        team_features[per_match_col] = (
            team_features[total_col]
            / team_features["season_matches_played_before"].replace(0, np.nan)
        )

    team_features = team_features.sort_values(
        ["Season", "Team", "Venue", "Kickoff", "MatchID"]
    ).copy()
    venue_group = team_features.groupby(["Season", "Team", "Venue"], group_keys=False)
    team_features["venue_season_matches_played_before"] = venue_group.cumcount()

    for col in VENUE_VALUE_COLUMNS:
        total_col = f"venue_season_{col}"
        per_match_col = f"venue_season_{col}_per_match"
        team_features[total_col] = venue_group[col].cumsum() - team_features[col]
        team_features[per_match_col] = (
            team_features[total_col]
            / team_features["venue_season_matches_played_before"].replace(0, np.nan)
        )

    return team_features


def empty_standing() -> dict[str, float]:
    return {
        "table_matches_played": 0,
        "table_points": 0,
        "table_wins": 0,
        "table_draws": 0,
        "table_losses": 0,
        "table_goals_for": 0,
        "table_goals_against": 0,
        "table_goal_diff": 0,
        "table_points_per_match": np.nan,
        "table_position": np.nan,
    }


def ranked_standings(standings: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    rows = []

    for team, stats in standings.items():
        row = {"Team": team, **stats}
        row["table_goal_diff"] = row["table_goals_for"] - row["table_goals_against"]
        row["table_points_per_match"] = (
            row["table_points"] / row["table_matches_played"]
            if row["table_matches_played"] > 0
            else np.nan
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["table_points", "table_goal_diff", "table_goals_for", "Team"],
        ascending=[False, False, False, True],
    )
    table["table_position"] = np.arange(1, len(table) + 1)

    return table.set_index("Team").to_dict(orient="index")


def add_league_table_features(matches: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for season, season_matches in matches.groupby("Season", sort=True):
        teams = sorted(set(season_matches["HomeTeam"]).union(season_matches["AwayTeam"]))
        standings = {team: empty_standing() for team in teams}
        season_matches = season_matches.sort_values(["Kickoff", "HomeTeam", "AwayTeam"])

        for _, kickoff_matches in season_matches.groupby("Kickoff", sort=True):
            ranked = ranked_standings(standings)

            for _, match in kickoff_matches.iterrows():
                row = {"MatchID": match["MatchID"]}

                for side, team_col in [("home", "HomeTeam"), ("away", "AwayTeam")]:
                    team_stats = ranked[match[team_col]]
                    for metric in TABLE_METRICS:
                        row[f"{side}_{metric}"] = team_stats[metric]

                feature_rows.append(row)

            for _, match in kickoff_matches.iterrows():
                home_team = match["HomeTeam"]
                away_team = match["AwayTeam"]
                home_goals = match["FTHG"]
                away_goals = match["FTAG"]

                standings[home_team]["table_matches_played"] += 1
                standings[away_team]["table_matches_played"] += 1
                standings[home_team]["table_goals_for"] += home_goals
                standings[home_team]["table_goals_against"] += away_goals
                standings[away_team]["table_goals_for"] += away_goals
                standings[away_team]["table_goals_against"] += home_goals

                if match["FTR"] == "H":
                    standings[home_team]["table_points"] += 3
                    standings[home_team]["table_wins"] += 1
                    standings[away_team]["table_losses"] += 1
                elif match["FTR"] == "A":
                    standings[away_team]["table_points"] += 3
                    standings[away_team]["table_wins"] += 1
                    standings[home_team]["table_losses"] += 1
                else:
                    standings[home_team]["table_points"] += 1
                    standings[away_team]["table_points"] += 1
                    standings[home_team]["table_draws"] += 1
                    standings[away_team]["table_draws"] += 1

                standings[home_team]["table_goal_diff"] = (
                    standings[home_team]["table_goals_for"]
                    - standings[home_team]["table_goals_against"]
                )
                standings[away_team]["table_goal_diff"] = (
                    standings[away_team]["table_goals_for"]
                    - standings[away_team]["table_goals_against"]
                )

    table_features = pd.DataFrame(feature_rows)
    model_df = model_df.merge(table_features, on="MatchID", how="left")

    for metric in TABLE_METRICS:
        home_col = f"home_{metric}"
        away_col = f"away_{metric}"
        model_df[f"diff_{metric}"] = model_df[home_col] - model_df[away_col]

    return model_df


def merge_team_features(
    matches: pd.DataFrame, team_features: pd.DataFrame
) -> pd.DataFrame:
    feature_cols = [
        col
        for col in team_features.columns
        if col.startswith("overall_")
        or col.startswith("venue_")
        or col.startswith("season_")
        or col
        in [
            "MatchID",
            "Team",
            "Venue",
            "team_matches_played_before",
            "days_since_last_match",
            "long_gap_since_last_match",
            "low_history_flag",
        ]
    ]

    home_features = team_features[team_features["Venue"] == "Home"][
        feature_cols
    ].copy()
    away_features = team_features[team_features["Venue"] == "Away"][
        feature_cols
    ].copy()

    home_features = home_features.drop(columns=["Venue"]).add_prefix("home_")
    home_features = home_features.rename(
        columns={"home_MatchID": "MatchID", "home_Team": "HomeTeam"}
    )

    away_features = away_features.drop(columns=["Venue"]).add_prefix("away_")
    away_features = away_features.rename(
        columns={"away_MatchID": "MatchID", "away_Team": "AwayTeam"}
    )

    model_df = matches.merge(home_features, on=["MatchID", "HomeTeam"], how="left")
    model_df = model_df.merge(away_features, on=["MatchID", "AwayTeam"], how="left")

    return model_df


def add_differential_features(model_df: pd.DataFrame) -> pd.DataFrame:
    model_df = model_df.copy()

    for metric in PAIRED_METRICS:
        home_col = f"home_{metric}"
        away_col = f"away_{metric}"

        if home_col in model_df.columns and away_col in model_df.columns:
            model_df[f"diff_{metric}"] = model_df[home_col] - model_df[away_col]

    return model_df


def add_season_context_features(model_df: pd.DataFrame) -> pd.DataFrame:
    model_df = model_df.copy()

    model_df["match_month"] = model_df["Date"].dt.month
    model_df["match_year"] = model_df["Date"].dt.year
    model_df["season_match_number"] = model_df.groupby("Season").cumcount() + 1
    model_df["season_progress"] = model_df["season_match_number"] / model_df.groupby(
        "Season"
    )["MatchID"].transform("count")
    model_df["season_stage"] = pd.cut(
        model_df["season_progress"],
        bins=[0, 1 / 3, 2 / 3, 1],
        labels=["early", "middle", "late"],
        include_lowest=True,
    )

    return model_df


def add_referee_features(matches: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    referee_history = matches[
        ["MatchID", "Kickoff", "Referee", "HF", "AF", "HY", "AY", "HR", "AR"]
    ].copy()
    referee_history["total_fouls"] = referee_history["HF"] + referee_history["AF"]
    referee_history["total_yellow_cards"] = (
        referee_history["HY"] + referee_history["AY"]
    )
    referee_history["total_red_cards"] = referee_history["HR"] + referee_history["AR"]
    referee_history = referee_history.sort_values(["Referee", "Kickoff", "MatchID"])

    referee_history["referee_matches_before"] = referee_history.groupby(
        "Referee"
    ).cumcount()

    for col in ["total_fouls", "total_yellow_cards", "total_red_cards"]:
        referee_history[f"referee_{col}_last_20"] = referee_history.groupby("Referee")[
            col
        ].transform(lambda s: s.shift(1).rolling(window=20, min_periods=1).mean())

    referee_feature_cols = [
        "MatchID",
        "referee_matches_before",
        "referee_total_fouls_last_20",
        "referee_total_yellow_cards_last_20",
        "referee_total_red_cards_last_20",
    ]

    return model_df.merge(referee_history[referee_feature_cols], on="MatchID", how="left")


def get_model_feature_cols(model_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in model_df.columns
        if col.startswith("home_overall_")
        or col.startswith("away_overall_")
        or col.startswith("home_venue_")
        or col.startswith("away_venue_")
        or col.startswith("home_season_")
        or col.startswith("away_season_")
        or col.startswith("home_table_")
        or col.startswith("away_table_")
        or col.startswith("diff_")
        or col.startswith("referee_")
        or col.startswith("market_")
        or col
        in [
            "home_team_matches_played_before",
            "away_team_matches_played_before",
            "home_days_since_last_match",
            "away_days_since_last_match",
            "home_long_gap_since_last_match",
            "away_long_gap_since_last_match",
            "home_low_history_flag",
            "away_low_history_flag",
            "match_month",
            "season_progress",
        ]
    ]


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    matches = prepare_matches(df)
    team_matches = build_team_matches(matches)

    team_features = add_rolling_features(
        data=team_matches,
        group_cols=["Team"],
        value_cols=ROLLING_BASE_COLUMNS,
        windows=[5, 10],
        prefix="overall",
    )
    team_features = add_rolling_features(
        data=team_features,
        group_cols=["Team", "Venue"],
        value_cols=VENUE_VALUE_COLUMNS,
        windows=[5],
        prefix="venue",
    )
    team_features = add_rest_features(team_features)
    team_features = add_season_to_date_features(team_features)

    model_df = merge_team_features(matches, team_features)
    model_df = add_differential_features(model_df)
    model_df = add_league_table_features(matches, model_df)
    model_df = add_season_context_features(model_df)
    model_df = add_referee_features(matches, model_df)

    model_feature_cols = get_model_feature_cols(model_df)

    id_cols = ["MatchID", "Season", "Date", "Time", "HomeTeam", "AwayTeam"]
    target_cols = [
        "target_result",
        "target_result_encoded",
        "target_home_win",
        "target_draw",
        "target_away_win",
    ]
    evaluation_cols = ["FTHG", "FTAG", "FTR"]

    final_cols = id_cols + target_cols + evaluation_cols + model_feature_cols

    return model_df[final_cols].copy()


def print_summary(features_df: pd.DataFrame) -> None:
    missing = features_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(features_df)}")
    print(f"Total columns: {len(features_df.columns)}")
    print(f"Duplicate MatchID rows: {features_df['MatchID'].duplicated().sum()}")
    print(f"Seasons included: {features_df['Season'].nunique()}")
    print("\nTop missing values by column:")
    print(missing.head(20))


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = load_matches(INPUT_PATH)
    features_df = build_feature_dataset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print_summary(features_df)


if __name__ == "__main__":
    main()
