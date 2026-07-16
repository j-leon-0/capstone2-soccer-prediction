from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/epl_combined_cleaned.csv")
OUTPUT_PATH = Path("data/processed/epl_model_predictions.csv")
VALIDATION_METRICS_PATH = Path("data/processed/epl_model_validation_metrics.csv")
VALIDATION_PREDICTIONS_PATH = Path(
    "data/processed/epl_model_validation_predictions.csv"
)
FINAL_METRICS_PATH = Path("data/processed/epl_model_final_metrics.csv")
DECISION_POLICY_RESULTS_PATH = Path(
    "data/processed/epl_model_decision_policy_results.csv"
)

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_LABEL_MAP = {0: "H", 1: "D", 2: "A"}
TARGET_NAMES = ["home", "draw", "away"]
TARGET_LABELS = [0, 1, 2]
TARGET_RESULT_LABELS = ["H", "D", "A"]
TEST_SEASON = "2025_2026"
ROLLING_WINDOWS = [3, 5, 10]
REST_GAP_THRESHOLD_DAYS = 120
LOW_HISTORY_THRESHOLD = 3
H2H_LOOKBACK_DAYS = 365 * 3

ODDS_PRECEDENCE = {
    "home": ["AvgCH", "AvgH", "B365CH", "B365H"],
    "draw": ["AvgCD", "AvgD", "B365CD", "B365D"],
    "away": ["AvgCA", "AvgA", "B365CA", "B365A"],
}

RAW_RESULT_COLUMNS = {
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
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
}


@dataclass(frozen=True)
class Fold:
    train_through: str
    validate: str


VALIDATION_FOLDS = [
    Fold("2020_2021", "2021_2022"),
    Fold("2021_2022", "2022_2023"),
    Fold("2022_2023", "2023_2024"),
    Fold("2023_2024", "2024_2025"),
]


def parse_kickoff(df: pd.DataFrame) -> pd.DataFrame:
    matches = df.copy()
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    time_text = matches["Time"].astype("string").fillna("00:00")
    time_text = time_text.mask(time_text.str.lower().isin(["nan", "<na>"]), "00:00")
    matches["Kickoff"] = pd.to_datetime(
        matches["Date"].dt.strftime("%Y-%m-%d") + " " + time_text,
        errors="coerce",
    )
    matches["Kickoff"] = matches["Kickoff"].fillna(matches["Date"])
    matches = matches.sort_values(["Kickoff", "HomeTeam", "AwayTeam"]).reset_index(
        drop=True
    )
    matches["MatchID"] = np.arange(len(matches))
    matches["target"] = matches["FTR"].map(TARGET_MAP).astype(int)
    return matches


def first_available_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in columns:
        if col in df.columns:
            values = values.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return values


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    matches = df.copy()
    for result in TARGET_NAMES:
        matches[f"market_{result}_odds"] = first_available_numeric(
            matches, ODDS_PRECEDENCE[result]
        )
        matches[f"market_{result}_implied_prob"] = np.where(
            matches[f"market_{result}_odds"] > 0,
            1.0 / matches[f"market_{result}_odds"],
            np.nan,
        )

    implied_cols = [f"market_{result}_implied_prob" for result in TARGET_NAMES]
    matches["market_overround"] = matches[implied_cols].sum(axis=1, min_count=3)
    for result in TARGET_NAMES:
        matches[f"market_{result}_norm_prob"] = (
            matches[f"market_{result}_implied_prob"] / matches["market_overround"]
        )
    return matches


def proxy_xg(shots: float, shots_on_target: float, goals: float) -> float:
    """Proxy xG because the source files do not include true expected goals."""
    return 0.04 * float(shots) + 0.18 * float(shots_on_target) + 0.12 * float(goals)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def weighted_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def empty_table_row() -> dict[str, float]:
    return {
        "played": 0.0,
        "points": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "losses": 0.0,
        "gf": 0.0,
        "ga": 0.0,
        "gd": 0.0,
    }


def table_positions(
    season_table: dict[str, dict[str, float]], season_teams: set[str]
) -> dict[str, float]:
    rows = []
    for team in season_teams:
        stats = season_table.get(team, empty_table_row())
        rows.append(
            (
                team,
                stats["points"],
                stats["gd"],
                stats["gf"],
            )
        )
    rows = sorted(rows, key=lambda row: (-row[1], -row[2], -row[3], row[0]))
    return {team: float(position) for position, (team, *_rest) in enumerate(rows, 1)}


def summarize_history(
    history: list[dict[str, float]], prefix: str, venue: str | None = None
) -> dict[str, float]:
    rows = [row for row in history if venue is None or row["venue"] == venue]
    features: dict[str, float] = {}
    venue_label = "overall" if venue is None else venue

    for window in ROLLING_WINDOWS:
        tail = rows[-window:]
        label = f"{prefix}_{venue_label}_last_{window}"
        for metric in [
            "points",
            "goals_for",
            "goals_against",
            "goal_diff",
            "shots_for",
            "shots_against",
            "shot_diff",
            "sot_for",
            "sot_against",
            "sot_diff",
            "proxy_xg_for",
            "proxy_xg_against",
            "proxy_xg_diff",
        ]:
            features[f"{label}_{metric}_weighted"] = weighted_mean(
                [float(row[metric]) for row in tail]
            )

        features[f"{label}_win_rate"] = safe_divide(
            sum(float(row["win"]) for row in tail), len(tail)
        )
        features[f"{label}_draw_rate"] = safe_divide(
            sum(float(row["draw"]) for row in tail), len(tail)
        )
        features[f"{label}_goals_per_proxy_xg"] = safe_divide(
            sum(float(row["goals_for"]) for row in tail),
            sum(float(row["proxy_xg_for"]) for row in tail),
        )
        features[f"{label}_proxy_xg_per_shot"] = safe_divide(
            sum(float(row["proxy_xg_for"]) for row in tail),
            sum(float(row["shots_for"]) for row in tail),
        )

    return features


def team_state_features(
    team: str,
    season: str,
    kickoff: pd.Timestamp,
    venue: str,
    team_history: dict[str, list[dict[str, float]]],
    season_table: dict[str, dict[str, float]],
    positions: dict[str, float],
    season_team_count: int,
    prefix: str,
) -> dict[str, float]:
    history = team_history[team]
    table_row = season_table.get(team, empty_table_row())
    days_since_last = np.nan
    if history:
        days_since_last = (kickoff - pd.Timestamp(history[-1]["kickoff"])).days

    features = {
        f"{prefix}_matches_before": float(len(history)),
        f"{prefix}_low_history_flag": float(len(history) < LOW_HISTORY_THRESHOLD),
        f"{prefix}_days_since_last_match": float(days_since_last)
        if not pd.isna(days_since_last)
        else np.nan,
        f"{prefix}_long_gap_since_last_match": float(
            not pd.isna(days_since_last) and days_since_last > REST_GAP_THRESHOLD_DAYS
        ),
        f"{prefix}_table_position": positions.get(team, float(season_team_count)),
        f"{prefix}_table_played": table_row["played"],
        f"{prefix}_table_points": table_row["points"],
        f"{prefix}_table_wins": table_row["wins"],
        f"{prefix}_table_draws": table_row["draws"],
        f"{prefix}_table_losses": table_row["losses"],
        f"{prefix}_table_gf": table_row["gf"],
        f"{prefix}_table_ga": table_row["ga"],
        f"{prefix}_table_gd": table_row["gd"],
        f"{prefix}_table_ppg": safe_divide(table_row["points"], table_row["played"]),
    }
    features.update(summarize_history(history, prefix, venue=None))
    features.update(summarize_history(history, prefix, venue=venue))
    return features


def h2h_features(
    home_team: str,
    away_team: str,
    kickoff: pd.Timestamp,
    h2h_history: dict[tuple[str, str], list[dict[str, object]]],
) -> dict[str, float]:
    pair = tuple(sorted([home_team, away_team]))
    cutoff = kickoff - pd.Timedelta(days=H2H_LOOKBACK_DAYS)
    rows = [
        row
        for row in h2h_history[pair]
        if cutoff <= pd.Timestamp(row["kickoff"]) < kickoff
    ]
    home_team_wins = 0
    away_team_wins = 0
    draws = 0
    for row in rows:
        result = row["result"]
        if result == "D":
            draws += 1
        elif row["home_team"] == home_team and result == "H":
            home_team_wins += 1
        elif row["away_team"] == home_team and result == "A":
            home_team_wins += 1
        else:
            away_team_wins += 1

    n_rows = len(rows)
    return {
        "h2h_matches_3y": float(n_rows),
        "h2h_home_team_win_rate_3y": safe_divide(home_team_wins, n_rows),
        "h2h_away_team_win_rate_3y": safe_divide(away_team_wins, n_rows),
        "h2h_draw_rate_3y": safe_divide(draws, n_rows),
    }


def add_difference_features(features: dict[str, float]) -> dict[str, float]:
    diff_features = {}
    for key in list(features):
        if not key.startswith("home_"):
            continue
        away_key = "away_" + key.removeprefix("home_")
        if away_key in features:
            diff_key = "diff_" + key.removeprefix("home_")
            diff_features[diff_key] = features[key] - features[away_key]
    features.update(diff_features)
    return features


def result_points(result: str, is_home: bool) -> tuple[int, int, int, int]:
    if result == "D":
        return 1, 0, 1, 0
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3, 1, 0, 0
    return 0, 0, 0, 1


def match_team_row(match: pd.Series, is_home: bool) -> dict[str, float]:
    if is_home:
        points, win, draw, loss = result_points(match["FTR"], is_home=True)
        gf, ga = match["FTHG"], match["FTAG"]
        sf, sa = match["HS"], match["AS"]
        sotf, sota = match["HST"], match["AST"]
        venue = "home"
    else:
        points, win, draw, loss = result_points(match["FTR"], is_home=False)
        gf, ga = match["FTAG"], match["FTHG"]
        sf, sa = match["AS"], match["HS"]
        sotf, sota = match["AST"], match["HST"]
        venue = "away"

    xgf = proxy_xg(sf, sotf, gf)
    xga = proxy_xg(sa, sota, ga)
    return {
        "kickoff": match["Kickoff"],
        "season": match["Season"],
        "venue": venue,
        "points": float(points),
        "win": float(win),
        "draw": float(draw),
        "loss": float(loss),
        "goals_for": float(gf),
        "goals_against": float(ga),
        "goal_diff": float(gf - ga),
        "shots_for": float(sf),
        "shots_against": float(sa),
        "shot_diff": float(sf - sa),
        "sot_for": float(sotf),
        "sot_against": float(sota),
        "sot_diff": float(sotf - sota),
        "proxy_xg_for": float(xgf),
        "proxy_xg_against": float(xga),
        "proxy_xg_diff": float(xgf - xga),
    }


def update_table_for_match(
    season_table: dict[str, dict[str, float]], match: pd.Series
) -> None:
    home_points, home_win, home_draw, home_loss = result_points(match["FTR"], True)
    away_points, away_win, away_draw, away_loss = result_points(match["FTR"], False)
    updates = [
        (
            match["HomeTeam"],
            home_points,
            home_win,
            home_draw,
            home_loss,
            match["FTHG"],
            match["FTAG"],
        ),
        (
            match["AwayTeam"],
            away_points,
            away_win,
            away_draw,
            away_loss,
            match["FTAG"],
            match["FTHG"],
        ),
    ]
    for team, points, wins, draws, losses, gf, ga in updates:
        row = season_table.setdefault(team, empty_table_row())
        row["played"] += 1
        row["points"] += points
        row["wins"] += wins
        row["draws"] += draws
        row["losses"] += losses
        row["gf"] += gf
        row["ga"] += ga
        row["gd"] = row["gf"] - row["ga"]


def build_dynamic_features(matches: pd.DataFrame) -> pd.DataFrame:
    team_history: dict[str, list[dict[str, float]]] = defaultdict(list)
    h2h_history: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    season_tables: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    season_teams = {
        season: set(group["HomeTeam"]).union(set(group["AwayTeam"]))
        for season, group in matches.groupby("Season", sort=False)
    }

    feature_rows = []
    # Matches with the exact same kickoff are scored before any of those results
    # update histories. This prevents same-time leakage on crowded matchdays.
    for _kickoff, kickoff_group in matches.groupby("Kickoff", sort=False):
        pending_rows = []
        for _, match in kickoff_group.iterrows():
            season = match["Season"]
            kickoff = match["Kickoff"]
            positions = table_positions(
                season_tables[season], season_teams.get(season, set())
            )
            season_team_count = len(season_teams.get(season, set())) or 20

            features = {
                "MatchID": match["MatchID"],
                "Season": season,
                "Date": match["Date"],
                "Kickoff": kickoff,
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "FTR": match["FTR"],
                "target": match["target"],
            }

            features.update(
                team_state_features(
                    match["HomeTeam"],
                    season,
                    kickoff,
                    venue="home",
                    team_history=team_history,
                    season_table=season_tables[season],
                    positions=positions,
                    season_team_count=season_team_count,
                    prefix="home",
                )
            )
            features.update(
                team_state_features(
                    match["AwayTeam"],
                    season,
                    kickoff,
                    venue="away",
                    team_history=team_history,
                    season_table=season_tables[season],
                    positions=positions,
                    season_team_count=season_team_count,
                    prefix="away",
                )
            )
            features.update(
                h2h_features(
                    match["HomeTeam"], match["AwayTeam"], kickoff, h2h_history
                )
            )
            features = add_difference_features(features)
            pending_rows.append(features)

        feature_rows.extend(pending_rows)

        for _, match in kickoff_group.iterrows():
            team_history[match["HomeTeam"]].append(match_team_row(match, is_home=True))
            team_history[match["AwayTeam"]].append(match_team_row(match, is_home=False))
            update_table_for_match(season_tables[match["Season"]], match)
            pair = tuple(sorted([match["HomeTeam"], match["AwayTeam"]]))
            h2h_history[pair].append(
                {
                    "kickoff": match["Kickoff"],
                    "home_team": match["HomeTeam"],
                    "away_team": match["AwayTeam"],
                    "result": match["FTR"],
                }
            )

    dynamic = pd.DataFrame(feature_rows).sort_values("MatchID").reset_index(drop=True)
    return dynamic


def build_modeling_table(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    if len(raw) != 3800:
        warnings.warn(f"Expected 3,800 rows, found {len(raw):,}.", stacklevel=2)
    matches = add_market_features(parse_kickoff(raw))
    dynamic = build_dynamic_features(matches)
    market_cols = [
        "MatchID",
        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",
        "market_home_norm_prob",
        "market_draw_norm_prob",
        "market_away_norm_prob",
        "market_overround",
    ]
    modeling = dynamic.merge(matches[market_cols], on="MatchID", how="left")
    assert not modeling["MatchID"].duplicated().any(), "Duplicate MatchID values found."
    return modeling


def feature_columns(modeling: pd.DataFrame) -> list[str]:
    blocked = {
        "MatchID",
        "Season",
        "Date",
        "Kickoff",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "target",
    }.union(RAW_RESULT_COLUMNS)
    return [
        col
        for col in modeling.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(modeling[col])
    ]


def split_train_calibration(
    modeling: pd.DataFrame, train_mask: pd.Series
) -> tuple[pd.Series, pd.Series]:
    train_rows = modeling.loc[train_mask].copy()
    ordered_seasons = sorted(train_rows["Season"].unique())
    calibration_season = ordered_seasons[-1]
    model_mask = train_mask & (modeling["Season"] != calibration_season)
    calibration_mask = train_mask & (modeling["Season"] == calibration_season)

    if modeling.loc[model_mask, "target"].nunique() < 3:
        cutoff = train_rows["Kickoff"].quantile(0.8)
        model_mask = train_mask & (modeling["Kickoff"] < cutoff)
        calibration_mask = train_mask & (modeling["Kickoff"] >= cutoff)

    return model_mask, calibration_mask


def make_estimators(random_state: int = 42) -> dict[str, Pipeline]:
    class_weights = {0: 1.0, 1: 1.45, 2: 1.0}
    return {
        "logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=class_weights,
                        C=0.35,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight=class_weights,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        num_class=3,
                        n_estimators=450,
                        learning_rate=0.025,
                        num_leaves=15,
                        max_depth=4,
                        min_child_samples=35,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_alpha=0.4,
                        reg_lambda=1.2,
                        class_weight=class_weights,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def fit_calibrated_model(
    estimator: Pipeline,
    X_model: pd.DataFrame,
    y_model: pd.Series,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibratedClassifierCV:
    estimator.fit(X_model, y_model)
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(estimator),
        method="sigmoid",
    )
    calibrator.fit(X_calibration, y_calibration)
    return calibrator


def ranked_probability_score(y_true: Iterable[int], probabilities: np.ndarray) -> float:
    y = np.asarray(list(y_true), dtype=int)
    one_hot = np.eye(len(TARGET_LABELS))[y]
    cumulative_prob = np.cumsum(probabilities, axis=1)
    cumulative_actual = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cumulative_prob - cumulative_actual) ** 2, axis=1) / 2))


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, name: str
) -> dict[str, float | str]:
    y_true_array = np.asarray(y_true, dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true_array, predictions),
        "macro_f1": f1_score(
            y_true_array, predictions, average="macro", labels=TARGET_LABELS
        ),
        "draw_recall": recall_score(
            y_true_array,
            predictions,
            labels=[TARGET_MAP["D"]],
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(y_true_array, probabilities, labels=TARGET_LABELS),
        "rps": ranked_probability_score(y_true_array, probabilities),
        "avg_confidence": float(np.max(probabilities, axis=1).mean()),
    }


def validation_fold_masks(modeling: pd.DataFrame, fold: Fold) -> tuple[pd.Series, pd.Series]:
    train_mask = modeling["Season"] <= fold.train_through
    validate_mask = modeling["Season"] == fold.validate
    assert modeling.loc[train_mask, "Kickoff"].max() < modeling.loc[
        validate_mask, "Kickoff"
    ].min(), f"Chronological leakage in fold {fold}"
    return train_mask, validate_mask


def train_validation_stack(
    modeling: pd.DataFrame, features: list[str]
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_blocks = []
    metric_rows = []

    for fold_number, fold in enumerate(VALIDATION_FOLDS, start=1):
        train_mask, validate_mask = validation_fold_masks(modeling, fold)
        model_mask, calibration_mask = split_train_calibration(modeling, train_mask)

        X_model = modeling.loc[model_mask, features]
        y_model = modeling.loc[model_mask, "target"]
        X_calibration = modeling.loc[calibration_mask, features]
        y_calibration = modeling.loc[calibration_mask, "target"]
        X_validation = modeling.loc[validate_mask, features]
        y_validation = modeling.loc[validate_mask, "target"]

        fold_prob_blocks = []
        for model_name, estimator in make_estimators(random_state=42 + fold_number).items():
            calibrated = fit_calibrated_model(
                estimator, X_model, y_model, X_calibration, y_calibration
            )
            probabilities = calibrated.predict_proba(X_validation)
            fold_prob_blocks.append(probabilities)
            metrics = evaluate_probabilities(
                y_validation, probabilities, f"{model_name}_fold_{fold.validate}"
            )
            metrics["fold"] = fold.validate
            metric_rows.append(metrics)

        stacked_features = np.hstack(fold_prob_blocks)
        block = pd.DataFrame(
            stacked_features,
            index=modeling.loc[validate_mask].index,
            columns=base_stack_feature_names(),
        )
        for target in TARGET_NAMES:
            block[f"market_prob_{target}"] = modeling.loc[
                validate_mask, f"market_{target}_norm_prob"
            ].values
        block["target"] = y_validation.values
        block["Season"] = fold.validate
        oof_blocks.append(block)

    oof = pd.concat(oof_blocks).sort_index()
    meta_model = LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight={0: 1.0, 1: 1.2, 2: 1.0},
        random_state=42,
    )
    meta_model.fit(oof[stack_feature_names()], oof["target"])
    oof_probabilities = meta_model.predict_proba(oof[stack_feature_names()])
    metric_rows.append(
        {
            **evaluate_probabilities(oof["target"], oof_probabilities, "stacked_oof"),
            "fold": "2021_2022_to_2024_2025",
        }
    )
    validation_predictions = make_prediction_frame(
        modeling.loc[oof.index],
        oof_probabilities,
        bookmaker_probabilities(modeling.loc[oof.index]),
    )
    return meta_model, oof, pd.DataFrame(metric_rows), validation_predictions


def base_stack_feature_names() -> list[str]:
    return [
        f"{model_name}_prob_{target}"
        for model_name in ["logistic", "random_forest", "lightgbm"]
        for target in TARGET_NAMES
    ]


def stack_feature_names() -> list[str]:
    return base_stack_feature_names() + [
        f"market_prob_{target}" for target in TARGET_NAMES
    ]


def fit_final_base_models(
    modeling: pd.DataFrame, features: list[str]
) -> dict[str, CalibratedClassifierCV]:
    train_mask = modeling["Season"] < TEST_SEASON
    test_mask = modeling["Season"] == TEST_SEASON
    assert modeling.loc[train_mask, "Kickoff"].max() < modeling.loc[
        test_mask, "Kickoff"
    ].min(), "Final test season is not chronologically after training data."
    model_mask, calibration_mask = split_train_calibration(modeling, train_mask)
    fitted = {}
    for model_name, estimator in make_estimators(random_state=99).items():
        fitted[model_name] = fit_calibrated_model(
            estimator,
            modeling.loc[model_mask, features],
            modeling.loc[model_mask, "target"],
            modeling.loc[calibration_mask, features],
            modeling.loc[calibration_mask, "target"],
        )
    return fitted


def bookmaker_probabilities(modeling: pd.DataFrame) -> np.ndarray:
    probs = modeling[
        ["market_home_norm_prob", "market_draw_norm_prob", "market_away_norm_prob"]
    ].to_numpy(dtype=float)
    return probs


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def argmax_labels(probabilities: np.ndarray) -> np.ndarray:
    return np.array([TARGET_LABEL_MAP[idx] for idx in np.argmax(probabilities, axis=1)])


def make_prediction_frame(
    rows: pd.DataFrame,
    stacked_probabilities: np.ndarray,
    bookmaker_probs: np.ndarray,
) -> pd.DataFrame:
    predictions = rows[
        ["MatchID", "Season", "Date", "HomeTeam", "AwayTeam", "FTR"]
    ].copy()
    for idx, target in enumerate(TARGET_NAMES):
        predictions[f"model_prob_{target}"] = stacked_probabilities[:, idx]
        predictions[f"bookmaker_prob_{target}"] = bookmaker_probs[:, idx]
        predictions[f"edge_{target}"] = (
            predictions[f"model_prob_{target}"] - predictions[f"bookmaker_prob_{target}"]
        )

    predictions["model_prediction"] = argmax_labels(stacked_probabilities)
    predictions["bookmaker_implied_favorite"] = argmax_labels(bookmaker_probs)
    return predictions


def save_predictions(
    test_rows: pd.DataFrame,
    stacked_probabilities: np.ndarray,
    bookmaker_probs: np.ndarray,
    selected_policy: pd.Series | None = None,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    predictions = make_prediction_frame(test_rows, stacked_probabilities, bookmaker_probs)
    if selected_policy is not None:
        policy_prediction = apply_selected_policy(
            predictions,
            selected_policy,
        )
        predictions["policy_prediction"] = policy_prediction
        predictions["policy_name"] = selected_policy["policy_name"]
    predictions.to_csv(output_path, index=False)
    return predictions


def prediction_probabilities(predictions: pd.DataFrame) -> np.ndarray:
    return predictions[
        ["model_prob_home", "model_prob_draw", "model_prob_away"]
    ].to_numpy(dtype=float)


def apply_threshold_policy(
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> np.ndarray:
    thresholds_array = np.asarray(list(thresholds), dtype=float)
    passes_threshold = probabilities >= thresholds_array
    policy_indexes = np.argmax(np.where(passes_threshold, probabilities, -1.0), axis=1)
    fallback_mask = ~passes_threshold.any(axis=1)
    policy_indexes[fallback_mask] = np.argmax(probabilities[fallback_mask], axis=1)
    return np.array([TARGET_RESULT_LABELS[idx] for idx in policy_indexes])


def apply_bias_policy(probabilities: np.ndarray, biases: Iterable[float]) -> np.ndarray:
    biases_array = np.asarray(list(biases), dtype=float)
    policy_indexes = np.argmax(probabilities + biases_array, axis=1)
    return np.array([TARGET_RESULT_LABELS[idx] for idx in policy_indexes])


def apply_selected_policy(
    predictions: pd.DataFrame,
    selected_policy: pd.Series,
) -> np.ndarray:
    probabilities = prediction_probabilities(predictions)
    policy_type = selected_policy["policy_type"]
    if policy_type == "argmax":
        return argmax_labels(probabilities)
    if policy_type == "threshold":
        return apply_threshold_policy(
            probabilities,
            [
                selected_policy["threshold_home"],
                selected_policy["threshold_draw"],
                selected_policy["threshold_away"],
            ],
        )
    if policy_type == "bias":
        return apply_bias_policy(
            probabilities,
            [
                selected_policy["bias_home"],
                selected_policy["bias_draw"],
                selected_policy["bias_away"],
            ],
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


def evaluate_label_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    policy_name: str,
) -> dict[str, float | int | str]:
    return {
        "model": policy_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=TARGET_RESULT_LABELS,
            average="macro",
            zero_division=0,
        ),
        "draw_precision": precision_score(
            y_true,
            y_pred,
            labels=["D"],
            average="macro",
            zero_division=0,
        ),
        "draw_recall": recall_score(
            y_true,
            y_pred,
            labels=["D"],
            average="macro",
            zero_division=0,
        ),
        "draw_f1": f1_score(
            y_true,
            y_pred,
            labels=["D"],
            average="macro",
            zero_division=0,
        ),
        "home_picks": int(np.sum(y_pred == "H")),
        "draw_picks": int(np.sum(y_pred == "D")),
        "away_picks": int(np.sum(y_pred == "A")),
    }


def tune_decision_policies(validation_predictions: pd.DataFrame) -> pd.DataFrame:
    probabilities = prediction_probabilities(validation_predictions)
    y_true = validation_predictions["FTR"].to_numpy()
    rows = []

    argmax_pred = argmax_labels(probabilities)
    rows.append(
        {
            **evaluate_label_predictions(y_true, argmax_pred, "argmax"),
            "policy_type": "argmax",
            "policy_name": "argmax",
            "threshold_home": np.nan,
            "threshold_draw": np.nan,
            "threshold_away": np.nan,
            "bias_home": 0.0,
            "bias_draw": 0.0,
            "bias_away": 0.0,
            "complexity": 0.0,
        }
    )

    threshold_values = np.round(np.arange(0.24, 0.501, 0.02), 3)
    for home_threshold in threshold_values:
        for draw_threshold in threshold_values:
            for away_threshold in threshold_values:
                policy_pred = apply_threshold_policy(
                    probabilities,
                    [home_threshold, draw_threshold, away_threshold],
                )
                rows.append(
                    {
                        **evaluate_label_predictions(
                            y_true,
                            policy_pred,
                            "threshold_policy",
                        ),
                        "policy_type": "threshold",
                        "policy_name": (
                            "threshold_"
                            f"h{home_threshold:.3f}_"
                            f"d{draw_threshold:.3f}_"
                            f"a{away_threshold:.3f}"
                        ),
                        "threshold_home": home_threshold,
                        "threshold_draw": draw_threshold,
                        "threshold_away": away_threshold,
                        "bias_home": 0.0,
                        "bias_draw": 0.0,
                        "bias_away": 0.0,
                        "complexity": float(
                            home_threshold + draw_threshold + away_threshold
                        ),
                    }
                )

    # Bias policies are identifiable only up to a shared constant, so home is
    # anchored at 0 and draw/away are searched relative to home.
    bias_values = np.round(np.arange(-0.08, 0.081, 0.005), 3)
    for draw_bias in bias_values:
        for away_bias in bias_values:
            policy_pred = apply_bias_policy(probabilities, [0.0, draw_bias, away_bias])
            rows.append(
                {
                    **evaluate_label_predictions(
                        y_true,
                        policy_pred,
                        "bias_policy",
                    ),
                    "policy_type": "bias",
                    "policy_name": f"bias_h0.000_d{draw_bias:.3f}_a{away_bias:.3f}",
                    "threshold_home": np.nan,
                    "threshold_draw": np.nan,
                    "threshold_away": np.nan,
                    "bias_home": 0.0,
                    "bias_draw": draw_bias,
                    "bias_away": away_bias,
                    "complexity": float(abs(draw_bias) + abs(away_bias)),
                }
            )

    results = pd.DataFrame(rows)
    return results.sort_values(
        ["accuracy", "macro_f1", "complexity"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def print_metrics(title: str, metrics: pd.DataFrame) -> None:
    print(f"\n{title}")
    display_cols = [
        "model",
        "fold",
        "accuracy",
        "macro_f1",
        "draw_recall",
        "log_loss",
        "rps",
        "avg_confidence",
    ]
    available = [col for col in display_cols if col in metrics.columns]
    print(metrics[available].round(4).to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )
    modeling = build_modeling_table(INPUT_PATH)
    features = feature_columns(modeling)

    assert len(modeling) == len(pd.read_csv(INPUT_PATH)), "Row count changed."
    assert not set(features).intersection(RAW_RESULT_COLUMNS), "Result columns leaked."
    print(f"Built leakage-safe modeling table: {modeling.shape[0]:,} rows")
    print(f"Using {len(features):,} numeric pre-match features")

    meta_model, oof, validation_metrics, validation_predictions = train_validation_stack(
        modeling, features
    )
    print_metrics("Chronological validation metrics", validation_metrics)
    validation_metrics.to_csv(VALIDATION_METRICS_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    decision_policy_results = tune_decision_policies(validation_predictions)
    selected_policy = decision_policy_results.iloc[0]
    decision_policy_results["selected"] = False
    decision_policy_results.loc[0, "selected"] = True
    decision_policy_results.to_csv(DECISION_POLICY_RESULTS_PATH, index=False)
    print("\nSelected validation-tuned decision policy")
    print(
        selected_policy[
            [
                "policy_name",
                "policy_type",
                "accuracy",
                "macro_f1",
                "draw_precision",
                "draw_recall",
                "draw_f1",
                "home_picks",
                "draw_picks",
                "away_picks",
            ]
        ].to_string()
    )

    final_models = fit_final_base_models(modeling, features)
    test_mask = modeling["Season"] == TEST_SEASON
    test_rows = modeling.loc[test_mask].copy()
    X_test = test_rows[features]
    y_test = test_rows["target"]

    base_probability_blocks = []
    final_metric_rows = []
    for model_name in ["logistic", "random_forest", "lightgbm"]:
        probabilities = normalize_probabilities(final_models[model_name].predict_proba(X_test))
        base_probability_blocks.append(probabilities)
        final_metric_rows.append(
            evaluate_probabilities(y_test, probabilities, model_name)
        )

    stacked_test_features = pd.DataFrame(
        np.hstack(base_probability_blocks),
        columns=base_stack_feature_names(),
        index=test_rows.index,
    )
    for target in TARGET_NAMES:
        stacked_test_features[f"market_prob_{target}"] = test_rows[
            f"market_{target}_norm_prob"
        ].values
    stacked_probabilities = normalize_probabilities(
        meta_model.predict_proba(stacked_test_features)
    )
    final_metric_rows.append(
        evaluate_probabilities(y_test, stacked_probabilities, "stacked_ensemble")
    )

    bookmaker_probs = normalize_probabilities(bookmaker_probabilities(test_rows))
    assert np.allclose(stacked_probabilities.sum(axis=1), 1.0)
    assert np.allclose(bookmaker_probs.sum(axis=1), 1.0)
    final_metric_rows.append(
        evaluate_probabilities(y_test, bookmaker_probs, "bookmaker_baseline")
    )
    selected_policy_predictions = apply_selected_policy(
        make_prediction_frame(test_rows, stacked_probabilities, bookmaker_probs),
        selected_policy,
    )
    selected_policy_metrics = evaluate_label_predictions(
        test_rows["FTR"].to_numpy(),
        selected_policy_predictions,
        "stacked_ensemble_policy",
    )
    selected_policy_metrics["log_loss"] = log_loss(
        y_test, stacked_probabilities, labels=TARGET_LABELS
    )
    selected_policy_metrics["rps"] = ranked_probability_score(y_test, stacked_probabilities)
    selected_policy_metrics["avg_confidence"] = float(
        np.max(stacked_probabilities, axis=1).mean()
    )
    selected_policy_metrics["policy_name"] = selected_policy["policy_name"]
    final_metric_rows.append(selected_policy_metrics)
    final_metrics = pd.DataFrame(final_metric_rows)
    final_metrics["fold"] = TEST_SEASON
    print_metrics(f"Final untouched test metrics ({TEST_SEASON})", final_metrics)
    final_metrics.to_csv(FINAL_METRICS_PATH, index=False)

    predictions = save_predictions(
        test_rows,
        stacked_probabilities,
        bookmaker_probs,
        selected_policy=selected_policy,
    )
    print(f"\nSaved predictions: {OUTPUT_PATH} ({len(predictions):,} rows)")
    print(f"Saved validation metrics: {VALIDATION_METRICS_PATH}")
    print(f"Saved validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Saved decision policy results: {DECISION_POLICY_RESULTS_PATH}")
    print(f"Saved final metrics: {FINAL_METRICS_PATH}")

    print("\nStacked ensemble confusion matrix")
    cm = confusion_matrix(
        y_test,
        np.argmax(stacked_probabilities, axis=1),
        labels=TARGET_LABELS,
    )
    print(pd.DataFrame(cm, index=["actual_H", "actual_D", "actual_A"], columns=["pred_H", "pred_D", "pred_A"]).to_string())

    print("\nStacked ensemble classification report")
    print(
        classification_report(
            y_test,
            np.argmax(stacked_probabilities, axis=1),
            labels=TARGET_LABELS,
            target_names=["home_win", "draw", "away_win"],
            zero_division=0,
        )
    )

    print("\nSelected policy confusion matrix")
    policy_cm = confusion_matrix(
        test_rows["FTR"].to_numpy(),
        predictions["policy_prediction"].to_numpy(),
        labels=TARGET_RESULT_LABELS,
    )
    print(
        pd.DataFrame(
            policy_cm,
            index=["actual_H", "actual_D", "actual_A"],
            columns=["pred_H", "pred_D", "pred_A"],
        ).to_string()
    )

    print("\nSelected policy classification report")
    print(
        classification_report(
            test_rows["FTR"].to_numpy(),
            predictions["policy_prediction"].to_numpy(),
            labels=TARGET_RESULT_LABELS,
            target_names=["home_win", "draw", "away_win"],
            zero_division=0,
        )
    )

    stacked_log_loss = final_metrics.loc[
        final_metrics["model"] == "stacked_ensemble", "log_loss"
    ].iloc[0]
    market_log_loss = final_metrics.loc[
        final_metrics["model"] == "bookmaker_baseline", "log_loss"
    ].iloc[0]
    delta = stacked_log_loss - market_log_loss
    print(
        "\nModel vs market log loss delta "
        f"(negative means model is better): {delta:.4f}"
    )


if __name__ == "__main__":
    main()
