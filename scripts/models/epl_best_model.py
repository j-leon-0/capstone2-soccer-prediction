"""Best-effort EPL 1X2 model: sharp-market-anchored calibrated ensemble.

Rationale (see project memories): the betting market is the accuracy ceiling
(~0.50 on the 2025-26 test season) and no legitimate feature set beats it by much.
So the design does not fight the market -- it *anchors* to the sharpest signal
(closing odds), adds independent signals (line movement, ClubElo, real Understat xG
form, squad value, current form), builds a calibrated ensemble, and then blends the
ensemble probabilities toward the market with a weight tuned on validation. Argmax of
the blended probabilities maximises plain accuracy.

Protocol: train <= 2023_2024, tune blend weight + choices on 2024_2025 validation,
report once on 2025_2026 test.

Outputs (for the dashboard):
* data/processed/epl_best_model_predictions.csv   test-season predictions + probs
* data/processed/epl_best_model_metrics.csv        metric table (model vs market)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RAW = Path("data/processed/epl_combined_cleaned.csv")
FEATURES = Path("data/processed/epl_features.csv")
EXTERNAL = Path("data/processed/external/epl_external_features.csv")
PRED_OUT = Path("data/processed/epl_best_model_predictions.csv")
METRICS_OUT = Path("data/processed/epl_best_model_metrics.csv")
BINARY_METRICS_OUT = Path("data/processed/epl_best_model_binary_metrics.csv")

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
CLASSES = [0, 1, 2]
LABELS = {0: "Home", 1: "Draw", 2: "Away"}


# --------------------------------------------------------------------------- #
# Market features: opening (Bet365) vs closing (avg), and the drift between.   #
# --------------------------------------------------------------------------- #
def implied(df, h, d, a):
    """Normalised (overround-removed) implied probs from three odds columns."""
    inv = pd.DataFrame({
        0: 1 / pd.to_numeric(df[h], errors="coerce"),
        1: 1 / pd.to_numeric(df[d], errors="coerce"),
        2: 1 / pd.to_numeric(df[a], errors="coerce"),
    })
    return inv.div(inv.sum(axis=1), axis=0)


def build_market(raw: pd.DataFrame) -> pd.DataFrame:
    m = raw.copy()
    # Opening: Bet365 (present all seasons). Closing: average closing, else B365 closing.
    open_p = implied(m, "B365H", "B365D", "B365A")
    close_p = implied(m, "AvgCH", "AvgCD", "AvgCA")
    b365c = implied(m, "B365CH", "B365CD", "B365CA")
    close_p = close_p.fillna(b365c).fillna(open_p)  # sharpest available per row

    out = m[["Season", "Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    for i, name in LABELS.items():
        out[f"mkt_{name.lower()}"] = close_p[i]
    # overround from raw closing (or opening) odds
    inv_close = pd.DataFrame({
        0: 1 / pd.to_numeric(m["AvgCH"], errors="coerce").fillna(pd.to_numeric(m["B365H"], errors="coerce")),
        1: 1 / pd.to_numeric(m["AvgCD"], errors="coerce").fillna(pd.to_numeric(m["B365D"], errors="coerce")),
        2: 1 / pd.to_numeric(m["AvgCA"], errors="coerce").fillna(pd.to_numeric(m["B365A"], errors="coerce")),
    })
    out["mkt_overround"] = inv_close.sum(axis=1)
    # Line movement: closing minus opening normalised prob (sharp-money signal, 2019+).
    drift = close_p - open_p
    for i, name in LABELS.items():
        out[f"drift_{name.lower()}"] = drift[i]
    out["target"] = out["FTR"].map(TARGET_MAP)
    return out


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def base_estimators(seed=42):
    num = lambda est: Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()), ("m", est)])
    tree = lambda est: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", est)])
    return {
        "lgbm": tree(LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                    min_child_samples=40, subsample=0.85, colsample_bytree=0.7,
                                    reg_lambda=2.0, random_state=seed, verbosity=-1, n_jobs=-1)),
        "rf": tree(RandomForestClassifier(n_estimators=500, min_samples_leaf=15,
                                          max_features="sqrt", random_state=seed, n_jobs=-1)),
        "hgb": tree(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                                   max_iter=400, l2_regularization=1.0,
                                                   random_state=seed)),
        "logit": num(LogisticRegression(C=0.5, max_iter=5000, random_state=seed)),
    }


def aligned_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = p[:, j]
    return out / out.sum(axis=1, keepdims=True)


def evaluate(y, proba, name):
    pred = proba.argmax(1)
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=CLASSES),
        "log_loss": log_loss(y, proba, labels=CLASSES),
        "draws_pred": int((pred == 1).sum()),
    }


def main() -> None:
    raw = pd.read_csv(RAW, parse_dates=["Date"])
    market = build_market(raw)
    feats = pd.read_csv(FEATURES)
    ext = pd.read_csv(EXTERNAL).drop(columns=["Date"])
    ext = ext.drop(columns=[c for c in ext.columns if c.startswith("understat_")])  # leakage

    df = market.merge(
        feats[["Season", "HomeTeam", "AwayTeam", "diff_season_points_per_match",
               "diff_table_position", "diff_overall_goal_diff_last_5",
               "diff_overall_points_last_5"]],
        on=["Season", "HomeTeam", "AwayTeam"], how="left",
    ).merge(ext, on=["Season", "HomeTeam", "AwayTeam"], how="left")

    market_cols = ["mkt_home", "mkt_draw", "mkt_away", "mkt_overround",
                   "drift_home", "drift_draw", "drift_away"]
    signal_cols = ["elo_diff", "elo_expected_home",
                   "diff_xg_xg_diff_last_5", "diff_xg_xg_diff_last_10",
                   "value_log_ratio", "value_rank_diff",
                   "diff_season_points_per_match", "diff_table_position",
                   "diff_overall_goal_diff_last_5", "diff_overall_points_last_5"]
    features = market_cols + signal_cols

    y = df["target"]
    train = df["Season"] <= "2023_2024"
    valid = df["Season"] == "2024_2025"
    test = df["Season"] == "2025_2026"
    trainval = df["Season"] <= "2024_2025"

    market_probs = df[["mkt_home", "mkt_draw", "mkt_away"]].values

    # Fit calibrated base models on train; build validation ensemble to tune blend.
    def fit_ensemble(fit_mask):
        models = {}
        for name, est in base_estimators().items():
            cal = CalibratedClassifierCV(est, method="sigmoid", cv=3)
            cal.fit(df.loc[fit_mask, features], y[fit_mask])
            models[name] = cal
        return models

    models_tr = fit_ensemble(train)
    def ensemble_proba(models, mask):
        return np.mean([aligned_proba(m, df.loc[mask, features]) for m in models.values()], axis=0)

    val_ens = ensemble_proba(models_tr, valid)
    val_mkt = market_probs[valid.values]
    yv = y[valid].values

    # Tune market-blend weight w (final = w*ensemble + (1-w)*market) for accuracy.
    best_w, best_acc = 0.0, -1
    for w in np.linspace(0, 1, 41):
        blend = w * val_ens + (1 - w) * val_mkt
        acc = accuracy_score(yv, blend.argmax(1))
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"Tuned blend weight w={best_w:.2f} (val accuracy {best_acc:.3f})")

    # Refit ensemble on train+val, evaluate once on test.
    models_final = fit_ensemble(trainval)
    test_ens = ensemble_proba(models_final, test)
    test_mkt = market_probs[test.values]
    yt = y[test].values
    blended = best_w * test_ens + (1 - best_w) * test_mkt

    rows = [
        evaluate(yt, test_mkt, "Market (closing odds)"),
        evaluate(yt, test_ens, "Calibrated ensemble"),
        evaluate(yt, blended, f"Ensemble x market blend (w={best_w:.2f})"),
    ]
    # individual calibrated base models on test (for the dashboard)
    for name, m in models_final.items():
        rows.append(evaluate(yt, aligned_proba(m, df.loc[test, features]), f"  base: {name}"))

    metrics = pd.DataFrame(rows)
    print("\n=== 2025_2026 TEST metrics ===")
    print(metrics.to_string(index=False,
          formatters={"accuracy": "{:.3f}".format, "balanced_acc": "{:.3f}".format,
                      "macro_f1": "{:.3f}".format, "log_loss": "{:.3f}".format}))

    # Save predictions for the dashboard.
    pred = df.loc[test, ["Season", "Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    pred["pred"] = pd.Series(blended.argmax(1), index=pred.index).map({0: "H", 1: "D", 2: "A"})
    pred[["p_home", "p_draw", "p_away"]] = blended
    pred["correct"] = (pred["pred"] == pred["FTR"]).astype(int)
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED_OUT, index=False)
    metrics.to_csv(METRICS_OUT, index=False)
    print(f"\nSaved predictions -> {PRED_OUT}")
    print(f"Saved metrics     -> {METRICS_OUT}")

    # ------------------------------------------------------------------ #
    # Binary heads: the same features answer practical yes/no questions   #
    # with genuine skill (AUC > 0.5), unlike the 3-class task.            #
    # ------------------------------------------------------------------ #
    def binary_head(pos_class: int) -> dict:
        yb = (y == pos_class).astype(int)
        probs = []
        for est in base_estimators().values():
            cal = CalibratedClassifierCV(est, method="sigmoid", cv=3)
            cal.fit(df.loc[trainval, features], yb[trainval])
            probs.append(cal.predict_proba(df.loc[test, features])[:, 1])
        p = np.mean(probs, axis=0)
        yb_test = yb[test].values
        pred_b = (p > 0.5).astype(int)
        return {
            "question": f"{LABELS[pos_class]} win vs not",
            "accuracy": accuracy_score(yb_test, pred_b),
            "auc": roc_auc_score(yb_test, p),
            "base_rate": yb_test.mean(),
            "skill": "yes" if roc_auc_score(yb_test, p) > 0.55 else "no (base-rate only)",
        }

    binary = pd.DataFrame([binary_head(c) for c in CLASSES])
    binary.to_csv(BINARY_METRICS_OUT, index=False)
    print("\n=== Binary reframings (real accuracy lives here) ===")
    print(binary.to_string(index=False,
          formatters={"accuracy": "{:.3f}".format, "auc": "{:.3f}".format,
                      "base_rate": "{:.3f}".format}))
    print(f"Saved binary metrics -> {BINARY_METRICS_OUT}")


if __name__ == "__main__":
    main()
