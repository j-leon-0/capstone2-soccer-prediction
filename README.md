# capstone2-soccer-prediction
A machine learning project focused on predicting soccer match outcomes using historical match data and team performance statistics. The repository includes data collection, preprocessing, exploratory analysis, feature engineering, and predictive modeling workflows.

dataset source: "https://www.football-data.co.uk/"

## Repository structure

```
data/
  raw/            football-data.co.uk season CSVs + raw/external/ scraped sources
  processed/      cleaned matches, engineered features, model outputs
scripts/
  pipeline/       data cleaning -> feature engineering (run from repo root)
  models/         model training & evaluation
  scrape/         external-data collectors (ClubElo, Transfermarkt, Understat, FBref)
notebooks/        exploration, feature engineering, and modeling experiments
dashboard/        self-contained static results dashboard (dashboard/index.html)
```

### scripts/

| Path | Purpose |
|---|---|
| `pipeline/epl_clean_raw.py` | Combine + clean the raw season CSVs |
| `pipeline/epl_validate_raw.py` | Sanity checks on the raw data |
| `pipeline/epl_feature_engineering.py` | Rolling form, table, rest, referee features |
| `pipeline/build_external_features.py` | Elo + xG + squad-value features (leakage-free) |
| `pipeline/build_congestion_features.py` | All-competition fixture-congestion features |
| `models/epl_best_model.py` | **Current best model** — market-anchored calibrated ensemble |
| `models/epl_production_model.py` | Earlier stacked-ensemble model (shots-based xG proxy) |
| `scrape/*` | One collector per external source; see the section below |

### notebooks/ (chronological history — kept intentionally)

| Notebook | What it is |
|---|---|
| `jero_eda.ipynb`, `wes_eda.ipynb` | Exploratory data analysis |
| `jero_feature_engineering.ipynb` | Feature-engineering walkthrough |
| `jero_baseline_modeling.ipynb` | First baseline models |
| `jero_market_subset_bayesian.ipynb` | Market-odds-only subset + Bayesian tuning |
| `jero_pca_autoencoder_rf.ipynb` | PCA + autoencoder + random-forest experiment |
| `jero_lightgbm_tuning.ipynb` | LightGBM hyperparameter tuning |
| `jero_ensemble_modeling.ipynb` | Stacked ensemble + decision policy (↔ `models/epl_production_model.py`) |
| `jero_external_features_modeling.ipynb` | **Current** — scraped Elo/xG/value features + honest evaluation |

## External scraped features

The base football-data.co.uk match/odds data is enriched with three externally
scraped, **strictly pre-match** signals. Scrapers live in `scripts/scrape/` and cache
raw responses under `data/raw/external/`; `scripts/pipeline/build_external_features.py` turns
them into leakage-free (`shift(1)`) model features in
`data/processed/external/epl_external_features.csv`.

| Source | Script | Feature | Notes |
|---|---|---|---|
| [ClubElo](http://clubelo.com) | `fetch_clubelo.py` | Pre-match Elo rating + win-expectancy | Free API; 0 missing across 3,800 matches |
| [Understat](https://understat.com) | `_understat_sink.py` (+ browser) | Rolling expected-goals (xG) form | Cloudflare-protected — see below |
| [Transfermarkt](https://www.transfermarkt.com) | `fetch_transfermarkt.py` | Season-start squad market value | Flaky over curl; script auto-retries |
| [FBref](https://fbref.com) | `_fbref_sink.py` (+ browser) | All-competition fixture dates → fixture-congestion features | Blocks headless fetch; scraped via browser, see below |

Fixture-congestion features (`scripts/pipeline/build_congestion_features.py` →
`epl_congestion_features.csv`) union every PL match with FA Cup / EFL Cup / Champions
League / Europa / Conference fixtures to compute *true* rest days, matches in the last
14 days, and a recent-European-tie flag. **Finding:** they do not improve the models —
the congestion signal is confounded with team quality (busy clubs are the good European
ones), which market odds + Elo + squad value already capture. See the modeling notebook.

Re-run the full external pipeline:

```bash
python scripts/scrape/fetch_clubelo.py          # Elo -> data/processed/external/epl_elo.csv
python scripts/scrape/fetch_transfermarkt.py    # squad value -> epl_squad_value.csv

# Understat is behind Cloudflare (headless fetch returns a stub), so xG is scraped
# through a real browser that POSTs each season's `datesData` to a local sink:
python scripts/scrape/_understat_sink.py &      # listens on localhost:8477
#   then open https://understat.com/league/EPL/<year> for years 2016..2025 in a
#   browser and POST `datesData` to http://localhost:8477/save (season\tTSV body).

python scripts/pipeline/build_external_features.py       # combine -> epl_external_features.csv
```

> **Important — leakage note:** Understat also publishes a per-match `forecast`
> (w/d/l). It is a Poisson *retrodiction* of the same match's realised xG
> (corr ≈ 0.96 with the match xG margin), **not** a pre-match prediction, and is
> dropped from all modeling. Modeling and honest results live in
> `notebooks/jero_external_features_modeling.ipynb`.

## Environment Setup

### 1. Create a Virtual Environment

From the project root directory:

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

After activation, your terminal should display:

```bash
(.venv)
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name capstone2-soccer --display-name "Python (capstone2-soccer)"
```

### 5. Select the Kernel in VS Code

Open any notebook and select:

```text
Python (capstone2-soccer)
```

### Updating Dependencies

If new packages are added to the project, update the dependency file:

```bash
pip freeze > requirements.txt
```