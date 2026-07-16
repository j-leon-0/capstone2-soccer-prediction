# Dashboard

A self-contained, static results dashboard for the Premier League 1X2 model
(`scripts/models/epl_best_model.py`). Single HTML file — no build step, no dependencies,
all CSS/JS inlined and charts hand-drawn as SVG.

## Open it

Either way works:

```bash
# 1. Just open the file in a browser
open dashboard/index.html            # macOS  (or double-click it)

# 2. Or serve it (identical result)
python3 -m http.server 8000 -d dashboard
#   then visit http://localhost:8000
```

It is theme-aware (follows your OS light/dark setting).

## Note: the numbers are a snapshot

The figures are baked into `index.html` — it does **not** read the CSVs live.
It reflects the results produced by:

```bash
python scripts/models/epl_best_model.py     # writes data/processed/epl_best_model_*.csv
```

If you re-run the model and the numbers change, the dashboard must be updated to
match (regenerate it, or edit the values in the `<script>` block and the KPI
tiles). A live, data-driven version would require a small server or a build step
and is intentionally not used here to keep the file portable.
