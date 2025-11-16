# Session-level Analysis (session_analysis.py)

This repository contains tools to extract session-level features from the `events.csv` dataset,
visualize them, and run an AutoML experiment using PyCaret to predict session conversion (purchase).

Quick start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the script (small sample run):
   - `python -m src.session_analysis --events dataset/events.csv --output-dir outputs --max-rows 100000`

Full run / AutoML usage
- To run the full dataset and PyCaret AutoML with a sample fraction (default 0.1):
   - `python -m src.session_analysis --events dataset/events.csv --output-dir outputs --automl-sample-frac 0.02`
- If you want to skip AutoML:
   - `python -m src.session_analysis --events dataset/events.csv --output-dir outputs --skip-automl`

Outputs
- `outputs/session_features.csv` — the session-level features (CSV)
- `outputs/figures/` — generated static plots
- `outputs/pycaret/best_pycaret_model.pkl` — the saved best model from the AutoML session

- The script processes `events.csv` and extracts session-level aggregates (num events, event type counts, duration, etc.).
- Visualizations are saved to `outputs/figures`.
- If PyCaret is not installed or you don't want AutoML, you can skip that step.

Derived session-level features
- session_id, user_id (if available)
- session_start, session_end, session_duration_seconds
- num_events, num_unique_uris
- num_product, num_cart, num_purchase, num_cancel, num_home, num_department (counts of event_type per session)
- browser, traffic_source, city, state (first seen in session)
- converted (boolean label: True if num_purchase > 0)

Next steps / ideas
- Run PyCaret AutoML to build a model that predicts 'converted' from session features.
- Add cross-session aggregation for user-level metrics (e.g., sessions per user, historic conversion rate).
- Add more complex features: time of day, sequence-level n-grams, product categories viewed, etc.

- The script processes `events.csv` and extracts session-level aggregates (num events, event type counts, duration, etc.).
- Visualizations are saved to `outputs/figures`.
- If PyCaret is not installed or you don't want AutoML, you can skip that step.
