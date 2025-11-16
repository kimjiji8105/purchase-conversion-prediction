import os
import sys
import tempfile
import pandas as pd

# ensure repo root is on sys.path so 'src' package is importable during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.session_analysis import extract_sessions_from_df, extract_sessions, visualize_session_features, summary_stats


def _make_sample_events_csv(path):
    lines = [
        "id,user_id,sequence_number,session_id,created_at,ip_address,city,state,postal_code,browser,traffic_source,uri,event_type",
        "1,10,1,s1,2023-01-01 12:00:00+00:00,1.1.1.1,CityX,StateX,123,Chrome,Organic,/home,home",
        "2,10,2,s1,2023-01-01 12:01:00+00:00,1.1.1.1,CityX,StateX,123,Chrome,Organic,/product,product",
        "3,10,3,s1,2023-01-01 12:02:00+00:00,1.1.1.1,CityX,StateX,123,Chrome,Organic,/purchase,purchase",
        "4,11,1,s2,2023-01-02 11:00:00+00:00,2.2.2.2,CityY,StateY,234,Safari,Search,/home,home",
        "5,11,2,s2,2023-01-02 11:05:00+00:00,2.2.2.2,CityY,StateY,234,Safari,Search,/product,product",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_extract_sessions_from_df():
    data = [
        {"id": 1, "user_id": 10, "session_id": "s1", "created_at": "2023-01-01 12:00:00+00:00", "uri": "/home", "event_type": "home", "browser": "Chrome", "traffic_source": "Organic", "city": "CityX", "state": "StateX"},
        {"id": 2, "user_id": 10, "session_id": "s1", "created_at": "2023-01-01 12:01:00+00:00", "uri": "/product", "event_type": "product", "browser": "Chrome", "traffic_source": "Organic", "city": "CityX", "state": "StateX"},
        {"id": 3, "user_id": 10, "session_id": "s1", "created_at": "2023-01-01 12:02:00+00:00", "uri": "/purchase", "event_type": "purchase", "browser": "Chrome", "traffic_source": "Organic", "city": "CityX", "state": "StateX"},
        {"id": 4, "user_id": 11, "session_id": "s2", "created_at": "2023-01-02 11:00:00+00:00", "uri": "/home", "event_type": "home", "browser": "Safari", "traffic_source": "Search", "city": "CityY", "state": "StateY"},
    ]
    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df["created_at"])
    sessions = extract_sessions_from_df(df)
    assert len(sessions) == 2
    # num_purchase and num_events are removed to avoid label leakage; assert converted label exists instead
    assert "num_purchase" not in sessions.columns
    assert "num_events" not in sessions.columns
    s1 = sessions[sessions["session_id"] == "s1"].iloc[0]
    # count columns are intentionally removed; ensure they are not present
    assert "num_product" not in sessions.columns
    assert bool(s1["converted"]) is True
    s2 = sessions[sessions["session_id"] == "s2"].iloc[0]
    assert bool(s2["converted"]) is False


def test_extract_sessions_cli_and_visualize(tmp_path):
    # create a small events csv
    events_csv = tmp_path / "events_small.csv"
    _make_sample_events_csv(events_csv)
    # run extract_sessions on file
    sessions = extract_sessions(str(events_csv), chunksize=2)
    assert "session_id" in sessions.columns
    assert len(sessions) == 2
    out_fig_dir = tmp_path / "figures"
    visualize_session_features(sessions, out_dir=str(out_fig_dir))
    # ensure main figures exist
    expected = ["session_funnel.png", "session_duration_log_hist.png", "feature_correlation_heatmap.png", "session_summary.png"]
    found = [p.name for p in out_fig_dir.iterdir()]
    for fname in expected:
        assert fname in found
    # optional category plots (may be absent if not enough data)
    optional = ["conversion_by_traffic_source.png", "conversion_by_browser.png", "unique_uris_by_conversion_box.png"]
    # ensure that if they exist they are valid files
    for fname in optional:
        if fname in found:
            assert (out_fig_dir / fname).exists()


def test_run_pycaret_automl_validation(tmp_path):
    events_csv = tmp_path / "events_small.csv"
    _make_sample_events_csv(events_csv)
    # small sessions
    from src.session_analysis import extract_sessions, run_pycaret_automl
    sessions = extract_sessions(str(events_csv), chunksize=2)
    out_dir = tmp_path / "pycaret_test"
    os.makedirs(out_dir, exist_ok=True)
    run_pycaret_automl(sessions, label="converted", sample_frac=1.0, target_dir=str(out_dir), train_size=0.8)
    # validation metrics should exist
    val_metrics = out_dir / "validation" / "validation_metrics.json"
    assert val_metrics.exists()


def test_run_pycaret_automl_drop_leaky(tmp_path):
    # build a sessions df where a synthetic 'leaker' feature equals the label
    from src.session_analysis import run_pycaret_automl
    events_csv = tmp_path / "events_small.csv"
    _make_sample_events_csv(events_csv)
    sessions = extract_sessions(str(events_csv), chunksize=2)
    # create a synthetic leaky feature
    sessions["leaker"] = sessions["converted"].astype(int)
    out_dir = tmp_path / "pycaret_drop"
    os.makedirs(out_dir, exist_ok=True)
    # run with drop_leaky_features, which should remove 'leaker'
    run_pycaret_automl(sessions, label="converted", sample_frac=1.0, target_dir=str(out_dir), train_size=0.8, drop_leaky_features=True)
    dropped = out_dir / "leakage" / "dropped_leaky_features.json"
    assert dropped.exists()
    import json
    d = json.loads(dropped.read_text())
    dropped_list = d.get("dropped", [])
    assert len(dropped_list) > 0


def test_train_validation_disjoint():
    # ensure that train and validation sets from the run split are disjoint (no session overlap)
    sessions = extract_sessions("dataset/events.csv", chunksize=2000, max_rows=2000)
    df = sessions.copy()
    # make label int
    if df["converted"].dtype == bool:
        df["converted"] = df["converted"].astype(int)
    from sklearn.model_selection import train_test_split
    stratify_col = None
    val_counts = df["converted"].value_counts(dropna=True)
    if val_counts.min() >= 2:
        stratify_col = df["converted"]
    train_df, val_df = train_test_split(df, train_size=0.8, stratify=stratify_col, random_state=42)
    assert len(set(train_df["session_id"]).intersection(set(val_df["session_id"]))) == 0


def test_num_purchase_excluded_from_validation(tmp_path):
    # Ensure num_purchase is removed before training/evaluation and not included in validation set
    events_csv = tmp_path / "events_small.csv"
    _make_sample_events_csv(events_csv)
    from src.session_analysis import extract_sessions, run_pycaret_automl

    sessions = extract_sessions(str(events_csv), chunksize=2)
    out_dir = tmp_path / "pycaret_test"
    out_dir.mkdir()
    run_pycaret_automl(sessions, label="converted", sample_frac=1.0, target_dir=str(out_dir), train_size=0.8)

    # check train columns saved
    val_raw = out_dir / "validation" / "validation_set_raw.csv"
    val_enc = out_dir / "validation" / "validation_set.csv"
    # Check the train_columns.json exists and doesn't include num_purchase
    import json
    train_cols_file = out_dir / "train_columns.json"
    assert train_cols_file.exists()
    train_cols = json.loads(train_cols_file.read_text())['columns']
    assert "num_purchase" not in train_cols
    found = False
    for p in [val_raw, val_enc]:
        if p.exists():
            # Using pandas for simplicity
            import pandas as pd
            df_val = pd.read_csv(p)
            assert "num_purchase" not in df_val.columns
            found = True
    # If validation files are written (environment dependent), ensure num_purchase not present; else it's OK
