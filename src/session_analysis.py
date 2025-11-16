"""
session_analysis.py

Utilities for extracting session-level features from event logs, visualizing them,
and running AutoML using PyCaret to predict purchase conversion at session level.

Intended usage:
- Extract features: create_session_features(events_path)
- Visualize: visualize_session_features(features_df)
- Run AutoML: run_pycaret_automl(features_df)

This module is written with chunked processing in mind so it can handle large event
logs without requiring the entire file in memory.
"""

from __future__ import annotations

import os
import math
from collections import defaultdict, Counter
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "t", "y", "yes"}


def _init_session_entry(session_id: str, row: pd.Series) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "user_id": row.get("user_id") if not pd.isna(row.get("user_id")) else None,
        "min_time": row["created_at"],
        "max_time": row["created_at"],
        "num_events": 0,
        "event_counts": Counter(),
        "unique_uris": set(),
        "browser": row.get("browser"),
        "traffic_source": row.get("traffic_source"),
        "city": row.get("city"),
        "state": row.get("state"),
    }


def _finalize_session_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    duration = None
    if entry["min_time"] is not None and entry["max_time"] is not None:
        try:
            min_t = pd.to_datetime(entry["min_time"]) if not pd.isna(entry["min_time"]) else None
            max_t = pd.to_datetime(entry["max_time"]) if not pd.isna(entry["max_time"]) else None
            if min_t is not None and max_t is not None:
                duration = (max_t - min_t).total_seconds()
        except Exception:
            duration = None
    return {
        "session_id": entry["session_id"],
        "user_id": entry["user_id"],
        "session_start": entry["min_time"],
        "session_end": entry["max_time"],
        "session_duration_seconds": duration if duration is not None else 0,
        "num_unique_uris": len(entry["unique_uris"]),
        "num_product": int(entry["event_counts"].get("product", 0)),
        "num_cart": int(entry["event_counts"].get("cart", 0)),
        "num_purchase": int(entry["event_counts"].get("purchase", 0)),
        "num_cancel": int(entry["event_counts"].get("cancel", 0)),
        "num_home": int(entry["event_counts"].get("home", 0)),
        "num_department": int(entry["event_counts"].get("department", 0)),
        "browser": entry.get("browser"),
        "traffic_source": entry.get("traffic_source"),
        "city": entry.get("city"),
        "state": entry.get("state"),
    }


def create_session_features(events_df: pd.DataFrame = None, events_path: str = None, chunksize: int = 100000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Wrapper that builds session features. Provide either an in-memory `events_df`
    or a path `events_path` to read in chunks. This keeps preprocessing centralized.
    """
    if events_df is not None:
        return extract_sessions_from_df(events_df)
    if events_path is not None:
        return extract_sessions(events_path, chunksize=chunksize, max_rows=max_rows)
    raise ValueError("Either events_df or events_path must be provided to create_session_features")


def extract_sessions_from_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an events DataFrame into a session-level DataFrame with aggregated features.

    Events DataFrame must contain at least: session_id, created_at, event_type, uri
    and optionally: user_id, browser, traffic_source, city, state
    """
    sessions: Dict[str, Dict[str, Any]] = {}

    # Ensure created_at is a datetime
    if not pd.api.types.is_datetime64_any_dtype(events_df["created_at"]):
        events_df["created_at"] = pd.to_datetime(
            events_df["created_at"], utc=True, infer_datetime_format=True, errors="coerce"
        )

    # Sort by session_id and created_at for consistent min/max
    events_df = events_df.sort_values(["session_id", "created_at"])  # stable ordering

    for _, row in events_df.iterrows():
        session_id = row["session_id"]
        if pd.isna(session_id):
            # skip rows without session id
            continue
        if session_id not in sessions:
            sessions[session_id] = _init_session_entry(session_id, row)
        entry = sessions[session_id]
        # user_id prefer first non-null
        if entry["user_id"] is None and not pd.isna(row.get("user_id")):
            entry["user_id"] = row.get("user_id")
        # min/max time
        if row["created_at"] < entry["min_time"]:
            entry["min_time"] = row["created_at"]
        if row["created_at"] > entry["max_time"]:
            entry["max_time"] = row["created_at"]
        # counts
        entry["num_events"] += 1
        event_type = row.get("event_type") or "unknown"
        entry["event_counts"][event_type] += 1
        # uri
        if not pd.isna(row.get("uri")):
            entry["unique_uris"].add(row.get("uri"))
        # browser/traffic/city/state: keep first non-null
        for col in ("browser", "traffic_source", "city", "state"):
            if entry.get(col) is None and not pd.isna(row.get(col)):
                entry[col] = row.get(col)

    # Finalize into DataFrame
    data = []
    for _, entry in sessions.items():
        data.append(_finalize_session_entry(entry))
    df = pd.DataFrame(data)

    # Define converted label (based on purchase events internally tracked)
    df["converted"] = df["num_purchase"] > 0
    # Merge user demographic info (age, gender, country) when available
    try:
        users_path = os.path.join("dataset", "users.csv")
        if os.path.exists(users_path):
            users_df = pd.read_csv(users_path)
            # normalize user_id types
            users_df["user_id"] = pd.to_numeric(users_df["user_id"], errors="coerce")
            df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
            users_keep = [c for c in ("user_id", "age", "gender", "country") if c in users_df.columns]
            if len(users_keep) > 1:
                df = df.merge(users_df[users_keep], on="user_id", how="left")
    except Exception:
        pass
    # Note: do not add an explicit is_member flag here to avoid label-leaking features
    # Remove all raw count columns (any column starting with 'num_') to avoid label leakage
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if num_cols:
        df = df.drop(columns=num_cols)
    return df


def extract_sessions(events_path: str, chunksize: int = 100000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Process events.csv in chunks and extract session-level features.

    Returns a DataFrame where each row is a session with aggregated features.
    """
    fields = [
        "id",
        "user_id",
        "sequence_number",
        "session_id",
        "created_at",
        "ip_address",
        "city",
        "state",
        "postal_code",
        "browser",
        "traffic_source",
        "uri",
        "event_type",
    ]

    # This function reads the CSV in chunks and aggregates into an in-memory dict.
    sessions: Dict[str, Dict[str, Any]] = {}

    read_rows = 0
    for chunk in pd.read_csv(events_path, usecols=fields, parse_dates=["created_at"], chunksize=chunksize):
        # Optionally limit rows for quick runs
        if max_rows is not None and read_rows >= max_rows:
            break
        # If chunk push beyond max_rows, trim
        if max_rows is not None and read_rows + len(chunk) > max_rows:
            chunk = chunk.head(max_rows - read_rows)
        read_rows += len(chunk)

        # iterate rows
        # Ensure created_at dtype is datetime
        if not pd.api.types.is_datetime64_any_dtype(chunk["created_at"]):
            chunk["created_at"] = pd.to_datetime(
                chunk["created_at"], utc=True, infer_datetime_format=True, errors="coerce"
            )
        for _, row in chunk.iterrows():
            session_id = row["session_id"]
            if pd.isna(session_id):
                continue
            session_id = str(session_id)
            if session_id not in sessions:
                sessions[session_id] = _init_session_entry(session_id, row)
            entry = sessions[session_id]
            # user_id prefer first non-null
            if entry["user_id"] is None and not pd.isna(row.get("user_id")):
                entry["user_id"] = row.get("user_id")
            # min/max time
            if row["created_at"] < entry["min_time"]:
                entry["min_time"] = row["created_at"]
            if row["created_at"] > entry["max_time"]:
                entry["max_time"] = row["created_at"]
            # counts
            entry["num_events"] += 1
            event_type = row.get("event_type") or "unknown"
            entry["event_counts"][event_type] += 1
            # uri
            if not pd.isna(row.get("uri")):
                entry["unique_uris"].add(row.get("uri"))
            # browser/traffic/city/state: keep first non-null
            for col in ("browser", "traffic_source", "city", "state"):
                if entry.get(col) is None and not pd.isna(row.get(col)):
                    entry[col] = row.get(col)

    # finalize
    results = []
    for _, entry in sessions.items():
        results.append(_finalize_session_entry(entry))
    df = pd.DataFrame(results)
    df["converted"] = df["num_purchase"] > 0
    # Merge user demographic info (age, gender, country) when available
    try:
        users_path = os.path.join("dataset", "users.csv")
        if os.path.exists(users_path):
            users_df = pd.read_csv(users_path)
            users_df["user_id"] = pd.to_numeric(users_df["user_id"], errors="coerce")
            df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
            users_keep = [c for c in ("user_id", "age", "gender", "country") if c in users_df.columns]
            if len(users_keep) > 1:
                df = df.merge(users_df[users_keep], on="user_id", how="left")
    except Exception:
        pass
    # Do not create an explicit membership flag here to avoid label-leaking features
    # Remove all raw count columns (any column starting with 'num_') to avoid label leakage
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if num_cols:
        df = df.drop(columns=num_cols)
    return df


def visualize_session_features(features_df: pd.DataFrame, out_dir: str = "outputs/figures", sample_frac: float = 0.2, train_columns_path: Optional[str] = None) -> None:
    """
    Create several readable visualizations for session features.
    - funnels of stages (home->product->cart->purchase)
    - session duration histogram/log distribution + boxplot
    - events distribution by conversion (violin/kde)
    - conversion by traffic_source and browser (top N) with counts annotated
    - correlation heatmap of numeric features

    Parameters
    - features_df: DataFrame returned from extract_sessions
    - out_dir: where to save figures
    - sample_frac: fraction of sessions to sample for plots that might be slow when data is large
    """
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style="whitegrid")

    df = features_df.copy()
    # If a train_columns_path is provided and exists, limit visualizations to features
    # that were actually used for training to avoid showing unused features.
    if train_columns_path:
        try:
            if os.path.exists(train_columns_path):
                import json
                train_cols = json.load(open(train_columns_path)).get("columns", [])
                # always keep label if present
                if "converted" in df.columns:
                    keep = [c for c in train_cols if c in df.columns] + ["converted"]
                else:
                    keep = [c for c in train_cols if c in df.columns]
                # also keep some plotting-friendly columns if present
                for extra in ("session_duration_seconds", "num_unique_uris", "browser", "traffic_source", "session_id"):
                    if extra in df.columns and extra not in keep:
                        keep.append(extra)
                df = df[keep].copy()
        except Exception:
            # if anything goes wrong reading train columns, fallback to full df
            df = features_df.copy()
    # Protect against label leakage: drop num_purchase which is the raw event count for purchase
    # (this directly correlates with the 'converted' label and should not be used as a predictor)
    if "num_purchase" in df.columns:
        print("Removing 'num_purchase' from features to prevent label leakage")
        df = df.drop(columns=["num_purchase"])
    # Ensure converted is boolean
    if "converted" in df.columns:
        df["converted"] = df["converted"].astype(bool)

    # Optionally sample for speed/clarity
    if sample_frac < 1.0 and len(df) > 5000:
        df_plot = df.sample(frac=max(0.1, sample_frac), random_state=42)
    else:
        df_plot = df

    # Funnel: compute conversion proportion for all event-type counts except purchase and cancel
    # Find all event count columns which start with 'num_' and exclude purchase/cancel
    count_cols = [c for c in df.columns if c.startswith("num_")]
    exclude = {"num_purchase", "num_cancel"}
    event_cols = [c for c in count_cols if c not in exclude]
    # Build staged counts and stacked percentages of converted vs not
    funnel_rows = []
    total_sessions = len(df)
    for col in event_cols:
        name = col.replace("num_", "").replace("_", " ").title()
        if col in df.columns:
            stage_mask = df[col] > 0
            stage_df = df[stage_mask]
            total_stage = len(stage_df)
            converted_count = int(stage_df[stage_df.get("converted", False) == True].shape[0])
            not_converted = total_stage - converted_count
        else:
            total_stage = 0
            converted_count = 0
            not_converted = 0
        funnel_rows.append({"stage": name, "total": total_stage, "converted": converted_count, "not_converted": not_converted})
    funnel_df = pd.DataFrame(funnel_rows)

    # Create stacked horizontal bar plot showing conversion proportion per stage
    funnel_available = not funnel_df.empty and funnel_df["total"].sum() > 0
    if funnel_available:
        plt.figure(figsize=(8, 4))
        y = funnel_df["stage"].values
        left = np.zeros(len(funnel_df))
        colors = ["#B3CDE3", "#33A02C"]  # not_converted, converted
        for i, col in enumerate(["not_converted", "converted"]):
            vals = funnel_df[col].values
            # convert to fraction of stage total to show proportion per stage
            frac = np.divide(vals, funnel_df["total"].replace(0, np.nan)).astype(float)
            frac = np.nan_to_num(frac, nan=0.0)
            plt.barh(y, frac, left=left, color=colors[i], edgecolor="k", label=("Not converted" if col=="not_converted" else "Converted"))
            left = left + frac
        plt.xlabel("Fraction of sessions in stage")
        plt.title("Conversion proportions by funnel stage (stacked)")
        plt.legend(loc="lower right")
        # annotate with counts and percent
        for i, row in funnel_df.iterrows():
            total = row["total"]
            if total == 0:
                continue
            # annotate not converted
            nc_frac = row["not_converted"] / total
            c_frac = row["converted"] / total
            # left edge
            plt.text(nc_frac / 2, i, f"{row['not_converted']} ({nc_frac:.0%})", va="center", ha="center", fontsize=9, color="black")
            # right edge
            plt.text(nc_frac + c_frac / 2, i, f"{row['converted']} ({c_frac:.0%})", va="center", ha="center", fontsize=9, color="white")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "session_funnel.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # placeholder figure so callers/tests that expect a file still find something
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No funnel data (event count columns removed)", ha="center", va="center")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "session_funnel.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Prepare additional columns (log transforms) used in several plots
    if "session_duration_seconds" in df.columns:
        df_plot["session_duration_log1p"] = np.log1p(df_plot["session_duration_seconds"].fillna(0).astype(float))
    if "session_duration_seconds" in df_plot.columns:
        durations = df_plot["session_duration_seconds"].fillna(0).astype(float)
        durations_pos = durations[durations > 0]
        if len(durations_pos) > 0:
            cap = min(durations_pos.quantile(0.99), durations_pos.max())
            x = durations_pos.clip(upper=cap)
            plt.figure(figsize=(8, 4))
            sns.histplot(x=np.log1p(x), bins=40, kde=True, color="#5A9" )
            plt.xlabel("log1p(Session duration seconds)")
            plt.title("Session duration distribution (log1p) — clipped at 99th percentile")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "session_duration_log_hist.png"), dpi=150, bbox_inches='tight')
            plt.close()

            # boxplot of log1p(duration) by conversion for summary
            if "converted" in df_plot.columns:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x="converted", y="session_duration_log1p", data=df_plot, palette=sns.color_palette(["#B2DF8A", "#33A02C"]))
                plt.xlabel("Converted")
                plt.ylabel("log1p(Session duration seconds)")
                plt.title("Session duration by conversion (log1p)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "session_duration_by_conversion_box.png"), dpi=150, bbox_inches='tight')
                plt.close()

    # (Removed) num_events-based plots to avoid leaking label-derived information.

    # Conversion rate by traffic_source and browser (top N categories for clarity)
    def plot_top_categories(cat_col: str, out_name: str, top_n: int = 10, min_count: int = 20):
        if cat_col not in df.columns:
            return
        agg = df.groupby(cat_col)["converted"].agg(["mean", "count"]).reset_index()
        agg = agg[agg["count"] >= min_count].sort_values("mean", ascending=False)
        if agg.empty:
            return
        top = agg.head(top_n).sort_values("mean")
        plt.figure(figsize=(8, max(3, 0.4 * len(top))))
        ax = sns.barplot(x="mean", y=cat_col, data=top, palette="crest")
        plt.xlabel("Conversion rate")
        plt.title(f"Conversion rate by {cat_col} (top {top_n}, min {min_count} sessions)")
        # annotate percentage and counts
        max_val = top["mean"].max()
        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row["mean"] + max_val * 0.01, i, f"{row['mean']:.1%} ({int(row['count'])})", va="center", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        ax.set_xlim(0, max_val * 1.12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, out_name), dpi=150, bbox_inches='tight')
        plt.close()

    plot_top_categories("traffic_source", "conversion_by_traffic_source.png")
    plot_top_categories("browser", "conversion_by_browser.png")

    # num_unique_uris by conversion
    if "num_unique_uris" in df_plot.columns and "converted" in df_plot.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="converted", y="num_unique_uris", data=df_plot, palette=sns.color_palette(["#FDB462", "#B3DE69"])) 
        plt.title("Unique URIs visited by conversion")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "unique_uris_by_conversion_box.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Correlation heatmap of numeric features
    numeric_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
    numeric_cols = [c for c in numeric_cols if c not in ("id", "user_id")]
    if len(numeric_cols) >= 2:
        corr = df_plot[numeric_cols].corr()
        plt.figure(figsize=(max(6, 0.6 * len(numeric_cols)), max(6, 0.6 * len(numeric_cols))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.5})
        plt.title("Correlation matrix of numeric features")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Summary dashboard: 2x2 (funnel, duration, events by conversion, traffic source)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Funnel (top-left)
        ax = axes[0, 0]
        # Use the same stacked fraction representation as session_funnel for consistency
        if funnel_available:
            y = funnel_df["stage"].values
            left_local = np.zeros(len(funnel_df))
            for i, col in enumerate(["not_converted", "converted"]):
                vals = funnel_df[col].values
                frac = np.divide(vals, funnel_df["total"].replace(0, np.nan)).astype(float)
                frac = np.nan_to_num(frac, nan=0.0)
                ax.barh(y, frac, left=left_local, color=("#B3CDE3" if col == "not_converted" else "#33A02C"), edgecolor="k")
                left_local = left_local + frac
            ax.set_xlabel("Fraction of sessions in stage")
            ax.set_title("Conversion proportions by funnel stage (stacked)")
            # annotate with counts/percent per stage (same as session_funnel)
            for i, row in funnel_df.iterrows():
                total = row["total"]
                if total == 0:
                    continue
                nc_frac = row["not_converted"] / total
                c_frac = row["converted"] / total
                ax.text(nc_frac / 2, i, f"{row['not_converted']} ({nc_frac:.0%})", va="center", ha="center", fontsize=9, color="black")
                ax.text(nc_frac + c_frac / 2, i, f"{row['converted']} ({c_frac:.0%})", va="center", ha="center", fontsize=9, color="white")
        else:
            ax.text(0.5, 0.5, "No funnel data (counts removed)", ha="center", va="center")
            ax.axis('off')

        # Duration (top-right)
        ax = axes[0, 1]
        if "session_duration_log1p" in df_plot.columns:
            sns.histplot(df_plot["session_duration_log1p"], bins=40, kde=True, ax=ax, color="#5A9")
            ax.set_xlabel("log1p(Session duration seconds)")
            ax.set_title("Session duration (log1p)")

        # Events by conversion (bottom-left) - removed num_events-based plot to avoid label leakage
        ax = axes[1, 0]
        ax.set_visible(False)

        # Traffic source conversion (bottom-right)
        ax = axes[1, 1]
        if "traffic_source" in df.columns:
            agg = df.groupby("traffic_source")["converted"].agg(["mean", "count"]).reset_index()
            agg = agg[agg["count"] >= 20].sort_values("mean", ascending=False).head(8)
            if not agg.empty:
                sns.barplot(x="mean", y="traffic_source", data=agg, ax=ax, palette="crest")
                ax.set_xlabel("Conversion rate")
                ax.set_title("Conversion by traffic source (top 8)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "session_summary.png"), dpi=180, bbox_inches='tight')
        plt.close()
    except Exception:
        # safe to ignore any dashboard creation failures; figures are still individually saved
        pass


def run_pycaret_automl(features_df: pd.DataFrame, label: str = "converted", sample_frac: float = 0.2, target_dir: str = "outputs/pycaret", train_size: float = 0.8, drop_leaky_features: bool = False):
    """
    Run a small AutoML experiment using PyCaret for classification.
    To keep runtime reasonable, sample a fraction of input if it's large.
    """
    try:
        from pycaret.classification import setup, compare_models, save_model, predict_model
    except Exception:
        raise RuntimeError("PyCaret is not installed. Please install 'pycaret' to use the AutoML feature.")

    os.makedirs(target_dir, exist_ok=True)
    df = features_df.copy()
    # Permanently remove any 'num_' count columns from modeling to avoid unintended signal/label leakage
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if num_cols:
        print(f"Removing count columns from features before modeling: {num_cols}")
        df = df.drop(columns=num_cols)
    # select useful columns
    cols = ["num_unique_uris", "session_duration_seconds", "browser", "traffic_source", label]
    # keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna(subset=[label])

    if 0 < sample_frac < 1.0 and len(df) > 1000:
        df = df.sample(frac=sample_frac, random_state=42)

    # convert boolean label to int
    if df[label].dtype == bool:
        df[label] = df[label].astype(int)

    # Save the raw column names used for modeling for reproducibility and tests
    try:
        cols_for_model = [c for c in df.columns if c != label]
        os.makedirs(target_dir, exist_ok=True)
        import json
        with open(os.path.join(target_dir, "train_columns.json"), "w") as f:
            json.dump({"columns": cols_for_model}, f, indent=2)
    except Exception:
        pass

    # optionally detect and drop leakage-prone features
    if drop_leaky_features:
        try:
            leak_report = detect_label_leakage(df, label=label)
            flagged = [f["feature"] for f in leak_report.get("flagged", [])]
            if flagged:
                print(f"Dropping potentially leaky features before training: {flagged}")
                df = df.drop(columns=flagged, errors="ignore")
                os.makedirs(os.path.join(target_dir, "leakage"), exist_ok=True)
                import json
                with open(os.path.join(target_dir, "leakage", "dropped_leaky_features.json"), "w") as f:
                    json.dump({"dropped": flagged}, f, indent=2)
        except Exception:
            pass

    # split into train / validation explicitly using sklearn and run AutoML on train set
    from sklearn.model_selection import train_test_split
    stratify_col = None
    if label in df.columns:
        val_counts = df[label].value_counts(dropna=True)
        if val_counts.min() >= 2:
            stratify_col = df[label]
    train_df, val_df = train_test_split(df, train_size=train_size, stratify=stratify_col, random_state=42)
    # Save explicit train/validation CSVs used for modeling (raw dataframes)
    try:
        os.makedirs(target_dir, exist_ok=True)
        train_path = os.path.join(target_dir, "train.csv")
        val_dir = os.path.join(target_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        # save raw validation set
        val_df.to_csv(os.path.join(val_dir, "validation_set_raw.csv"), index=False)
        # also save a canonical validation_set.csv (may be overwritten later with encoded version)
        val_df.to_csv(os.path.join(val_dir, "validation_set.csv"), index=False)
    except Exception:
        pass
    # PyCaret setup uses train set only
    # For tiny datasets, PyCaret may not behave well — fallback to a simple sklearn pipeline
    use_sklearn_fallback = len(train_df) < 10 or len(val_df) < 2
    if not use_sklearn_fallback:
        clf_setup = setup(
            data=train_df,
            target=label,
            train_size=0.8 if train_size >= 0.8 else train_size,
            session_id=42,
            log_experiment=False,
            verbose=False,
            html=False,
            n_jobs=-1,
        )
        # compare_models prints a leaderboard to stdout; capture it to avoid duplicate
        # printing in the CLI logs
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                best = compare_models()
            # Optionally save the captured leaderboard to file for debugging
            try:
                with open(os.path.join(target_dir, "pycaret_leaderboard.txt"), "w") as f:
                    f.write(buf.getvalue())
            except Exception:
                pass
        except Exception:
            # Fallback: if capture fails, just run normally
            best = compare_models()
    else:
        # fallback to sklearn LogisticRegression pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X_train = train_df.drop(columns=[label]) if label in train_df.columns else train_df
        y_train = train_df[label].astype(int) if label in train_df.columns else None
        X_val = val_df.drop(columns=[label]) if label in val_df.columns else val_df
        # basic encoding for small dataset: get dummies and align
        X_train_enc = pd.get_dummies(X_train)
        X_val_enc = pd.get_dummies(X_val)
        # align columns
        X_train_enc, X_val_enc = X_train_enc.align(X_val_enc, join='left', axis=1, fill_value=0)
        # If y_train has only a single class, fallback to DummyClassifier to avoid errors
        from sklearn.dummy import DummyClassifier
        if len(np.unique(y_train)) < 2:
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", DummyClassifier(strategy="most_frequent"))])
        else:
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        pipeline.fit(X_train_enc, y_train)
        import joblib
        joblib.dump(pipeline, os.path.join(target_dir, "best_pycaret_model.pkl"))
        # Save the aligned val set as we will use it for evaluation
        try:
            os.makedirs(os.path.join(target_dir, "validation"), exist_ok=True)
            X_val_enc.to_csv(os.path.join(target_dir, "validation", "validation_set.csv"), index=False)
            # also save original val_df for downstream
            val_df.to_csv(os.path.join(target_dir, "validation", "validation_set_raw.csv"), index=False)
        except Exception:
            pass
        best = pipeline
        # Evaluate on validation set for sklearn fallback and save validation metrics
        try:
            y_true = val_df[label].astype(int)
            y_pred = pipeline.predict(X_val_enc)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_val_enc)[:, 1]
                except Exception:
                    y_proba = None
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
            }
            os.makedirs(os.path.join(target_dir, "validation"), exist_ok=True)
            import json
            with open(os.path.join(target_dir, "validation", "validation_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            # confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Validation Confusion Matrix")
            plt.tight_layout()
            fig.savefig(os.path.join(target_dir, "validation", "confusion_matrix.png"), dpi=150)
            plt.close(fig)
            if y_proba is not None:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Validation ROC Curve")
                ax.legend()
                plt.tight_layout()
                fig.savefig(os.path.join(target_dir, "validation", "roc_curve.png"), dpi=150)
                plt.close(fig)
        except Exception:
            pass
    # save a model
    try:
        save_model(best, os.path.join(target_dir, "best_pycaret_model"))
    except Exception:
        # if best is sklearn pipeline, already saved above
        pass
    # Evaluate on validation set and save validation metrics (only validation results saved)
    try:
        pred = predict_model(best, data=val_df)
        # write out validation metrics using the evaluate helper for consistency
        os.makedirs(os.path.join(target_dir, "validation"), exist_ok=True)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        y_true = val_df[label].astype(int)
        # detect predicted label column
        if "Label" in pred.columns:
            y_pred = pred["Label"].astype(int)
        elif "prediction_label" in pred.columns:
            y_pred = pred["prediction_label"].astype(int)
        else:
            y_pred = pred.iloc[:, -1].astype(int)
        prob_cols = [c for c in pred.columns if c.lower().startswith("score") or c.lower().startswith("prob") or c == "Score"]
        y_proba = pred[prob_cols[0]] if prob_cols else None
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        import json
        # Save a copy of validation set used
        try:
            os.makedirs(os.path.join(target_dir, "validation"), exist_ok=True)
            val_df.to_csv(os.path.join(target_dir, "validation", "validation_set.csv"), index=False)
        except Exception:
            pass
        with open(os.path.join(target_dir, "validation", "validation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        # Save confusion matrix and ROC if prob exists
        cm = confusion_matrix(y_true, y_pred)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Validation Confusion Matrix")
        plt.tight_layout()
        fig.savefig(os.path.join(target_dir, "validation", "confusion_matrix.png"), dpi=150)
        plt.close(fig)
        if y_proba is not None:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Validation ROC Curve")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(target_dir, "validation", "roc_curve.png"), dpi=150)
            plt.close(fig)
    except Exception:
        pass
    return best


def evaluate_pycaret_model(model_path: str, features_df: pd.DataFrame = None, val_df: pd.DataFrame = None, label: str = "converted", test_frac: float = 0.2, random_state: int = 42, target_dir: str = "outputs/analysis"):
    """
    Evaluate a saved PyCaret model on a holdout test set; save ROC/Confusion matrix and basic metrics.
    """
    try:
        from pycaret.classification import load_model, predict_model
    except Exception:
        raise RuntimeError("PyCaret is not available for evaluation.")

    os.makedirs(target_dir, exist_ok=True)
    model = load_model(model_path)

    # Prepare test set: prefer val_df if provided; else split features_df
    if val_df is not None:
        test = val_df
    else:
        if features_df is None:
            raise ValueError("Either features_df or val_df must be provided")
        if label not in features_df.columns:
            raise ValueError(f"Label column '{label}' not found in features dataframe")
        df = features_df.dropna(subset=[label])
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_frac, random_state=random_state, stratify=df[label])

    # Use pycaret.predict_model for preprocessing and predictions
    pred = predict_model(model, data=test)
    # predict_model returns a DataFrame with 'Label' column for predicted labels and 'Score' prob
    y_true = test[label].astype(int)
    # Attempt to detect prediction column
    if "Label" in pred.columns:
        y_pred = pred["Label"].astype(int)
    elif "prediction_label" in pred.columns:
        y_pred = pred["prediction_label"].astype(int)
    else:
        # fallback to 'Label'
        y_pred = pred.iloc[:, -1].astype(int)

    # If there's a probability column like 'Score'
    prob_cols = [c for c in pred.columns if c.lower().startswith("score") or c.lower().startswith("prob") or c == "Score"]
    y_proba = pred[prob_cols[0]] if prob_cols else None

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # Save metrics
    import json
    with open(os.path.join(target_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(target_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    if y_proba is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(target_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)

    # Save classification report
    from sklearn.metrics import classification_report
    with open(os.path.join(target_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred))

    return metrics


def compute_feature_importance_shap(model_path: str, features_df: pd.DataFrame = None, val_df: pd.DataFrame = None, label: str = "converted", sample_size: int = 2000, target_dir: str = "outputs/analysis"):
    """
    Compute feature importances using coefficient-based approach (if linear) and SHAP if available.
    """
    os.makedirs(target_dir, exist_ok=True)
    try:
        from pycaret.classification import load_model
    except Exception:
        raise RuntimeError("PyCaret is not available for feature importance analysis.")

    model = load_model(model_path)

    # get sample from provided validation set if available, otherwise use features_df
    if val_df is not None:
        X = val_df.drop(columns=[label]) if label in val_df.columns else val_df.copy()
    else:
        if features_df is None:
            raise ValueError("Either features_df or val_df must be provided to compute feature importance")
        X = features_df.drop(columns=[label]) if label in features_df.columns else features_df.copy()
    # If train_columns.json exists next to the model, limit SHAP inputs to the actual train columns
    try:
        model_dir = os.path.dirname(model_path)
        train_cols_path = os.path.join(model_dir, "train_columns.json")
        if os.path.exists(train_cols_path):
            import json
            train_cols = json.load(open(train_cols_path)).get("columns", [])
            # keep only columns that exist in X
            keep = [c for c in train_cols if c in X.columns]
            if keep:
                X = X[keep]
    except Exception:
        pass
    if len(X) > sample_size:
        Xs = X.sample(n=sample_size, random_state=42)
    else:
        Xs = X

    # Try SHAP explainer
    try:
        import shap
        from sklearn.pipeline import Pipeline
        # separate preprocessor and estimator if pipeline
        if isinstance(model, Pipeline) and len(model.steps) > 1:
            preprocessor = Pipeline(model.steps[:-1])
            estimator = model.steps[-1][1]
            Xs_proc = preprocessor.transform(Xs)
        else:
            # direct estimator
            estimator = model
            Xs_proc = Xs.values

        # choose appropriate explainer
        if hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, Xs_proc, feature_perturbation="interventional")
        else:
            try:
                explainer = shap.Explainer(estimator, Xs_proc)
            except Exception:
                explainer = shap.KernelExplainer(estimator.predict, shap.sample(Xs_proc, 100))

        # get shap values robustly and produce a summary plot
        try:
            try:
                shap_values = explainer.shap_values(Xs_proc)
            except Exception:
                ev = explainer(Xs_proc)
                if hasattr(ev, "values"):
                    shap_values = ev.values
                else:
                    shap_values = ev

            import matplotlib.pyplot as plt

            # attempt to extract feature names
            feature_names = None
            try:
                if 'preprocessor' in locals() and hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out(Xs.columns)
            except Exception:
                feature_names = None
            if feature_names is None and hasattr(Xs, 'columns'):
                feature_names = list(Xs.columns)

            use_names = None
            if feature_names is not None and hasattr(Xs_proc, 'shape') and len(feature_names) == Xs_proc.shape[1]:
                use_names = feature_names

            # If shap_values contains multiple outputs (e.g., multi-class), arrange
            # the summary plots in a single row so they appear on one line instead of 2x2.
            try:
                # determine number of plots
                if isinstance(shap_values, (list, tuple)):
                    n_plots = len(shap_values)
                elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                    n_plots = shap_values.shape[0]
                else:
                    n_plots = 1

                if n_plots == 1:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, Xs_proc, feature_names=use_names, show=False)
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150)
                    plt.close()
                else:
                    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
                    if n_plots == 1:
                        axes = [axes]
                    for i in range(n_plots):
                        ax = axes[i]
                        plt.sca(ax)
                        vals = shap_values[i] if isinstance(shap_values, (list, tuple)) else shap_values[i]
                        shap.summary_plot(vals, Xs_proc, feature_names=use_names, show=False)
                        ax.set_title(f"SHAP summary (class {i})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150)
                    plt.close()
            except Exception:
                # fallback to default single plot
                try:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, Xs_proc, feature_names=use_names, show=False)
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150)
                    plt.close()
                except Exception:
                    try:
                        plt.close()
                    except Exception:
                        pass
        except Exception:
            try:
                plt.close()
            except Exception:
                pass

    except Exception as e:
        # fallback to permutation importance
        from sklearn.inspection import permutation_importance
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline) and len(model.steps) > 1:
            preprocessor = Pipeline(model.steps[:-1])
            estimator = model.steps[-1][1]
            X_proc = preprocessor.transform(Xs)
        else:
            estimator = model
            X_proc = Xs.values
        # get baseline predictions
        r = permutation_importance(estimator, X_proc, features_df[label].loc[Xs.index].astype(int) if label in features_df.columns else None, n_repeats=5, random_state=42, n_jobs=-1)
        imp = r.importances_mean
        importances = sorted(zip(Xs.columns if hasattr(Xs,'columns') else list(range(len(imp))), imp), key=lambda x: x[1], reverse=True)
        # plot
        names = [i[0] for i in importances]
        vals = [i[1] for i in importances]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, max(3, 0.3 * len(names))))
        sns.barplot(x=vals, y=names, palette="mako")
        plt.title("Permutation feature importances (fall-back)")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "permutation_importances.png"), dpi=150)
        plt.close()

    # Also save logistic coefficients if present
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline) and len(model.steps) > 1:
            estimator = model.steps[-1][1]
        else:
            estimator = model
        if hasattr(estimator, "coef_"):
            coefs = estimator.coef_.ravel()
            names = Xs.columns if hasattr(Xs, 'columns') else [f"f{i}" for i in range(len(coefs))]
            df_coef = pd.DataFrame({"feature": names, "coef": coefs})
            df_coef = df_coef.reindex(df_coef.coef.abs().sort_values(ascending=False).index)
            # plot
            plt.figure(figsize=(10, max(3, 0.3 * len(names))))
            sns.barplot(x="coef", y="feature", data=df_coef, palette="viridis")
            plt.title("Model coefficients (absolute values)")
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, "model_coefficients.png"), dpi=150)
            plt.close()
            df_coef.to_csv(os.path.join(target_dir, "model_coefficients.csv"), index=False)
    except Exception:
        pass


def detect_label_leakage(features_df: pd.DataFrame, label: str = "converted", corr_threshold: float = 0.9, sample_frac: float = 0.1) -> Dict[str, Any]:
    """
    Detect potential label leakage in feature set. Returns a report (dictionary) with flagged features.
    Heuristics:
    - Exact equality with boolean label
    - High correlation (>corr_threshold)
    """
    report = {"flagged": [], "summary": {}}
    if label not in features_df.columns:
        raise ValueError("Label column not found")
    y = features_df[label].astype(int)
    # Candidate features exclude ids and label
    exclude = {label, "session_id", "user_id"}
    candidate_cols = [c for c in features_df.columns if c not in exclude and c != label]
    # simple checks
    for col in candidate_cols:
        ser = features_df[col]
        reason = []
        if ser.nunique() == 2:
            # compare exact equality mapping
            try:
                if np.array_equal(ser.astype(int).fillna(0).values, y.values):
                    reason.append("exact_match_to_label")
            except Exception:
                pass
        # numeric correlation
        if pd.api.types.is_numeric_dtype(ser):
            try:
                corr = abs(ser.fillna(0).astype(float).corr(y))
                if corr >= corr_threshold:
                    reason.append(f"high_corr_{corr:.3f}")
            except Exception:
                corr = None
        else:
            corr = None
        if len(reason) > 0:
            report["flagged"].append({"feature": col, "reasons": reason, "corr": corr})
    report["count_flagged"] = len(report["flagged"])
    return report


def match_orders_to_sessions(orders_path: str, sessions_df: pd.DataFrame, time_window_hours: int = 1, order_by_user: bool = True) -> pd.DataFrame:
    """
    Attempt to match orders to sessions using user_id and nearest prior session within a time window.
    Returns sessions_df with a new column `converted_by_order` (bool) and an `matched_order_id` if matched.
    """
    import pandas as pd
    orders = pd.read_csv(orders_path, parse_dates=["created_at"])  # order creation time
    # drop missing user_id orders
    orders = orders.dropna(subset=["user_id"]).copy()
    # ensure datetimes and normalize to naive tz
    sessions_df["session_start"] = pd.to_datetime(sessions_df["session_start"], utc=True, errors="coerce").dt.tz_convert(None)
    sessions_df["session_end"] = pd.to_datetime(sessions_df["session_end"], utc=True, errors="coerce").dt.tz_convert(None)
    # If session_end NaT, fallback to session_start
    sessions_df["session_end"] = sessions_df["session_end"].fillna(sessions_df["session_start"]) 

    # sort for merge_asof
    # normalize order timestamps
    orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True, errors="coerce").dt.tz_convert(None)
    orders = orders.dropna(subset=["created_at"]).copy()
    # normalize user_id as string to avoid dtype mismatch
    orders["user_id"] = orders["user_id"].astype("Int64").astype(str).fillna("")
    sessions_df["user_id"] = sessions_df["user_id"].astype("Int64").astype(str).fillna("")
    # Ensure not-null user_id for merge: numeric conversion, drop invalid then stringify
    orders["user_id"] = pd.to_numeric(orders["user_id"], errors="coerce")
    orders = orders.dropna(subset=["user_id"]).copy()
    orders["user_id"] = orders["user_id"].astype(int).astype(str)
    sessions_df["user_id"] = pd.to_numeric(sessions_df["user_id"], errors="coerce")
    sessions_df = sessions_df.dropna(subset=["user_id"]).copy()
    sessions_df["user_id"] = sessions_df["user_id"].astype(int).astype(str)
    sessions_df = sessions_df.dropna(subset=["session_start"]).copy()
    orders_sorted = orders.sort_values(["user_id", "created_at"])  # created_at order
    sessions_sorted = sessions_df.sort_values(["user_id", "session_start"]).copy()
    # Drop any residual null/empty user ids before merge
    orders_sorted = orders_sorted[orders_sorted["user_id"].notna() & (orders_sorted["user_id"] != "")].copy()
    sessions_sorted = sessions_sorted[sessions_sorted["user_id"].notna() & (sessions_sorted["user_id"] != "")].copy()
    # Use per-user merge_asof to avoid global sort issues and improve robustness
    merged_list = []
    user_groups = orders_sorted.groupby("user_id")
    # We'll iterate users but stop if too slow; this is more robust for dtype/sort issues
    for uid, og in user_groups:
        sg = sessions_sorted[sessions_sorted["user_id"] == uid]
        if sg.empty:
            continue
        og_sorted = og.sort_values("created_at")
        sg_sorted = sg.sort_values("session_start")
        m = pd.merge_asof(og_sorted, sg_sorted, left_on="created_at", right_on="session_start", by="user_id", direction="backward", suffixes=("_order", "_session"))
        merged_list.append(m)
    if merged_list:
        merged = pd.concat(merged_list, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=list(orders.columns) + list(sessions_df.columns))
    # compute time delta between order created and session_end
    merged["delta_seconds"] = (merged["created_at"] - merged["session_end"]).dt.total_seconds()
    # match if order.created_at between session.start and session.end + time_window
    time_window_secs = time_window_hours * 3600
    matched = merged[merged["delta_seconds"] >= 0]
    matched = matched[matched["delta_seconds"] <= time_window_secs]
    # get first matching order per session
    # Build dict session_id -> matched order
    matched_map = matched.groupby("session_id").agg({"order_id": "first"}).to_dict()["order_id"]
    # apply to sessions_df
    sessions = sessions_df.copy()
    sessions["matched_order_id"] = sessions["session_id"].map(matched_map).astype(pd.Int64Dtype())
    sessions["converted_by_order"] = ~sessions["matched_order_id"].isna()
    return sessions


def summary_stats(df: pd.DataFrame) -> pd.Series:
    out = {
        "n_sessions": len(df),
        "n_converted": int(df["converted"].sum()),
        "conversion_rate": float(df["converted"].mean()),
        "median_session_duration_secs": float(df["session_duration_seconds"].median()),
        # mean events per session computed from available event-type count columns (exclude ids/label)
        "mean_events_per_session": float(df[[c for c in df.columns if c.startswith('num_')]].sum(axis=1).mean()) if any(c.startswith('num_') for c in df.columns) else 0.0,
    }
    return pd.Series(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="dataset/events.csv", help="Path to events.csv")
    parser.add_argument("--output-dir", default="outputs", help="Where to write outputs")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to read from events.csv (for quick runs)")
    parser.add_argument("--automl-sample-frac", type=float, default=1.0, help="Sample fraction of sessions for AutoML (0-1)")
    parser.add_argument("--train-size", type=float, default=0.8, help="Train fraction of data to use for AutoML; validation will be 1-train-size interval (default 0.8)")
    parser.add_argument("--drop-leaky-features", action="store_true", help="Drop label-leaky features detected by detect_label_leakage before training")
    # Note: leak threshold was removed; detect_label_leakage uses its default corr threshold
    parser.add_argument("--skip-automl", action="store_true", help="Skip running AutoML even if PyCaret installed")
    parser.add_argument("--analysis", action="store_true", help="Run additional analysis: leakage detection, model evaluation, SHAP, order matching")
    parser.add_argument("--order-window-hours", type=int, default=1, help="Time window in hours for matching orders to sessions")
    args = parser.parse_args()

    print("Reading events and extracting session features. This may take time...")
    features = extract_sessions(args.events, chunksize=200000, max_rows=args.max_rows)
    print("Done. Summary:")
    print(summary_stats(features))
    # NOTE: Do not persist the full session features CSV to avoid accidental use of
    # label-derived count columns downstream. The train/validation splits used for
    # modeling will be saved by run_pycaret_automl (train.csv and validation/validation_set*.csv).
    print("Session-level features extracted (not saved as session_features.csv).")

    print("Generating visualizations...")
    visualize_session_features(features, out_dir=os.path.join(args.output_dir, "figures"))

    try:
        if not args.skip_automl:
            print("Running PyCaret AutoML on a sample (this may take a while)...")
            run_pycaret_automl(features, sample_frac=args.automl_sample_frac, target_dir=os.path.join(args.output_dir, "pycaret"), train_size=args.train_size, drop_leaky_features=args.drop_leaky_features)
            print("PyCaret finished. Best model saved to outputs/pycaret.")
            # If training columns were saved, produce a model-specific visualization limited
            # to the features used for training. Save these in a separate folder.
            train_cols_path = os.path.join(args.output_dir, "pycaret", "train_columns.json")
            if os.path.exists(train_cols_path):
                try:
                    model_fig_dir = os.path.join(args.output_dir, "figures_model")
                    visualize_session_features(features, out_dir=model_fig_dir, sample_frac=0.2, train_columns_path=train_cols_path)
                except Exception:
                    pass
        else:
            print("Skipping AutoML as requested.")
    except RuntimeError as e:
        print(e)

    if args.analysis:
        analysis_dir = os.path.join(args.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        print("Running leakage detection...")
        leak_report = detect_label_leakage(features)
        import json
        with open(os.path.join(analysis_dir, "leak_report.json"), "w") as f:
            json.dump(leak_report, f, indent=2)
        print("Saved leak report.")

        # Evaluate original saved model (if exists)
        model_path = os.path.join(args.output_dir, "pycaret", "best_pycaret_model")
        # Load validation set used by run_pycaret_automl if present (CSV)
        val_path = os.path.join(args.output_dir, "pycaret", "validation", "validation_set.csv")
        val_df = None
        if os.path.exists(val_path):
            try:
                val_df = pd.read_csv(val_path)
            except Exception:
                val_df = None
        try:
            print("Evaluating saved PyCaret model on holdout (converted label)...")
            metrics = evaluate_pycaret_model(model_path, features_df=None, val_df=val_df if val_df is not None else None, label="converted", test_frac=1-args.train_size, random_state=42, target_dir=os.path.join(analysis_dir, "orig_model"))
            print("Saved evaluation metrics for original model.")
        except Exception as e:
            print("Evaluation of original model failed:", e)

        # SHAP for original model
        try:
            print("Computing SHAP / feature importance for original model...")
            compute_feature_importance_shap(model_path, features, val_df=val_df, label="converted", sample_size=2000, target_dir=os.path.join(analysis_dir, "orig_model"))
            print("Saved SHAP and coeffs for original model.")
        except Exception as e:
            print("SHAP for original model failed:", e)

        # (order-based AutoML and re-run steps removed as per configuration)
