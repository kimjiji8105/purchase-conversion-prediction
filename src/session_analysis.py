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


# ==============================================================================
# DATA PIPELINE: Event Processing → Session Aggregation → Feature Engineering
# ==============================================================================

def _process_event_row(row: pd.Series, sessions: Dict[str, Dict[str, Any]]) -> None:
    """
    Process a single event row and update the sessions dictionary.
    This consolidates the duplicate row-processing logic.
    """
    session_id = row["session_id"]
    if pd.isna(session_id):
        return
    
    session_id = str(session_id)
    if session_id not in sessions:
        sessions[session_id] = _init_session_entry(session_id, row)
    
    entry = sessions[session_id]
    
    # Update user_id (prefer first non-null)
    if entry["user_id"] is None and not pd.isna(row.get("user_id")):
        entry["user_id"] = row.get("user_id")
    
    # Update time range
    if row["created_at"] < entry["min_time"]:
        entry["min_time"] = row["created_at"]
    if row["created_at"] > entry["max_time"]:
        entry["max_time"] = row["created_at"]
    
    # Update event counts
    entry["num_events"] += 1
    event_type = row.get("event_type") or "unknown"
    entry["event_counts"][event_type] += 1
    
    # Track unique URIs
    if not pd.isna(row.get("uri")):
        entry["unique_uris"].add(row.get("uri"))
    
    # Update session attributes (keep first non-null)
    for col in ("browser", "traffic_source", "city", "state"):
        if entry.get(col) is None and not pd.isna(row.get(col)):
            entry[col] = row.get(col)


def _merge_user_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user demographic data (age, gender, country) from users.csv.
    This consolidates duplicate user merge logic.
    """
    try:
        users_path = os.path.join("dataset", "users.csv")
        if not os.path.exists(users_path):
            return df
        
        users_df = pd.read_csv(users_path)
        users_df["user_id"] = pd.to_numeric(users_df["user_id"], errors="coerce")
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        
        # Only keep safe demographic columns (no membership flags)
        users_keep = [c for c in ("user_id", "age", "gender", "country") if c in users_df.columns]
        if len(users_keep) > 1:
            df = df.merge(users_df[users_keep], on="user_id", how="left")
    except Exception:
        pass
    
    return df


def _apply_label_leakage_protection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that could leak label information.
    - Creates 'converted' label from num_purchase (before removal)
    - Removes all num_* count columns to prevent leakage
    """
    # Define converted label (must be done before dropping num_purchase)
    if "num_purchase" in df.columns:
        df["converted"] = df["num_purchase"] > 0
    else:
        df["converted"] = False
    
    # Remove all raw count columns to avoid label leakage
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if num_cols:
        df = df.drop(columns=num_cols)
    
    return df


def apply_onehot_encoding(df: pd.DataFrame, categorical_cols: list = None, drop_first: bool = False) -> pd.DataFrame:
    """
    PIPELINE STAGE: Apply one-hot encoding to categorical features.
    
    This ensures all downstream operations (visualization, modeling, SHAP) work with
    consistently encoded data.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of columns to encode. If None, auto-detect object/category dtypes
        drop_first: Whether to drop first dummy variable (for linear models to avoid multicollinearity)
    
    Returns:
        DataFrame with one-hot encoded categorical features
    """
    df = df.copy()
    
    # Auto-detect categorical columns if not specified
    if categorical_cols is None:
        categorical_cols = [
            col for col in df.columns 
            if col != 'converted' and (
                pd.api.types.is_object_dtype(df[col]) or 
                pd.api.types.is_categorical_dtype(df[col])
            )
        ]
    
    # Filter to only existing columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if not categorical_cols:
        return df
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first, dtype=int)
    
    return df_encoded


def extract_sessions_from_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    PIPELINE STAGE 1: Convert events DataFrame to session-level features.
    
    Input: events DataFrame with columns: session_id, created_at, event_type, uri
           (optional: user_id, browser, traffic_source, city, state)
    Output: Session-level DataFrame with aggregated features and 'converted' label
    
    Pipeline:
    1. Normalize timestamps
    2. Aggregate events by session
    3. Merge user demographics
    4. Create label and remove leaking features
    """
    sessions: Dict[str, Dict[str, Any]] = {}

    # Normalize timestamps
    if not pd.api.types.is_datetime64_any_dtype(events_df["created_at"]):
        events_df["created_at"] = pd.to_datetime(
            events_df["created_at"], utc=True, errors="coerce"
        )

    # Sort for consistent aggregation
    events_df = events_df.sort_values(["session_id", "created_at"])

    # Aggregate events into sessions
    for _, row in events_df.iterrows():
        _process_event_row(row, sessions)

    # Convert to DataFrame
    data = [_finalize_session_entry(entry) for entry in sessions.values()]
    df = pd.DataFrame(data)

    # Enrich with user demographics
    df = _merge_user_demographics(df)
    
    # Apply label leakage protection
    df = _apply_label_leakage_protection(df)
    
    return df


def extract_sessions(events_path: str, chunksize: int = 100000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    PIPELINE STAGE 1: Process events.csv in chunks and extract session-level features.
    
    This is a memory-efficient version of extract_sessions_from_df that processes
    large files in chunks.
    
    Pipeline:
    1. Read events in chunks
    2. Aggregate events by session (streaming)
    3. Merge user demographics
    4. Create label and remove leaking features
    """
    fields = [
        "id", "user_id", "sequence_number", "session_id", "created_at",
        "ip_address", "city", "state", "postal_code", "browser",
        "traffic_source", "uri", "event_type",
    ]

    sessions: Dict[str, Dict[str, Any]] = {}
    read_rows = 0
    
    # Process chunks
    for chunk in pd.read_csv(events_path, usecols=fields, parse_dates=["created_at"], chunksize=chunksize):
        if max_rows is not None and read_rows >= max_rows:
            break
        
        if max_rows is not None and read_rows + len(chunk) > max_rows:
            chunk = chunk.head(max_rows - read_rows)
        read_rows += len(chunk)

        # Normalize timestamps
        if not pd.api.types.is_datetime64_any_dtype(chunk["created_at"]):
            chunk["created_at"] = pd.to_datetime(
                chunk["created_at"], utc=True, errors="coerce"
            )
        
        # Aggregate events into sessions
        for _, row in chunk.iterrows():
            _process_event_row(row, sessions)

    # Convert to DataFrame
    data = [_finalize_session_entry(entry) for entry in sessions.values()]
    df = pd.DataFrame(data)
    
    # Enrich with user demographics
    df = _merge_user_demographics(df)
    
    # Apply label leakage protection
    df = _apply_label_leakage_protection(df)
    
    return df


def create_session_features(events_df: pd.DataFrame = None, events_path: str = None, chunksize: int = 100000, max_rows: Optional[int] = None, apply_encoding: bool = True) -> pd.DataFrame:
    """
    MAIN ENTRY POINT: Build session-level features from events.
    
    This is the recommended interface for feature extraction.
    
    Pipeline:
    1. Extract session features from events
    2. Merge user demographics
    3. Create label and remove leaking features
    4. Apply one-hot encoding to categorical features
    
    Args:
        events_df: In-memory events DataFrame (for small datasets or testing)
        events_path: Path to events.csv (for production, large datasets)
        chunksize: Number of rows to process at once when reading from file
        max_rows: Maximum rows to read (for quick testing)
        apply_encoding: If True, apply one-hot encoding to categorical features (default: True)
    
    Returns:
        Session-level DataFrame with features and 'converted' label (one-hot encoded if apply_encoding=True)
    """
    if events_df is not None:
        df = extract_sessions_from_df(events_df)
    elif events_path is not None:
        df = extract_sessions(events_path, chunksize=chunksize, max_rows=max_rows)
    else:
        raise ValueError("Either events_df or events_path must be provided")
    
    # Apply one-hot encoding to categorical features
    if apply_encoding:
        df = apply_onehot_encoding(df, categorical_cols=['browser', 'traffic_source'])
    
    return df


# ==============================================================================
# VISUALIZATION: Session Feature Analysis
# ==============================================================================

def _plot_conversion_by_category(df: pd.DataFrame, cat_col: str, out_name: str, out_dir: str, top_n: int = 10, min_count: int = 20) -> None:
    """Helper: Plot conversion rate by categorical feature with annotations.
    
    Works with both one-hot encoded and raw categorical data.
    """
    if "converted" not in df.columns:
        return
    
    # Check if we have one-hot encoded columns (e.g., browser_Chrome, browser_Firefox)
    onehot_cols = [col for col in df.columns if col.startswith(f"{cat_col}_")]
    
    if onehot_cols:
        # Reconstruct categorical data from one-hot encoding for visualization
        category_data = []
        for idx in df.index:
            found = False
            for col in onehot_cols:
                if df.loc[idx, col] == 1:
                    category_data.append(col.replace(f"{cat_col}_", ""))
                    found = True
                    break
            if not found:
                # If no 1 found (e.g., drop_first=True case), infer the dropped category
                category_data.append("Other")
        
        temp_df = df.copy()
        temp_df[cat_col] = category_data
        
        agg = temp_df.groupby(cat_col)["converted"].agg(["mean", "count"]).reset_index()
    elif cat_col in df.columns:
        # Raw categorical column exists
        agg = df.groupby(cat_col)["converted"].agg(["mean", "count"]).reset_index()
    else:
        return
    
    agg = agg[agg["count"] >= min_count].sort_values("mean", ascending=False)
    
    if agg.empty:
        return
    
    top = agg.head(top_n).sort_values("mean")
    
    plt.figure(figsize=(10, max(4, 0.4 * len(top))))
    ax = sns.barplot(x="mean", y=cat_col, data=top, palette="viridis")
    
    plt.xlabel("Conversion Rate", fontsize=12, fontweight='bold')
    plt.ylabel(cat_col.replace("_", " ").title(), fontsize=12, fontweight='bold')
    plt.title(f"Conversion Rate by {cat_col.replace('_', ' ').title()}\n(Top {top_n}, Min {min_count} sessions)", 
              fontsize=14, fontweight='bold', pad=20)
    
    # Annotate bars with percentage and counts
    max_val = top["mean"].max()
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["mean"] + max_val * 0.02, i, 
                f"{row['mean']:.1%} (n={int(row['count']):,})", 
                va="center", fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, max_val * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_name), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_session_duration_analysis(df_plot: pd.DataFrame, out_dir: str) -> None:
    """Helper: Plot session duration distributions and comparisons."""
    if "session_duration_seconds" not in df_plot.columns:
        return
    
    # Add log-transformed column
    df_plot["session_duration_log1p"] = np.log1p(df_plot["session_duration_seconds"].fillna(0).astype(float))
    
    durations = df_plot["session_duration_seconds"].fillna(0).astype(float)
    durations_pos = durations[durations > 0]
    
    if len(durations_pos) == 0:
        return
    
    # Clip at 99th percentile for cleaner visualization
    cap = min(durations_pos.quantile(0.99), durations_pos.max())
    x = durations_pos.clip(upper=cap)
    
    # 1. Distribution plot
    plt.figure(figsize=(10, 5))
    sns.histplot(x=np.log1p(x), bins=50, kde=True, color="#3498db", edgecolor='black', alpha=0.7)
    plt.xlabel("log1p(Session Duration in Seconds)", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.title("Session Duration Distribution (Log-Transformed)\nClipped at 99th Percentile", 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "session_duration_log_hist.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Comparison by conversion status
    if "converted" in df_plot.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="converted", y="session_duration_log1p", data=df_plot, hue="converted",
                   palette={False: "#e74c3c", True: "#2ecc71"}, legend=False)
        plt.xlabel("Converted", fontsize=12, fontweight='bold')
        plt.ylabel("log1p(Session Duration in Seconds)", fontsize=12, fontweight='bold')
        plt.title("Session Duration by Conversion Status\n(Log-Transformed)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "session_duration_by_conversion_box.png"), dpi=150, bbox_inches='tight')
        plt.close()


def visualize_session_features(features_df: pd.DataFrame, out_dir: str = "outputs/figures", sample_frac: float = 0.2, train_columns_path: Optional[str] = None) -> None:
    """
    PIPELINE STAGE 2: Create comprehensive visualizations of session features.
    
    Generates:
    - Conversion funnel (stacked bar chart)
    - Session duration distributions
    - Conversion rates by traffic source and browser
    - Feature correlation heatmap
    - Summary dashboard
    
    Args:
        features_df: Session-level features from extract_sessions()
        out_dir: Output directory for figures
        sample_frac: Sampling fraction for large datasets (0-1)
        train_columns_path: Optional path to train_columns.json to limit viz to trained features
    """
    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_palette("husl")

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
                # If a train CSV exists next to train_columns.json, prefer using that
                # dataset for numeric correlation calculation because it represents
                # the exact data the model trained on (sampling, filtering, encoding
                # aside). This helps show correlations as seen by the model.
                train_csv = os.path.join(os.path.dirname(train_columns_path), "train.csv")
                train_df_for_corr = None
                if os.path.exists(train_csv):
                    try:
                        train_df_for_corr = pd.read_csv(train_csv)
                    except Exception:
                        train_df_for_corr = None
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

    # Session duration analysis
    _plot_session_duration_analysis(df_plot, out_dir)

    # Conversion rate by categorical features
    _plot_conversion_by_category(df, "traffic_source", "conversion_by_traffic_source.png", out_dir)
    _plot_conversion_by_category(df, "browser", "conversion_by_browser.png", out_dir)

    # num_unique_uris by conversion
    if "num_unique_uris" in df_plot.columns and "converted" in df_plot.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="converted", y="num_unique_uris", data=df_plot, palette=sns.color_palette(["#FDB462", "#B3DE69"])) 
        plt.title("Unique URIs visited by conversion")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "unique_uris_by_conversion_box.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Correlation heatmap of numeric features
    # If we loaded a train CSV (train_df_for_corr), prefer that for correlation
    # since it reflects the exact data used for modeling. Otherwise fall back to
    # the sampled df_plot.
    corr_source = None
    try:
        if 'train_df_for_corr' in locals() and train_df_for_corr is not None:
            corr_source = train_df_for_corr
        else:
            corr_source = df_plot
        numeric_cols = [c for c in corr_source.columns if pd.api.types.is_numeric_dtype(corr_source[c])]
        numeric_cols = [c for c in numeric_cols if c not in ("id", "user_id")]
        if len(numeric_cols) >= 2:
            corr = corr_source[numeric_cols].corr()
            plt.figure(figsize=(max(6, 0.6 * len(numeric_cols)), max(6, 0.6 * len(numeric_cols))))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.5})
            title = "Correlation matrix of numeric features"
            if 'train_df_for_corr' in locals() and train_df_for_corr is not None:
                title += " (model train set)"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "feature_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()
    except Exception:
        # If correlation plotting fails, continue without aborting visualizations
        pass

    # Summary dashboard: 2x1 (duration, traffic source)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Duration (left)
        ax = axes[0]
        if "session_duration_log1p" in df_plot.columns:
            sns.histplot(df_plot["session_duration_log1p"], bins=40, kde=True, ax=ax, color="#5A9")
            ax.set_xlabel("log1p(Session duration seconds)")
            ax.set_title("Session duration (log1p)")

        # Traffic source conversion (right)
        ax = axes[1]
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


# ==============================================================================
# MODELING: AutoML Training Pipeline
# ==============================================================================

def _save_validation_metrics_and_plots(y_true, y_pred, y_proba, target_dir: str, subdir: str = "validation", 
                                       metrics_filename: str = "validation_metrics.json", 
                                       save_classification_report: bool = False):
    """Helper: Save validation metrics, confusion matrix, ROC curve, and optionally classification report."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
    import matplotlib.pyplot as plt
    import json
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
    }
    
    # Determine output directory
    out_dir = os.path.join(target_dir, subdir) if subdir else target_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Save metrics JSON
    with open(os.path.join(out_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    
    # ROC curve
    if y_proba is not None:
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
        fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)
    
    # Classification report
    if save_classification_report:
        with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred))
    
    return metrics


def run_pycaret_automl(features_df: pd.DataFrame, label: str = "converted", sample_frac: float = 0.2, target_dir: str = "outputs/pycaret", train_size: float = 0.8):
    """
    PIPELINE STAGE 3: Train classification model using AutoML.
    
    Pipeline steps:
    1. Remove label-leaking features (num_* columns)
    2. Split into train/validation sets
    3. Train model using PyCaret AutoML (or sklearn fallback)
    4. Save model, training data, and evaluation metrics
    
    Args:
        features_df: Session features with 'converted' label
        label: Target column name (default: "converted")
        sample_frac: Fraction to sample for faster training (0-1)
        target_dir: Output directory for model and artifacts
        train_size: Train split ratio (default: 0.8)
    
    Returns:
        Trained model object
    """
    pycaret_available = True
    try:
        from pycaret.classification import setup, compare_models, save_model, predict_model
    except Exception:
        # PyCaret not available in this environment — fall back to sklearn pipeline below.
        pycaret_available = False

    os.makedirs(target_dir, exist_ok=True)
    df = features_df.copy()
    
    # Permanently remove any 'num_' count columns from modeling to avoid unintended signal/label leakage
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if num_cols:
        print(f"Removing count columns from features before modeling: {num_cols}")
        df = df.drop(columns=num_cols)
    
    # Select useful columns (including one-hot encoded columns)
    # Keep numeric features and all one-hot encoded categorical features
    # Exclude IDs and datetime columns
    exclude_cols = ['session_id', 'user_id', 'session_start', 'session_end', 'city', 'state']
    feature_cols = [c for c in df.columns if c != label and c not in exclude_cols]
    cols = feature_cols + [label]
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

    # split into train / validation explicitly using sklearn and run AutoML on train set
    from sklearn.model_selection import train_test_split
    stratify_col = None
    if label in df.columns:
        val_counts = df[label].value_counts(dropna=True)
        if val_counts.min() >= 2:
            stratify_col = df[label]
    train_df, val_df = train_test_split(df, train_size=train_size, stratify=stratify_col, random_state=42)
    # Save explicit train/validation CSVs used for modeling
    try:
        os.makedirs(target_dir, exist_ok=True)
        train_path = os.path.join(target_dir, "train.csv")
        val_dir = os.path.join(target_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(os.path.join(val_dir, "validation_set.csv"), index=False)
    except Exception:
        pass
    # PyCaret setup uses train set only
    # For tiny datasets, PyCaret may not behave well — fallback to a simple sklearn pipeline
    use_sklearn_fallback = len(train_df) < 10 or len(val_df) < 2
    if pycaret_available and not use_sklearn_fallback:
        # Since data is already one-hot encoded, we tell PyCaret not to re-encode
        # by treating all columns as numeric (which they are after one-hot encoding)
        clf_setup = setup(
            data=train_df,
            target=label,
            train_size=0.8 if train_size >= 0.8 else train_size,
            session_id=42,
            log_experiment=False,
            verbose=False,
            html=False,
            n_jobs=-1,
            categorical_features=[],  # All features already encoded as numeric
            numeric_features=[c for c in train_df.columns if c != label],
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
        
        # Data is already one-hot encoded, just ensure alignment
        X_train_enc = X_train.copy()
        X_val_enc = X_val.copy()
        # align columns (in case of any mismatch)
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
        # Note: validation_set.csv already saved above with full validation data
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
            _save_validation_metrics_and_plots(y_true, y_pred, y_proba, target_dir)
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
        _save_validation_metrics_and_plots(y_true, y_pred, y_proba, target_dir)
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

    metrics = _save_validation_metrics_and_plots(
        y_true, y_pred, y_proba, target_dir, 
        subdir=None,  # Save directly in target_dir
        metrics_filename="evaluation_metrics.json",
        save_classification_report=True
    )
    return metrics


def compute_feature_importance_shap(model_path: str, features_df: pd.DataFrame = None, val_df: pd.DataFrame = None, label: str = "converted", sample_size: int = 2000, target_dir: str = "outputs/analysis"):
    """
    Compute feature importances using coefficient-based approach (if linear) and SHAP if available.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Try PyCaret first, fallback to joblib
    model = None
    try:
        from pycaret.classification import load_model
        model = load_model(model_path)
    except Exception:
        # Try loading with joblib (for sklearn pipelines saved directly)
        try:
            import joblib
            pkl_path = model_path if model_path.endswith('.pkl') else model_path + '.pkl'
            model = joblib.load(pkl_path)
        except Exception as e:
            raise RuntimeError(f"Could not load model from {model_path}: {e}")

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
                # Prefer using the original train.csv if present so SHAP uses the exact
                # data the model trained on (sampling/filters aside). Otherwise fall
                # back to the provided features/val set limited to train columns.
                train_csv = os.path.join(os.path.dirname(train_cols_path), "train.csv")
                try:
                    if os.path.exists(train_csv):
                        train_df = pd.read_csv(train_csv)
                        # keep train columns that exist in train_df
                        train_keep = [c for c in train_cols if c in train_df.columns]
                        if train_keep:
                            X = train_df[train_keep].copy()
                        else:
                            X = X[keep]
                    else:
                        X = X[keep]
                except Exception:
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
            # let the pipeline preprocessor handle categorical encoding/transform
            Xs_proc = preprocessor.transform(Xs)
        else:
            # direct estimator: ensure Xs_proc is numeric (encode categoricals)
            estimator = model
            Xs_proc_df = Xs.copy()
            cat_cols = [c for c in Xs_proc_df.columns if pd.api.types.is_object_dtype(Xs_proc_df[c]) or pd.api.types.is_categorical_dtype(Xs_proc_df[c])]
            if len(cat_cols) > 0:
                try:
                    Xs_proc_df = pd.get_dummies(Xs_proc_df, columns=cat_cols, drop_first=False)
                except Exception:
                    # fallback: convert categoricals to codes
                    for c in cat_cols:
                        Xs_proc_df[c] = Xs_proc_df[c].astype('category').cat.codes
            # final numeric array for explainer
            try:
                Xs_proc = Xs_proc_df.values
            except Exception:
                Xs_proc = np.asarray(Xs_proc_df)

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
            if feature_names is None:
                # if we used a DataFrame and didn't let preprocessor handle encoding,
                # try to infer names from Xs (after get_dummies) or Xs_proc_df
                if 'Xs_proc_df' in locals() and hasattr(Xs_proc_df, 'columns'):
                    feature_names = list(Xs_proc_df.columns)
                elif hasattr(Xs, 'columns'):
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

                # Calculate dynamic height based on number of features
                n_features = Xs_proc.shape[1] if hasattr(Xs_proc, 'shape') else len(use_names) if use_names else 10
                fig_height = max(6, n_features * 0.4)  # At least 6 inches, 0.4 inch per feature
                
                if n_plots == 1:
                    plt.figure(figsize=(12, fig_height))
                    shap.summary_plot(shap_values, Xs_proc, feature_names=use_names, show=False, max_display=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150, pad_inches=0.5)
                    plt.close()
                else:
                    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, fig_height))
                    if n_plots == 1:
                        axes = [axes]
                    for i in range(n_plots):
                        ax = axes[i]
                        plt.sca(ax)
                        vals = shap_values[i] if isinstance(shap_values, (list, tuple)) else shap_values[i]
                        shap.summary_plot(vals, Xs_proc, feature_names=use_names, show=False, max_display=20)
                        ax.set_title(f"SHAP summary (class {i})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150, pad_inches=0.5)
                    plt.close()
            except Exception:
                # fallback to default single plot
                try:
                    n_features = Xs_proc.shape[1] if hasattr(Xs_proc, 'shape') else 10
                    fig_height = max(6, n_features * 0.4)
                    plt.figure(figsize=(12, fig_height))
                    shap.summary_plot(shap_values, Xs_proc, feature_names=use_names, show=False, max_display=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(target_dir, "shap_summary.png"), bbox_inches='tight', dpi=150, pad_inches=0.5)
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
        # plot with dynamic height
        names = [i[0] for i in importances]
        vals = [i[1] for i in importances]
        import matplotlib.pyplot as plt
        fig_height = max(4, len(names) * 0.35)
        plt.figure(figsize=(12, fig_height))
        sns.barplot(x=vals, y=names, palette="mako")
        plt.title("Permutation Feature Importances", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Importance", fontsize=11, fontweight='bold')
        plt.ylabel("Feature", fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "permutation_importances.png"), dpi=150, bbox_inches='tight', pad_inches=0.3)
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
            # plot with dynamic height
            fig_height = max(4, len(names) * 0.35)
            plt.figure(figsize=(12, fig_height))
            sns.barplot(x="coef", y="feature", data=df_coef, palette="viridis")
            plt.title("Model Coefficients", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Coefficient Value", fontsize=11, fontweight='bold')
            plt.ylabel("Feature", fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, "model_coefficients.png"), dpi=150, bbox_inches='tight', pad_inches=0.3)
            plt.close()
            df_coef.to_csv(os.path.join(target_dir, "model_coefficients.csv"), index=False)
    except Exception:
        pass


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
    parser.add_argument("--skip-automl", action="store_true", help="Skip running AutoML even if PyCaret installed")
    parser.add_argument("--analysis", action="store_true", help="Run additional analysis: leakage detection, model evaluation, SHAP, order matching")
    parser.add_argument("--order-window-hours", type=int, default=1, help="Time window in hours for matching orders to sessions")
    args = parser.parse_args()

    print("Reading events and extracting session features. This may take time...")
    print("Pipeline: Events → Sessions → One-Hot Encoding")
    features = create_session_features(events_path=args.events, chunksize=200000, max_rows=args.max_rows, apply_encoding=True)
    print("Done. Summary:")
    print(summary_stats(features))
    print(f"Features extracted with shape: {features.shape}")
    print(f"Columns: {list(features.columns[:10])}..." if len(features.columns) > 10 else f"Columns: {list(features.columns)}")
    # NOTE: Do not persist the full session features CSV to avoid accidental use of
    # label-derived count columns downstream. The train/validation splits used for
    # modeling will be saved by run_pycaret_automl (train.csv and validation/validation_set*.csv).
    print("Session-level features extracted with one-hot encoding applied.")

    print("Generating visualizations from one-hot encoded features...")
    visualize_session_features(features, out_dir=os.path.join(args.output_dir, "figures"))

    try:
        if not args.skip_automl:
            print("Running PyCaret AutoML on a sample (this may take a while)...")
            run_pycaret_automl(features, sample_frac=args.automl_sample_frac, target_dir=os.path.join(args.output_dir, "pycaret"), train_size=args.train_size)
            print("PyCaret finished. Best model saved to outputs/pycaret.")
        else:
            print("Skipping AutoML as requested.")
    except RuntimeError as e:
        print(e)

    if args.analysis:
        analysis_dir = os.path.join(args.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

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
