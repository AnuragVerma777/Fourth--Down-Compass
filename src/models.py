"""
src/models.py
─────────────
HistGradientBoosting models for the NFL 4th-down decision tool:
  1. Win Probability (WP) classifier
  2. 4th-down Conversion (CONV) classifier
  3. Field-Goal Make (FG) classifier
  4. Punt Net Field-Position (PUNT) regressor

Handles model definition, training with monotonic constraints,
evaluation, persistence (joblib), and artifact reloading.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import log_loss

from src.config import (
    MODEL_DIR,
    WP_FEATURES,
    WP_MONOTONIC,
    WP_PARAMS,
    CONV_FEATURES,
    CONV_MONOTONIC,
    CONV_PARAMS,
    FG_FEATURES,
    FG_MONOTONIC,
    FG_PARAMS,
    PUNT_FEATURES,
    PUNT_MONOTONIC,
    PUNT_PARAMS,
)


# ────────────────────────────────────────────────────────────────────────────
# 1. Win Probability Model (Classification)
# ────────────────────────────────────────────────────────────────────────────
def train_wp_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> HistGradientBoostingClassifier:
    """Train the win probability (WP) model with monotonic constraints.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Training feature matrix containing WP_FEATURES.
    y : pd.Series | np.ndarray
        Binary target (1 = possession team won, 0 = loss).

    Returns
    -------
    HistGradientBoostingClassifier
        Fitted win probability model.
    """
    print("Training Win Probability (WP) model...")
    if isinstance(X, pd.DataFrame):
        X_mat = X[WP_FEATURES]
    else:
        X_mat = X

    model = HistGradientBoostingClassifier(
        monotonic_cst=WP_MONOTONIC,
        **WP_PARAMS,
    )
    model.fit(X_mat, y)

    # Attach training metadata
    model.n_samples_fit_ = len(y)
    model.feature_names_ = list(WP_FEATURES)

    # Training diagnostics
    y_prob = model.predict_proba(X_mat)
    train_logloss = float(log_loss(y, y_prob))
    n_iters = getattr(model, "n_iter_", len(getattr(model, "train_score_", [])))
    print(f"  WP model trained in {n_iters} iterations | Train log-loss: {train_logloss:.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# 2. 4th-Down Conversion Model (Classification)
# ────────────────────────────────────────────────────────────────────────────
def train_conv_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> HistGradientBoostingClassifier:
    """Train the 4th-down conversion model with monotonic constraints.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Training feature matrix containing CONV_FEATURES.
    y : pd.Series | np.ndarray
        Binary target (1 = converted, 0 = failed).

    Returns
    -------
    HistGradientBoostingClassifier
        Fitted conversion model.
    """
    print("Training 4th-down Conversion (CONV) model...")
    if isinstance(X, pd.DataFrame):
        X_mat = X[CONV_FEATURES]
    else:
        X_mat = X

    model = HistGradientBoostingClassifier(
        monotonic_cst=CONV_MONOTONIC,
        **CONV_PARAMS,
    )
    model.fit(X_mat, y)

    # Attach training metadata
    model.n_samples_fit_ = len(y)
    model.feature_names_ = list(CONV_FEATURES)

    # Training diagnostics
    y_prob = model.predict_proba(X_mat)
    train_logloss = float(log_loss(y, y_prob))
    n_iters = getattr(model, "n_iter_", len(getattr(model, "train_score_", [])))
    print(f"  CONV model trained in {n_iters} iterations | Train log-loss: {train_logloss:.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# 3. Field-Goal Make Model (Classification)
# ────────────────────────────────────────────────────────────────────────────
def train_fg_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> HistGradientBoostingClassifier:
    """Train the field-goal make probability model with monotonic constraints.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Training feature matrix containing FG_FEATURES.
    y : pd.Series | np.ndarray
        Binary target (1 = field goal made, 0 = missed/blocked).

    Returns
    -------
    HistGradientBoostingClassifier
        Fitted field goal model.
    """
    print("Training Field Goal (FG) model...")
    if isinstance(X, pd.DataFrame):
        X_mat = X[FG_FEATURES]
    else:
        X_mat = X

    model = HistGradientBoostingClassifier(
        monotonic_cst=FG_MONOTONIC,
        **FG_PARAMS,
    )
    model.fit(X_mat, y)

    # Attach training metadata
    model.n_samples_fit_ = len(y)
    model.feature_names_ = list(FG_FEATURES)

    # Training diagnostics
    y_prob = model.predict_proba(X_mat)
    train_logloss = float(log_loss(y, y_prob))
    n_iters = getattr(model, "n_iter_", len(getattr(model, "train_score_", [])))
    print(f"  FG model trained in {n_iters} iterations | Train log-loss: {train_logloss:.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# 4. Punt Resulting Field-Position Model (Regression)
# ────────────────────────────────────────────────────────────────────────────
def train_punt_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> HistGradientBoostingRegressor:
    """Train the punt opponent starting yardline model (regression).

    Predicts the receiving team's expected yardline_100 after the punt
    (distance from their own end zone, where higher values mean deeper
    in opponent territory and thus better for the kicking team).

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Training feature matrix containing PUNT_FEATURES.
    y : pd.Series | np.ndarray
        Continuous target: opponent resulting yardline_100.

    Returns
    -------
    HistGradientBoostingRegressor
        Fitted punt regression model.
    """
    print("Training Punt (PUNT) model (regression)...")
    if isinstance(X, pd.DataFrame):
        X_mat = X[PUNT_FEATURES]
    else:
        X_mat = X

    model = HistGradientBoostingRegressor(
        monotonic_cst=PUNT_MONOTONIC,
        **PUNT_PARAMS,
    )
    model.fit(X_mat, y)

    # Attach training metadata
    model.n_samples_fit_ = len(y)
    model.feature_names_ = list(PUNT_FEATURES)

    # Training diagnostics
    y_pred = model.predict(X_mat)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    n_iters = getattr(model, "n_iter_", len(getattr(model, "train_score_", [])))
    print(f"  PUNT model trained in {n_iters} iterations | Train RMSE: {rmse:.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────────
# 5. Master Model Trainer
# ────────────────────────────────────────────────────────────────────────────
def train_all_models(
    datasets: Mapping[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, HistGradientBoostingClassifier | HistGradientBoostingRegressor]:
    """Train all four decision models using the datasets dictionary.

    Parameters
    ----------
    datasets : Mapping[str, tuple[pd.DataFrame, pd.Series]]
        Dictionary containing ("wp", "conv", "fg", "punt") tuples of (X, y).

    Returns
    -------
    dict[str, HistGradientBoostingClassifier | HistGradientBoostingRegressor]
        Mapping of model names to fitted model instances:
        {"wp": wp_model, "conv": conv_model, "fg": fg_model, "punt": punt_model}.
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL 4TH-DOWN DECISION MODELS")
    print("=" * 60)

    X_wp, y_wp = datasets["wp"]
    wp_model = train_wp_model(X_wp, y_wp)

    X_conv, y_conv = datasets["conv"]
    conv_model = train_conv_model(X_conv, y_conv)

    X_fg, y_fg = datasets["fg"]
    fg_model = train_fg_model(X_fg, y_fg)

    X_punt, y_punt = datasets["punt"]
    punt_model = train_punt_model(X_punt, y_punt)

    print("=" * 60)
    print("All models successfully trained.")
    print("=" * 60 + "\n")

    return {
        "wp": wp_model,
        "conv": conv_model,
        "fg": fg_model,
        "punt": punt_model,
    }


# ────────────────────────────────────────────────────────────────────────────
# 6. Persistence & Metadata
# ────────────────────────────────────────────────────────────────────────────
def save_models(
    models: Mapping[str, Any],
    path: Path | str | None = None,
    row_counts: Mapping[str, int] | None = None,
    **kwargs: Any,
) -> None:
    """Save trained models and training metadata to disk.

    Parameters
    ----------
    models : Mapping[str, Any]
        Dictionary of models to save (typically "wp", "conv", "fg", "punt").
    path : Path | str | None, optional
        Target directory to save artifacts. If None, defaults to MODEL_DIR.
    row_counts : Mapping[str, int] | None, optional
        Explicit mapping of training row counts per model. If not provided,
        inferred from model attributes when available.
    **kwargs : Any
        Additional metadata key-value pairs stored in model_metadata.json.
    """
    target_dir = Path(path) if path is not None else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Determine row counts
    final_row_counts: dict[str, int | None] = {}
    for name, model in models.items():
        if row_counts is not None and name in row_counts:
            final_row_counts[name] = int(row_counts[name])
        else:
            n_samples = getattr(model, "n_samples_fit_", None)
            final_row_counts[name] = int(n_samples) if n_samples is not None else None

    # Save each model using joblib
    for name, model in models.items():
        model_path = target_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved {name} model to {model_path}")

    # Build and save metadata
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "models": list(models.keys()),
        "row_counts": final_row_counts,
    }
    if kwargs:
        metadata.update(kwargs)

    meta_path = target_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {meta_path}")


# ────────────────────────────────────────────────────────────────────────────
# 7. Loading
# ────────────────────────────────────────────────────────────────────────────
def load_models(
    path: Path | str | None = None,
) -> dict[str, HistGradientBoostingClassifier | HistGradientBoostingRegressor]:
    """Load all four decision models from the model directory.

    Parameters
    ----------
    path : Path | str | None, optional
        Directory containing the saved model files. Defaults to MODEL_DIR.

    Returns
    -------
    dict[str, HistGradientBoostingClassifier | HistGradientBoostingRegressor]
        Dictionary containing loaded models keyed by "wp", "conv", "fg", "punt".
    """
    source_dir = Path(path) if path is not None else MODEL_DIR
    model_names = ["wp", "conv", "fg", "punt"]
    loaded_models: dict[str, Any] = {}

    print(f"Loading models from {source_dir}...")
    for name in model_names:
        model_path = source_dir / f"{name}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Train and save models first."
            )
        loaded_models[name] = joblib.load(model_path)
        print(f"  Loaded {name} model from {model_path.name}")

    return loaded_models
