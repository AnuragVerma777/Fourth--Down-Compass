"""
src/data_pipeline.py
────────────────────
Load nflverse parquet data, clean it, and build the four training
datasets needed by the modelling layer:

    1. wp_data      – every real play  →  WP model
    2. conv_data    – 4th-down go-for-it plays  →  conversion model
    3. fg_data      – field-goal attempts  →  FG-make model
    4. punt_data    – punts  →  punt net field-position model
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    RAW_DIR, PROC_DIR,
    WP_FEATURES, CONV_FEATURES, FG_FEATURES, PUNT_FEATURES,
    FG_SNAP_OFFSET,
)


# ────────────────────────────────────────────────────────────────────────
# 1. Loading
# ────────────────────────────────────────────────────────────────────────
def load_seasons(years: list[int]) -> pd.DataFrame:
    """Concatenate nflverse PBP parquet files for the requested seasons."""
    frames = []
    for yr in years:
        fp = RAW_DIR / f"play_by_play_{yr}.parquet"
        if not fp.exists():
            raise FileNotFoundError(
                f"{fp} not found – run  python scripts/download_data.py  first."
            )
        frames.append(pd.read_parquet(fp))
    return pd.concat(frames, ignore_index=True)


# ────────────────────────────────────────────────────────────────────────
# 2. Cleaning helpers
# ────────────────────────────────────────────────────────────────────────
def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns the models need but that aren't in the raw data."""
    df = df.copy()

    # Binary: is the possession team the home team?
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(np.int8)

    # Binary: is the venue a dome / closed roof?
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(np.int8)

    # Binary: does possession team receive the 2nd half kickoff?
    home_rec_2h = (df.get("home_opening_kickoff", 1) == 1)
    df["receive_2h_ko"] = np.where(
        df["qtr"] <= 2,
        np.where(home_rec_2h, df["is_home"] == 1, df["is_home"] == 0).astype(np.int8),
        np.int8(0),
    )

    # Goal-to-go flag
    df["is_goal_to_go"] = (
        (df["yardline_100"] <= df["ydstogo"])
    ).astype(np.int8)

    # Kick distance (FG snaps are ≈17 yards behind the line of scrimmage)
    df["kick_distance_calc"] = df["yardline_100"] + FG_SNAP_OFFSET

    # Did the possession team win this game?  (for WP target)
    # result = home_score − away_score at game end
    home_win = df["result"] > 0
    away_win = df["result"] < 0
    is_home  = df["posteam"] == df["home_team"]
    df["posteam_win"] = np.where(
        df["result"] == 0,
        np.nan,                            # drop ties
        np.where(is_home, home_win, away_win).astype(np.float32),
    )

    return df


def _real_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only real, non-penalty snaps with valid game state."""
    mask = (
        df["play_type"].isin(["run", "pass", "punt", "field_goal",
                               "qb_kneel", "qb_spike"])
        & df["down"].notna()
        & df["yardline_100"].notna()
        & df["game_seconds_remaining"].notna()
        & df["score_differential"].notna()
        & df["posteam"].notna()
    )
    return df.loc[mask].copy()


# ────────────────────────────────────────────────────────────────────────
# 3. Dataset builders
# ────────────────────────────────────────────────────────────────────────
def build_wp_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """All real plays → features + binary win label."""
    cols_needed = WP_FEATURES + ["posteam_win"]
    sub = df.dropna(subset=cols_needed).copy()
    X = sub[WP_FEATURES].astype(np.float32)
    y = sub["posteam_win"].astype(np.float32)
    return X, y


def build_conv_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fourth-down go-for-it plays → features + converted flag."""
    mask = (
        (df["down"] == 4)
        & df["play_type"].isin(["run", "pass"])
        & (df["fourth_down_converted"].notna() | df["fourth_down_failed"].notna())
    )
    sub = df.loc[mask].copy()
    sub["converted"] = sub["fourth_down_converted"].fillna(0).astype(np.float32)

    cols_needed = CONV_FEATURES + ["converted"]
    sub = sub.dropna(subset=cols_needed)
    X = sub[CONV_FEATURES].astype(np.float32)
    y = sub["converted"]
    return X, y


def build_fg_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Field-goal attempts → features + made flag."""
    mask = (
        (df["play_type"] == "field_goal")
        & df["field_goal_result"].isin(["made", "missed", "blocked"])
    )
    sub = df.loc[mask].copy()
    sub["fg_made"] = (sub["field_goal_result"] == "made").astype(np.float32)

    # Use actual kick_distance where available, else calculated
    sub["kick_distance"] = sub["kick_distance"].fillna(sub["kick_distance_calc"])
    sub = sub.dropna(subset=FG_FEATURES + ["fg_made"])

    X = sub[FG_FEATURES].astype(np.float32)
    y = sub["fg_made"]
    return X, y


def build_punt_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Punts → features + opponent's resulting field position.

    Target: opponent_yardline_100 = how far the receiving team is from
    *their own* end-zone after the punt (i.e. 100 − final_yardline_100
    in the kicking team's frame).  Higher = better for the kicking team.
    """
    mask = (
        (df["play_type"] == "punt")
        & df["kick_distance"].notna()
        & df["yardline_100"].notna()
    )
    sub = df.loc[mask].copy()

    # Net punt yardage: gross − return yards (touchback → return=0)
    sub["return_yards"] = sub["return_yards"].fillna(0)
    sub["net_yards"] = sub["kick_distance"] - sub["return_yards"]

    # Where does the ball end up (in the kicking team's yardline_100 frame)?
    sub["ball_at"] = sub["yardline_100"] - sub["net_yards"]

    # Touchback → opponent starts at their 25  ⟹  ball_at = 25 in kicker's frame
    sub.loc[sub["touchback"] == 1, "ball_at"] = 25

    # Clamp to [1, 99]
    sub["ball_at"] = sub["ball_at"].clip(1, 99)

    # Opponent's yardline_100 (from *their* end-zone) = 100 − ball_at
    sub["opp_yardline_100"] = 100 - sub["ball_at"]

    sub = sub.dropna(subset=PUNT_FEATURES + ["opp_yardline_100"])
    X = sub[PUNT_FEATURES].astype(np.float32)
    y = sub["opp_yardline_100"].astype(np.float32)
    return X, y


# ────────────────────────────────────────────────────────────────────────
# 4. Master loader
# ────────────────────────────────────────────────────────────────────────
def prepare_data(
    years: list[int],
) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline: load → clean → derive → build four datasets.

    Returns
    -------
    df_clean : pd.DataFrame   (the cleaned master frame – useful for auditing)
    datasets : dict            {"wp": (X, y), "conv": (X, y), ...}
    """
    print(f"Loading seasons {years[0]}-{years[-1]} ...")
    df = load_seasons(years)
    print(f"  Raw rows: {len(df):,}")

    df = _add_derived_columns(df)
    df = _real_plays(df)
    print(f"  After cleaning: {len(df):,}")

    datasets = {
        "wp":   build_wp_dataset(df),
        "conv": build_conv_dataset(df),
        "fg":   build_fg_dataset(df),
        "punt": build_punt_dataset(df),
    }
    for name, (X, y) in datasets.items():
        print(f"  {name:6s}  ->  {len(X):>7,} rows,  {X.shape[1]} features")

    return df, datasets
