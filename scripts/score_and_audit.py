"""
scripts/score_and_audit.py
──────────────────────────
Comprehensive evaluation of NFL 4th-down decisions against the AQX models:
  1. Score every 2025 decision (WP branches, optimal recommendations, margins).
  2. 95% Cluster Bootstrap Intervals (resampling whole games).
  3. Analysis of coach conservatism by situation (field position, yards-to-go).
  4. Team Ledger for all 32 teams.
  5. Sensitivity analysis (5 percentage point haircut on conversion probability).
  6. Multi-year trend across 2020-2025.
  7. Publication-grade figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODEL_DIR,
    REPORT_DIR,
    WP_FEATURES,
    CONV_FEATURES,
    FG_FEATURES,
    PUNT_FEATURES,
    KICKOFF_YARDLINE,
    FG_SNAP_OFFSET,
    PAT_POINTS,
    FG_POINTS,
    MAX_FG_DISTANCE,
)
from src.data_pipeline import _add_derived_columns
from src.models import load_models


def evaluate_plays_vectorized(
    df: pd.DataFrame,
    models: dict,
    conv_haircut: float = 0.0,
) -> pd.DataFrame:
    """Vectorized evaluation of 4th-down decisions for a dataframe of plays.

    Parameters
    ----------
    df : pd.DataFrame
        4th-down plays with valid state columns.
    models : dict
        Fitted models dictionary ("wp", "conv", "fg", "punt").
    conv_haircut : float
        Subtract this value from P(convert) for sensitivity testing (e.g. 0.05).

    Returns
    -------
    pd.DataFrame
        df enriched with model evaluation metrics.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    yl = df["yardline_100"].values.astype(np.float32)
    ytg = df["ydstogo"].values.astype(np.float32)
    sd = df["score_differential"].values.astype(np.float32)
    gsr = df["game_seconds_remaining"].values.astype(np.float32)
    hsr = df["half_seconds_remaining"].values.astype(np.float32)
    pto = df["posteam_timeouts_remaining"].values.astype(np.float32)
    dto = df["defteam_timeouts_remaining"].values.astype(np.float32)
    r2h = df["receive_2h_ko"].values.astype(np.float32)
    ish = df["is_home"].values.astype(np.float32)
    spl = df["spread_line"].values.astype(np.float32)
    is_dome = df["is_dome"].values.astype(np.float32)
    is_gtg = np.where(yl <= ytg, 1.0, 0.0).astype(np.float32)

    # 1. P(convert)
    conv_X = pd.DataFrame({
        "ydstogo": ytg,
        "yardline_100": yl,
        "score_differential": sd,
        "game_seconds_remaining": gsr,
        "is_goal_to_go": is_gtg,
    })
    p_conv_raw = models["conv"].predict_proba(conv_X)[:, 1]
    p_conv = np.maximum(0.0, p_conv_raw - conv_haircut).astype(np.float32)

    # 2. WP(convert)
    is_td = yl <= ytg
    X_conv = pd.DataFrame({
        "score_differential": np.where(is_td, -(sd + PAT_POINTS), sd),
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": hsr,
        "yardline_100": np.where(is_td, KICKOFF_YARDLINE, yl - ytg),
        "down": np.ones(n, dtype=np.float32),
        "ydstogo": np.where(is_td, 10.0, np.minimum(10.0, yl - ytg)),
        "posteam_timeouts_remaining": np.where(is_td, dto, pto),
        "defteam_timeouts_remaining": np.where(is_td, pto, dto),
        "receive_2h_ko": np.where(is_td, 1.0 - r2h, r2h),
        "is_home": np.where(is_td, 1.0 - ish, ish),
        "spread_line": np.where(is_td, -spl, spl),
    })
    wp_conv_model = models["wp"].predict_proba(X_conv)[:, 1]
    wp_convert = np.where(is_td, 1.0 - wp_conv_model, wp_conv_model)

    # 3. WP(fail)
    X_fail = pd.DataFrame({
        "score_differential": -sd,
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": hsr,
        "yardline_100": 100.0 - yl,
        "down": np.ones(n, dtype=np.float32),
        "ydstogo": np.full(n, 10.0, dtype=np.float32),
        "posteam_timeouts_remaining": dto,
        "defteam_timeouts_remaining": pto,
        "receive_2h_ko": 1.0 - r2h,
        "is_home": 1.0 - ish,
        "spread_line": -spl,
    })
    wp_fail = 1.0 - models["wp"].predict_proba(X_fail)[:, 1]

    # Expected Go WP
    go_wp = p_conv * wp_convert + (1.0 - p_conv) * wp_fail

    # 4. Punt
    punt_X = pd.DataFrame({"yardline_100": yl})
    opp_punt_yl = np.clip(models["punt"].predict(punt_X), 1.0, 99.0)
    X_punt = pd.DataFrame({
        "score_differential": -sd,
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": hsr,
        "yardline_100": opp_punt_yl,
        "down": np.ones(n, dtype=np.float32),
        "ydstogo": np.full(n, 10.0, dtype=np.float32),
        "posteam_timeouts_remaining": dto,
        "defteam_timeouts_remaining": pto,
        "receive_2h_ko": 1.0 - r2h,
        "is_home": 1.0 - ish,
        "spread_line": -spl,
    })
    punt_wp = 1.0 - models["wp"].predict_proba(X_punt)[:, 1]

    # 5. Field Goal
    if "kick_distance" in df.columns:
        kd = np.where(df["kick_distance"].notna(), df["kick_distance"].values, yl + FG_SNAP_OFFSET).astype(np.float32)
    else:
        kd = (yl + FG_SNAP_OFFSET).astype(np.float32)

    fg_X = pd.DataFrame({"kick_distance": kd, "is_dome": is_dome})
    p_fg_make = models["fg"].predict_proba(fg_X)[:, 1]

    X_fg_make = pd.DataFrame({
        "score_differential": -(sd + FG_POINTS),
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": hsr,
        "yardline_100": np.full(n, float(KICKOFF_YARDLINE), dtype=np.float32),
        "down": np.ones(n, dtype=np.float32),
        "ydstogo": np.full(n, 10.0, dtype=np.float32),
        "posteam_timeouts_remaining": dto,
        "defteam_timeouts_remaining": pto,
        "receive_2h_ko": 1.0 - r2h,
        "is_home": 1.0 - ish,
        "spread_line": -spl,
    })
    wp_fg_make = 1.0 - models["wp"].predict_proba(X_fg_make)[:, 1]

    X_fg_miss = pd.DataFrame({
        "score_differential": -sd,
        "game_seconds_remaining": gsr,
        "half_seconds_remaining": hsr,
        "yardline_100": 100.0 - np.maximum(yl, 20.0),
        "down": np.ones(n, dtype=np.float32),
        "ydstogo": np.full(n, 10.0, dtype=np.float32),
        "posteam_timeouts_remaining": dto,
        "defteam_timeouts_remaining": pto,
        "receive_2h_ko": 1.0 - r2h,
        "is_home": 1.0 - ish,
        "spread_line": -spl,
    })
    wp_fg_miss = 1.0 - models["wp"].predict_proba(X_fg_miss)[:, 1]

    valid_fg = kd <= MAX_FG_DISTANCE
    fg_wp = np.where(valid_fg, p_fg_make * wp_fg_make + (1.0 - p_fg_make) * wp_fg_miss, -1e9)

    # 6. Break-even conversion rate
    best_alt = np.maximum(np.where(valid_fg, fg_wp, -1e9), punt_wp)
    denom = wp_convert - wp_fail
    safe_denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    break_even = np.where(
        np.abs(denom) < 1e-9,
        np.where(best_alt <= wp_fail, 0.0, 1.0),
        (best_alt - wp_fail) / safe_denom,
    )
    break_even = np.clip(break_even, 0.0, 1.0)

    # 7. Action mapping and recommendation
    # Options matrix: [Go, FG, Punt]
    options_matrix = np.column_stack([go_wp, np.where(valid_fg, fg_wp, -1e9), punt_wp])
    choice_idx = np.argmax(options_matrix, axis=1)
    rec_map = {0: "go", 1: "fg", 2: "punt"}
    recommendations = [rec_map[i] for i in choice_idx]

    # Compute margin (best WP - 2nd best WP)
    sorted_opts = np.sort(options_matrix, axis=1)[:, ::-1]
    # If FG is invalid, 2nd best is options_matrix[:, 2] (punt) or go
    second_best = np.where(
        valid_fg,
        sorted_opts[:, 1],
        np.minimum(go_wp, punt_wp),
    )
    margins = sorted_opts[:, 0] - second_best

    # Actual coach action
    act_series = df["play_type"].map({
        "run": "go",
        "pass": "go",
        "field_goal": "fg",
        "punt": "punt",
    })
    actual_actions = act_series.values

    # WP of actual action
    actual_wps = np.where(
        actual_actions == "go",
        go_wp,
        np.where(actual_actions == "fg", np.where(valid_fg, fg_wp, 0.0), punt_wp),
    )
    optimal_wps = sorted_opts[:, 0]
    wp_left = np.maximum(0.0, optimal_wps - actual_wps)
    is_optimal = (actual_actions == recommendations).astype(int)

    # Clear Go flag (margin >= 0.01 / 1.0 pp)
    is_clear_go = ((np.array(recommendations) == "go") & (margins >= 0.01)).astype(int)
    went_on_clear_go = ((actual_actions == "go") & (is_clear_go == 1)).astype(int)
    conservative_error = ((is_clear_go == 1) & (actual_actions != "go")).astype(int)

    df["p_convert"] = p_conv
    df["wp_convert"] = wp_convert
    df["wp_fail"] = wp_fail
    df["go_wp"] = go_wp
    df["fg_wp"] = np.where(valid_fg, fg_wp, np.nan)
    df["punt_wp"] = punt_wp
    df["break_even_conv"] = break_even
    df["recommendation"] = recommendations
    df["margin"] = margins
    df["actual_action"] = actual_actions
    df["optimal_wp"] = optimal_wps
    df["actual_wp"] = actual_wps
    df["wp_left"] = wp_left
    df["is_optimal"] = is_optimal
    df["is_clear_go"] = is_clear_go
    df["went_on_clear_go"] = went_on_clear_go
    df["conservative_error"] = conservative_error

    return df


def load_clean_4th_downs(years: list[int]) -> pd.DataFrame:
    """Load and prepare 4th down decision plays for given seasons."""
    frames = []
    for yr in years:
        fp = ROOT / "data" / "raw" / f"play_by_play_{yr}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing {fp}")
        raw = pd.read_parquet(fp)
        raw["season"] = yr
        raw = _add_derived_columns(raw)

        mask = (
            (raw["down"] == 4)
            & raw["play_type"].isin(["run", "pass", "field_goal", "punt"])
            & raw["posteam"].notna()
            & raw["yardline_100"].notna()
            & raw["ydstogo"].notna()
            & raw["score_differential"].notna()
            & raw["game_seconds_remaining"].notna()
            & raw["half_seconds_remaining"].notna()
            & raw["posteam_timeouts_remaining"].notna()
            & raw["defteam_timeouts_remaining"].notna()
        )
        frames.append(raw.loc[mask].copy())

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def cluster_bootstrap_game(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Perform cluster bootstrap resampling whole games (game_id).

    Computes 95% bootstrap intervals for:
      - Go-rate on clear go (margin >= 1 pt)
      - WP lost per team-game (in percentage points)
      - Overall decision agreement (%)
      - Total WP lost per game (pp/game)
    """
    rng = np.random.default_rng(seed)
    unique_games = df["game_id"].unique()
    n_games = len(unique_games)

    # Pre-aggregate game-level statistics for rapid bootstrap resampling
    # For each game:
    #   sum_clear_go, sum_went_clear_go, sum_optimal, n_plays, sum_wp_left, n_team_games
    game_stats = []
    for gid, grp in df.groupby("game_id"):
        n_tg = grp["posteam"].nunique()  # usually 2 teams
        game_stats.append({
            "game_id": gid,
            "clear_go": grp["is_clear_go"].sum(),
            "went_clear_go": grp["went_on_clear_go"].sum(),
            "optimal": grp["is_optimal"].sum(),
            "n_plays": len(grp),
            "wp_left": grp["wp_left"].sum(),
            "n_team_games": n_tg,
        })
    gdf = pd.DataFrame(game_stats)

    clear_go_arr = gdf["clear_go"].values
    went_clear_go_arr = gdf["went_clear_go"].values
    optimal_arr = gdf["optimal"].values
    n_plays_arr = gdf["n_plays"].values
    wp_left_arr = gdf["wp_left"].values
    n_tg_arr = gdf["n_team_games"].values

    boot_go_rate = np.zeros(n_boot)
    boot_wp_lost_tg = np.zeros(n_boot)
    boot_agreement = np.zeros(n_boot)
    boot_wp_lost_game = np.zeros(n_boot)

    for b in range(n_boot):
        indices = rng.integers(0, n_games, size=n_games)

        c_go = clear_go_arr[indices].sum()
        w_go = went_clear_go_arr[indices].sum()
        boot_go_rate[b] = (w_go / c_go) * 100.0 if c_go > 0 else np.nan

        tot_wp = wp_left_arr[indices].sum() * 100.0  # in percentage points
        tot_tg = n_tg_arr[indices].sum()
        boot_wp_lost_tg[b] = tot_wp / tot_tg if tot_tg > 0 else np.nan

        tot_opt = optimal_arr[indices].sum()
        tot_p = n_plays_arr[indices].sum()
        boot_agreement[b] = (tot_opt / tot_p) * 100.0 if tot_p > 0 else np.nan

        boot_wp_lost_game[b] = tot_wp / n_games

    results = {}
    metrics = {
        "go_rate_on_clear_go": boot_go_rate,
        "wp_lost_per_team_game": boot_wp_lost_tg,
        "optimal_agreement_pct": boot_agreement,
        "wp_lost_per_game": boot_wp_lost_game,
    }

    for name, vals in metrics.items():
        v = vals[~np.isnan(vals)]
        results[name] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "ci_lower": float(np.percentile(v, 2.5)),
            "ci_upper": float(np.percentile(v, 97.5)),
        }

    return results


def analyze_conservative_situations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze which situations exhibit the highest coach conservatism."""
    # Define categorical buckets
    # 1. Yards to go
    ytg_bins = [0, 1, 3, 6, 100]
    ytg_labels = ["4th & 1", "4th & 2-3", "4th & 4-6", "4th & 7+"]
    df["ytg_cat"] = pd.cut(df["ydstogo"], bins=ytg_bins, labels=ytg_labels, right=True)

    # 2. Field position zone
    # yardline_100: 100 is own goal line, 0 is opponent goal line
    def get_zone(yl):
        if yl <= 10:
            return "Red Zone (OPP 1-10)"
        elif yl <= 35:
            return "FG Range (OPP 11-35)"
        elif yl <= 60:
            return "Midfield (OPP 36 to OWN 40)"
        else:
            return "Deep Own Terr (OWN 1-39)"

    df["zone"] = df["yardline_100"].apply(get_zone)

    # Grouping
    records = []
    for (zone, ytg_cat), grp in df.groupby(["zone", "ytg_cat"], observed=True):
        n_tot = len(grp)
        n_clear_go = int(grp["is_clear_go"].sum())
        n_rec_go = int((grp["recommendation"] == "go").sum())
        n_act_go = int((grp["actual_action"] == "go").sum())

        act_go_rate = (n_act_go / n_tot * 100.0) if n_tot > 0 else 0.0
        rec_go_rate = (n_rec_go / n_tot * 100.0) if n_tot > 0 else 0.0

        # When model clearly says GO: how often did they kick/punt?
        if n_clear_go > 0:
            went_clear = int(grp["went_on_clear_go"].sum())
            go_rate_clear = (went_clear / n_clear_go) * 100.0
            conservative_errs = int(grp["conservative_error"].sum())
            cons_err_rate = (conservative_errs / n_clear_go) * 100.0
        else:
            go_rate_clear = np.nan
            cons_err_rate = 0.0
            conservative_errs = 0

        tot_wp_lost = grp["wp_left"].sum() * 100.0
        avg_wp_lost_per_play = (tot_wp_lost / n_tot) if n_tot > 0 else 0.0

        # Punts vs FGs when clear GO rejected
        rejected_grp = grp[grp["conservative_error"] == 1]
        punts_on_go = int((rejected_grp["actual_action"] == "punt").sum())
        fgs_on_go = int((rejected_grp["actual_action"] == "fg").sum())

        records.append({
            "zone": zone,
            "distance": ytg_cat,
            "total_plays": n_tot,
            "model_rec_go": n_rec_go,
            "actual_go": n_act_go,
            "actual_go_pct": act_go_rate,
            "model_go_pct": rec_go_rate,
            "clear_go_opps": n_clear_go,
            "go_rate_on_clear_pct": go_rate_clear,
            "conservative_errors": conservative_errs,
            "punts_on_clear_go": punts_on_go,
            "fgs_on_clear_go": fgs_on_go,
            "total_wp_lost_pp": tot_wp_lost,
            "avg_wp_lost_pp": avg_wp_lost_per_play,
        })

    sit_df = pd.DataFrame(records).sort_values(by="total_wp_lost_pp", ascending=False).reset_index(drop=True)
    return sit_df


def compute_team_ledger(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Compute comprehensive team ledger with bootstrap CI for WP lost/game."""
    rng = np.random.default_rng(42)
    team_rows = []

    for team, grp in df.groupby("posteam"):
        n_plays = len(grp)
        team_games = grp["game_id"].unique()
        n_games = len(team_games)

        n_opt = int(grp["is_optimal"].sum())
        opt_pct = (n_opt / n_plays * 100.0) if n_plays > 0 else 0.0

        n_clear_go = int(grp["is_clear_go"].sum())
        went_clear = int(grp["went_on_clear_go"].sum())
        clear_go_rate = (went_clear / n_clear_go * 100.0) if n_clear_go > 0 else np.nan

        tot_wp_lost = grp["wp_left"].sum() * 100.0  # in pp
        wp_lost_per_game = tot_wp_lost / n_games if n_games > 0 else 0.0

        # Cluster bootstrap within team's games for team-level CI
        # Aggregate plays by game for this team
        g_wp = grp.groupby("game_id")["wp_left"].sum().values * 100.0
        boot_team_wpg = np.zeros(n_boot)
        for b in range(n_boot):
            sampled_g = rng.choice(g_wp, size=len(g_wp), replace=True)
            boot_team_wpg[b] = np.mean(sampled_g)

        ci_low = float(np.percentile(boot_team_wpg, 2.5))
        ci_high = float(np.percentile(boot_team_wpg, 97.5))

        act_go = int((grp["actual_action"] == "go").sum())
        act_fg = int((grp["actual_action"] == "fg").sum())
        act_punt = int((grp["actual_action"] == "punt").sum())

        rec_go = int((grp["recommendation"] == "go").sum())
        rec_fg = int((grp["recommendation"] == "fg").sum())
        rec_punt = int((grp["recommendation"] == "punt").sum())

        team_rows.append({
            "team": team,
            "games": n_games,
            "decisions": n_plays,
            "optimal_pct": opt_pct,
            "clear_go_opps": n_clear_go,
            "went_clear_go": went_clear,
            "clear_go_rate_pct": clear_go_rate,
            "total_wp_lost_pp": tot_wp_lost,
            "wp_lost_per_game_pp": wp_lost_per_game,
            "ci_low_pp": ci_low,
            "ci_high_pp": ci_high,
            "act_go": act_go,
            "act_fg": act_fg,
            "act_punt": act_punt,
            "rec_go": rec_go,
            "rec_fg": rec_fg,
            "rec_punt": rec_punt,
        })

    team_df = pd.DataFrame(team_rows).sort_values(
        by="wp_lost_per_game_pp", ascending=True
    ).reset_index(drop=True)
    team_df["rank"] = np.arange(1, len(team_df) + 1)
    return team_df


def run_multi_year_trend(years: list[int], models: dict) -> pd.DataFrame:
    """Evaluate 4th-down decision trends from 2020 through 2025."""
    trend_records = []
    print("\n" + "=" * 70)
    print("CALCULATING MULTI-YEAR TREND (2020 - 2025)")
    print("=" * 70)

    for yr in years:
        df_yr = load_clean_4th_downs([yr])
        df_scored = evaluate_plays_vectorized(df_yr, models, conv_haircut=0.0)
        boot_res = cluster_bootstrap_game(df_scored, n_boot=1000)

        n_plays = len(df_scored)
        n_games = df_scored["game_id"].nunique()
        n_tg = df_scored.groupby("game_id")["posteam"].nunique().sum()

        act_go_cnt = (df_scored["actual_action"] == "go").sum()
        act_go_rate = act_go_cnt / n_plays * 100.0

        clear_go_opps = int(df_scored["is_clear_go"].sum())
        went_clear = int(df_scored["went_on_clear_go"].sum())
        clear_go_rate = (went_clear / clear_go_opps * 100.0) if clear_go_opps > 0 else np.nan

        opt_agreement = df_scored["is_optimal"].mean() * 100.0
        tot_wp_lost = df_scored["wp_left"].sum() * 100.0
        wp_lost_per_tg = tot_wp_lost / n_tg if n_tg > 0 else 0.0

        c_go_boot = boot_res["go_rate_on_clear_go"]
        wpl_boot = boot_res["wp_lost_per_team_game"]

        print(f"Season {yr}: {n_plays:,} plays in {n_games} games | "
              f"Go-Rate Clear: {clear_go_rate:.1f}% [{c_go_boot['ci_lower']:.1f}%, {c_go_boot['ci_upper']:.1f}%] | "
              f"WP Lost/TG: {wp_lost_per_tg:.2f}pp [{wpl_boot['ci_lower']:.2f}, {wpl_boot['ci_upper']:.2f}]")

        trend_records.append({
            "season": yr,
            "games": n_games,
            "team_games": n_tg,
            "decisions": n_plays,
            "overall_go_rate_pct": act_go_rate,
            "clear_go_opps": clear_go_opps,
            "went_clear_go": went_clear,
            "clear_go_rate_pct": clear_go_rate,
            "clear_go_ci_low": c_go_boot["ci_lower"],
            "clear_go_ci_high": c_go_boot["ci_upper"],
            "optimal_agreement_pct": opt_agreement,
            "total_wp_lost_pp": tot_wp_lost,
            "wp_lost_per_team_game_pp": wp_lost_per_tg,
            "wp_lost_ci_low": wpl_boot["ci_lower"],
            "wp_lost_ci_high": wpl_boot["ci_upper"],
        })

    return pd.DataFrame(trend_records)


def plot_audit_figures(
    team_df: pd.DataFrame,
    sit_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    base_boot: dict,
    sens_boot: dict,
    base_clear_rate: float,
    sens_clear_rate: float,
) -> None:
    """Generate high-resolution plots for the audit report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Team Ledger Forest Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(team_df))
    # Order: best team at top (rank 1)
    team_rev = team_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(team_rev))

    wpl = team_rev["wp_lost_per_game_pp"].values
    err_low = np.maximum(0, wpl - team_rev["ci_low_pp"].values)
    err_high = np.maximum(0, team_rev["ci_high_pp"].values - wpl)

    colors = [
        "#2ca02c" if r <= 8 else ("#1f77b4" if r <= 24 else "#d62728")
        for r in team_rev["rank"].values
    ]

    ax.errorbar(
        wpl,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color="#333333",
        ecolor="#888888",
        elinewidth=1.5,
        capsize=3,
        zorder=3,
    )
    ax.scatter(wpl, y_pos, c=colors, s=70, zorder=4, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.team} (#{r.rank})" for _, r in team_rev.iterrows()], fontsize=9)
    ax.set_xlabel("Win Probability Lost per Game (Percentage Points)", fontsize=11, fontweight="bold")
    ax.set_title("2025 NFL Team 4th-Down Ledger: WP Sacrificed per Game (with 95% Game-Resampled CIs)", fontsize=12, fontweight="bold", pad=12)
    ax.axvline(team_df["wp_lost_per_game_pp"].median(), color="grey", linestyle="--", alpha=0.7, label=f"Median ({team_df['wp_lost_per_game_pp'].median():.2f} pp)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "team_ledger_2025.png", dpi=300)
    plt.close(fig)

    # 2. Conservative Situations Bar Chart
    top_sit = sit_df.sort_values(by="total_wp_lost_pp", ascending=True).tail(8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    labels = [f"{r.zone}\n[{r.distance}]" for _, r in top_sit.iterrows()]
    y_idx = np.arange(len(top_sit))

    # Panel 1: Total WP lost
    ax1.barh(y_idx, top_sit["total_wp_lost_pp"], color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_idx)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Total WP Sacrificed Across League (Percentage Points)", fontsize=11, fontweight="bold")
    ax1.set_title("Most Costly Situations (Total WP Lost in 2025)", fontsize=12, fontweight="bold")

    # Panel 2: Clear-Go Discrepancy
    x_rec = top_sit["clear_go_opps"].values
    x_went = (top_sit["clear_go_opps"] - top_sit["conservative_errors"]).values
    rate = top_sit["go_rate_on_clear_pct"].values

    ax2.barh(y_idx - 0.18, [100.0] * len(top_sit), height=0.35, color="#aec7e8", label="Model Clear-Go (100%)", edgecolor="black", linewidth=0.5)
    ax2.barh(y_idx + 0.18, rate, height=0.35, color="#1f77b4", label="Actual Coach Go Rate %", edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Go Rate on Clear-Go Recommendations (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Coach Compliance on Clear-Go Recommendations", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "conservative_situations.png", dpi=300)
    plt.close(fig)

    # 3. 2020-2025 Trend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    years = trend_df["season"].values
    c_rate = trend_df["clear_go_rate_pct"].values
    c_low = trend_df["clear_go_ci_low"].values
    c_high = trend_df["clear_go_ci_high"].values

    ax1.plot(years, c_rate, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Coach Go Rate on Clear Go")
    ax1.fill_between(years, c_low, c_high, color="#1f77b4", alpha=0.2, label="95% Bootstrap CI")
    ax1.set_xlabel("NFL Season", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Go Rate on Clear-Go (Margin >= 1 pp) [%]", fontsize=11, fontweight="bold")
    ax1.set_title("Multi-Year Trend: Compliance with Clear-Go Situations", fontsize=12, fontweight="bold")
    ax1.set_ylim([45, 85])
    ax1.legend(loc="lower right")

    wpl = trend_df["wp_lost_per_team_game_pp"].values
    w_low = trend_df["wp_lost_ci_low"].values
    w_high = trend_df["wp_lost_ci_high"].values

    ax2.plot(years, wpl, "s-", color="#d62728", linewidth=2.5, markersize=8, label="WP Lost per Team-Game")
    ax2.fill_between(years, w_low, w_high, color="#d62728", alpha=0.2, label="95% Bootstrap CI")
    ax2.set_xlabel("NFL Season", fontsize=11, fontweight="bold")
    ax2.set_ylabel("WP Lost per Team-Game (pp)", fontsize=11, fontweight="bold")
    ax2.set_title("Multi-Year Trend: Average WP Sacrificed per Team-Game", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "trend_2020_2025.png", dpi=300)
    plt.close(fig)

    # 4. Sensitivity Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ["Baseline Model\n(Standard P_conv)", "Sensitivity Model\n(P_conv - 5 pp)"]
    rates = [base_clear_rate, sens_clear_rate]
    lows = [base_boot["go_rate_on_clear_go"]["ci_lower"], sens_boot["go_rate_on_clear_go"]["ci_lower"]]
    highs = [base_boot["go_rate_on_clear_go"]["ci_upper"], sens_boot["go_rate_on_clear_go"]["ci_upper"]]
    yerr = [
        [rates[0] - lows[0], rates[1] - lows[1]],
        [highs[0] - rates[0], highs[1] - rates[1]],
    ]

    bars = ax.bar(scenarios, rates, yerr=yerr, capsize=6, color=["#1f77b4", "#ff7f0e"], width=0.45, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2, f"{val:.1f}%", ha="center", va="center", color="white", fontsize=13, fontweight="bold")

    ax.set_ylabel("Coach Go Rate on Clear Go (%)", fontsize=11, fontweight="bold")
    ax.set_title("Sensitivity Test: Impact of -5 pp Conversion Rate on Clear-Go Compliance", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 100])
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "sensitivity_comparison.png", dpi=300)
    plt.close(fig)

    print(f"All 4 audit visualization figures saved to: {REPORT_DIR}")


def main():
    print("=" * 70)
    print("AQX 4TH-DOWN DECISION AUDIT & BOOTSTRAP INFERENCE")
    print("=" * 70)

    # 1. Load trained models
    models = load_models()

    # 2. Load 2025 data
    print("\nLoading and scoring 2025 4th-down decisions...")
    df_2025 = load_clean_4th_downs([2025])
    print(f"  Valid 4th-down decision plays identified in 2025: {len(df_2025):,}")

    # 3. Score baseline
    df_scored = evaluate_plays_vectorized(df_2025, models, conv_haircut=0.0)

    # 4. Bootstrap intervals for 2025 baseline
    print("Running 2,000 cluster bootstrap iterations (resampling whole games)...")
    boot_res = cluster_bootstrap_game(df_scored, n_boot=2000, seed=42)

    n_clear_go = int(df_scored["is_clear_go"].sum())
    went_clear_go = int(df_scored["went_on_clear_go"].sum())
    point_go_rate = (went_clear_go / n_clear_go * 100.0)

    n_games = df_scored["game_id"].nunique()
    n_tg = df_scored.groupby("game_id")["posteam"].nunique().sum()
    point_wp_lost_tg = (df_scored["wp_left"].sum() * 100.0) / n_tg

    print("\n" + "=" * 70)
    print("CORE 2025 AUDIT RESULTS (95% GAME-RESAMPLED BOOTSTRAP CIS)")
    print("=" * 70)
    c_res = boot_res["go_rate_on_clear_go"]
    print(f"Coach Go Rate on Clear Go (margin >= 1.0 pp):")
    print(f"  Point Estimate : {point_go_rate:.1f}% ({went_clear_go}/{n_clear_go} plays)")
    print(f"  95% CI         : [{c_res['ci_lower']:.1f}%, {c_res['ci_upper']:.1f}%]")

    w_res = boot_res["wp_lost_per_team_game"]
    print(f"\nWin Probability Lost per Team per Game:")
    print(f"  Point Estimate : {point_wp_lost_tg:.2f} percentage points")
    print(f"  95% CI         : [{w_res['ci_lower']:.2f} pp, {w_res['ci_upper']:.2f} pp]")

    a_res = boot_res["optimal_agreement_pct"]
    print(f"\nOverall Optimal Decision Agreement Rate:")
    print(f"  Point Estimate : {df_scored['is_optimal'].mean() * 100.0:.1f}%")
    print(f"  95% CI         : [{a_res['ci_lower']:.1f}%, {a_res['ci_upper']:.1f}%]")

    # 5. Sensitivity test: haircut = 5 pp (0.05)
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS (-5 PP CONVERSION ODDS HAIRCUT)")
    print("=" * 70)
    df_sens = evaluate_plays_vectorized(df_2025, models, conv_haircut=0.05)
    sens_boot = cluster_bootstrap_game(df_sens, n_boot=2000, seed=42)

    sens_clear_go = int(df_sens["is_clear_go"].sum())
    sens_went_clear = int(df_sens["went_on_clear_go"].sum())
    sens_point_go_rate = (sens_went_clear / sens_clear_go * 100.0)
    sens_point_wpl = (df_sens["wp_left"].sum() * 100.0) / n_tg

    s_c_res = sens_boot["go_rate_on_clear_go"]
    s_w_res = sens_boot["wp_lost_per_team_game"]
    print(f"Sensitivity Clear-Go Opportunities   : {sens_clear_go} (vs {n_clear_go} baseline)")
    print(f"Sensitivity Go Rate on Clear Go      : {sens_point_go_rate:.1f}% [{s_c_res['ci_lower']:.1f}%, {s_c_res['ci_upper']:.1f}%]")
    print(f"Sensitivity WP Lost per Team-Game     : {sens_point_wpl:.2f} pp [{s_w_res['ci_lower']:.2f} pp, {s_w_res['ci_upper']:.2f} pp]")

    # 6. Conservative situations
    print("\nAnalyzing most conservative situation categories...")
    sit_df = analyze_conservative_situations(df_scored)

    # 7. Team ledger
    print("Computing 32-team ledger with team-game CIs...")
    team_df = compute_team_ledger(df_scored, n_boot=1000)

    # 8. 2020-2025 Trend
    years_to_run = list(range(2020, 2026))
    trend_df = run_multi_year_trend(years_to_run, models)

    # 9. Plotting
    print("\nGenerating figures...")
    plot_audit_figures(
        team_df=team_df,
        sit_df=sit_df,
        trend_df=trend_df,
        base_boot=boot_res,
        sens_boot=sens_boot,
        base_clear_rate=point_go_rate,
        sens_clear_rate=sens_point_go_rate,
    )

    # Save structured results to json & csv for artifacts
    results_summary = {
        "baseline": {
            "decisions_analyzed": len(df_scored),
            "games_analyzed": n_games,
            "team_games_analyzed": int(n_tg),
            "clear_go_opportunities": n_clear_go,
            "went_on_clear_go": went_clear_go,
            "go_rate_on_clear_go_pct": point_go_rate,
            "go_rate_95_ci": [c_res["ci_lower"], c_res["ci_upper"]],
            "wp_lost_per_team_game_pp": point_wp_lost_tg,
            "wp_lost_95_ci": [w_res["ci_lower"], w_res["ci_upper"]],
            "optimal_agreement_pct": float(df_scored["is_optimal"].mean() * 100.0),
            "optimal_agreement_95_ci": [a_res["ci_lower"], a_res["ci_upper"]],
        },
        "sensitivity_5pp_haircut": {
            "clear_go_opportunities": sens_clear_go,
            "went_on_clear_go": sens_went_clear,
            "go_rate_on_clear_go_pct": sens_point_go_rate,
            "go_rate_95_ci": [s_c_res["ci_lower"], s_c_res["ci_upper"]],
            "wp_lost_per_team_game_pp": sens_point_wpl,
            "wp_lost_95_ci": [s_w_res["ci_lower"], s_w_res["ci_upper"]],
        },
    }

    out_json = ROOT / "reports" / "audit_metrics_2025.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    team_df.to_csv(ROOT / "reports" / "team_ledger_2025.csv", index=False)
    sit_df.to_csv(ROOT / "reports" / "conservative_situations_2025.csv", index=False)
    trend_df.to_csv(ROOT / "reports" / "trend_2020_2025.csv", index=False)

    print("\nAudit completed and artifacts generated successfully.")


if __name__ == "__main__":
    main()
