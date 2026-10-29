"""
src/audit.py
────────────
Audit frozen NFL 4th-down models on held-out 2025 season data:
  1. Win Probability (WP) model audit against nflfastR's vegas_wp benchmark
     (Brier score, AUC-ROC, calibration curves).
  2. Fourth-down coaching decision audit comparing real-world 2025 choices
     against optimal decision recommendations.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import (
    WP_FEATURES,
    CONV_FEATURES,
    FG_FEATURES,
    PUNT_FEATURES,
    REPORT_DIR,
    ROOT,
    KICKOFF_YARDLINE,
    FG_SNAP_OFFSET,
    PAT_POINTS,
    FG_POINTS,
    MAX_FG_DISTANCE,
)
from src.data_pipeline import _add_derived_columns, _real_plays
from src.engine import FourthDownEngine


def row_to_state(row) -> dict:
    """
    Extract game state dictionary from a DataFrame row or mapping.

    Returns the standard dictionary format expected by FourthDownEngine.
    """
    return {
        "score_differential": float(row["score_differential"]),
        "game_seconds_remaining": float(row["game_seconds_remaining"]),
        "half_seconds_remaining": float(row["half_seconds_remaining"]),
        "yardline_100": float(row["yardline_100"]),
        "down": int(row["down"]),
        "ydstogo": float(row["ydstogo"]),
        "posteam_timeouts_remaining": float(row["posteam_timeouts_remaining"]),
        "defteam_timeouts_remaining": float(row["defteam_timeouts_remaining"]),
        "receive_2h_ko": float(row.get("receive_2h_ko", 0) if pd.notna(row.get("receive_2h_ko", 0)) else 0),
        "is_home": float(row.get("is_home", 0) if pd.notna(row.get("is_home", 0)) else 0),
        "spread_line": float(row.get("spread_line", 0) if pd.notna(row.get("spread_line", 0)) else 0),
        "is_dome": float(row.get("is_dome", 0) if pd.notna(row.get("is_dome", 0)) else 0),
    }


def audit_wp_model(models: dict, df_2025: pd.DataFrame) -> dict:
    """
    Evaluate the WP model against nflfastR's vegas_wp on 2025 data.

    1. Prepare X from df_2025 using WP_FEATURES
    2. Get model predictions: y_pred = wp_model.predict_proba(X)[:, 1]
    3. Get y_true = df_2025['posteam_win'] (binary)
    4. Get vegas_wp = df_2025['vegas_wp']
    5. Compute for BOTH our model and vegas_wp:
       - Brier score: mean((y_pred - y_true)²)
       - AUC-ROC
       - Calibration curve (n_bins=10)
    6. Print comparison table
    7. Plot calibration curves side-by-side and save to reports/figures/calibration_2025.png
    8. Return dict with metrics
    """
    df = df_2025.copy()
    if "posteam_win" not in df.columns or "is_home" not in df.columns:
        df = _add_derived_columns(df)
    if "play_type" in df.columns and "down" in df.columns:
        df = _real_plays(df)

    eval_cols = WP_FEATURES + ["posteam_win", "vegas_wp"]
    mask = df["posteam_win"].notna() & df["vegas_wp"].notna()
    for col in WP_FEATURES:
        mask = mask & df[col].notna()

    df_eval = df.loc[mask].copy()
    if len(df_eval) == 0:
        raise ValueError("No valid plays found in df_2025 with complete WP_FEATURES and targets.")

    X = df_eval[WP_FEATURES].astype(np.float32)
    y_true = df_eval["posteam_win"].astype(int).to_numpy()
    y_pred = models["wp"].predict_proba(X)[:, 1]
    vegas_wp = df_eval["vegas_wp"].astype(float).to_numpy()

    # 5. Compute metrics for both
    model_brier = float(brier_score_loss(y_true, y_pred))
    vegas_brier = float(brier_score_loss(y_true, vegas_wp))

    model_auc = float(roc_auc_score(y_true, y_pred))
    vegas_auc = float(roc_auc_score(y_true, vegas_wp))

    prob_true_model, prob_pred_model = calibration_curve(y_true, y_pred, n_bins=10)
    prob_true_vegas, prob_pred_vegas = calibration_curve(y_true, vegas_wp, n_bins=10)

    # 6. Print comparison table
    print("\n" + "=" * 65)
    print(" WIN PROBABILITY MODEL AUDIT (HELD-OUT 2025 SEASON)")
    print("=" * 65)
    print(f"{'Metric':<28} {'Our Model':<18} {'nflfastR Vegas WP':<18}")
    print("-" * 65)
    print(f"{'Brier Score (lower=better)':<28} {model_brier:<18.5f} {vegas_brier:<18.5f}")
    print(f"{'AUC-ROC (higher=better)':<28} {model_auc:<18.5f} {vegas_auc:<18.5f}")
    print(f"{'Evaluated Plays':<28} {len(df_eval):<18,d} {len(df_eval):<18,d}")
    print("=" * 65 + "\n")

    # 7. Plot calibration curves side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Subplot 1: Our WP Model
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfectly Calibrated")
    ax1.plot(
        prob_pred_model,
        prob_true_model,
        "s-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        label=f"Our Model (Brier = {model_brier:.4f})",
    )
    ax1.set_xlabel("Mean Predicted Win Probability", fontsize=11)
    ax1.set_ylabel("Fraction of Positives (Posteam Won)", fontsize=11)
    ax1.set_title("Our Monotonic HistGB WP Model", fontsize=12, fontweight="bold")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Subplot 2: Vegas WP benchmark
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfectly Calibrated")
    ax2.plot(
        prob_pred_vegas,
        prob_true_vegas,
        "o-",
        color="#d62728",
        linewidth=2,
        markersize=6,
        label=f"Vegas WP (Brier = {vegas_brier:.4f})",
    )
    ax2.set_xlabel("Mean Predicted Win Probability", fontsize=11)
    ax2.set_title("nflfastR Vegas WP Benchmark", fontsize=12, fontweight="bold")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle("Win Probability Calibration Comparison (2025 Season)", fontsize=14, y=1.02)
    fig.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = REPORT_DIR / "calibration_2025.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plot successfully saved to: {plot_path}")

    return {
        "model_brier": model_brier,
        "vegas_brier": vegas_brier,
        "model_auc": model_auc,
        "vegas_auc": vegas_auc,
        "prob_true_model": prob_true_model.tolist(),
        "prob_pred_model": prob_pred_model.tolist(),
        "prob_true_vegas": prob_true_vegas.tolist(),
        "prob_pred_vegas": prob_pred_vegas.tolist(),
        "n_plays": len(df_eval),
        "plot_path": str(plot_path),
    }


def audit_fourth_down_decisions(engine: FourthDownEngine, df_2025: pd.DataFrame) -> pd.DataFrame:
    """
    For every 4th-down play in 2025 where the team actually went for it, kicked FG, or punted:
      1. Build the game state dict from the play's columns
      2. Call engine.evaluate(state)
      3. Compare what the coach actually did vs what the model recommends
      4. Compute: % of plays where coach chose optimally, total WP left on table
      5. Break down by team
      6. Print a summary table
      7. Save team-level results and return as DataFrame
    """
    df = df_2025.copy()
    if "is_dome" not in df.columns or "is_home" not in df.columns:
        df = _add_derived_columns(df)

    valid_types = ["run", "pass", "field_goal", "punt"]
    fourth_mask = (
        (df["down"] == 4)
        & df["play_type"].isin(valid_types)
        & df["posteam"].notna()
        & df["yardline_100"].notna()
        & df["ydstogo"].notna()
        & df["score_differential"].notna()
        & df["game_seconds_remaining"].notna()
        & df["half_seconds_remaining"].notna()
        & df["posteam_timeouts_remaining"].notna()
        & df["defteam_timeouts_remaining"].notna()
    )
    df_4th = df.loc[fourth_mask].copy()

    models = engine.models
    n = len(df_4th)
    yl = df_4th["yardline_100"].values.astype(np.float32)
    ytg = df_4th["ydstogo"].values.astype(np.float32)
    sd = df_4th["score_differential"].values.astype(np.float32)
    gsr = df_4th["game_seconds_remaining"].values.astype(np.float32)
    hsr = df_4th["half_seconds_remaining"].values.astype(np.float32)
    pto = df_4th["posteam_timeouts_remaining"].values.astype(np.float32)
    dto = df_4th["defteam_timeouts_remaining"].values.astype(np.float32)
    r2h = df_4th.get("receive_2h_ko", pd.Series(0, index=df_4th.index)).values.astype(np.float32)
    ish = df_4th.get("is_home", pd.Series(0, index=df_4th.index)).values.astype(np.float32)
    spl = df_4th.get("spread_line", pd.Series(0, index=df_4th.index)).values.astype(np.float32)
    is_dome = df_4th.get("is_dome", pd.Series(0, index=df_4th.index)).values.astype(np.float32)
    is_gtg = np.where(yl <= ytg, 1.0, 0.0).astype(np.float32)

    # 1. P(convert)
    conv_X = pd.DataFrame({"ydstogo": ytg, "yardline_100": yl, "score_differential": sd, "game_seconds_remaining": gsr, "is_goal_to_go": is_gtg})
    p_conv = models["conv"].predict_proba(conv_X)[:, 1]

    # WP(convert)
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
    wp_conv_raw = models["wp"].predict_proba(X_conv)[:, 1]
    wp_convert = np.where(is_td, 1.0 - wp_conv_raw, wp_conv_raw)

    # WP(fail)
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
    go_wp = p_conv * wp_convert + (1.0 - p_conv) * wp_fail

    # Punt
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

    # FG
    if "kick_distance" in df_4th.columns:
        kd = np.where(df_4th["kick_distance"].notna(), df_4th["kick_distance"].values, yl + FG_SNAP_OFFSET).astype(np.float32)
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

    opts = np.column_stack([go_wp, np.where(valid_fg, fg_wp, -1e9), punt_wp])
    choice_idx = np.argmax(opts, axis=1)
    rec_map = {0: "go", 1: "fg", 2: "punt"}
    recommended = [rec_map[i] for i in choice_idx]

    sorted_opts = np.sort(opts, axis=1)[:, ::-1]
    second_best = np.where(valid_fg, sorted_opts[:, 1], np.minimum(go_wp, punt_wp))
    margins = sorted_opts[:, 0] - second_best

    act_map = {"run": "go", "pass": "go", "field_goal": "fg", "punt": "punt"}
    actual_actions = df_4th["play_type"].map(act_map).values

    actual_wps = np.where(actual_actions == "go", go_wp, np.where(actual_actions == "fg", np.where(valid_fg, fg_wp, 0.0), punt_wp))
    optimal_wps = sorted_opts[:, 0]
    wp_lefts = np.maximum(0.0, optimal_wps - actual_wps)
    is_optimals = (actual_actions == np.array(recommended)).astype(int)

    play_audit_df = pd.DataFrame({
        "posteam": df_4th["posteam"].values,
        "actual_action": actual_actions,
        "recommended": recommended,
        "is_optimal": is_optimals,
        "optimal_wp": optimal_wps,
        "actual_wp": actual_wps,
        "wp_left": wp_lefts,
        "margin": margins,
        "go_wp": go_wp,
        "fg_wp": np.where(valid_fg, fg_wp, np.nan),
        "punt_wp": punt_wp,
    })

    # 4 & 5. Team breakdown
    team_rows = []
    for team, grp in play_audit_df.groupby("posteam"):
        n_plays = len(grp)
        n_optimal = int(grp["is_optimal"].sum())
        optimal_pct = (n_optimal / n_plays) * 100.0 if n_plays > 0 else 0.0
        tot_wp_left = float(grp["wp_left"].sum())
        avg_wp_left = float(grp["wp_left"].mean()) if n_plays > 0 else 0.0

        team_rows.append({
            "team": team,
            "total_plays": n_plays,
            "optimal_decisions": n_optimal,
            "optimal_pct": optimal_pct,
            "total_wp_left": tot_wp_left,
            "avg_wp_left": avg_wp_left,
            "actual_go": int((grp["actual_action"] == "go").sum()),
            "actual_fg": int((grp["actual_action"] == "fg").sum()),
            "actual_punt": int((grp["actual_action"] == "punt").sum()),
            "rec_go": int((grp["recommended"] == "go").sum()),
            "rec_fg": int((grp["recommended"] == "fg").sum()),
            "rec_punt": int((grp["recommended"] == "punt").sum()),
        })

    team_summary_df = pd.DataFrame(team_rows).sort_values(
        by=["optimal_pct", "total_wp_left"], ascending=[False, True]
    ).reset_index(drop=True)

    # 6. Print summary table
    league_plays = int(team_summary_df["total_plays"].sum())
    league_optimal = int(team_summary_df["optimal_decisions"].sum())
    league_pct = (league_optimal / league_plays * 100.0) if league_plays > 0 else 0.0
    league_total_wp_left = float(team_summary_df["total_wp_left"].sum())
    league_avg_wp_left = (league_total_wp_left / league_plays) if league_plays > 0 else 0.0

    print("\n" + "=" * 92)
    print(" 2025 NFL FOURTH-DOWN COACH DECISION AUDIT")
    print("=" * 92)
    print(f"Total 4th-Down Decisions Analyzed : {league_plays:,}")
    print(f"Optimal Coach Decisions          : {league_optimal:,} ({league_pct:.1f}%)")
    print(f"Total WP Left on the Table       : {league_total_wp_left:.2f} ({league_total_wp_left * 100:.1f} percentage points)")
    print(f"Average WP Sacrificed / Decision : {league_avg_wp_left * 100:.2f} pp")
    print("-" * 92)
    print(f"{'Team':<6} {'Plays':<7} {'Optimal':<9} {'Opt %':<9} {'Total WP Lost':<15} {'Avg WP Lost':<13} {'Act (Go/FG/P)':<15} {'Rec (Go/FG/P)':<15}")
    print("-" * 92)
    for _, r in team_summary_df.iterrows():
        act_str = f"({int(r['actual_go'])}/{int(r['actual_fg'])}/{int(r['actual_punt'])})"
        rec_str = f"({int(r['rec_go'])}/{int(r['rec_fg'])}/{int(r['rec_punt'])})"
        print(
            f"{r['team']:<6} {int(r['total_plays']):<7} {int(r['optimal_decisions']):<9} "
            f"{r['optimal_pct']:<8.1f}% {r['total_wp_left']:<14.2f} {r['avg_wp_left'] * 100:<11.2f}pp "
            f"{act_str:<15} {rec_str:<15}"
        )
    print("=" * 92 + "\n")

    # 7. Save team-level results
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path1 = REPORT_DIR / "fourth_down_audit_2025.csv"
    csv_path2 = ROOT / "reports" / "fourth_down_audit_2025.csv"
    csv_path2.parent.mkdir(parents=True, exist_ok=True)

    team_summary_df.to_csv(csv_path1, index=False)
    team_summary_df.to_csv(csv_path2, index=False)
    print(f"Team audit results saved to: {csv_path1}")

    return team_summary_df


def run_full_audit(
    models: dict, df_2025: pd.DataFrame, engine: FourthDownEngine | None = None
) -> dict:
    """
    Orchestrator: calls audit_wp_model and audit_fourth_down_decisions.
    Prints a banner and returns all audit results.
    """
    print("\n" + "#" * 70)
    print("   RUNNING FULL 2025 AUDIT: WIN PROBABILITY & 4TH DOWN DECISIONS")
    print("#" * 70)

    # 1. Audit Win Probability model
    wp_metrics = audit_wp_model(models, df_2025)

    # 2. Audit 4th Down decisions
    if engine is None:
        engine = FourthDownEngine(models)
    team_audit_df = audit_fourth_down_decisions(engine, df_2025)

    print("#" * 70)
    print("   FULL AUDIT FINISHED SUCCESSFULLY")
    print("#" * 70 + "\n")

    return {
        "wp_metrics": wp_metrics,
        "team_audit": team_audit_df,
    }
