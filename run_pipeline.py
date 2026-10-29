"""
run_pipeline.py
───────────────
End-to-end orchestrator for the AQX NFL 4th-Down Decision Tool.

    python run_pipeline.py            # full pipeline
    python run_pipeline.py --sanity   # sanity checks only (after models are saved)

Steps:
  1. Download data (if needed)
  2. Prepare train & audit datasets
  3. Train all four models on 2020-2024
  4. Freeze models to disk
  5. Audit on the unseen 2025 season
  6. Run sanity-check scenarios
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TRAIN_SEASONS, AUDIT_SEASON, MODEL_DIR, REPORT_DIR,
    KICKOFF_YARDLINE, FG_SNAP_OFFSET, PAT_POINTS, FG_POINTS,
    MAX_FG_DISTANCE,
)


# ====================================================================
# Sanity-check scenarios a coach would instantly recognise
# ====================================================================
SANITY_SCENARIOS = [
    {
        "label": "4th & 1, OPP 2, tied, Q3 5:00 — MUST GO",
        "state": dict(
            score_differential=0, game_seconds_remaining=1500,
            half_seconds_remaining=300, yardline_100=2, down=4, ydstogo=1,
            posteam_timeouts_remaining=3, defteam_timeouts_remaining=3,
            receive_2h_ko=0, is_home=1, spread_line=-3.0, is_dome=0,
        ),
        "expected": "go",
    },
    {
        "label": "4th & 15, OWN 20, down 3, Q2 8:00 — PUNT",
        "state": dict(
            score_differential=-3, game_seconds_remaining=2280,
            half_seconds_remaining=480, yardline_100=80, down=4, ydstogo=15,
            posteam_timeouts_remaining=3, defteam_timeouts_remaining=3,
            receive_2h_ko=1, is_home=0, spread_line=3.0, is_dome=0,
        ),
        "expected": "punt",
    },
    {
        "label": "4th & Goal at 1, down 4, Q4 0:30, 0 TO — MUST GO (need TD)",
        "state": dict(
            score_differential=-4, game_seconds_remaining=30,
            half_seconds_remaining=30, yardline_100=1, down=4, ydstogo=1,
            posteam_timeouts_remaining=0, defteam_timeouts_remaining=0,
            receive_2h_ko=0, is_home=1, spread_line=-3.0, is_dome=0,
        ),
        "expected": "go",
    },
    {
        "label": "4th & 8, OPP 30, up 14, Q3 10:00 — FG (protect lead)",
        "state": dict(
            score_differential=14, game_seconds_remaining=2400,
            half_seconds_remaining=600, yardline_100=30, down=4, ydstogo=8,
            posteam_timeouts_remaining=3, defteam_timeouts_remaining=3,
            receive_2h_ko=0, is_home=1, spread_line=-7.0, is_dome=0,
        ),
        "expected": "fg",
    },
    {
        "label": "4th & 1, midfield, tied, Q1 12:00 — Analytics says GO",
        "state": dict(
            score_differential=0, game_seconds_remaining=3420,
            half_seconds_remaining=720, yardline_100=50, down=4, ydstogo=1,
            posteam_timeouts_remaining=3, defteam_timeouts_remaining=3,
            receive_2h_ko=0, is_home=1, spread_line=-1.0, is_dome=0,
        ),
        "expected": "go",
    },
    {
        "label": "4th & 3, OPP 35, tied, Q4 2:00 — borderline go-for-it",
        "state": dict(
            score_differential=0, game_seconds_remaining=120,
            half_seconds_remaining=120, yardline_100=35, down=4, ydstogo=3,
            posteam_timeouts_remaining=2, defteam_timeouts_remaining=2,
            receive_2h_ko=0, is_home=1, spread_line=-2.5, is_dome=0,
        ),
        "expected": "go",
    },
    {
        "label": "4th & 3, OPP 5, down 1, Q4 3:00 — FG ties it up",
        "state": dict(
            score_differential=-3, game_seconds_remaining=180,
            half_seconds_remaining=180, yardline_100=5, down=4, ydstogo=3,
            posteam_timeouts_remaining=1, defteam_timeouts_remaining=1,
            receive_2h_ko=0, is_home=0, spread_line=2.0, is_dome=0,
        ),
        # Either go or fg is defensible here
        "expected": None,  # no strict expectation — just inspect
    },
    {
        "label": "4th & 10, OWN 30, up 3, Q4 1:30 — PUNT (run clock)",
        "state": dict(
            score_differential=3, game_seconds_remaining=90,
            half_seconds_remaining=90, yardline_100=70, down=4, ydstogo=10,
            posteam_timeouts_remaining=0, defteam_timeouts_remaining=2,
            receive_2h_ko=0, is_home=1, spread_line=-3.0, is_dome=0,
        ),
        "expected": "punt",
    },
]


def run_sanity_checks(engine) -> tuple[int, int]:
    """
    Run pre-defined sanity scenarios.  Returns (passed, total).
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS — known 4th-down situations")
    print("=" * 70)

    passed = 0
    total  = 0

    for sc in SANITY_SCENARIOS:
        label    = sc["label"]
        state    = sc["state"]
        expected = sc["expected"]

        result = engine.evaluate(state)
        rec    = result["recommendation"]

        if expected is not None:
            total += 1
            ok = rec == expected
            if ok:
                passed += 1
            icon = "[OK]" if ok else "[FAIL]"
        else:
            icon = "[?]"          # no strict expectation

        fg_str = f"{result['fg_wp']:.1%}" if result['fg_wp'] is not None else "N/A"
        print(f"\n  {icon}  {label}")
        print(f"     Rec: {rec.upper():6s}  |  "
              f"Go={result['go_wp']:.1%}  "
              f"FG={fg_str}  "
              f"Punt={result['punt_wp']:.1%}")
        print(f"     P(conv)={result['p_convert']:.0%}  "
              f"Break-even={result['break_even_conv_rate']:.0%}  "
              f"Margin={result['margin']:+.1%}")
        if expected is not None and not ok:
            print(f"     WARNING: Expected {expected.upper()}, got {rec.upper()}")

    print(f"\n  Passed {passed}/{total} strict checks")
    return passed, total


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="AQX 4th-Down Decision Tool")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity checks only (models must exist)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download step")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip 2025 audit")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── If --sanity, just load models and run checks ────────────────
    if args.sanity:
        from src.models import load_models
        from src.engine import FourthDownEngine
        models = load_models()
        engine = FourthDownEngine(models)
        run_sanity_checks(engine)
        return

    # ── Step 1: Download ────────────────────────────────────────────
    if not args.skip_download:
        print("\n" + "=" * 70)
        print("STEP 1 / 5 — Download data")
        print("=" * 70)
        # Import from scripts directory
        sys.path.insert(0, str(ROOT / "scripts"))
        from download_data import download
        download()

    # ── Step 2: Prepare data ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 / 5 — Prepare training data (2020-2024)")
    print("=" * 70)
    from src.data_pipeline import prepare_data
    t0 = time.time()
    df_train, datasets = prepare_data(TRAIN_SEASONS)
    print(f"  Prepared in {time.time() - t0:.1f}s")

    # ── Step 3: Train models ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 / 5 — Train models")
    print("=" * 70)
    from src.models import train_all_models, save_models
    t0 = time.time()
    models = train_all_models(datasets)
    print(f"  All models trained in {time.time() - t0:.1f}s")

    # ── Step 4: Freeze to disk ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 / 5 — Freeze models to disk")
    print("=" * 70)
    save_models(models, row_counts={
        k: len(v[0]) for k, v in datasets.items()
    })

    # ── Step 5: Build engine & sanity checks ────────────────────────
    from src.engine import FourthDownEngine
    engine = FourthDownEngine(models)
    run_sanity_checks(engine)

    # ── Step 6: Audit on 2025 (optional) ────────────────────────────
    if not args.skip_audit:
        print("\n" + "=" * 70)
        print("STEP 5 / 5 — Audit on unseen 2025 season")
        print("=" * 70)
        from src.data_pipeline import prepare_data as prep
        df_2025, _ = prep([AUDIT_SEASON])
        from src.audit import run_full_audit
        run_full_audit(models, df_2025, engine)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Models saved in:   {MODEL_DIR}")
    print(f"  Figures saved in:  {REPORT_DIR}")
    print("  Run  python run_pipeline.py --sanity  to re-test scenarios.\n")


if __name__ == "__main__":
    main()
