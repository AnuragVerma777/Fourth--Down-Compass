"""
scripts/export_and_test.py
──────────────────────────
Exports the trained HistGradientBoosting models to JSON for client-side JS evaluation,
and verifies mathematical parity against Python on 500 randomized game situations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    MODEL_DIR,
    ROOT,
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
from src.models import load_models
from src.engine import FourthDownEngine


def export_models_to_dict(models: dict) -> dict:
    """Export the 4 models into compact tree representations."""
    exported = {}
    for name, is_reg, feats in [
        ("wp", False, WP_FEATURES),
        ("conv", False, CONV_FEATURES),
        ("fg", False, FG_FEATURES),
        ("punt", True, PUNT_FEATURES),
    ]:
        m = models[name]
        trees = []
        for pred_list in m._predictors:
            nodes = pred_list[0].nodes
            tree_repr = []
            for n in nodes:
                if n["is_leaf"]:
                    # Leaf node: store value
                    tree_repr.append([round(float(n["value"]), 7)])
                else:
                    # Split node: [feature_idx, threshold, left_child, right_child]
                    tree_repr.append([
                        int(n["feature_idx"]),
                        round(float(n["num_threshold"]), 5),
                        int(n["left"]),
                        int(n["right"]),
                    ])
            trees.append(tree_repr)

        bp = float(np.squeeze(m._baseline_prediction))
        exported[name] = {
            "baseline": round(bp, 7),
            "is_reg": is_reg,
            "features": list(feats),
            "trees": trees,
        }
    return exported


def generate_500_scenarios(seed: int = 42) -> list[dict]:
    """Generate 500 diverse, realistic, and stress-testing 4th-down game situations."""
    rng = np.random.default_rng(seed)
    scenarios = []

    for i in range(500):
        # Quarter & Time
        qtr = int(rng.choice([1, 2, 3, 4, 5], p=[0.24, 0.24, 0.24, 0.26, 0.02]))
        if qtr in (1, 3):
            # Q1 or Q3: 15 mins
            sec_in_qtr = float(rng.integers(1, 900))
            if qtr == 1:
                gsr = 2700.0 + sec_in_qtr
                hsr = 900.0 + sec_in_qtr
                r2h = int(rng.choice([0, 1]))
            else:
                gsr = 900.0 + sec_in_qtr
                hsr = 900.0 + sec_in_qtr
                r2h = 0
        elif qtr in (2, 4):
            sec_in_qtr = float(rng.integers(1, 900))
            if qtr == 2:
                gsr = 1800.0 + sec_in_qtr
                hsr = sec_in_qtr
                r2h = int(rng.choice([0, 1]))
            else:
                gsr = sec_in_qtr
                hsr = sec_in_qtr
                r2h = 0
        else: # OT
            sec_in_qtr = float(rng.integers(1, 600))
            gsr = sec_in_qtr
            hsr = sec_in_qtr
            r2h = 0

        # Score diff
        diff = int(rng.choice(np.arange(-28, 29)))
        # Yardline
        yl = float(rng.integers(1, 99))
        # Ydstogo
        # Bias towards short yardage (4th & 1 to 4th & 10)
        ydstogo = float(rng.choice(
            [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        ))
        # Goal to go
        is_gtg = 1.0 if yl <= ydstogo else 0.0

        p_to = int(rng.choice([0, 1, 2, 3]))
        d_to = int(rng.choice([0, 1, 2, 3]))
        ish = int(rng.choice([0, 1]))
        spread = float(rng.choice(np.arange(-14.0, 14.5, 0.5)))
        is_dome = int(rng.choice([0, 1]))

        scenarios.append({
            "id": i + 1,
            "score_differential": diff,
            "game_seconds_remaining": gsr,
            "half_seconds_remaining": hsr,
            "yardline_100": yl,
            "down": 4,
            "ydstogo": ydstogo,
            "posteam_timeouts_remaining": p_to,
            "defteam_timeouts_remaining": d_to,
            "receive_2h_ko": r2h,
            "is_home": ish,
            "spread_line": spread,
            "is_dome": is_dome,
            "is_goal_to_go": is_gtg,
        })

    return scenarios


def evaluate_with_python_engine(engine: FourthDownEngine, scenarios: list[dict]) -> list[dict]:
    """Run Python FourthDownEngine on all scenarios."""
    results = []
    for sc in scenarios:
        res = engine.evaluate(sc)
        results.append({
            "id": sc["id"],
            "state": sc,
            "go_wp": round(res["go_wp"], 6),
            "fg_wp": round(res["fg_wp"], 6) if res["fg_wp"] is not None else None,
            "punt_wp": round(res["punt_wp"], 6),
            "p_convert": round(res["p_convert"], 6),
            "p_fg_make": round(res["p_fg_make"], 6) if res["p_fg_make"] is not None else None,
            "break_even": round(res["break_even_conv_rate"], 6),
            "recommendation": res["recommendation"],
            "margin": round(res["margin"], 6),
        })
    return results


def run_node_verification(models_json_path: Path, fixtures_path: Path) -> dict:
    """Run Node.js script to execute the JS engine and verify against fixtures."""
    js_test_code = f"""
    const fs = require('fs');
    const modelsData = JSON.parse(fs.readFileSync({json.dumps(str(models_json_path))}, 'utf8'));
    const testCases = JSON.parse(fs.readFileSync({json.dumps(str(fixtures_path))}, 'utf8'));

    // --- JS Tree Traversal & Decision Engine ---
    function predictModel(modelSpec, featureValues) {{
        let raw = modelSpec.baseline;
        const trees = modelSpec.trees;
        for (let i = 0; i < trees.length; i++) {{
            const nodes = trees[i];
            let idx = 0;
            while (nodes[idx].length > 1) {{
                const featIdx = nodes[idx][0];
                const thresh = nodes[idx][1];
                const leftChild = nodes[idx][2];
                const rightChild = nodes[idx][3];
                const val = featureValues[featIdx];
                if (val <= thresh) {{
                    idx = leftChild;
                }} else {{
                    idx = rightChild;
                }}
            }}
            raw += nodes[idx][0];
        }}
        if (modelSpec.is_reg) {{
            return raw;
        }}
        return 1.0 / (1.0 + Math.exp(-raw));
    }}

    function evaluateState(models, state, customPConv = null) {{
        const yl100 = state.yardline_100;
        const ydstogo = state.ydstogo;
        const sd = state.score_differential;
        const gsr = state.game_seconds_remaining;
        const hsr = state.half_seconds_remaining;
        const pto = state.posteam_timeouts_remaining;
        const dto = state.defteam_timeouts_remaining;
        const r2h = state.receive_2h_ko || 0;
        const isHome = state.is_home || 0;
        const spread = state.spread_line || 0;
        const isDome = state.is_dome || 0;
        const isGtg = (yl100 <= ydstogo) ? 1.0 : 0.0;

        // 1. P(convert)
        let pConv;
        if (customPConv !== null) {{
            pConv = customPConv;
        }} else {{
            pConv = predictModel(models.conv, [ydstogo, yl100, sd, gsr, isGtg]);
        }}

        // WP(convert)
        let wpConvert;
        if (yl100 > ydstogo) {{
            const newYl = yl100 - ydstogo;
            const newYtg = Math.min(10.0, newYl);
            wpConvert = predictModel(models.wp, [
                sd, gsr, hsr, newYl, 1.0, newYtg, pto, dto, r2h, isHome, spread
            ]);
        }} else {{
            // TD
            const flippedSd = -(sd + 7.0);
            const oppWp = predictModel(models.wp, [
                flippedSd, gsr, hsr, 75.0, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread
            ]);
            wpConvert = 1.0 - oppWp;
        }}

        // WP(fail)
        const failOppYl = 100.0 - yl100;
        const oppWpFail = predictModel(models.wp, [
            -sd, gsr, hsr, failOppYl, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread
        ]);
        const wpFail = 1.0 - oppWpFail;

        const goWp = pConv * wpConvert + (1.0 - pConv) * wpFail;

        // 2. Field Goal
        const kickDist = yl100 + 17.0;
        let fgWp = null;
        let pMake = null;
        if (kickDist <= 66.0) {{
            pMake = predictModel(models.fg, [kickDist, isDome]);
            const makeOppWp = predictModel(models.wp, [
                -(sd + 3.0), gsr, hsr, 75.0, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread
            ]);
            const wpMake = 1.0 - makeOppWp;

            const missSpot = Math.max(yl100, 20.0);
            const missOppYl = 100.0 - missSpot;
            const missOppWp = predictModel(models.wp, [
                -sd, gsr, hsr, missOppYl, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread
            ]);
            const wpMiss = 1.0 - missOppWp;

            fgWp = pMake * wpMake + (1.0 - pMake) * wpMiss;
        }}

        // 3. Punt
        const rawPuntYl = predictModel(models.punt, [yl100]);
        const oppPuntYl = Math.max(1.0, Math.min(99.0, rawPuntYl));
        const puntOppWp = predictModel(models.wp, [
            -sd, gsr, hsr, oppPuntYl, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread
        ]);
        const puntWp = 1.0 - puntOppWp;

        // 4. Break-even
        const bestAlt = (fgWp !== null) ? Math.max(fgWp, puntWp) : puntWp;
        const denom = wpConvert - wpFail;
        let breakEven;
        if (Math.abs(denom) < 1e-9) {{
            breakEven = (bestAlt <= wpFail) ? 0.0 : 1.0;
        }} else {{
            breakEven = (bestAlt - wpFail) / denom;
            breakEven = Math.max(0.0, Math.min(1.0, breakEven));
        }}

        // 5. Recommendation & Margin
        let rec = 'go';
        let maxWp = goWp;
        let secondWp = puntWp;

        if (fgWp !== null) {{
            if (fgWp > maxWp) {{
                secondWp = maxWp;
                maxWp = fgWp;
                rec = 'fg';
            }} else if (fgWp > secondWp) {{
                secondWp = fgWp;
            }}
            if (puntWp > maxWp) {{
                secondWp = maxWp;
                maxWp = puntWp;
                rec = 'punt';
            }} else if (puntWp > secondWp) {{
                secondWp = puntWp;
            }}
        }} else {{
            if (puntWp > maxWp) {{
                secondWp = maxWp;
                maxWp = puntWp;
                rec = 'punt';
            }} else {{
                secondWp = puntWp;
            }}
        }}
        const margin = maxWp - secondWp;

        return {{
            go_wp: goWp,
            fg_wp: fgWp,
            punt_wp: puntWp,
            p_convert: pConv,
            p_fg_make: pMake,
            break_even: breakEven,
            recommendation: rec,
            margin: margin,
        }};
    }}

    // Verify 500 test cases
    let recMatches = 0;
    let maxDiffGo = 0;
    let maxDiffPunt = 0;
    let maxDiffFg = 0;
    let maxDiffConv = 0;
    let maxDiffMargin = 0;

    for (let i = 0; i < testCases.length; i++) {{
        const tc = testCases[i];
        const jsRes = evaluateState(modelsData, tc.state);

        if (jsRes.recommendation === tc.recommendation) {{
            recMatches++;
        }}

        const dGo = Math.abs(jsRes.go_wp - tc.go_wp);
        const dPunt = Math.abs(jsRes.punt_wp - tc.punt_wp);
        const dConv = Math.abs(jsRes.p_convert - tc.p_convert);
        const dMargin = Math.abs(jsRes.margin - tc.margin);

        if (dGo > maxDiffGo) maxDiffGo = dGo;
        if (dPunt > maxDiffPunt) maxDiffPunt = dPunt;
        if (dConv > maxDiffConv) maxDiffConv = dConv;
        if (dMargin > maxDiffMargin) maxDiffMargin = dMargin;

        if (tc.fg_wp !== null && jsRes.fg_wp !== null) {{
            const dFg = Math.abs(jsRes.fg_wp - tc.fg_wp);
            if (dFg > maxDiffFg) maxDiffFg = dFg;
        }}
    }}

    const out = {{
        total_tests: testCases.length,
        recommendation_matches: recMatches,
        match_rate_pct: (recMatches / testCases.length) * 100,
        max_diff_go_wp: maxDiffGo,
        max_diff_fg_wp: maxDiffFg,
        max_diff_punt_wp: maxDiffPunt,
        max_diff_p_conv: maxDiffConv,
        max_diff_margin: maxDiffMargin,
    }};

    console.log(JSON.stringify(out));
    """

    runner_script = ROOT / "tests" / "_temp_node_runner.js"
    runner_script.parent.mkdir(parents=True, exist_ok=True)
    with open(runner_script, "w", encoding="utf-8") as f:
        f.write(js_test_code)

    res = subprocess.run(
        ["node", str(runner_script)],
        capture_output=True,
        text=True,
        check=True,
    )
    runner_script.unlink(missing_ok=True)
    return json.loads(res.stdout.strip())


def main():
    print("=" * 70)
    print("EXPORTING TRAINED MODELS & VERIFYING PARITY ON 500 SITUATIONS")
    print("=" * 70)

    # 1. Load Python models & engine
    models = load_models()
    engine = FourthDownEngine(models)

    # 2. Export models to compact dict
    print("Exporting models to compact tree representation...")
    models_dict = export_models_to_dict(models)

    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "models_exported.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(models_dict, f)
    print(f"  Exported models JSON saved to: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

    # 3. Generate 500 test scenarios
    print("Generating 500 randomized game situations...")
    scenarios = generate_500_scenarios(seed=42)

    # 4. Run Python engine
    print("Evaluating 500 situations with Python FourthDownEngine...")
    fixtures = evaluate_with_python_engine(engine, scenarios)

    tests_dir = ROOT / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    fixtures_path = tests_dir / "fixtures_500.json"
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)
    print(f"  Saved 500 evaluation fixtures to: {fixtures_path}")

    # 5. Run Node.js verification
    print("Running Node.js client-side inference engine on all 500 situations...")
    verify_res = run_node_verification(json_path, fixtures_path)

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS (PYTHON vs JAVASCRIPT INFERENCE)")
    print("=" * 70)
    print(f"Total Test Situations Evaluated : {verify_res['total_tests']}")
    print(f"Recommendation Matches          : {verify_res['recommendation_matches']} / {verify_res['total_tests']} ({verify_res['match_rate_pct']:.2f}%)")
    print(f"Max Difference in Go WP         : {verify_res['max_diff_go_wp']:.8f}")
    print(f"Max Difference in FG WP         : {verify_res['max_diff_fg_wp']:.8f}")
    print(f"Max Difference in Punt WP       : {verify_res['max_diff_punt_wp']:.8f}")
    print(f"Max Difference in P(convert)    : {verify_res['max_diff_p_conv']:.8f}")
    print(f"Max Difference in Margin        : {verify_res['max_diff_margin']:.8f}")
    print("=" * 70 + "\n")

    assert verify_res["recommendation_matches"] == 500, "Not all recommendations matched!"
    assert verify_res["max_diff_go_wp"] < 1e-4, "Go WP difference exceeded tolerance!"
    print("SUCCESS: 100% PARITY CONFIRMED ON 500 RANDOM SITUATIONS!")


if __name__ == "__main__":
    main()
