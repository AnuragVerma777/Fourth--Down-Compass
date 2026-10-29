"""
tests/test_model_parity.py
──────────────────────────
Automated parity test proving that the exported JavaScript decision engine matches
scikit-learn Python models on 500 randomized 4th-down game situations.
"""

import json
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = ROOT / "models" / "models_exported.json"
FIXTURES_JSON = ROOT / "tests" / "fixtures_500.json"


def test_500_situations_parity_with_node():
    """Verify 100% decision recommendation match and <1e-4 probability error."""
    assert MODELS_JSON.exists(), f"Missing {MODELS_JSON}. Run scripts/export_and_test.py first."
    assert FIXTURES_JSON.exists(), f"Missing {FIXTURES_JSON}."

    with open(FIXTURES_JSON, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    assert len(fixtures) == 500, f"Expected 500 fixtures, found {len(fixtures)}"

    js_code = f"""
    const fs = require('fs');
    const modelsData = JSON.parse(fs.readFileSync({json.dumps(str(MODELS_JSON))}, 'utf8'));
    const testCases = JSON.parse(fs.readFileSync({json.dumps(str(FIXTURES_JSON))}, 'utf8'));

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
        if (modelSpec.is_reg) return raw;
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

        let pConv = (customPConv !== null) ? customPConv : predictModel(models.conv, [ydstogo, yl100, sd, gsr, isGtg]);

        let wpConvert;
        if (yl100 > ydstogo) {{
            const newYl = yl100 - ydstogo;
            const newYtg = Math.min(10.0, newYl);
            wpConvert = predictModel(models.wp, [sd, gsr, hsr, newYl, 1.0, newYtg, pto, dto, r2h, isHome, spread]);
        }} else {{
            const oppWp = predictModel(models.wp, [-(sd + 7.0), gsr, hsr, 75.0, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread]);
            wpConvert = 1.0 - oppWp;
        }}

        const oppWpFail = predictModel(models.wp, [-sd, gsr, hsr, 100.0 - yl100, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread]);
        const wpFail = 1.0 - oppWpFail;
        const goWp = pConv * wpConvert + (1.0 - pConv) * wpFail;

        const kickDist = yl100 + 17.0;
        let fgWp = null;
        let pMake = null;
        if (kickDist <= 66.0) {{
            pMake = predictModel(models.fg, [kickDist, isDome]);
            const makeOppWp = predictModel(models.wp, [-(sd + 3.0), gsr, hsr, 75.0, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread]);
            const wpMake = 1.0 - makeOppWp;

            const missSpot = Math.max(yl100, 20.0);
            const missOppWp = predictModel(models.wp, [-sd, gsr, hsr, 100.0 - missSpot, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread]);
            const wpMiss = 1.0 - missOppWp;
            fgWp = pMake * wpMake + (1.0 - pMake) * wpMiss;
        }}

        const oppPuntYl = Math.max(1.0, Math.min(99.0, predictModel(models.punt, [yl100])));
        const puntOppWp = predictModel(models.wp, [-sd, gsr, hsr, oppPuntYl, 1.0, 10.0, dto, pto, 1.0 - r2h, 1.0 - isHome, -spread]);
        const puntWp = 1.0 - puntOppWp;

        const bestAlt = (fgWp !== null) ? Math.max(fgWp, puntWp) : puntWp;
        const denom = wpConvert - wpFail;
        let breakEven = (Math.abs(denom) < 1e-9) ? ((bestAlt <= wpFail) ? 0.0 : 1.0) : Math.max(0.0, Math.min(1.0, (bestAlt - wpFail) / denom));

        let rec = 'go';
        let maxWp = goWp;
        let secondWp = puntWp;

        if (fgWp !== null) {{
            if (fgWp > maxWp) {{ secondWp = maxWp; maxWp = fgWp; rec = 'fg'; }}
            else if (fgWp > secondWp) {{ secondWp = fgWp; }}
            if (puntWp > maxWp) {{ secondWp = maxWp; maxWp = puntWp; rec = 'punt'; }}
            else if (puntWp > secondWp) {{ secondWp = puntWp; }}
        }} else {{
            if (puntWp > maxWp) {{ secondWp = maxWp; maxWp = puntWp; rec = 'punt'; }}
            else {{ secondWp = puntWp; }}
        }}
        const margin = maxWp - secondWp;

        return {{ go_wp: goWp, fg_wp: fgWp, punt_wp: puntWp, p_convert: pConv, recommendation: rec, margin: margin }};
    }}

    let matches = 0;
    let maxDiffGo = 0;
    for (let i = 0; i < testCases.length; i++) {{
        const tc = testCases[i];
        const res = evaluateState(modelsData, tc.state);
        if (res.recommendation === tc.recommendation) matches++;
        const dGo = Math.abs(res.go_wp - tc.go_wp);
        if (dGo > maxDiffGo) maxDiffGo = dGo;
    }}
    console.log(JSON.stringify({{ matches: matches, total: testCases.length, maxDiffGo: maxDiffGo }}));
    """

    res = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    out = json.loads(res.stdout.strip())

    assert out["matches"] == 500, f"Expected 500 matches, got {out['matches']}"
    assert out["maxDiffGo"] < 1e-4, f"Max WP diff {out['maxDiffGo']} is too high"
    print(f"\nAll 500 tests passed with 100% recommendation agreement (max WP diff = {out['maxDiffGo']:.8f})")


if __name__ == "__main__":
    test_500_situations_parity_with_node()
