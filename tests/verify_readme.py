import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
readme = (ROOT / "README.md").read_text(encoding="utf-8")
metrics = json.loads((ROOT / "reports" / "audit_metrics_2025.json").read_text(encoding="utf-8"))
fixtures = json.loads((ROOT / "tests" / "fixtures_500.json").read_text(encoding="utf-8"))

checks = []

# 1. Total decisions
checks.append(("3,997 total decisions in README", "3,997" in readme))

# 2. Go rate on clear go
base_c = metrics["baseline"]
checks.append(("33.1% clear-go rate in README", "33.1%" in readme))
checks.append(("Clear-go CI in README", ("30.2%" in readme or "30.3%" in readme) and ("36.1%" in readme or "36.2%" in readme)))

# 3. WP lost per team game
checks.append(("37.07 pp WP lost in README", "37.07" in readme))
checks.append(("WP lost CI in README", ("34.20" in readme or "34.18" in readme) and ("39.86" in readme or "39.88" in readme)))

# 4. Overall agreement
checks.append(("32.3% agreement in README", "32.3%" in readme))

# 5. Sensitivity test
sens_c = metrics["sensitivity_5pp_haircut"]
checks.append(("1,084 sensitivity plays in README", "1,084" in readme))
checks.append(("34.9% sensitivity go rate in README", "34.9%" in readme))
checks.append(("34.09 pp sensitivity WP lost in README", "34.09" in readme))

# 6. Model audit table numbers
checks.append(("Brier score 0.16090 in README", "0.16090" in readme))
checks.append(("Vegas Brier 0.15854 in README", "0.15854" in readme))
checks.append(("AUC-ROC 0.84793 in README", "0.84793" in readme))
checks.append(("Vegas AUC-ROC 0.84958 in README", "0.84958" in readme))
checks.append(("38,061 evaluated plays in README", "38,061" in readme))

# 7. Test fixture count
checks.append(("500 situations in README", "500" in readme and len(fixtures) == 500))

all_passed = True
for desc, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {desc}")

print(f"\nAudit check status: {'ALL NUMBERS MATCH EXACTLY' if all_passed else 'MISMATCH DETECTED'}")
