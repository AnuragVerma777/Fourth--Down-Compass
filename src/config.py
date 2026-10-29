"""
src/config.py
─────────────
Central configuration: feature lists, monotonic constraints, hyperparameters.
Every model reads its spec from here so nothing is hard-coded twice.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
MODEL_DIR   = ROOT / "models"
REPORT_DIR  = ROOT / "reports" / "figures"

TRAIN_SEASONS = list(range(2020, 2025))   # 2020-2024  (train)
AUDIT_SEASON  = 2025                       # held-out

# ── Win-Probability model ───────────────────────────────────────────────
WP_FEATURES = [
    "score_differential",           # posteam − defteam
    "game_seconds_remaining",
    "half_seconds_remaining",
    "yardline_100",                 # yards from opponent end-zone
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "receive_2h_ko",
    "is_home",
    "spread_line",                  # Vegas spread (negative = posteam favored)
]

# +1 = prediction must increase with feature, −1 = decrease, 0 = free
WP_MONOTONIC = [
     1,   # score_differential ↑  →  WP ↑
     0,   # game_seconds_remaining (complex)
     0,   # half_seconds_remaining (complex)
    -1,   # yardline_100 ↑  →  farther from scoring  →  WP ↓
    -1,   # down ↑  →  harder situation  →  WP ↓
    -1,   # ydstogo ↑  →  harder  →  WP ↓
     1,   # posteam_timeouts ↑  →  WP ↑
    -1,   # defteam_timeouts ↑  →  WP ↓
     0,   # receive_2h_ko (binary flag)
     0,   # is_home (binary flag)
     0,   # spread_line (could constrain −1 but Vegas line already has sign info)
]

WP_PARAMS = dict(
    max_iter=600,
    max_depth=6,
    min_samples_leaf=250,
    learning_rate=0.05,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=30,
    random_state=42,
)

# ── Fourth-down conversion model ────────────────────────────────────────
CONV_FEATURES = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "is_goal_to_go",
]

CONV_MONOTONIC = [
    -1,   # ydstogo ↑  →  harder  →  P(convert) ↓
     0,   # yardline_100
     0,   # score_differential
     0,   # game_seconds_remaining
     0,   # is_goal_to_go (binary)
]

CONV_PARAMS = dict(
    max_iter=400,
    max_depth=4,
    min_samples_leaf=80,
    learning_rate=0.05,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42,
)

# ── Field-goal make model ───────────────────────────────────────────────
FG_FEATURES = [
    "kick_distance",
    "is_dome",
]

FG_MONOTONIC = [
    -1,   # kick_distance ↑  →  P(make) ↓
     0,   # is_dome (binary)
]

FG_PARAMS = dict(
    max_iter=300,
    max_depth=3,
    min_samples_leaf=50,
    learning_rate=0.05,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)

# ── Punt expected field-position model (regression) ─────────────────────
PUNT_FEATURES = [
    "yardline_100",
]

PUNT_MONOTONIC = [
    1,    # farther from opponent EZ → opponent starts farther from ours
]

PUNT_PARAMS = dict(
    max_iter=300,
    max_depth=3,
    min_samples_leaf=60,
    learning_rate=0.05,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)

# ── Decision-engine constants ───────────────────────────────────────────
KICKOFF_YARDLINE   = 75    # average opp starting field pos after kickoff (their 25)
FG_SNAP_OFFSET     = 17    # kick_distance ≈ yardline_100 + 17
PAT_POINTS         = 7     # assume PAT made on TD
FG_POINTS          = 3
MAX_FG_DISTANCE    = 66    # longest plausible FG attempt
