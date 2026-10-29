"""
src/engine.py
─────────────
NFL 4th-down decision engine.
Evaluates the three possible decisions on 4th down:
  1. Go for it (convert vs turnover on downs / touchdown)
  2. Field goal attempt (make vs miss)
  3. Punt (expected opponent starting field position)

Computes Win Probability (WP) for each branch, break-even conversion rates,
and recommends the expected-WP-maximizing choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
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


class FourthDownEngine:
    """Core evaluation engine for NFL 4th-down decisions."""

    def __init__(self, models: dict):
        """
        Store the trained models dictionary.

        Expected keys in models:
          - 'wp':   HistGradientBoostingClassifier predicting P(posteam wins)
          - 'conv': HistGradientBoostingClassifier predicting P(4th down converted)
          - 'fg':   HistGradientBoostingClassifier predicting P(FG made)
          - 'punt': HistGradientBoostingRegressor predicting expected opponent yardline_100
        """
        self.models = models

    def wp(self, state: dict) -> float:
        """
        Given a game-state dict with keys matching WP_FEATURES, return P(posteam wins).

        State dict keys:
          score_differential, game_seconds_remaining, half_seconds_remaining,
          yardline_100, down, ydstogo, posteam_timeouts_remaining,
          defteam_timeouts_remaining, receive_2h_ko, is_home, spread_line.
        """
        p_to = state.get("posteam_timeouts_remaining", state.get("posteam_timeouts", 3))
        d_to = state.get("defteam_timeouts_remaining", state.get("defteam_timeouts", 3))

        row = {
            "score_differential": float(state["score_differential"]),
            "game_seconds_remaining": float(state["game_seconds_remaining"]),
            "half_seconds_remaining": float(state["half_seconds_remaining"]),
            "yardline_100": float(state["yardline_100"]),
            "down": float(state["down"]),
            "ydstogo": float(state["ydstogo"]),
            "posteam_timeouts_remaining": float(p_to),
            "defteam_timeouts_remaining": float(d_to),
            "receive_2h_ko": float(state.get("receive_2h_ko", 0)),
            "is_home": float(state.get("is_home", 0)),
            "spread_line": float(state.get("spread_line", 0)),
        }
        df = pd.DataFrame([row], columns=WP_FEATURES).astype(np.float32)
        prob = self.models["wp"].predict_proba(df)[0, 1]
        return float(prob)

    def p_convert(self, state: dict) -> float:
        """
        Return P(converting on 4th down).

        CONV_FEATURES: ydstogo, yardline_100, score_differential,
                       game_seconds_remaining, is_goal_to_go.
        """
        is_gtg = state.get("is_goal_to_go")
        if is_gtg is None:
            is_gtg = 1.0 if float(state["yardline_100"]) <= float(state["ydstogo"]) else 0.0

        row = {
            "ydstogo": float(state["ydstogo"]),
            "yardline_100": float(state["yardline_100"]),
            "score_differential": float(state["score_differential"]),
            "game_seconds_remaining": float(state["game_seconds_remaining"]),
            "is_goal_to_go": float(is_gtg),
        }
        df = pd.DataFrame([row], columns=CONV_FEATURES).astype(np.float32)
        prob = self.models["conv"].predict_proba(df)[0, 1]
        return float(prob)

    def p_fg_make(self, state: dict) -> float:
        """
        Return P(FG made).

        FG_FEATURES: kick_distance, is_dome.
        kick_distance = yardline_100 + FG_SNAP_OFFSET.
        """
        kick_dist = float(state.get("kick_distance", state["yardline_100"] + FG_SNAP_OFFSET))
        is_dome = float(state.get("is_dome", 0))

        row = {
            "kick_distance": kick_dist,
            "is_dome": is_dome,
        }
        df = pd.DataFrame([row], columns=FG_FEATURES).astype(np.float32)
        prob = self.models["fg"].predict_proba(df)[0, 1]
        return float(prob)

    def expected_punt_opp_yardline(self, state: dict) -> float:
        """
        Return expected opponent yardline_100 (from their endzone) after a punt.
        Uses punt model with yardline_100.
        """
        row = {"yardline_100": float(state["yardline_100"])}
        df = pd.DataFrame([row], columns=PUNT_FEATURES).astype(np.float32)
        pred = float(self.models["punt"].predict(df)[0])
        return float(np.clip(pred, 1.0, 99.0))

    def _flip_state(self, state: dict, new_yardline_100: float, score_change: int = 0) -> dict:
        """
        Create a NEW state dict from the opponent's perspective after a possession change.

        Parameters
        ----------
        state : dict
            Current game state before possession flip.
        new_yardline_100 : float
            Resulting yardline_100 from the opponent's perspective (distance to scoring EZ).
        score_change : int
            Points scored on the play before possession flip (e.g. +7 for TD, +3 for FG).
        """
        p_to = state.get("posteam_timeouts_remaining", state.get("posteam_timeouts", 3))
        d_to = state.get("defteam_timeouts_remaining", state.get("defteam_timeouts", 3))

        flipped = {
            "score_differential": -(state["score_differential"] + score_change),
            "yardline_100": float(new_yardline_100),
            "down": 1,
            "ydstogo": 10,
            "game_seconds_remaining": state["game_seconds_remaining"],
            "half_seconds_remaining": state["half_seconds_remaining"],
            "posteam_timeouts_remaining": d_to,
            "defteam_timeouts_remaining": p_to,
            "posteam_timeouts": d_to,
            "defteam_timeouts": p_to,
            "receive_2h_ko": 1 - state.get("receive_2h_ko", 0),
            "is_home": 1 - state.get("is_home", 0),
            "spread_line": -state.get("spread_line", 0),
            "is_dome": state.get("is_dome", 0),
        }
        return flipped

    def evaluate(self, state: dict) -> dict:
        """
        Evaluate 4th-down decision options: Go, Field Goal, or Punt.

        Returns
        -------
        dict with keys:
            go_wp, fg_wp, punt_wp, recommendation, p_convert, p_fg_make,
            break_even_conv_rate, wp_convert, wp_fail, wp_fg_make, wp_fg_miss,
            wp_punt, margin
        """
        yl100 = float(state["yardline_100"])
        ydstogo = float(state["ydstogo"])

        # 1. GO FOR IT
        p_conv = self.p_convert(state)

        if yl100 > ydstogo:
            # Conversion succeeds: 1st down at yardline_100 - ydstogo
            converted_state = state.copy()
            new_yl = yl100 - ydstogo
            converted_state["down"] = 1
            converted_state["ydstogo"] = min(10.0, new_yl)
            converted_state["yardline_100"] = new_yl
            wp_convert = self.wp(converted_state)
        else:
            # Conversion succeeds: Touchdown! Opponent takes over after kickoff
            post_td_state = self._flip_state(
                state, new_yardline_100=KICKOFF_YARDLINE, score_change=PAT_POINTS
            )
            wp_convert = 1.0 - self.wp(post_td_state)

        # Conversion fails: Turnover on downs at current spot
        fail_opp_yl100 = 100.0 - yl100
        fail_state = self._flip_state(state, new_yardline_100=fail_opp_yl100, score_change=0)
        wp_fail = 1.0 - self.wp(fail_state)

        go_wp = p_conv * wp_convert + (1.0 - p_conv) * wp_fail

        # 2. FIELD GOAL
        kick_distance = float(state.get("kick_distance", yl100 + FG_SNAP_OFFSET))

        if kick_distance > MAX_FG_DISTANCE:
            fg_wp = None
            p_make = None
            wp_fg_make = None
            wp_fg_miss = None
        else:
            p_make = self.p_fg_make(state)
            # FG made: +3 points, opponent gets kickoff at KICKOFF_YARDLINE
            make_state = self._flip_state(
                state, new_yardline_100=KICKOFF_YARDLINE, score_change=FG_POINTS
            )
            wp_fg_make = 1.0 - self.wp(make_state)

            # FG missed: Opponent gets ball at spot = max(yardline_100, 20)
            miss_opp_yl100 = 100.0 - max(yl100, 20.0)
            miss_state = self._flip_state(
                state, new_yardline_100=miss_opp_yl100, score_change=0
            )
            wp_fg_miss = 1.0 - self.wp(miss_state)

            fg_wp = p_make * wp_fg_make + (1.0 - p_make) * wp_fg_miss

        # 3. PUNT
        opp_punt_yl100 = self.expected_punt_opp_yardline(state)
        punt_state = self._flip_state(state, new_yardline_100=opp_punt_yl100, score_change=0)
        wp_punt = 1.0 - self.wp(punt_state)
        punt_wp = wp_punt

        # 4. BREAK-EVEN conversion rate
        best_alt = max(fg_wp, punt_wp) if fg_wp is not None else punt_wp
        denom = wp_convert - wp_fail
        if abs(denom) < 1e-9:
            break_even = 0.0 if best_alt <= wp_fail else 1.0
        else:
            break_even = (best_alt - wp_fail) / denom
            break_even = float(np.clip(break_even, 0.0, 1.0))

        # 5. RECOMMENDATION & MARGIN
        options = {"go": go_wp, "punt": punt_wp}
        if fg_wp is not None:
            options["fg"] = fg_wp

        recommendation = max(options, key=options.get)
        sorted_wps = sorted(options.values(), reverse=True)
        margin = float(sorted_wps[0] - sorted_wps[1])

        return {
            "go_wp": float(go_wp),
            "fg_wp": float(fg_wp) if fg_wp is not None else None,
            "punt_wp": float(punt_wp),
            "recommendation": recommendation,
            "p_convert": float(p_conv),
            "p_fg_make": float(p_make) if p_make is not None else None,
            "break_even_conv_rate": float(break_even),
            "wp_convert": float(wp_convert),
            "wp_fail": float(wp_fail),
            "wp_fg_make": float(wp_fg_make) if wp_fg_make is not None else None,
            "wp_fg_miss": float(wp_fg_miss) if wp_fg_miss is not None else None,
            "wp_punt": float(wp_punt),
            "margin": float(margin),
        }

    def recommend_str(self, state: dict) -> str:
        """
        Return a formatted string summarizing the 4th-down decision recommendation.

        Example format:
        4th & 3 at OPP 35 | Score: +7 | Q3 5:00
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ★ GO FOR IT          WP = 72.3%
          Field Goal         WP = 69.1%   (P make = 52%)
          Punt               WP = 65.4%

        P(convert) = 58%  |  Break-even = 41%  |  Margin = +3.2 pp
        """
        res = self.evaluate(state)

        ydstogo = int(state.get("ydstogo", 10))
        yl = float(state.get("yardline_100", 50))
        if yl == 50:
            yl_str = "50"
        elif yl < 50:
            yl_str = f"OPP {int(yl)}"
        else:
            yl_str = f"OWN {int(100 - yl)}"

        diff = int(state.get("score_differential", 0))
        score_str = f"+{diff}" if diff > 0 else str(diff)

        sec = float(state.get("game_seconds_remaining", 3600))
        if sec > 2700:
            quarter, q_sec = "Q1", sec - 2700
        elif sec > 1800:
            quarter, q_sec = "Q2", sec - 1800
        elif sec > 900:
            quarter, q_sec = "Q3", sec - 900
        elif sec >= 0:
            quarter, q_sec = "Q4", sec
        else:
            quarter, q_sec = "OT", 0
        m = int(q_sec) // 60
        s = int(q_sec) % 60
        time_str = f"{quarter} {m}:{s:02d}"

        header = f"4th & {ydstogo} at {yl_str} | Score: {score_str} | {time_str}"
        divider = "━" * max(len(header), 39)

        rec = res["recommendation"]

        # Option 1: Go for it
        go_star = "★ " if rec == "go" else "  "
        go_label = "GO FOR IT" if rec == "go" else "Go for it"
        go_line = f"{go_star}{go_label:<18} WP = {res['go_wp'] * 100:.1f}%"

        # Option 2: Field Goal
        fg_star = "★ " if rec == "fg" else "  "
        fg_label = "Field Goal"
        if res["fg_wp"] is not None:
            p_fg_str = (
                f"   (P make = {res['p_fg_make'] * 100:.0f}%)"
                if res["p_fg_make"] is not None
                else ""
            )
            fg_line = f"{fg_star}{fg_label:<18} WP = {res['fg_wp'] * 100:.1f}%{p_fg_str}"
        else:
            kick_dist = yl + FG_SNAP_OFFSET
            fg_line = (
                f"{fg_star}{fg_label:<18} WP = N/A   "
                f"(Distance {kick_dist:.0f} yds > {MAX_FG_DISTANCE} max)"
            )

        # Option 3: Punt
        punt_star = "★ " if rec == "punt" else "  "
        punt_label = "Punt"
        punt_line = f"{punt_star}{punt_label:<18} WP = {res['punt_wp'] * 100:.1f}%"

        summary = (
            f"P(convert) = {res['p_convert'] * 100:.0f}%  |  "
            f"Break-even = {res['break_even_conv_rate'] * 100:.0f}%  |  "
            f"Margin = +{res['margin'] * 100:.1f} pp"
        )

        lines = [
            header,
            divider,
            go_line,
            fg_line,
            punt_line,
            "",
            summary,
        ]
        return "\n".join(lines)
