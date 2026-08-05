"""
prediction_engine.py
Elite-tier live basketball prediction engine.

ELITE FEATURES ADDED:
- LiveGameFeatureExtractor: computes momentum, fatigue, foul trouble,
  garbage time, pace pressure, scoring runs from raw AllSportsAPI data
- All features feed as adjustments into existing Poisson/NegBinom/
  Bayesian models — math foundation unchanged, accuracy improved
- Features implemented (from AllSportsAPI data only, no external deps):
  * scoring_pace_current (live pace vs quarter average)
  * point_run_momentum (home/away scoring dominance in recent quarters)
  * fatigue_index (per player: minutes played / expected max minutes)
  * foul_trouble_key_player (starter in foul trouble -> scoring penalty)
  * garbage_time_indicator (large lead + little time -> pace drops)
  * spread_vs_live_pressure (live diff vs pregame expectation divergence)
  * time_score_pressure (urgency: trailing team's pressure index)
  * to_rate_live (turnovers per minute from player stats)
  * orb_rate_live (offensive rebounds from player stats)
  * three_pa_rate_live (3-point attempt rate from player stats)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nbinom, norm
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# =========================================================
# SHIN'S METHOD DE-VIG
# =========================================================
class ShinDevig:
    @staticmethod
    def _shin_equation(z, implied_probs):
        n = len(implied_probs)
        total = 0.0
        for p in implied_probs:
            inner = z**2 + 4 * (1 - z) * (p**2) / n
            total += (np.sqrt(inner) - z) / (2 * (1 - z))
        return total - 1.0

    @classmethod
    def devig(cls, decimal_odds: List[float]) -> List[float]:
        implied = [1.0 / o for o in decimal_odds]
        overround = sum(implied) - 1.0
        if overround <= 0.001:
            total = sum(implied)
            return [p / total for p in implied]
        try:
            z = brentq(cls._shin_equation, 1e-6, 0.5 - 1e-6, args=(implied,))
        except ValueError:
            total = sum(implied)
            return [p / total for p in implied]
        n = len(implied)
        fair_probs = []
        for p in implied:
            inner = z**2 + 4 * (1 - z) * (p**2) / n
            fair_p = (np.sqrt(inner) - z) / (2 * (1 - z))
            fair_probs.append(fair_p)
        total = sum(fair_probs)
        return [p / total for p in fair_probs]

    @staticmethod
    def hk_to_decimal(hk_odds: float) -> float:
        return round(hk_odds + 1.0, 4)


# =========================================================
# SHARED NEGBINOM HELPER
# =========================================================
class QuarterScoringModel:
    @staticmethod
    def _nbinom_params(mean: float, dispersion: float):
        r = dispersion
        p = r / (r + mean)
        return r, p


# =========================================================
# ELITE FEATURE EXTRACTOR
# =========================================================
class LiveGameFeatureExtractor:
    """
    Extracts elite-tier live features from raw AllSportsAPI Livescore
    event data. All features are dimensionless scalars in [0, 1] or
    signed floats, ready to be used as multipliers/penalties in the
    Poisson/NegBinom scoring models.

    DESIGN PRINCIPLE: Each feature is computed independently and
    gracefully degrades to a neutral value (0.0 or 1.0) when data
    is missing or insufficient — ensuring the core math is never
    corrupted by a bad/missing stat.
    """

    # Foul thresholds for NBA-style rules (adjust per league if needed)
    FOUL_TROUBLE_THRESHOLD = 3  # fouls at which penalty starts
    FOUL_OUT_THRESHOLD = 6      # fouls at which player is disqualified

    # Garbage time definition: lead >= this with <= this many minutes left
    GARBAGE_TIME_LEAD = 20
    GARBAGE_TIME_MINUTES = 8.0

    @staticmethod
    def _safe_int(value, default=0) -> int:
        try:
            if value in ("-", "", None):
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        try:
            if value in ("-", "", None):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def _parse_minutes(cls, mins_str) -> float:
        try:
            parts = str(mins_str).split(":")
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0

    @classmethod
    def extract_all(cls, event: Dict, match_state: Dict,
                     pregame_elo_diff: float = 0.0) -> Dict:
        """
        Main entry point. Returns a dict of all elite features, each
        with a brief description and its computed value.
        """
        player_stats_home = event.get("player_statistics", {}).get("home_team", [])
        player_stats_away = event.get("player_statistics", {}).get("away_team", [])
        all_players = [("home", p) for p in player_stats_home] + \
                      [("away", p) for p in player_stats_away]

        features = {}
        features.update(cls._scoring_pace_features(match_state))
        features.update(cls._momentum_features(match_state))
        features.update(cls._garbage_time_feature(match_state))
        features.update(cls._time_score_pressure(match_state))
        features.update(cls._spread_vs_live_pressure(match_state, pregame_elo_diff))
        features.update(cls._foul_trouble_features(all_players, match_state))
        features.update(cls._player_efficiency_features(all_players))

        return features

    @classmethod
    def _scoring_pace_features(cls, match_state: Dict) -> Dict:
        """
        scoring_pace_ratio: current quarter pace vs overall game pace.
        > 1.0 = game accelerating (more scoring recently)
        < 1.0 = game slowing down
        """
        completed = match_state.get("completed_quarters", [])
        if len(completed) < 1:
            return {"scoring_pace_ratio": 1.0, "pace_trend": 0.0}

        overall_avg = np.mean([q["home"] + q["away"] for q in completed])
        if overall_avg <= 0:
            return {"scoring_pace_ratio": 1.0, "pace_trend": 0.0}

        # Current quarter partial pace (annualized to 12 min)
        min_elapsed = max(match_state.get("minutes_elapsed_current_q", 6.0), 0.5)
        current_total = match_state.get("current_total", 0)
        prior_total = sum(q["home"] + q["away"] for q in completed)
        current_q_so_far = current_total - prior_total
        current_q_pace = (current_q_so_far / min_elapsed) * 12.0

        ratio = float(np.clip(current_q_pace / overall_avg, 0.5, 2.0))

        # Pace trend: is scoring accelerating across quarters?
        if len(completed) >= 2:
            trend = (completed[-1]["home"] + completed[-1]["away"]) - \
                    (completed[0]["home"] + completed[0]["away"])
            trend_normalized = float(np.clip(trend / max(overall_avg, 1), -1.0, 1.0))
        else:
            trend_normalized = 0.0

        return {"scoring_pace_ratio": ratio, "pace_trend": trend_normalized}

    @classmethod
    def _momentum_features(cls, match_state: Dict) -> Dict:
        """
        point_run_home / point_run_away: measures if one team has been
        dominant across recent quarters (a "run"). Range [0, 1].
        High home run → home team on fire → adjust home win prob upward.
        """
        completed = match_state.get("completed_quarters", [])
        if len(completed) < 1:
            return {"momentum_home": 0.0, "momentum_away": 0.0}

        # Use last 2 quarters if available (shorter window = more live)
        recent = completed[-2:] if len(completed) >= 2 else completed

        home_recent = sum(q["home"] for q in recent)
        away_recent = sum(q["away"] for q in recent)
        total_recent = home_recent + away_recent

        if total_recent == 0:
            return {"momentum_home": 0.0, "momentum_away": 0.0}

        momentum_home = float(np.clip((home_recent - away_recent) / total_recent, -1.0, 1.0))
        momentum_away = -momentum_home  # mirror

        return {"momentum_home": momentum_home, "momentum_away": momentum_away}

    @classmethod
    def _garbage_time_feature(cls, match_state: Dict) -> Dict:
        """
        garbage_time: 1.0 = blowout (game effectively over, starters rest,
        pace drops dramatically). 0.0 = competitive game.
        This is the single most important pace-correction feature in
        live basketball models.
        """
        score_diff = abs(match_state.get("score_diff", 0))
        minutes_remaining = match_state.get("minutes_remaining", 48.0)

        if (score_diff >= cls.GARBAGE_TIME_LEAD and
                minutes_remaining <= cls.GARBAGE_TIME_MINUTES):
            # Scale: larger lead + less time = deeper garbage time
            lead_factor = min((score_diff - cls.GARBAGE_TIME_LEAD) / 20.0, 1.0)
            time_factor = 1.0 - (minutes_remaining / cls.GARBAGE_TIME_MINUTES)
            garbage = float(np.clip(0.5 + 0.5 * lead_factor + 0.5 * time_factor, 0.0, 1.0))
        elif score_diff >= cls.GARBAGE_TIME_LEAD * 1.5:
            # Very large lead even with time remaining (e.g. up 30 in Q3)
            garbage = float(np.clip((score_diff - cls.GARBAGE_TIME_LEAD) / 40.0, 0.0, 0.6))
        else:
            garbage = 0.0

        return {"garbage_time": garbage}

    @classmethod
    def _time_score_pressure(cls, match_state: Dict) -> Dict:
        """
        time_score_pressure: urgency index for the trailing team.
        High value → trailing team must score faster → pace increases.
        Used to adjust pace upward in close, late-game situations.
        Range [0, 1].
        """
        score_diff = match_state.get("score_diff", 0)  # positive = home leads
        minutes_remaining = max(match_state.get("minutes_remaining", 48.0), 0.1)

        # Points per minute needed to catch up (rough approximation)
        if abs(score_diff) == 0:
            pressure = 0.0
        else:
            points_needed = abs(score_diff)
            # Pressure scales with points_needed / minutes_remaining
            raw_pressure = points_needed / (minutes_remaining * 2.0)
            pressure = float(np.clip(raw_pressure, 0.0, 1.0))

        # Direction: positive = home needs to press, negative = away needs to press
        direction = -1.0 if score_diff > 0 else (1.0 if score_diff < 0 else 0.0)

        return {
            "time_score_pressure": pressure,
            "pressure_direction": direction,  # which team is pressing
        }

    @classmethod
    def _spread_vs_live_pressure(cls, match_state: Dict, pregame_elo_diff: float) -> Dict:
        """
        spread_divergence: how much the live score diverges from the
        pregame expectation. High divergence = underdog overperforming
        or favorite underperforming → regression to mean likely.
        Range [-1, 1]: positive = home outperforming pregame expectation.
        """
        if pregame_elo_diff == 0.0:
            return {"spread_divergence": 0.0}

        # Pregame expected margin (from Elo diff)
        expected_margin = pregame_elo_diff * 0.04
        actual_diff = match_state.get("score_diff", 0)
        minutes_remaining = match_state.get("minutes_remaining", 48.0)
        time_elapsed_fraction = 1.0 - (minutes_remaining / 48.0)

        if time_elapsed_fraction < 0.1:
            return {"spread_divergence": 0.0}  # too early to assess

        # Expected diff at this point in time (linear extrapolation of pregame)
        expected_now = expected_margin * time_elapsed_fraction
        divergence = actual_diff - expected_now

        # Normalize to [-1, 1]
        normalized = float(np.clip(divergence / 20.0, -1.0, 1.0))
        return {"spread_divergence": normalized}

    @classmethod
    def _foul_trouble_features(cls, all_players: List[Tuple], match_state: Dict) -> Dict:
        """
        foul_trouble_home / foul_trouble_away: fraction of team scoring
        capacity at risk due to foul trouble. Range [0, 1].
        0.0 = no foul trouble; 1.0 = entire team in foul trouble.

        fatigue_index_home / fatigue_index_away: average minutes played
        by starters relative to expected (proxy for fatigue).
        """
        home_fouls, home_pts, home_mins = 0.0, 0.0, 0.0
        away_fouls, away_pts, away_mins = 0.0, 0.0, 0.0
        home_count, away_count = 0, 0

        quarters_completed = match_state.get("quarters_completed", 0)
        expected_mins = max(quarters_completed * 12.0, 1.0)

        for team_side, p in all_players:
            fouls = cls._safe_int(p.get("player_personal_fouls") or p.get("player_fouls"))
            pts = cls._safe_int(p.get("player_points"))
            mins = cls._parse_minutes(p.get("player_minutes", "0:00"))

            if team_side == "home":
                home_fouls += max(0, fouls - cls.FOUL_TROUBLE_THRESHOLD) * pts
                home_pts += pts
                home_mins += mins
                home_count += 1
            else:
                away_fouls += max(0, fouls - cls.FOUL_TROUBLE_THRESHOLD) * pts
                away_pts += pts
                away_mins += mins
                away_count += 1

        # Foul trouble: weighted penalty (more fouls on high scorers = worse)
        home_foul_penalty = float(np.clip(home_fouls / max(home_pts, 1), 0.0, 0.5))
        away_foul_penalty = float(np.clip(away_fouls / max(away_pts, 1), 0.0, 0.5))

        # Fatigue: average mins relative to expected
        home_fatigue = float(np.clip((home_mins / max(home_count, 1)) / expected_mins - 1.0, 0.0, 0.5))
        away_fatigue = float(np.clip((away_mins / max(away_count, 1)) / expected_mins - 1.0, 0.0, 0.5))

        return {
            "foul_trouble_home": home_foul_penalty,
            "foul_trouble_away": away_foul_penalty,
            "fatigue_home": home_fatigue,
            "fatigue_away": away_fatigue,
        }

    @classmethod
    def _player_efficiency_features(cls, all_players: List[Tuple]) -> Dict:
        """
        to_rate_live: combined turnovers per minute (both teams).
        High TO rate → possession changes often → pace faster but scoring uncertain.

        orb_rate_live: offensive rebounds / total rebounds.
        High ORB% → more second-chance points → scoring rate elevated.

        three_pa_rate_live: 3-point attempts / total field goal attempts.
        High 3PA rate → higher variance, higher scoring potential.
        """
        total_turnovers = 0
        total_rebounds = 0
        total_off_rebounds = 0
        total_fga = 0
        total_3pa = 0
        total_mins = 0.0

        for _, p in all_players:
            total_turnovers += cls._safe_int(p.get("player_turnovers"))
            total_rebounds += cls._safe_int(p.get("player_rebounds"))
            total_off_rebounds += cls._safe_int(p.get("player_offensive_rebounds") or
                                                  p.get("player_oreb"))
            total_fga += cls._safe_int(p.get("player_field_goals_att") or
                                        p.get("player_fga"))
            total_3pa += cls._safe_int(p.get("player_threepoint_goals_att") or
                                        p.get("player_3pa"))
            total_mins += cls._parse_minutes(p.get("player_minutes", "0:00"))

        to_rate = float(np.clip(total_turnovers / max(total_mins / 10, 1), 0.0, 1.0))
        orb_rate = float(np.clip(total_off_rebounds / max(total_rebounds, 1), 0.0, 0.6))
        three_pa_rate = float(np.clip(total_3pa / max(total_fga, 1), 0.0, 1.0))

        return {
            "to_rate_live": to_rate,
            "orb_rate_live": orb_rate,
            "three_pa_rate_live": three_pa_rate,
        }


# =========================================================
# ELITE FEATURE → MODEL ADJUSTMENT TRANSLATOR
# =========================================================
class EliteFeatureAdjuster:
    """
    Translates raw elite feature values into concrete parameter
    adjustments for the Poisson/NegBinom/Bayesian models.

    Design: each adjustment is bounded and additive/multiplicative,
    so a single bad feature can never catastrophically distort the
    prediction — it can only nudge it within a reasonable range.
    """

    @staticmethod
    def combined_pace_multiplier(features: Dict) -> float:
        """
        Adjusts the expected scoring pace up or down based on game state.
        Returns a multiplier (1.0 = no change, >1.0 = faster, <1.0 = slower).
        """
        multiplier = 1.0

        # Garbage time: reduces pace significantly
        garbage = features.get("garbage_time", 0.0)
        multiplier *= (1.0 - 0.35 * garbage)  # up to -35% pace in full garbage time

        # Scoring pace ratio: if current quarter is faster/slower than average
        pace_ratio = features.get("scoring_pace_ratio", 1.0)
        # Regress 30% toward the ratio (don't fully trust one partial quarter)
        multiplier *= (1.0 + 0.3 * (pace_ratio - 1.0))

        # Pace trend: scoring increasing over game → slight upward adjustment
        trend = features.get("pace_trend", 0.0)
        multiplier *= (1.0 + 0.1 * trend)

        # Time-score pressure: trailing team presses → pace increases
        pressure = features.get("time_score_pressure", 0.0)
        multiplier *= (1.0 + 0.1 * pressure)

        # High TO rate → more possessions but not always more scoring
        to_rate = features.get("to_rate_live", 0.0)
        multiplier *= (1.0 - 0.05 * to_rate)  # slight negative (TOs kill scoring)

        # High ORB% → more second chances → slightly more scoring
        orb = features.get("orb_rate_live", 0.0)
        multiplier *= (1.0 + 0.08 * orb)

        return float(np.clip(multiplier, 0.4, 1.8))

    @staticmethod
    def home_win_prob_adjustment(features: Dict, minutes_remaining: float) -> float:
        """
        Returns an additive adjustment to home win probability (in log-odds
        space, then converted back) based on momentum and pressure features.
        Range: roughly [-0.05, +0.05] — a nudge, not a takeover.
        """
        adjustment = 0.0

        # Momentum: recent quarter dominance
        momentum_home = features.get("momentum_home", 0.0)
        # Momentum matters more late in the game
        time_weight = 1.0 - (minutes_remaining / 48.0)
        adjustment += 0.06 * momentum_home * time_weight

        # Spread divergence: underdog overperforming → regression to mean
        # (away team more likely to regress if they're way ahead of expectation)
        divergence = features.get("spread_divergence", 0.0)
        # Negative feedback: if home is over-performing (divergence > 0),
        # expect slight regression (reduce home advantage adjustment)
        adjustment -= 0.03 * divergence

        return float(np.clip(adjustment, -0.07, 0.07))

    @staticmethod
    def player_scoring_multipliers(features: Dict, team_side: str) -> Tuple[float, float]:
        """
        Returns (foul_trouble_penalty, garbage_time_penalty) for a specific
        team's players — used directly in PlayerScoringModel.
        """
        key = f"foul_trouble_{team_side}"
        foul_penalty = float(np.clip(features.get(key, 0.0), 0.0, 0.4))
        garbage_penalty = float(np.clip(features.get("garbage_time", 0.0) * 0.5, 0.0, 0.3))
        return foul_penalty, garbage_penalty


# =========================================================
# LIVE-ADAPTIVE GAME TOTAL MODEL
# =========================================================
class LiveAdaptivePaceModel:
    NEUTRAL_QUARTER_TOTAL = 52.0

    def __init__(self):
        self.quarter_totals: List[float] = []

    def add_completed_quarter(self, home_pts: int, away_pts: int):
        self.quarter_totals.append(home_pts + away_pts)

    def current_pace_per_minute(self, pace_multiplier: float = 1.0) -> float:
        if not self.quarter_totals:
            base = self.NEUTRAL_QUARTER_TOTAL / 12.0
        else:
            n = len(self.quarter_totals)
            weights = np.linspace(0.7, 1.3, n)
            weighted_avg = np.average(self.quarter_totals, weights=weights)
            base = weighted_avg / 12.0
        return base * pace_multiplier

    def pace_confidence(self) -> float:
        n = len(self.quarter_totals)
        if n == 0: return 0.15
        elif n == 1: return 0.45
        elif n == 2: return 0.70
        elif n == 3: return 0.90
        return 0.95


class LiveOnlyGameTotalModel:
    def __init__(self, dispersion: float = 25.0):
        self.dispersion = dispersion
        self.pace_model = LiveAdaptivePaceModel()

    def ingest_quarter(self, home_pts: int, away_pts: int):
        self.pace_model.add_completed_quarter(home_pts, away_pts)

    def over_under_prob(self, current_total: int, quarters_completed: int,
                         minutes_elapsed_current_q: float, line: float,
                         pace_multiplier: float = 1.0) -> Dict:
        per_minute_rate = self.pace_model.current_pace_per_minute(pace_multiplier)
        confidence = self.pace_model.pace_confidence()

        minutes_played_total = quarters_completed * 12.0 + minutes_elapsed_current_q
        minutes_remaining = max(48.0 - minutes_played_total, 0.0)

        expected_remaining = max(per_minute_rate * minutes_remaining, 0.5)
        adjusted_dispersion = self.dispersion * confidence if confidence > 0 else self.dispersion * 0.2

        r, p = QuarterScoringModel._nbinom_params(expected_remaining, max(adjusted_dispersion, 1.0))
        dist = nbinom(r, p)

        points_needed = line - current_total
        if points_needed < 0:
            return {"p_over": 1.0, "p_under": 0.0, "already_settled": True}

        threshold = int(np.floor(points_needed))
        p_under = dist.cdf(threshold)
        p_over = 1.0 - p_under

        return {
            "p_over": round(float(p_over), 4),
            "p_under": round(float(1 - p_over), 4),
            "already_settled": False,
            "expected_final_total": round(float(current_total + dist.mean()), 1),
            "std_dev": round(float(dist.std()), 2),
            "pace_confidence": round(confidence, 2),
            "observed_pace_per_min": round(per_minute_rate, 3),
        }


# =========================================================
# BAYESIAN LIVE WIN PROBABILITY (enhanced with elite features)
# =========================================================
class LiveWinProbabilityModel:
    def __init__(self, pregame_elo_diff: float, elo_to_points_scale: float = 0.04):
        self.pregame_elo_diff = pregame_elo_diff
        self.expected_pregame_margin = pregame_elo_diff * elo_to_points_scale

    @staticmethod
    def _volatility_per_minute(minutes_remaining: float) -> float:
        return 0.9 * np.sqrt(max(minutes_remaining, 0.01))

    def win_probability(self, current_score_diff: int, minutes_remaining: float,
                         elite_adjustment: float = 0.0) -> Dict:
        if minutes_remaining <= 0:
            p_home = 1.0 if current_score_diff > 0 else (0.0 if current_score_diff < 0 else 0.5)
            return {"p_home_win": p_home, "p_away_win": 1 - p_home}

        time_fraction_remaining = minutes_remaining / 48.0
        drift_weight = time_fraction_remaining
        expected_remaining_margin_shift = drift_weight * (
            self.expected_pregame_margin - current_score_diff * (1 - time_fraction_remaining)
        )

        projected_final_diff = current_score_diff + expected_remaining_margin_shift
        vol = self._volatility_per_minute(minutes_remaining)

        p_home_win = 1 - norm.cdf(0, loc=projected_final_diff, scale=vol)

        # Apply elite feature adjustment (bounded nudge)
        p_home_win = float(np.clip(p_home_win + elite_adjustment, 0.001, 0.999))

        return {
            "p_home_win": round(p_home_win, 4),
            "p_away_win": round(1 - p_home_win, 4),
            "projected_final_margin": round(float(projected_final_diff), 2),
        }


# =========================================================
# PLAYER SCORING MODEL (enhanced with foul/garbage features)
# =========================================================
class PlayerScoringModel:
    NEUTRAL_PTS_PER_MIN = 0.45
    FULL_TRUST_MINUTES = 30.0
    MAX_PLAUSIBLE_RATE = 0.75

    def __init__(self, dispersion: float = 6.0):
        self.dispersion = dispersion

    def _effective_rate(self, points_so_far: int, minutes_played: float) -> float:
        if minutes_played <= 0:
            return self.NEUTRAL_PTS_PER_MIN
        observed_rate = min(points_so_far / minutes_played, self.MAX_PLAUSIBLE_RATE)
        weight_observed = minutes_played / (minutes_played + self.FULL_TRUST_MINUTES)
        return (observed_rate * weight_observed) + (self.NEUTRAL_PTS_PER_MIN * (1 - weight_observed))

    def milestone_over_under_prob(self, points_so_far: int, minutes_played: float,
                                   minutes_remaining_in_game: float, line: float,
                                   foul_trouble_penalty: float = 0.0,
                                   garbage_time_penalty: float = 0.0) -> Dict:
        rate = self._effective_rate(points_so_far, minutes_played)
        rate *= (1.0 - foul_trouble_penalty) * (1.0 - garbage_time_penalty)
        expected_remaining = max(rate * minutes_remaining_in_game, 0.05)
        r, p = QuarterScoringModel._nbinom_params(expected_remaining, self.dispersion)
        dist = nbinom(r, p)

        points_needed = line - points_so_far
        if points_needed < 0:
            return {"p_over": 1.0, "p_under": 0.0, "already_settled": True}
        threshold = int(np.floor(points_needed))
        p_over = 1.0 - dist.cdf(threshold)
        return {
            "p_over": round(float(p_over), 4), "p_under": round(float(1 - p_over), 4),
            "already_settled": False,
            "expected_final_points": round(float(points_so_far + dist.mean()), 2),
            "std_dev": round(float(dist.std()), 2),
        }


# =========================================================
# ENSEMBLE BLENDING
# =========================================================
class EnsembleBlender:
    def __init__(self):
        self.source_weights = {
            "market_devigged": 0.45,
            "statistical_model": 0.35,
            "player_adjusted": 0.20,
        }

    def blend(self, probabilities: Dict[str, float]) -> float:
        available = {k: v for k, v in probabilities.items() if k in self.source_weights}
        total_weight = sum(self.source_weights[k] for k in available)
        if total_weight == 0:
            raise ValueError("No valid probability sources provided")
        blended = sum(
            probabilities[k] * (self.source_weights[k] / total_weight)
            for k in available
        )
        return round(blended, 4)


# =========================================================
# KELLY CRITERION STAKING
# =========================================================
class KellyStaking:
    def __init__(self, kelly_fraction: float = 0.25, max_stake_pct: float = 0.05):
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct

    def calculate_stake(self, fair_probability: float, decimal_odds: float,
                         bankroll: float) -> Dict:
        b = decimal_odds - 1.0
        q = 1 - fair_probability
        full_kelly_fraction = (b * fair_probability - q) / b if b > 0 else 0.0

        if full_kelly_fraction <= 0:
            return {"recommended_stake": 0.0, "edge": round(fair_probability - (1 / decimal_odds), 4),
                    "reason": "No positive edge — do not bet"}

        applied_fraction = min(full_kelly_fraction * self.kelly_fraction, self.max_stake_pct)
        stake = round(bankroll * applied_fraction, 2)
        edge = fair_probability - (1 / decimal_odds)

        return {
            "recommended_stake": stake,
            "stake_pct_of_bankroll": round(applied_fraction * 100, 2),
            "full_kelly_pct": round(full_kelly_fraction * 100, 2),
            "edge": round(edge, 4),
            "expected_value_per_unit": round((fair_probability * b) - q, 4),
        }


# =========================================================
# TEAM STRENGTH ESTIMATOR
# =========================================================
class SimpleTeamStrengthEstimator:
    @staticmethod
    def estimate_elo_diff_from_records(home_wins: int, home_losses: int,
                                        away_wins: int, away_losses: int,
                                        elo_scale: float = 400.0) -> float:
        home_win_pct = np.clip(home_wins / max(home_wins + home_losses, 1), 0.05, 0.95)
        away_win_pct = np.clip(away_wins / max(away_wins + away_losses, 1), 0.05, 0.95)
        home_logit = np.log(home_win_pct / (1 - home_win_pct))
        away_logit = np.log(away_win_pct / (1 - away_win_pct))
        return (home_logit - away_logit) * elo_scale / 4.0


def sanity_check_disagreement(model_prob: float, market_prob: float,
                               max_allowed_diff: float = 0.25) -> bool:
    return abs(model_prob - market_prob) <= max_allowed_diff


# =========================================================
# ORCHESTRATOR: FULL-GAME HOME/AWAY (with elite features)
# =========================================================
class HomeAwayOrchestrator:
    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.03,
                 pregame_elo_diff: float = 0.0,
                 max_model_market_disagreement: float = 0.25):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_disagreement = max_model_market_disagreement
        self.wp_model = LiveWinProbabilityModel(pregame_elo_diff=pregame_elo_diff)
        self.ensemble = EnsembleBlender()
        self.kelly = KellyStaking()

    def evaluate_full_game(self, match_state: Dict, odds: Dict,
                            elite_features: Optional[Dict] = None) -> Optional[Dict]:
        features = elite_features or {}

        # Elite feature adjustment to win probability
        elite_adj = EliteFeatureAdjuster.home_win_prob_adjustment(
            features, match_state.get("minutes_remaining", 24.0)
        ) if features else 0.0

        wp = self.wp_model.win_probability(
            current_score_diff=match_state["score_diff"],
            minutes_remaining=match_state["minutes_remaining"],
            elite_adjustment=elite_adj,
        )

        fair_probs = ShinDevig.devig([odds["home_decimal"], odds["away_decimal"]])
        market_p_home = fair_probs[0]

        if not sanity_check_disagreement(wp["p_home_win"], market_p_home, self.max_disagreement):
            return None

        blended_p_home = self.ensemble.blend({
            "statistical_model": wp["p_home_win"],
            "market_devigged": market_p_home,
        })
        return self._check_both_sides(match_state, odds, blended_p_home, "Full Game Home/Away")

    def _check_both_sides(self, match_state: Dict, odds: Dict, blended_p_home: float,
                           market_label: str) -> Optional[Dict]:
        blended_p_away = 1 - blended_p_home
        home_implied = 1 / odds["home_decimal"]
        away_implied = 1 / odds["away_decimal"]
        home_edge = blended_p_home - home_implied
        away_edge = blended_p_away - away_implied

        best_side, best_edge, best_odds, best_prob = None, 0.0, 0.0, 0.0
        if home_edge > self.ev_alert_threshold and home_edge > away_edge:
            best_side, best_edge, best_odds, best_prob = "Home", home_edge, odds["home_decimal"], blended_p_home
        elif away_edge > self.ev_alert_threshold:
            best_side, best_edge, best_odds, best_prob = "Away", away_edge, odds["away_decimal"], blended_p_away

        if best_side is None:
            return None

        stake_info = self.kelly.calculate_stake(best_prob, best_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None

        return {
            "match_id": match_state["match_id"], "market": market_label,
            "side": best_side, "blended_probability": round(best_prob, 4),
            "offered_odds": best_odds, "edge_pct": round(best_edge * 100, 2),
            "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 VALUE BET DETECTED",
        }


# =========================================================
# ORCHESTRATOR: QUARTER HOME/AWAY (with elite features)
# =========================================================
class QuarterHomeAwayOrchestrator:
    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.03,
                 max_model_market_disagreement: float = 0.30,
                 volatility_constant: float = 2.2):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_disagreement = max_model_market_disagreement
        self.volatility_constant = volatility_constant
        self.ensemble = EnsembleBlender()
        self.kelly = KellyStaking()

    def evaluate_current_quarter(self, quarter_score_diff: int, minutes_elapsed_in_q: float,
                                  odds: Dict, quarter_number: int,
                                  elite_features: Optional[Dict] = None) -> Optional[Dict]:
        features = elite_features or {}
        minutes_remaining_in_q = max(12.0 - minutes_elapsed_in_q, 0.01)

        rate_per_min = quarter_score_diff / minutes_elapsed_in_q if minutes_elapsed_in_q > 0 else 0.0
        regressed_rate = rate_per_min * 0.5
        projected_final_q_diff = quarter_score_diff + (regressed_rate * minutes_remaining_in_q)
        vol = self.volatility_constant * np.sqrt(minutes_remaining_in_q)

        # Apply momentum nudge to within-quarter projection
        momentum_home = features.get("momentum_home", 0.0)
        projected_final_q_diff += momentum_home * 1.5  # small nudge from recent momentum

        p_home_wins_quarter = float(np.clip(
            1 - norm.cdf(0, loc=projected_final_q_diff, scale=vol), 0.02, 0.98
        ))

        fair_probs = ShinDevig.devig([odds["home_decimal"], odds["away_decimal"]])
        market_p_home = fair_probs[0]

        if abs(p_home_wins_quarter - market_p_home) > self.max_disagreement:
            return None

        blended_p_home = self.ensemble.blend({
            "statistical_model": p_home_wins_quarter, "market_devigged": market_p_home,
        })
        blended_p_away = 1 - blended_p_home

        home_implied = 1 / odds["home_decimal"]
        away_implied = 1 / odds["away_decimal"]
        home_edge = blended_p_home - home_implied
        away_edge = blended_p_away - away_implied

        best_side, best_edge, best_odds, best_prob = None, 0.0, 0.0, 0.0
        if home_edge > self.ev_alert_threshold and home_edge > away_edge:
            best_side, best_edge, best_odds, best_prob = "Home", home_edge, odds["home_decimal"], blended_p_home
        elif away_edge > self.ev_alert_threshold:
            best_side, best_edge, best_odds, best_prob = "Away", away_edge, odds["away_decimal"], blended_p_away

        if best_side is None:
            return None

        stake_info = self.kelly.calculate_stake(best_prob, best_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None

        return {
            "market": f"Home/Away - Quarter {quarter_number}", "side": best_side,
            "blended_probability": round(best_prob, 4), "offered_odds": best_odds,
            "edge_pct": round(best_edge * 100, 2),
            "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 VALUE BET DETECTED",
        }


# =========================================================
# ORCHESTRATOR: PLAYER PROPS (with foul/garbage features)
# =========================================================
class PlayerPropsOrchestrator:
    MAX_PLAUSIBLE_EDGE = 0.25

    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.04,
                 max_relative_std: float = 0.35):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_relative_std = max_relative_std
        self.model = PlayerScoringModel()
        self.kelly = KellyStaking()

    def evaluate_milestone(self, player_name: str, points_so_far: int, minutes_played: float,
                            minutes_remaining_in_game: float, threshold_odds: Dict,
                            team_side: str = "home",
                            elite_features: Optional[Dict] = None) -> Optional[Dict]:
        features = elite_features or {}
        foul_penalty, garbage_penalty = EliteFeatureAdjuster.player_scoring_multipliers(
            features, team_side
        )

        result = self.model.milestone_over_under_prob(
            points_so_far=points_so_far, minutes_played=minutes_played,
            minutes_remaining_in_game=minutes_remaining_in_game,
            line=threshold_odds["threshold"],
            foul_trouble_penalty=foul_penalty,
            garbage_time_penalty=garbage_penalty,
        )

        if result.get("already_settled"):
            return None

        relative_std = result["std_dev"] / max(result["expected_final_points"], 1)
        if relative_std > self.max_relative_std:
            return None

        offered_odds = threshold_odds["over_decimal"]
        implied_prob = 1 / offered_odds
        edge = result["p_over"] - implied_prob

        if edge < self.ev_alert_threshold or edge > self.MAX_PLAUSIBLE_EDGE:
            return None

        stake_info = self.kelly.calculate_stake(result["p_over"], offered_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None

        return {
            "player": player_name, "market": f"Over {threshold_odds['threshold']} points",
            "model_probability": result["p_over"], "offered_odds": offered_odds,
            "edge_pct": round(edge * 100, 2), "expected_final_points": result["expected_final_points"],
            "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 PLAYER PROP VALUE DETECTED",
        }


# =========================================================
# ORCHESTRATOR: ODD/EVEN (with pace multiplier)
# =========================================================
class OddEvenOrchestrator:
    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.04):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.kelly = KellyStaking()

    def evaluate(self, live_total_model: LiveOnlyGameTotalModel, current_total: int,
                 quarters_completed: int, minutes_elapsed_current_q: float,
                 odds: Dict, pace_multiplier: float = 1.0) -> Optional[Dict]:
        per_minute_rate = live_total_model.pace_model.current_pace_per_minute(pace_multiplier)
        confidence = live_total_model.pace_model.pace_confidence()
        minutes_played_total = quarters_completed * 12.0 + minutes_elapsed_current_q
        minutes_remaining = max(48.0 - minutes_played_total, 0.0)
        expected_remaining = max(per_minute_rate * minutes_remaining, 0.5)
        adjusted_dispersion = max(live_total_model.dispersion * confidence, 1.0)
        r, p = QuarterScoringModel._nbinom_params(expected_remaining, adjusted_dispersion)
        dist = nbinom(r, p)

        max_k = int(dist.mean() + 6 * dist.std()) + 1
        p_odd, p_even = 0.0, 0.0
        for k in range(0, max_k + 1):
            p_k = dist.pmf(k)
            if (current_total + k) % 2 == 0:
                p_even += p_k
            else:
                p_odd += p_k
        total_p = p_odd + p_even
        if total_p <= 0:
            return None
        p_odd, p_even = p_odd / total_p, p_even / total_p

        fair_probs = ShinDevig.devig([odds["odd_decimal"], odds["even_decimal"]])
        blended_p_odd = 0.5 * p_odd + 0.5 * fair_probs[0]
        blended_p_even = 1 - blended_p_odd

        odd_edge = blended_p_odd - (1 / odds["odd_decimal"])
        even_edge = blended_p_even - (1 / odds["even_decimal"])

        side, edge, side_odds, prob = None, 0.0, 0.0, 0.0
        if odd_edge > self.ev_alert_threshold and odd_edge > even_edge:
            side, edge, side_odds, prob = "Odd", odd_edge, odds["odd_decimal"], blended_p_odd
        elif even_edge > self.ev_alert_threshold:
            side, edge, side_odds, prob = "Even", even_edge, odds["even_decimal"], blended_p_even

        if side is None:
            return None
        stake_info = self.kelly.calculate_stake(prob, side_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None
        return {
            "market": "Odd/Even Total", "side": side, "blended_probability": round(prob, 4),
            "offered_odds": side_odds, "edge_pct": round(edge * 100, 2),
            "recommended_stake": stake_info["recommended_stake"], "alert": "🔥 VALUE BET DETECTED",
        }


# =========================================================
# ORCHESTRATOR: HIGHEST SCORING QUARTER
# =========================================================
class HighestScoringQuarterOrchestrator:
    NEUTRAL_QUARTER_PRIORS = {1: 0.23, 2: 0.26, 3: 0.24, 4: 0.27}

    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.05):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.kelly = KellyStaking()

    def evaluate(self, completed_quarter_totals: Dict[int, int],
                 odds_by_quarter: Dict[int, float]) -> Optional[Dict]:
        remaining = [q for q in [1, 2, 3, 4] if q not in completed_quarter_totals]
        if not remaining:
            return None

        current_leader = max(completed_quarter_totals, key=completed_quarter_totals.get) \
            if completed_quarter_totals else None

        remaining_mass = sum(self.NEUTRAL_QUARTER_PRIORS[q] for q in remaining)
        model_probs = {q: self.NEUTRAL_QUARTER_PRIORS[q] / remaining_mass for q in remaining}
        if current_leader is not None:
            model_probs[current_leader] = model_probs.get(current_leader, 0.0) + 0.05
        total_p = sum(model_probs.values())
        model_probs = {q: p / total_p for q, p in model_probs.items()}

        best_q, best_edge, best_odds, best_prob = None, 0.0, 0.0, 0.0
        for q, dec_odds in odds_by_quarter.items():
            if q not in model_probs:
                continue
            edge = model_probs[q] - (1 / dec_odds)
            if edge > self.ev_alert_threshold and edge > best_edge:
                best_q, best_edge, best_odds, best_prob = q, edge, dec_odds, model_probs[q]

        if best_q is None:
            return None
        stake_info = self.kelly.calculate_stake(best_prob, best_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None
        return {
            "market": "Highest Scoring Quarter", "side": f"Q{best_q}",
            "blended_probability": round(best_prob, 4), "offered_odds": best_odds,
            "edge_pct": round(best_edge * 100, 2),
            "recommended_stake": stake_info["recommended_stake"], "alert": "🔥 VALUE BET DETECTED",
        }
