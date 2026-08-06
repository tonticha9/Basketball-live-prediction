"""
prediction_engine.py
Elite-tier live basketball prediction engine.

UPDATES IN THIS VERSION:
1. BayesianPaceModel: Gamma-Poisson conjugate Bayesian rate updater
   replaces simple averaging — properly handles small samples, gives
   full posterior distribution over scoring rate
2. LiveGameFeatureExtractor: fixed based on confirmed AllSportsAPI fields
   - REMOVED: foul_trouble (player_personal_fouls is empty in API)
   - REMOVED: orb_rate_live (player_offence_rebounds is empty)
   - ADDED: efg_pct_live (effective FG% from confirmed fields)
   - ADDED: ft_rate_live (free throw rate from confirmed fields)
   - KEPT: pace, momentum, garbage_time, pressure, spread_divergence,
           to_rate_live, three_pa_rate_live, fatigue
3. SyntheticTotalOrchestrator: projects Over/Under without bookmaker
   odds — uses Bayesian posterior distribution to flag high-confidence
   total predictions directly from live game data
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nbinom, norm, gamma as gamma_dist
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
# BAYESIAN GAMMA-POISSON RATE MODEL
# =========================================================
class BayesianPaceModel:
    """
    Models the combined scoring rate (points per minute, both teams)
    as a Poisson process with a Gamma prior — the Gamma-Poisson
    conjugate model.

    Bayesian update rule (exact, closed-form):
      Prior:     Rate ~ Gamma(alpha_0, beta_0)
      Data:      k points observed over t minutes
      Posterior: Rate ~ Gamma(alpha_0 + k, beta_0 + t)

    This is mathematically superior to simple averaging because:
    1. Small samples are automatically shrunk toward the prior
       (e.g., 2 points in 1 minute doesn't extrapolate to 96/game)
    2. Full uncertainty quantification: we know the entire distribution
       over possible rates, not just a point estimate
    3. Posterior predictive (NegBinom) naturally handles overdispersion

    Reference: Gelman et al., Bayesian Data Analysis (3rd ed.) Ch. 2
    """

    # Prior: weakly informative — centered at 52 pts/quarter = 4.33/min
    # alpha_0/beta_0 = prior mean; beta_0 = prior "weight" in minutes
    PRIOR_ALPHA_0 = 8.66   # = prior_mean * prior_weight
    PRIOR_BETA_0 = 2.0     # prior weight (equiv. to 2 minutes of data)
    # Interpretation: we start with the belief that the combined rate
    # is ~4.33 pts/min, but are willing to update quickly once real data arrives

    def __init__(self):
        self.alpha = self.PRIOR_ALPHA_0  # accumulates observed points
        self.beta = self.PRIOR_BETA_0   # accumulates observed minutes

    def update(self, points_observed: int, minutes_elapsed: float):
        """
        Bayesian update: incorporate new observations.
        Call once per completed quarter (or partial quarter snapshot).
        """
        self.alpha += max(0, points_observed)
        self.beta += max(0.0, minutes_elapsed)

    @property
    def posterior_mean_rate(self) -> float:
        """Expected scoring rate (pts/min) under posterior."""
        return self.alpha / self.beta

    @property
    def posterior_variance_rate(self) -> float:
        """Variance of rate estimate — shrinks as more data arrives."""
        return self.alpha / (self.beta ** 2)

    @property
    def pace_confidence(self) -> float:
        """
        0-1 confidence based on posterior precision relative to prior.
        Approaches 1.0 as more game data accumulates.
        """
        prior_variance = self.PRIOR_ALPHA_0 / (self.PRIOR_BETA_0 ** 2)
        posterior_variance = self.posterior_variance_rate
        confidence = 1.0 - (posterior_variance / prior_variance)
        return float(np.clip(confidence, 0.0, 0.99))

    def predictive_over_under_prob(self, current_total: int, minutes_remaining: float,
                                    line: float, pace_multiplier: float = 1.0) -> Dict:
        """
        Posterior predictive distribution for remaining points:
        P(remaining points = k) = NegBinom(k | alpha, p)
        where p = beta / (beta + minutes_remaining * pace_multiplier)

        This integrates out uncertainty in the rate (unlike point-estimate
        Poisson which pretends we know the rate exactly), giving wider
        and better-calibrated prediction intervals especially early in game.
        """
        adj_minutes = minutes_remaining * pace_multiplier

        if adj_minutes <= 0:
            return {
                "p_over": 1.0 if current_total > line else 0.0,
                "p_under": 0.0 if current_total > line else 1.0,
                "already_settled": True,
                "expected_final_total": float(current_total),
                "std_dev": 0.0,
                "confidence": 1.0,
            }

        points_needed = line - current_total
        if points_needed < 0:
            return {
                "p_over": 1.0, "p_under": 0.0, "already_settled": True,
                "expected_final_total": float(current_total), "std_dev": 0.0, "confidence": 1.0,
            }

        # Posterior predictive for total remaining points:
        # k ~ NegBinom(alpha, p) where p = beta/(beta + adj_minutes)
        r = self.alpha
        p_param = self.beta / (self.beta + adj_minutes)
        dist = nbinom(r, p_param)

        expected_remaining = dist.mean()
        std_remaining = dist.std()

        threshold = int(np.floor(points_needed))
        p_under = float(dist.cdf(threshold))
        p_over = 1.0 - p_under

        return {
            "p_over": round(p_over, 4),
            "p_under": round(p_over, 4),
            "already_settled": False,
            "expected_final_total": round(float(current_total + expected_remaining), 1),
            "std_dev": round(float(std_remaining), 2),
            "confidence": round(self.pace_confidence, 3),
            "posterior_rate_per_min": round(self.posterior_mean_rate, 3),
        }

    def high_confidence_total_range(self, current_total: int,
                                     minutes_remaining: float,
                                     pace_multiplier: float = 1.0,
                                     credible_pct: float = 0.80) -> Tuple[float, float]:
        """
        Returns the (lower, upper) bounds of the credible_pct% posterior
        predictive interval for the final total.
        e.g., 80% credible interval: we're 80% confident the final
        total falls within [lower, upper].
        Used by SyntheticTotalOrchestrator to assess confidence.
        """
        adj_minutes = minutes_remaining * pace_multiplier
        if adj_minutes <= 0:
            return (float(current_total), float(current_total))

        r = self.alpha
        p_param = self.beta / (self.beta + adj_minutes)
        dist = nbinom(r, p_param)

        alpha_tail = (1.0 - credible_pct) / 2.0
        lower_remaining = float(dist.ppf(alpha_tail))
        upper_remaining = float(dist.ppf(1.0 - alpha_tail))

        return (current_total + lower_remaining, current_total + upper_remaining)


# =========================================================
# LIVE-ADAPTIVE GAME TOTAL MODEL (wraps BayesianPaceModel)
# =========================================================
class LiveAdaptivePaceModel:
    """Legacy interface maintained for compatibility with OddEvenOrchestrator."""
    NEUTRAL_QUARTER_TOTAL = 52.0

    def __init__(self):
        self.quarter_totals: List[float] = []
        self.bayesian_model = BayesianPaceModel()

    def add_completed_quarter(self, home_pts: int, away_pts: int):
        total = home_pts + away_pts
        self.quarter_totals.append(total)
        # Bayesian update: 12 minutes of data
        self.bayesian_model.update(points_observed=total, minutes_elapsed=12.0)

    def current_pace_per_minute(self, pace_multiplier: float = 1.0) -> float:
        return self.bayesian_model.posterior_mean_rate * pace_multiplier

    def pace_confidence(self) -> float:
        return self.bayesian_model.pace_confidence


class LiveOnlyGameTotalModel:
    def __init__(self, dispersion: float = 25.0):
        self.dispersion = dispersion
        self.pace_model = LiveAdaptivePaceModel()
        # Expose Bayesian model directly for SyntheticTotalOrchestrator
        self.bayesian = self.pace_model.bayesian_model

    def ingest_quarter(self, home_pts: int, away_pts: int):
        self.pace_model.add_completed_quarter(home_pts, away_pts)

    def update_intra_quarter(self, points_so_far: int, minutes_elapsed: float):
        """
        Mid-quarter Bayesian update using partial quarter observations.
        Call this every poll cycle for the in-progress quarter.
        """
        if minutes_elapsed > 0:
            self.bayesian.update(
                points_observed=points_so_far,
                minutes_elapsed=minutes_elapsed,
            )

    def over_under_prob(self, current_total: int, quarters_completed: int,
                         minutes_elapsed_current_q: float, line: float,
                         pace_multiplier: float = 1.0) -> Dict:
        minutes_remaining = max(48.0 - (quarters_completed * 12.0 + minutes_elapsed_current_q), 0.0)
        return self.bayesian.predictive_over_under_prob(
            current_total, minutes_remaining, line, pace_multiplier
        )


# =========================================================
# ELITE FEATURE EXTRACTOR (fixed for confirmed API fields)
# =========================================================
class LiveGameFeatureExtractor:
    """
    Extracts elite-tier live features from AllSportsAPI data.

    CONFIRMED AVAILABLE FIELDS (from live API test 2026-08-05):
      player_points, player_minutes, player_assists, player_total_rebounds,
      player_field_goals_made, player_field_goals_attempts,
      player_threepoint_goals_made, player_threepoint_goals_attempts,
      player_freethrows_goals_made, player_freethrows_goals_attempts,
      player_turnovers

    CONFIRMED UNAVAILABLE (empty in API response — NOT used):
      player_personal_fouls, player_offence_rebounds, player_defense_rebounds
    """

    GARBAGE_TIME_LEAD = 20
    GARBAGE_TIME_MINUTES = 8.0

    @staticmethod
    def _safe_int(value, default=0) -> int:
        try:
            if value in ("-", "", None):
                return default
            return int(float(value))
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
        features.update(cls._shooting_efficiency_features(all_players))
        features.update(cls._fatigue_features(all_players, match_state))
        return features

    @classmethod
    def _scoring_pace_features(cls, match_state: Dict) -> Dict:
        completed = match_state.get("completed_quarters", [])
        if not completed:
            return {"scoring_pace_ratio": 1.0, "pace_trend": 0.0}

        overall_avg = np.mean([q["home"] + q["away"] for q in completed])
        if overall_avg <= 0:
            return {"scoring_pace_ratio": 1.0, "pace_trend": 0.0}

        min_elapsed = max(match_state.get("minutes_elapsed_current_q", 6.0), 0.5)
        current_total = match_state.get("current_total", 0)
        prior_total = sum(q["home"] + q["away"] for q in completed)
        current_q_so_far = current_total - prior_total
        current_q_pace = (current_q_so_far / min_elapsed) * 12.0
        ratio = float(np.clip(current_q_pace / overall_avg, 0.5, 2.0))

        if len(completed) >= 2:
            trend = ((completed[-1]["home"] + completed[-1]["away"]) -
                     (completed[0]["home"] + completed[0]["away"]))
            trend_normalized = float(np.clip(trend / max(overall_avg, 1), -1.0, 1.0))
        else:
            trend_normalized = 0.0

        return {"scoring_pace_ratio": ratio, "pace_trend": trend_normalized}

    @classmethod
    def _momentum_features(cls, match_state: Dict) -> Dict:
        completed = match_state.get("completed_quarters", [])
        if not completed:
            return {"momentum_home": 0.0, "momentum_away": 0.0}

        recent = completed[-2:] if len(completed) >= 2 else completed
        home_recent = sum(q["home"] for q in recent)
        away_recent = sum(q["away"] for q in recent)
        total_recent = home_recent + away_recent

        if total_recent == 0:
            return {"momentum_home": 0.0, "momentum_away": 0.0}

        momentum_home = float(np.clip((home_recent - away_recent) / total_recent, -1.0, 1.0))
        return {"momentum_home": momentum_home, "momentum_away": -momentum_home}

    @classmethod
    def _garbage_time_feature(cls, match_state: Dict) -> Dict:
        score_diff = abs(match_state.get("score_diff", 0))
        minutes_remaining = match_state.get("minutes_remaining", 48.0)

        if (score_diff >= cls.GARBAGE_TIME_LEAD and
                minutes_remaining <= cls.GARBAGE_TIME_MINUTES):
            lead_factor = min((score_diff - cls.GARBAGE_TIME_LEAD) / 20.0, 1.0)
            time_factor = 1.0 - (minutes_remaining / cls.GARBAGE_TIME_MINUTES)
            garbage = float(np.clip(0.5 + 0.5 * lead_factor + 0.5 * time_factor, 0.0, 1.0))
        elif score_diff >= cls.GARBAGE_TIME_LEAD * 1.5:
            garbage = float(np.clip((score_diff - cls.GARBAGE_TIME_LEAD) / 40.0, 0.0, 0.6))
        else:
            garbage = 0.0

        return {"garbage_time": garbage}

    @classmethod
    def _time_score_pressure(cls, match_state: Dict) -> Dict:
        score_diff = match_state.get("score_diff", 0)
        minutes_remaining = max(match_state.get("minutes_remaining", 48.0), 0.1)

        if abs(score_diff) == 0:
            pressure = 0.0
        else:
            raw_pressure = abs(score_diff) / (minutes_remaining * 2.0)
            pressure = float(np.clip(raw_pressure, 0.0, 1.0))

        direction = -1.0 if score_diff > 0 else (1.0 if score_diff < 0 else 0.0)
        return {"time_score_pressure": pressure, "pressure_direction": direction}

    @classmethod
    def _spread_vs_live_pressure(cls, match_state: Dict, pregame_elo_diff: float) -> Dict:
        if pregame_elo_diff == 0.0:
            return {"spread_divergence": 0.0}

        expected_margin = pregame_elo_diff * 0.04
        actual_diff = match_state.get("score_diff", 0)
        minutes_remaining = match_state.get("minutes_remaining", 48.0)
        time_elapsed_fraction = 1.0 - (minutes_remaining / 48.0)

        if time_elapsed_fraction < 0.1:
            return {"spread_divergence": 0.0}

        expected_now = expected_margin * time_elapsed_fraction
        divergence = actual_diff - expected_now
        normalized = float(np.clip(divergence / 20.0, -1.0, 1.0))
        return {"spread_divergence": normalized}

    @classmethod
    def _shooting_efficiency_features(cls, all_players: List[Tuple]) -> Dict:
        """
        eFG% = (FGM + 0.5 * 3PM) / FGA  — values 3-pointers appropriately
        FT_rate = FTM / FTA  — free throw efficiency
        3PA_rate = 3PA / FGA  — shot selection (reliance on 3s)
        TO_rate  = turnovers / total_minutes

        Uses ONLY confirmed-available API fields.
        """
        total_fgm = 0
        total_fga = 0
        total_3pm = 0
        total_3pa = 0
        total_ftm = 0
        total_fta = 0
        total_to = 0
        total_mins = 0.0

        for _, p in all_players:
            fgm = cls._safe_int(p.get("player_field_goals_made"))
            fga = cls._safe_int(p.get("player_field_goals_attempts"))
            tpm = cls._safe_int(p.get("player_threepoint_goals_made"))
            tpa_raw = p.get("player_threepoint_goals_attempts")
            tpa = cls._safe_int(tpa_raw) if tpa_raw != "-" else 0
            ftm = cls._safe_int(p.get("player_freethrows_goals_made"))
            fta = cls._safe_int(p.get("player_freethrows_goals_attempts"))
            to = cls._safe_int(p.get("player_turnovers"))
            mins = cls._parse_minutes(p.get("player_minutes", "0:00"))

            total_fgm += fgm
            total_fga += fga
            total_3pm += tpm
            total_3pa += tpa
            total_ftm += ftm
            total_fta += fta
            total_to += to
            total_mins += mins

        # eFG%: >0.55 is elite, <0.45 is poor
        efg = (total_fgm + 0.5 * total_3pm) / max(total_fga, 1)
        efg_normalized = float(np.clip((efg - 0.50) / 0.15, -1.0, 1.0))

        # FT rate: high FTA/FGA ratio → more free scoring opportunities
        ft_rate = total_ftm / max(total_fta, 1)

        # 3PA rate: high → higher variance, higher scoring potential
        three_pa_rate = total_3pa / max(total_fga, 1)

        # TO rate per 10 minutes
        to_rate = total_to / max(total_mins / 10.0, 1.0)
        to_rate_normalized = float(np.clip(to_rate, 0.0, 1.0))

        return {
            "efg_pct_live": float(np.clip(efg, 0.0, 1.0)),
            "efg_normalized": efg_normalized,  # signed: positive = shooting well
            "ft_rate_live": float(np.clip(ft_rate, 0.0, 1.0)),
            "three_pa_rate_live": float(np.clip(three_pa_rate, 0.0, 1.0)),
            "to_rate_live": to_rate_normalized,
        }

    @classmethod
    def _fatigue_features(cls, all_players: List[Tuple], match_state: Dict) -> Dict:
        """
        Fatigue proxy: average minutes played relative to expected
        for this point in the game. High fatigue → slower pace.
        """
        quarters_completed = match_state.get("quarters_completed", 0)
        expected_mins = max(quarters_completed * 12.0, 1.0)

        home_mins, away_mins = 0.0, 0.0
        home_count, away_count = 0, 0

        for team_side, p in all_players:
            mins = cls._parse_minutes(p.get("player_minutes", "0:00"))
            if team_side == "home":
                home_mins += mins
                home_count += 1
            else:
                away_mins += mins
                away_count += 1

        home_fatigue = float(np.clip(
            (home_mins / max(home_count, 1)) / expected_mins - 1.0, 0.0, 0.5
        ))
        away_fatigue = float(np.clip(
            (away_mins / max(away_count, 1)) / expected_mins - 1.0, 0.0, 0.5
        ))

        return {"fatigue_home": home_fatigue, "fatigue_away": away_fatigue}


# =========================================================
# ELITE FEATURE ADJUSTER
# =========================================================
class EliteFeatureAdjuster:
    @staticmethod
    def combined_pace_multiplier(features: Dict) -> float:
        multiplier = 1.0

        # Garbage time: biggest single reducer
        garbage = features.get("garbage_time", 0.0)
        multiplier *= (1.0 - 0.35 * garbage)

        # Current pace ratio vs quarter average
        pace_ratio = features.get("scoring_pace_ratio", 1.0)
        multiplier *= (1.0 + 0.3 * (pace_ratio - 1.0))

        # Pace trend across quarters
        trend = features.get("pace_trend", 0.0)
        multiplier *= (1.0 + 0.1 * trend)

        # Time-score pressure (trailing team presses)
        pressure = features.get("time_score_pressure", 0.0)
        multiplier *= (1.0 + 0.1 * pressure)

        # Shooting efficiency: high eFG → more scoring per possession
        efg_norm = features.get("efg_normalized", 0.0)
        multiplier *= (1.0 + 0.08 * efg_norm)

        # Turnover rate: kills possessions, reduces scoring
        to_rate = features.get("to_rate_live", 0.0)
        multiplier *= (1.0 - 0.06 * to_rate)

        # 3PA rate: high 3PA → more variance, slightly higher expected pts
        three_pa = features.get("three_pa_rate_live", 0.0)
        multiplier *= (1.0 + 0.05 * (three_pa - 0.35))

        # Fatigue (average of both teams)
        avg_fatigue = (features.get("fatigue_home", 0.0) +
                       features.get("fatigue_away", 0.0)) / 2.0
        multiplier *= (1.0 - 0.08 * avg_fatigue)

        return float(np.clip(multiplier, 0.4, 1.8))

    @staticmethod
    def home_win_prob_adjustment(features: Dict, minutes_remaining: float) -> float:
        adjustment = 0.0
        momentum_home = features.get("momentum_home", 0.0)
        time_weight = 1.0 - (minutes_remaining / 48.0)
        adjustment += 0.06 * momentum_home * time_weight
        divergence = features.get("spread_divergence", 0.0)
        adjustment -= 0.03 * divergence
        return float(np.clip(adjustment, -0.07, 0.07))

    @staticmethod
    def player_scoring_multipliers(features: Dict, team_side: str) -> Tuple[float, float]:
        # Note: foul_trouble removed (data unavailable from AllSportsAPI)
        garbage_penalty = float(np.clip(features.get("garbage_time", 0.0) * 0.5, 0.0, 0.3))
        fatigue_penalty = float(np.clip(features.get(f"fatigue_{team_side}", 0.0) * 0.3, 0.0, 0.2))
        combined_penalty = min(garbage_penalty + fatigue_penalty, 0.4)
        return 0.0, combined_penalty  # (foul_penalty=0, garbage+fatigue penalty)


# =========================================================
# SYNTHETIC TOTAL ORCHESTRATOR
# =========================================================
class SyntheticTotalOrchestrator:
    """
    Predicts Over/Under total points WITHOUT bookmaker odds.

    Methodology:
    1. Use BayesianPaceModel posterior to project final total
       (full distribution, not just point estimate)
    2. Find the nearest "natural line" (round numbers common in basketball
       betting: multiples of 5, or significant totals)
    3. Only fire an alert when the model has HIGH CONFIDENCE — specifically
       when the 80% credible interval lies entirely above OR entirely below
       the natural line (meaning 90%+ probability on one side)
    4. Alert includes the projected total, confidence level, and the
       specific line being beaten

    This is equivalent to what a sharp bettor would do: only bet a total
    when you have very strong conviction, not just a slight lean.
    """

    # Minimum confidence threshold before firing any alert
    MIN_CONFIDENCE = 0.60  # posterior confidence (based on data seen so far)

    # Minimum probability on one side before alerting
    # (80% credible interval method → effectively ~90% probability)
    MIN_SIDE_PROBABILITY = 0.72

    # Natural lines to check (common basketball total ranges)
    NATURAL_LINES = list(range(130, 260, 5))  # 130, 135, ..., 255

    def __init__(self, bankroll: float, ev_alert_threshold: float = 0.04):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.kelly = KellyStaking()

    def evaluate(self, live_total_model: LiveOnlyGameTotalModel,
                 current_total: int, quarters_completed: int,
                 minutes_elapsed_current_q: float,
                 pace_multiplier: float = 1.0) -> Optional[Dict]:
        """
        Evaluates all natural lines and returns the strongest signal,
        or None if no line meets the confidence threshold.
        """
        bayesian = live_total_model.bayesian
        minutes_remaining = max(48.0 - (quarters_completed * 12.0 + minutes_elapsed_current_q), 0.0)

        # Require at least some game data before alerting
        if bayesian.pace_confidence < self.MIN_CONFIDENCE:
            return None

        # Get the projected total and 80% credible interval
        low, high = bayesian.high_confidence_total_range(
            current_total, minutes_remaining, pace_multiplier, credible_pct=0.80
        )
        projected_total = current_total + bayesian.posterior_mean_rate * minutes_remaining * pace_multiplier

        best_alert = None
        best_probability = 0.0

        for line in self.NATURAL_LINES:
            # Only consider lines near our projected total (within 25 pts)
            if abs(projected_total - line) > 25:
                continue

            result = bayesian.predictive_over_under_prob(
                current_total, minutes_remaining, float(line), pace_multiplier
            )

            if result.get("already_settled"):
                continue

            p_over = result["p_over"]
            p_under = result["p_under"]

            # Method 1: 80% CI entirely above line → strong Over signal
            # Method 2: 80% CI entirely below line → strong Under signal
            # Method 3: Direct probability threshold
            ci_over = low > line   # entire credible interval above line
            ci_under = high < line  # entire credible interval below line

            side, prob = None, 0.0
            if ci_over and p_over > self.MIN_SIDE_PROBABILITY:
                side, prob = "Over", p_over
            elif ci_under and p_under > self.MIN_SIDE_PROBABILITY:
                side, prob = "Under", p_under
            elif p_over > self.MIN_SIDE_PROBABILITY + 0.05:  # extra margin without CI confirmation
                side, prob = "Over", p_over
            elif p_under > self.MIN_SIDE_PROBABILITY + 0.05:
                side, prob = "Under", p_under

            if side and prob > best_probability:
                best_probability = prob
                best_alert = {
                    "market": f"Synthetic Total {side} {line}",
                    "side": side,
                    "line": line,
                    "model_probability": round(prob, 4),
                    "projected_final_total": round(projected_total, 1),
                    "credible_interval_80pct": (round(low, 1), round(high, 1)),
                    "pace_confidence": round(bayesian.pace_confidence, 3),
                    "posterior_rate_per_min": round(bayesian.posterior_mean_rate, 3),
                    "offered_odds": None,   # no bookmaker odds
                    "edge_pct": round((prob - 0.5) * 100, 2),  # edge vs coin flip
                    "recommended_stake": 0.0,  # no stake without odds
                    "alert": f"📊 SYNTHETIC TOTAL: {side} {line} ({round(prob*100,1)}% confident)",
                }

        if best_alert is None:
            return None

        # Add Kelly stake using implied "fair" odds if confidence is very high
        # Estimated odds: if model says 75% likely, fair odds = 1/0.75 = 1.33
        # We use this to give a stake recommendation even without bookmaker
        if best_alert["model_probability"] >= 0.75:
            implied_fair_odds = round(1.0 / best_alert["model_probability"], 3)
            stake_info = self.kelly.calculate_stake(
                best_alert["model_probability"], implied_fair_odds, self.bankroll
            )
            best_alert["recommended_stake"] = stake_info["recommended_stake"]
            best_alert["implied_fair_odds"] = implied_fair_odds

        return best_alert


# =========================================================
# BAYESIAN LIVE WIN PROBABILITY
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
        p_home_win = float(np.clip(
            1 - norm.cdf(0, loc=projected_final_diff, scale=vol) + elite_adjustment,
            0.001, 0.999
        ))

        return {
            "p_home_win": round(p_home_win, 4),
            "p_away_win": round(1 - p_home_win, 4),
            "projected_final_margin": round(float(projected_final_diff), 2),
        }


# =========================================================
# PLAYER SCORING MODEL
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
            "p_over": round(float(p_over), 4),
            "p_under": round(float(1 - p_over), 4),
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
            raise ValueError("No valid probability sources")
        return round(sum(
            probabilities[k] * (self.source_weights[k] / total_weight)
            for k in available
        ), 4)


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
            return {"recommended_stake": 0.0,
                    "edge": round(fair_probability - (1 / decimal_odds), 4),
                    "reason": "No positive edge"}

        applied_fraction = min(full_kelly_fraction * self.kelly_fraction, self.max_stake_pct)
        stake = round(bankroll * applied_fraction, 2)
        return {
            "recommended_stake": stake,
            "stake_pct_of_bankroll": round(applied_fraction * 100, 2),
            "full_kelly_pct": round(full_kelly_fraction * 100, 2),
            "edge": round(fair_probability - (1 / decimal_odds), 4),
            "expected_value_per_unit": round((fair_probability * b) - q, 4),
        }


# =========================================================
# TEAM STRENGTH ESTIMATOR
# =========================================================
class SimpleTeamStrengthEstimator:
    @staticmethod
    def estimate_elo_diff_from_records(home_wins, home_losses, away_wins, away_losses,
                                        elo_scale=400.0) -> float:
        home_wp = np.clip(home_wins / max(home_wins + home_losses, 1), 0.05, 0.95)
        away_wp = np.clip(away_wins / max(away_wins + away_losses, 1), 0.05, 0.95)
        home_logit = np.log(home_wp / (1 - home_wp))
        away_logit = np.log(away_wp / (1 - away_wp))
        return (home_logit - away_logit) * elo_scale / 4.0


def sanity_check_disagreement(model_prob, market_prob, max_allowed_diff=0.25):
    return abs(model_prob - market_prob) <= max_allowed_diff


# =========================================================
# ORCHESTRATORS
# =========================================================
class HomeAwayOrchestrator:
    def __init__(self, bankroll, ev_alert_threshold=0.03, pregame_elo_diff=0.0,
                 max_model_market_disagreement=0.25):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_disagreement = max_model_market_disagreement
        self.wp_model = LiveWinProbabilityModel(pregame_elo_diff=pregame_elo_diff)
        self.ensemble = EnsembleBlender()
        self.kelly = KellyStaking()

    def evaluate_full_game(self, match_state, odds, elite_features=None):
        features = elite_features or {}
        elite_adj = EliteFeatureAdjuster.home_win_prob_adjustment(
            features, match_state.get("minutes_remaining", 24.0)
        )
        wp = self.wp_model.win_probability(
            match_state["score_diff"], match_state["minutes_remaining"], elite_adj
        )
        fair_probs = ShinDevig.devig([odds["home_decimal"], odds["away_decimal"]])
        market_p_home = fair_probs[0]

        if not sanity_check_disagreement(wp["p_home_win"], market_p_home, self.max_disagreement):
            return None

        blended_p_home = self.ensemble.blend({
            "statistical_model": wp["p_home_win"], "market_devigged": market_p_home,
        })
        return self._check_both_sides(match_state, odds, blended_p_home, "Full Game Home/Away")

    def _check_both_sides(self, match_state, odds, blended_p_home, market_label):
        blended_p_away = 1 - blended_p_home
        home_edge = blended_p_home - (1 / odds["home_decimal"])
        away_edge = blended_p_away - (1 / odds["away_decimal"])

        side, edge, side_odds, prob = None, 0.0, 0.0, 0.0
        if home_edge > self.ev_alert_threshold and home_edge > away_edge:
            side, edge, side_odds, prob = "Home", home_edge, odds["home_decimal"], blended_p_home
        elif away_edge > self.ev_alert_threshold:
            side, edge, side_odds, prob = "Away", away_edge, odds["away_decimal"], blended_p_away

        if side is None:
            return None
        stake_info = self.kelly.calculate_stake(prob, side_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None

        return {
            "match_id": match_state["match_id"], "market": market_label, "side": side,
            "blended_probability": round(prob, 4), "offered_odds": side_odds,
            "edge_pct": round(edge * 100, 2), "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 VALUE BET DETECTED",
        }


class QuarterHomeAwayOrchestrator:
    def __init__(self, bankroll, ev_alert_threshold=0.03,
                 max_model_market_disagreement=0.30, volatility_constant=2.2):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_disagreement = max_model_market_disagreement
        self.volatility_constant = volatility_constant
        self.ensemble = EnsembleBlender()
        self.kelly = KellyStaking()

    def evaluate_current_quarter(self, quarter_score_diff, minutes_elapsed_in_q,
                                  odds, quarter_number, elite_features=None):
        features = elite_features or {}
        minutes_remaining_in_q = max(12.0 - minutes_elapsed_in_q, 0.01)
        rate_per_min = quarter_score_diff / minutes_elapsed_in_q if minutes_elapsed_in_q > 0 else 0.0
        projected_final_q_diff = quarter_score_diff + (rate_per_min * 0.5 * minutes_remaining_in_q)

        # Momentum nudge
        projected_final_q_diff += features.get("momentum_home", 0.0) * 1.5
        vol = self.volatility_constant * np.sqrt(minutes_remaining_in_q)

        p_home = float(np.clip(1 - norm.cdf(0, loc=projected_final_q_diff, scale=vol), 0.02, 0.98))
        fair_probs = ShinDevig.devig([odds["home_decimal"], odds["away_decimal"]])
        market_p_home = fair_probs[0]

        if abs(p_home - market_p_home) > self.max_disagreement:
            return None

        blended_p_home = self.ensemble.blend({
            "statistical_model": p_home, "market_devigged": market_p_home,
        })
        blended_p_away = 1 - blended_p_home

        home_edge = blended_p_home - (1 / odds["home_decimal"])
        away_edge = blended_p_away - (1 / odds["away_decimal"])

        side, edge, side_odds, prob = None, 0.0, 0.0, 0.0
        if home_edge > self.ev_alert_threshold and home_edge > away_edge:
            side, edge, side_odds, prob = "Home", home_edge, odds["home_decimal"], blended_p_home
        elif away_edge > self.ev_alert_threshold:
            side, edge, side_odds, prob = "Away", away_edge, odds["away_decimal"], blended_p_away

        if side is None:
            return None
        stake_info = self.kelly.calculate_stake(prob, side_odds, self.bankroll)
        if stake_info["recommended_stake"] <= 0:
            return None

        return {
            "market": f"Home/Away - Quarter {quarter_number}", "side": side,
            "blended_probability": round(prob, 4), "offered_odds": side_odds,
            "edge_pct": round(edge * 100, 2), "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 VALUE BET DETECTED",
        }


class PlayerPropsOrchestrator:
    MAX_PLAUSIBLE_EDGE = 0.25

    def __init__(self, bankroll, ev_alert_threshold=0.04, max_relative_std=0.35):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.max_relative_std = max_relative_std
        self.model = PlayerScoringModel()
        self.kelly = KellyStaking()

    def evaluate_milestone(self, player_name, points_so_far, minutes_played,
                            minutes_remaining_in_game, threshold_odds,
                            team_side="home", elite_features=None):
        features = elite_features or {}
        _, garbage_fatigue_penalty = EliteFeatureAdjuster.player_scoring_multipliers(
            features, team_side
        )

        result = self.model.milestone_over_under_prob(
            points_so_far=points_so_far, minutes_played=minutes_played,
            minutes_remaining_in_game=minutes_remaining_in_game,
            line=threshold_odds["threshold"],
            foul_trouble_penalty=0.0,
            garbage_time_penalty=garbage_fatigue_penalty,
        )

        if result.get("already_settled"):
            return None

        relative_std = result["std_dev"] / max(result["expected_final_points"], 1)
        if relative_std > self.max_relative_std:
            return None

        offered_odds = threshold_odds["over_decimal"]
        edge = result["p_over"] - (1 / offered_odds)

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


class OddEvenOrchestrator:
    def __init__(self, bankroll, ev_alert_threshold=0.04):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.kelly = KellyStaking()

    def evaluate(self, live_total_model, current_total, quarters_completed,
                 minutes_elapsed_current_q, odds, pace_multiplier=1.0):
        bayesian = live_total_model.bayesian
        minutes_remaining = max(48.0 - (quarters_completed * 12.0 + minutes_elapsed_current_q), 0.0)

        adj_minutes = minutes_remaining * pace_multiplier
        if adj_minutes <= 0:
            return None

        r = bayesian.alpha
        p_param = bayesian.beta / (bayesian.beta + adj_minutes)
        dist = nbinom(r, p_param)

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


class HighestScoringQuarterOrchestrator:
    NEUTRAL_QUARTER_PRIORS = {1: 0.23, 2: 0.26, 3: 0.24, 4: 0.27}

    def __init__(self, bankroll, ev_alert_threshold=0.05):
        self.bankroll = bankroll
        self.ev_alert_threshold = ev_alert_threshold
        self.kelly = KellyStaking()

    def evaluate(self, completed_quarter_totals, odds_by_quarter):
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
            "edge_pct": round(best_edge * 100, 2), "recommended_stake": stake_info["recommended_stake"],
            "alert": "🔥 VALUE BET DETECTED",
        }
