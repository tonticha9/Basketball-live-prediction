"""
prediction_engine.py
All mathematical models: de-vig, scoring models, win probability,
ensemble blending, Kelly staking, CLV tracking. Consolidated from
Kaggle testing into production-ready module.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nbinom, norm
from dataclasses import dataclass
from typing import List, Dict, Optional
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
# QUARTER / PLAYER SCORING HELPERS
# =========================================================
class QuarterScoringModel:
    @staticmethod
    def _nbinom_params(mean: float, dispersion: float):
        r = dispersion
        p = r / (r + mean)
        return r, p


# =========================================================
# LIVE-ADAPTIVE GAME TOTAL MODEL (zero historical data)
# =========================================================
class LiveAdaptivePaceModel:
    NEUTRAL_QUARTER_TOTAL = 52.0

    def __init__(self):
        self.quarter_totals: List[float] = []

    def add_completed_quarter(self, home_pts: int, away_pts: int):
        self.quarter_totals.append(home_pts + away_pts)

    def current_pace_per_minute(self) -> float:
        if not self.quarter_totals:
            return self.NEUTRAL_QUARTER_TOTAL / 12.0
        n = len(self.quarter_totals)
        weights = np.linspace(0.7, 1.3, n)
        weighted_avg = np.average(self.quarter_totals, weights=weights)
        return weighted_avg / 12.0

    def pace_confidence(self) -> float:
        n = len(self.quarter_totals)
        if n == 0:
            return 0.15
        elif n == 1:
            return 0.45
        elif n == 2:
            return 0.70
        elif n == 3:
            return 0.90
        return 0.95


class LiveOnlyGameTotalModel:
    def __init__(self, dispersion: float = 25.0):
        self.dispersion = dispersion
        self.pace_model = LiveAdaptivePaceModel()

    def ingest_quarter(self, home_pts: int, away_pts: int):
        self.pace_model.add_completed_quarter(home_pts, away_pts)

    def over_under_prob(self, current_total: int, quarters_completed: int,
                         minutes_elapsed_current_q: float, line: float) -> Dict:
        per_minute_rate = self.pace_model.current_pace_per_minute()
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
# BAYESIAN LIVE WIN PROBABILITY
# =========================================================
class LiveWinProbabilityModel:
    def __init__(self, pregame_elo_diff: float, elo_to_points_scale: float = 0.04):
        self.pregame_elo_diff = pregame_elo_diff
        self.expected_pregame_margin = pregame_elo_diff * elo_to_points_scale

    @staticmethod
    def _volatility_per_minute(minutes_remaining: float) -> float:
        base_vol_per_min = 0.9
        return base_vol_per_min * np.sqrt(max(minutes_remaining, 0.01))

    def win_probability(self, current_score_diff: int, minutes_remaining: float,
                         possession_adjustment: float = 0.0) -> Dict:
        if minutes_remaining <= 0:
            p_home = 1.0 if current_score_diff > 0 else (0.0 if current_score_diff < 0 else 0.5)
            return {"p_home_win": p_home, "p_away_win": 1 - p_home}

        time_fraction_remaining = minutes_remaining / 48.0
        drift_weight = time_fraction_remaining
        expected_remaining_margin_shift = drift_weight * (
            self.expected_pregame_margin - current_score_diff * (1 - time_fraction_remaining)
        )

        projected_final_diff = current_score_diff + expected_remaining_margin_shift + possession_adjustment
        vol = self._volatility_per_minute(minutes_remaining)

        p_home_win = 1 - norm.cdf(0, loc=projected_final_diff, scale=vol)

        return {
            "p_home_win": round(float(np.clip(p_home_win, 0.001, 0.999)), 4),
            "p_away_win": round(float(np.clip(1 - p_home_win, 0.001, 0.999)), 4),
            "projected_final_margin": round(float(projected_final_diff), 2),
        }


# =========================================================
# PLAYER SCORING MODEL (fixed: continuous shrinkage + rate cap)
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

    def project_remaining_points(self, points_so_far: int, minutes_played: float,
                                  minutes_remaining_in_game: float,
                                  foul_trouble_penalty: float = 0.0,
                                  garbage_time_penalty: float = 0.0):
        rate = self._effective_rate(points_so_far, minutes_played)
        rate *= (1.0 - foul_trouble_penalty) * (1.0 - garbage_time_penalty)
        expected_remaining = max(rate * minutes_remaining_in_game, 0.05)
        r, p = QuarterScoringModel._nbinom_params(expected_remaining, self.dispersion)
        return nbinom(r, p)

    def milestone_over_under_prob(self, points_so_far: int, minutes_played: float,
                                   minutes_remaining_in_game: float, line: float,
                                   foul_trouble_penalty: float = 0.0,
                                   garbage_time_penalty: float = 0.0) -> Dict:
        dist = self.project_remaining_points(
            points_so_far, minutes_played, minutes_remaining_in_game,
            foul_trouble_penalty, garbage_time_penalty
        )
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

    def update_weights_from_backtest(self, brier_scores: Dict[str, float]):
        inv_scores = {k: 1.0 / max(v, 0.001) for k, v in brier_scores.items()}
        total = sum(inv_scores.values())
        self.source_weights = {k: round(v / total, 4) for k, v in inv_scores.items()}


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
            return {
                "recommended_stake": 0.0,
                "edge": round(fair_probability - (1 / decimal_odds), 4),
                "reason": "No positive edge — do not bet",
            }

        applied_fraction = min(full_kelly_fraction * self.kelly_fraction, self.max_stake_pct)
        stake = round(bankroll * applied_fraction, 2)
        implied_prob = 1 / decimal_odds
        edge = fair_probability - implied_prob

        return {
            "recommended_stake": stake,
            "stake_pct_of_bankroll": round(applied_fraction * 100, 2),
            "full_kelly_pct": round(full_kelly_fraction * 100, 2),
            "edge": round(edge, 4),
            "expected_value_per_unit": round((fair_probability * b) - q, 4),
        }


# =========================================================
# TEAM STRENGTH ESTIMATOR (from Standings, no ML training)
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
# ORCHESTRATORS
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

    def evaluate_full_game(self, match_state: Dict, odds: Dict) -> Optional[Dict]:
        wp = self.wp_model.win_probability(
            current_score_diff=match_state["score_diff"],
            minutes_remaining=match_state["minutes_remaining"],
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
                                  odds: Dict, quarter_number: int) -> Optional[Dict]:
        minutes_remaining_in_q = max(12.0 - minutes_elapsed_in_q, 0.01)
        rate_per_min = quarter_score_diff / minutes_elapsed_in_q if minutes_elapsed_in_q > 0 else 0.0
        regressed_rate = rate_per_min * 0.5
        projected_final_q_diff = quarter_score_diff + (regressed_rate * minutes_remaining_in_q)
        vol = self.volatility_constant * np.sqrt(minutes_remaining_in_q)

        p_home_wins_quarter = float(np.clip(
            1 - norm.cdf(0, loc=projected_final_q_diff, scale=vol), 0.02, 0.98
        ))

        fair_probs = ShinDevig.devig([odds["home_decimal"], odds["away_decimal"]])
        market_p_home = fair_probs[0]

        if abs(p_home_wins_quarter - market_p_home) > self.max_disagreement:
            return None

        blended_p_home = self.ensemble.blend({
            "statistical_model": p_home_wins_quarter,
            "market_devigged": market_p_home,
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
                            foul_trouble_penalty: float = 0.0,
                            garbage_time_penalty: float = 0.0) -> Optional[Dict]:
        result = self.model.milestone_over_under_prob(
            points_so_far=points_so_far, minutes_played=minutes_played,
            minutes_remaining_in_game=minutes_remaining_in_game,
            line=threshold_odds["threshold"],
            foul_trouble_penalty=foul_trouble_penalty,
            garbage_time_penalty=garbage_time_penalty,
        )

        if result.get("already_settled"):
            return None

        relative_std = result["std_dev"] / max(result["expected_final_points"], 1)
        if relative_std > self.max_relative_std:
            return None

        offered_odds = threshold_odds["over_decimal"]
        implied_prob = 1 / offered_odds
        edge = result["p_over"] - implied_prob

        if edge < self.ev_alert_threshold:
            return None
        if edge > self.MAX_PLAUSIBLE_EDGE:
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
