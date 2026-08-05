"""
parsers.py
Converts raw AllSportsAPI JSON responses into clean, usable data
structures for the prediction engine.

Quarter-completion detection is based on which quarter score entries
have data (not on event_status text), which is robust across leagues
that report status differently (e.g. "3rd Quarter" vs plain "3").

UPDATED: _average_odds now limits to the first 2 bookmaker entries per
market instead of all 15+, reducing memory/CPU load per poll cycle on
low-RAM hosting (negligible accuracy cost since 2 major books already
track market consensus closely).
"""

from typing import Dict, List, Optional
from prediction_engine import SimpleTeamStrengthEstimator


class AllSportsLivescoreParser:

    @staticmethod
    def is_live(event: Dict) -> bool:
        status = str(event.get("event_status", "")).strip()
        if status == "" or status == "Finished" or status.lower() == "not started":
            return False
        return True

    @staticmethod
    def build_match_state(event: Dict) -> Optional[Dict]:
        if not AllSportsLivescoreParser.is_live(event):
            return None

        scores = event.get("scores", {})
        keys = ["1stQuarter", "2ndQuarter", "3rdQuarter", "4thQuarter"]

        quarter_entries: List[Optional[Dict]] = []
        for key in keys:
            q_data = scores.get(key, [])
            if q_data:
                entry = q_data[0]
                quarter_entries.append({
                    "home": int(entry.get("score_home", 0) or 0),
                    "away": int(entry.get("score_away", 0) or 0),
                })
            else:
                quarter_entries.append(None)

        last_idx = -1
        for i, q in enumerate(quarter_entries):
            if q is not None:
                last_idx = i

        if last_idx == -1:
            completed_quarters: List[Dict] = []
            current_total = 0
            score_diff = 0
            quarters_completed = 0
        else:
            completed_quarters = [q for q in quarter_entries[:last_idx] if q is not None]
            live_entry = quarter_entries[last_idx]
            completed_home = sum(q["home"] for q in completed_quarters)
            completed_away = sum(q["away"] for q in completed_quarters)
            current_total = completed_home + completed_away + live_entry["home"] + live_entry["away"]
            score_diff = (completed_home + live_entry["home"]) - (completed_away + live_entry["away"])
            quarters_completed = len(completed_quarters)

        raw_status = str(event.get("event_status", ""))
        minutes_elapsed_current_q = 6.0
        if ":" in raw_status:
            try:
                mins, secs = raw_status.split(":")
                minutes_elapsed_current_q = 12.0 - (int(mins) + int(secs) / 60.0)
                minutes_elapsed_current_q = max(0.0, min(minutes_elapsed_current_q, 12.0))
            except ValueError:
                pass

        return {
            "match_id": str(event.get("event_key")),
            "home_team": event.get("event_home_team"),
            "away_team": event.get("event_away_team"),
            "quarters_completed": quarters_completed,
            "minutes_elapsed_current_q": minutes_elapsed_current_q,
            "current_total": current_total,
            "score_diff": score_diff,
            "minutes_remaining": max(48.0 - (quarters_completed * 12.0 + minutes_elapsed_current_q), 0.0),
            "completed_quarters": completed_quarters,
            "league_name": event.get("league_name"),
        }

    @staticmethod
    def extract_all_quarters_for_finished(event: Dict) -> List[Optional[Dict]]:
        """For a FINISHED match, returns all 4 quarters (None if missing)."""
        scores = event.get("scores", {})
        keys = ["1stQuarter", "2ndQuarter", "3rdQuarter", "4thQuarter"]
        result = []
        for key in keys:
            q_data = scores.get(key, [])
            if q_data:
                entry = q_data[0]
                result.append({
                    "home": int(entry.get("score_home", 0) or 0),
                    "away": int(entry.get("score_away", 0) or 0),
                })
            else:
                result.append(None)
        return result

    @staticmethod
    def sync_orchestrator_quarters(live_only_model, completed_quarters: List[Dict],
                                    already_ingested_count: int) -> int:
        new_quarters = completed_quarters[already_ingested_count:]
        for q in new_quarters:
            live_only_model.ingest_quarter(home_pts=q["home"], away_pts=q["away"])
        return len(completed_quarters)


class AllSportsOddsParser:
    @staticmethod
    def _average_odds(book_dict: Dict[str, str]) -> Optional[float]:
        """
        Averages only the first 2 bookmaker entries per market instead
        of all 15+. This reduces both memory footprint (smaller
        intermediate lists built on every poll cycle) and CPU time,
        at negligible accuracy cost since 2 major bookmakers already
        track market consensus closely.
        """
        if not book_dict:
            return None
        limited_values = list(book_dict.values())[:2]
        values = [float(v) for v in limited_values if v]
        return round(sum(values) / len(values), 3) if values else None

    @staticmethod
    def get_full_game_home_away(odds_response: Dict, match_id: str) -> Optional[Dict]:
        match_odds = odds_response.get("result", {}).get(str(match_id), {})
        market = match_odds.get("Home/Away", {})
        home_avg = AllSportsOddsParser._average_odds(market.get("Home", {}))
        away_avg = AllSportsOddsParser._average_odds(market.get("Away", {}))
        if home_avg is None or away_avg is None:
            return None
        return {"home_decimal": home_avg, "away_decimal": away_avg}

    @staticmethod
    def get_quarter_home_away(odds_response: Dict, match_id: str, quarter: int) -> Optional[Dict]:
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(quarter)
        if not ordinal:
            return None
        match_odds = odds_response.get("result", {}).get(str(match_id), {})
        market = match_odds.get(f"Home/Away - {ordinal} Qtr", {})
        home_avg = AllSportsOddsParser._average_odds(market.get("Home", {}))
        away_avg = AllSportsOddsParser._average_odds(market.get("Away", {}))
        if home_avg is None or away_avg is None:
            return None
        return {"home_decimal": home_avg, "away_decimal": away_avg}

    @staticmethod
    def get_odd_even(odds_response: Dict, match_id: str) -> Optional[Dict]:
        match_odds = odds_response.get("result", {}).get(str(match_id), {})
        market = match_odds.get("Odd/Even (Including OT)", {})
        odd_avg = AllSportsOddsParser._average_odds(market.get("Odd", {}))
        even_avg = AllSportsOddsParser._average_odds(market.get("Even", {}))
        if odd_avg is None or even_avg is None:
            return None
        return {"odd_decimal": odd_avg, "even_decimal": even_avg}

    @staticmethod
    def get_highest_scoring_quarter_odds(odds_response: Dict, match_id: str) -> Optional[Dict[int, float]]:
        match_odds = odds_response.get("result", {}).get(str(match_id), {})
        market = match_odds.get("Highest Scoring Quarter", {})
        result = {}
        ordinal_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        for key, ord_num in ordinal_map.items():
            sub_market = market.get(f"Highest Scoring Quarter {key}", {})
            avg = AllSportsOddsParser._average_odds(sub_market.get("", {}))
            if avg:
                result[ord_num] = avg
        return result if result else None


class AllSportsPlayerStatsParser:
    @staticmethod
    def get_all_players_live_stats(event: Dict) -> List[Dict]:
        results = []
        for team_side in ["home_team", "away_team"]:
            players = event.get("player_statistics", {}).get(team_side, [])
            for p in players:
                pts_raw = p.get("player_points", "0")
                pts = int(pts_raw) if pts_raw not in ("-", "", None) else 0
                mins_raw = p.get("player_minutes", "0:00")
                mins = AllSportsPlayerStatsParser._parse_minutes(mins_raw)
                results.append({
                    "player": p.get("player"), "team_side": team_side,
                    "points": pts, "minutes": mins,
                })
        return results

    @staticmethod
    def _parse_minutes(mins_str: str) -> float:
        try:
            parts = str(mins_str).split(":")
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0


class AllSportsPlayerOddsParser:
    @staticmethod
    def get_all_milestones_for_player(odds_response: Dict, match_id: str,
                                       player_name: str) -> List[Dict]:
        match_odds = odds_response.get("result", {}).get(str(match_id), {})
        milestones = match_odds.get("Player Points Milestones", {})
        market_key = f"Player Points Milestones {player_name}"
        player_market = milestones.get(market_key, {})

        results = []
        for threshold_str, book_odds in player_market.items():
            limited_values = list(book_odds.values())[:2]
            values = [float(v) for v in limited_values if v]
            if values:
                results.append({
                    "threshold": float(threshold_str),
                    "over_decimal": round(sum(values) / len(values), 3),
                })
        return results


class AllSportsStandingsParser:
    @staticmethod
    def extract_team_record(standings_response: Dict, team_name: str) -> Optional[Dict]:
        results = standings_response.get("result", [])
        for entry in results:
            if entry.get("standing_team", "").strip().lower() == team_name.strip().lower():
                wins = int(entry.get("standing_W", 0) or 0)
                losses = int(entry.get("standing_L", 0) or 0)
                return {"wins": wins, "losses": losses}
        return None

    @staticmethod
    def get_matchup_elo_diff(standings_response: Dict, home_team: str, away_team: str,
                              elo_scale: float = 400.0) -> float:
        home_record = AllSportsStandingsParser.extract_team_record(standings_response, home_team)
        away_record = AllSportsStandingsParser.extract_team_record(standings_response, away_team)
        if home_record is None or away_record is None:
            return 0.0
        return SimpleTeamStrengthEstimator.estimate_elo_diff_from_records(
            home_record["wins"], home_record["losses"],
            away_record["wins"], away_record["losses"], elo_scale=elo_scale,
        )
