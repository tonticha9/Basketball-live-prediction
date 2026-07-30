"""
scheduler.py
Polling loop that runs every POLL_INTERVAL_SECONDS, fetching live matches
from AllSportsAPI, running them through the orchestrators, storing
snapshots, and saving any alerts to the database (visible on dashboard).
"""

import requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from config import Config
from models import get_db_session, LiveMatchSnapshot, ValueBetAlert, TeamStrengthCache
from parsers import (
    AllSportsLivescoreParser, AllSportsOddsParser,
    AllSportsPlayerStatsParser, AllSportsPlayerOddsParser,
    AllSportsStandingsParser,
)
from prediction_engine import (
    HomeAwayOrchestrator, QuarterHomeAwayOrchestrator, PlayerPropsOrchestrator,
)


class LiveMatchManager:
    def __init__(self, api_key: str, bankroll: float):
        self.api_key = api_key
        self.bankroll = bankroll
        self.base_url = Config.ALLSPORTS_BASE_URL
        self.active_matches: dict = {}
        self._standings_cache: dict = {}  # {league_key: standings_json}

    def _fetch(self, params: dict) -> dict:
        params["APIkey"] = self.api_key
        resp = requests.get(self.base_url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def poll_cycle(self):
        """Main entry point — called every POLL_INTERVAL_SECONDS by scheduler."""
        try:
            livescore_data = self._fetch({"met": "Livescore"})
        except requests.RequestException as e:
            print(f"[poll_cycle] Livescore fetch failed: {e}")
            return

        live_events = [
            e for e in livescore_data.get("result", [])
            if AllSportsLivescoreParser.is_live(e)
        ]

        for event in live_events:
            match_id = str(event.get("event_key"))
            self._ensure_match_initialized(match_id)
            try:
                self._process_match(event, match_id)
            except Exception as e:
                print(f"[poll_cycle] Error processing match {match_id}: {e}")

    def _ensure_match_initialized(self, match_id: str):
        if match_id not in self.active_matches:
            self.active_matches[match_id] = {
                "home_away_orch": HomeAwayOrchestrator(bankroll=self.bankroll,
                                                        ev_alert_threshold=Config.EV_ALERT_THRESHOLD),
                "quarter_orch": QuarterHomeAwayOrchestrator(bankroll=self.bankroll,
                                                             ev_alert_threshold=Config.EV_ALERT_THRESHOLD),
                "player_orch": PlayerPropsOrchestrator(bankroll=self.bankroll),
            }

    def _get_elo_diff(self, event: dict) -> float:
        """Fetches (and caches per-poll-cycle) standings-based team strength."""
        league_key = event.get("league_key")
        if league_key not in self._standings_cache:
            try:
                self._standings_cache[league_key] = self._fetch({
                    "met": "Standings", "leagueId": league_key
                })
            except requests.RequestException:
                self._standings_cache[league_key] = {"result": []}

        standings = self._standings_cache[league_key]
        return AllSportsStandingsParser.get_matchup_elo_diff(
            standings, event.get("event_home_team", ""), event.get("event_away_team", "")
        )

    def _process_match(self, event: dict, match_id: str):
        match_state = AllSportsLivescoreParser.build_match_state(event)
        if match_state is None:
            return

        try:
            odds_data = self._fetch({"met": "Odds", "matchId": match_id})
        except requests.RequestException as e:
            print(f"[_process_match] Odds fetch failed for {match_id}: {e}")
            return

        session = get_db_session()
        try:
            snapshot = LiveMatchSnapshot(
                match_id=match_id, league_name=match_state.get("league_name"),
                home_team=match_state.get("home_team"), away_team=match_state.get("away_team"),
                quarters_completed=match_state["quarters_completed"],
                minutes_elapsed_current_q=match_state["minutes_elapsed_current_q"],
                current_total=match_state["current_total"], score_diff=match_state["score_diff"],
                minutes_remaining=match_state["minutes_remaining"],
                raw_odds_json=odds_data, polled_at=datetime.utcnow(),
            )
            session.add(snapshot)
            session.commit()

            # --- Full-game Home/Away ---
            home_away_odds = AllSportsOddsParser.get_full_game_home_away(odds_data, match_id)
            if home_away_odds:
                elo_diff = self._get_elo_diff(event)
                orch = self.active_matches[match_id]["home_away_orch"]
                orch.wp_model.pregame_elo_diff = elo_diff
                orch.wp_model.expected_pregame_margin = elo_diff * 0.04
                alert = orch.evaluate_full_game(match_state, home_away_odds)
                if alert:
                    self._save_alert(session, alert)

            # --- Player Props ---
            player_stats = AllSportsPlayerStatsParser.get_all_players_live_stats(event)
            player_orch = self.active_matches[match_id]["player_orch"]
            for player in player_stats:
                if player["minutes"] < 3.0:
                    continue  # too early to evaluate this player meaningfully
                milestones = AllSportsPlayerOddsParser.get_all_milestones_for_player(
                    odds_data, match_id, player["player"]
                )
                minutes_remaining_game = match_state["minutes_remaining"]
                for m in milestones:
                    p_alert = player_orch.evaluate_milestone(
                        player_name=player["player"], points_so_far=player["points"],
                        minutes_played=player["minutes"],
                        minutes_remaining_in_game=minutes_remaining_game,
                        threshold_odds=m,
                    )
                    if p_alert:
                        self._save_alert(session, p_alert, match_id=match_id)

        finally:
            session.close()

    def _save_alert(self, session, alert: dict, match_id: str = None):
        db_alert = ValueBetAlert(
            match_id=alert.get("match_id", match_id),
            market=alert["market"],
            side=alert.get("side") or alert.get("player", ""),
            blended_probability=alert.get("blended_probability") or alert.get("model_probability", 0.0),
            offered_odds=alert["offered_odds"],
            edge_pct=alert["edge_pct"],
            recommended_stake=alert["recommended_stake"],
            fired_at=datetime.utcnow(),
        )
        session.add(db_alert)
        session.commit()
        print(f"[ALERT] {alert}")


def create_scheduler():
    manager = LiveMatchManager(
        api_key=Config.ALLSPORTS_API_KEY, bankroll=Config.STARTING_BANKROLL,
    )
    scheduler = BackgroundScheduler()
    scheduler.add_job(manager.poll_cycle, "interval",
                       seconds=Config.POLL_INTERVAL_SECONDS, id="live_poll")
    scheduler.start()
    return scheduler
