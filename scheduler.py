"""
scheduler.py
Polling loop — reads API key from database (with expiry check),
processes all live matches across: Full-game Home/Away, Quarter
Home/Away, Player Props, Odd/Even, Highest Scoring Quarter.
Updates LiveMatchStatus for dashboard display.

Includes:
- Safe _fetch() handling non-JSON / non-dict responses
- Alert deduplication (cooldown per match+market+side)
"""

import requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from config import Config
from models import (
    get_db_session, LiveMatchSnapshot, ValueBetAlert, LiveMatchStatus,
    get_or_create_api_key_settings,
)
from parsers import (
    AllSportsLivescoreParser, AllSportsOddsParser,
    AllSportsPlayerStatsParser, AllSportsPlayerOddsParser,
    AllSportsStandingsParser,
)
from prediction_engine import (
    HomeAwayOrchestrator, QuarterHomeAwayOrchestrator, PlayerPropsOrchestrator,
    OddEvenOrchestrator, HighestScoringQuarterOrchestrator, LiveOnlyGameTotalModel,
)


class LiveMatchManager:
    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        self.base_url = Config.ALLSPORTS_BASE_URL
        self.active_matches = {}
        self._standings_cache = {}
        self._recent_alert_keys = {}

    def _get_active_api_key(self) -> str:
        session = get_db_session()
        try:
            settings = get_or_create_api_key_settings(session)
            if settings.api_key and settings.is_active:
                if settings.expires_at and settings.expires_at < datetime.utcnow():
                    print(f"[api_key] Key expired at {settings.expires_at} — clearing.")
                    settings.api_key = None
                    settings.is_active = False
                    session.commit()
                    return Config.ALLSPORTS_API_KEY
                return settings.api_key
            return Config.ALLSPORTS_API_KEY
        finally:
            session.close()

    def _fetch(self, params, api_key):
        params["APIkey"] = api_key
        resp = requests.get(self.base_url, params=params, timeout=15)
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            print(f"[_fetch] Response was not valid JSON. Raw text: {resp.text[:300]}")
            return {"result": {}}

        if not isinstance(data, dict):
            print(f"[_fetch] Unexpected response type ({type(data).__name__}). "
                  f"Content: {repr(data)[:300]}")
            return {"result": {}}

        if data.get("success") == 0:
            print(f"[_fetch] API returned success=0. Full response: {repr(data)[:300]}")

        return data

    def _should_fire_alert(self, match_id, market, side, cooldown_seconds=300):
        key = f"{match_id}:{market}:{side}"
        now = datetime.utcnow()
        last_fired = self._recent_alert_keys.get(key)
        if last_fired and (now - last_fired).total_seconds() < cooldown_seconds:
            return False
        self._recent_alert_keys[key] = now
        return True

    def poll_cycle(self):
        print(f"[heartbeat] Poll cycle running at {datetime.utcnow().isoformat()}")

        api_key = self._get_active_api_key()
        if not api_key:
            print("[poll_cycle] No active API key — skipping this cycle.")
            return

        try:
            livescore_data = self._fetch({"met": "Livescore"}, api_key)
        except requests.RequestException as e:
            print(f"[poll_cycle] Livescore fetch failed: {e}")
            return

        live_events = [
            e for e in livescore_data.get("result", [])
            if isinstance(e, dict) and AllSportsLivescoreParser.is_live(e)
        ]

        live_match_ids_this_cycle = set()

        for event in live_events:
            match_id = str(event.get("event_key"))
            live_match_ids_this_cycle.add(match_id)
            self._ensure_match_initialized(match_id)
            try:
                self._process_match(event, match_id, api_key)
            except Exception as e:
                print(f"[poll_cycle] Error processing match {match_id}: {e}")

        self._cleanup_stale_matches(live_match_ids_this_cycle)

    def _ensure_match_initialized(self, match_id):
        if match_id not in self.active_matches:
            self.active_matches[match_id] = {
                "home_away_orch": HomeAwayOrchestrator(
                    bankroll=self.bankroll, ev_alert_threshold=Config.EV_ALERT_THRESHOLD),
                "quarter_orch": QuarterHomeAwayOrchestrator(
                    bankroll=self.bankroll, ev_alert_threshold=Config.EV_ALERT_THRESHOLD),
                "player_orch": PlayerPropsOrchestrator(bankroll=self.bankroll),
                "odd_even_orch": OddEvenOrchestrator(bankroll=self.bankroll),
                "hsq_orch": HighestScoringQuarterOrchestrator(bankroll=self.bankroll),
                "live_total_model": LiveOnlyGameTotalModel(),
                "quarters_ingested": 0,
            }

    def _get_elo_diff(self, event, api_key):
        league_key = event.get("league_key")
        if league_key not in self._standings_cache:
            try:
                self._standings_cache[league_key] = self._fetch(
                    {"met": "Standings", "leagueId": league_key}, api_key
                )
            except requests.RequestException:
                self._standings_cache[league_key] = {"result": []}

        standings = self._standings_cache[league_key]
        return AllSportsStandingsParser.get_matchup_elo_diff(
            standings, event.get("event_home_team", ""), event.get("event_away_team", "")
        )

    def _process_match(self, event, match_id, api_key):
        match_state = AllSportsLivescoreParser.build_match_state(event)
        if match_state is None:
            return

        try:
            odds_data = self._fetch({"met": "Odds", "matchId": match_id}, api_key)
        except requests.RequestException as e:
            print(f"[_process_match] Odds fetch failed for {match_id}: {e}")
            odds_data = {"result": {}}

        session = get_db_session()
        try:
            status_row = session.query(LiveMatchStatus).filter_by(match_id=match_id).first()
            if status_row is None:
                status_row = LiveMatchStatus(match_id=match_id, last_updated=datetime.utcnow())
                session.add(status_row)

            status_row.home_team = match_state.get("home_team")
            status_row.away_team = match_state.get("away_team")
            status_row.league_name = match_state.get("league_name")
            status_row.quarters_completed = match_state["quarters_completed"]
            status_row.current_total = match_state["current_total"]
            status_row.score_diff = match_state["score_diff"]
            status_row.last_updated = datetime.utcnow()
            has_alert_this_cycle = False

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

            match_bundle = self.active_matches[match_id]

            home_away_odds = AllSportsOddsParser.get_full_game_home_away(odds_data, match_id)
            if home_away_odds:
                elo_diff = self._get_elo_diff(event, api_key)
                orch = match_bundle["home_away_orch"]
                orch.wp_model.pregame_elo_diff = elo_diff
                orch.wp_model.expected_pregame_margin = elo_diff * 0.04
                alert = orch.evaluate_full_game(match_state, home_away_odds)
                if alert and self._save_alert(session, alert, match_id):
                    has_alert_this_cycle = True

            current_q = match_state["quarters_completed"] + 1
            if current_q <= 4:
                q_odds = AllSportsOddsParser.get_quarter_home_away(odds_data, match_id, current_q)
                if q_odds and match_state["minutes_elapsed_current_q"] >= 2.0:
                    completed = match_state["completed_quarters"]
                    prior_diff = sum(q["home"] - q["away"] for q in completed)
                    q_diff = match_state["score_diff"] - prior_diff
                    q_alert = match_bundle["quarter_orch"].evaluate_current_quarter(
                        quarter_score_diff=q_diff,
                        minutes_elapsed_in_q=match_state["minutes_elapsed_current_q"],
                        odds=q_odds, quarter_number=current_q,
                    )
                    if q_alert and self._save_alert(session, q_alert, match_id):
                        has_alert_this_cycle = True

            player_stats = AllSportsPlayerStatsParser.get_all_players_live_stats(event)
            player_orch = match_bundle["player_orch"]
            for player in player_stats:
                if player["minutes"] < 3.0:
                    continue
                milestones = AllSportsPlayerOddsParser.get_all_milestones_for_player(
                    odds_data, match_id, player["player"]
                )
                for m in milestones:
                    p_alert = player_orch.evaluate_milestone(
                        player_name=player["player"], points_so_far=player["points"],
                        minutes_played=player["minutes"],
                        minutes_remaining_in_game=match_state["minutes_remaining"],
                        threshold_odds=m,
                    )
                    if p_alert and self._save_alert(session, p_alert, match_id):
                        has_alert_this_cycle = True

            already_ingested = match_bundle["quarters_ingested"]
            new_count = AllSportsLivescoreParser.sync_orchestrator_quarters(
                match_bundle["live_total_model"], match_state["completed_quarters"], already_ingested
            )
            match_bundle["quarters_ingested"] = new_count

            odd_even_odds = AllSportsOddsParser.get_odd_even(odds_data, match_id)
            if odd_even_odds and match_state["quarters_completed"] >= 1:
                oe_alert = match_bundle["odd_even_orch"].evaluate(
                    match_bundle["live_total_model"], match_state["current_total"],
                    match_state["quarters_completed"], match_state["minutes_elapsed_current_q"],
                    odd_even_odds,
                )
                if oe_alert and self._save_alert(session, oe_alert, match_id):
                    has_alert_this_cycle = True

            hsq_odds = AllSportsOddsParser.get_highest_scoring_quarter_odds(odds_data, match_id)
            if hsq_odds and match_state["completed_quarters"]:
                completed_totals = {
                    i + 1: q["home"] + q["away"]
                    for i, q in enumerate(match_state["completed_quarters"])
                }
                hsq_alert = match_bundle["hsq_orch"].evaluate(completed_totals, hsq_odds)
                if hsq_alert and self._save_alert(session, hsq_alert, match_id):
                    has_alert_this_cycle = True

            status_row.has_active_alert = has_alert_this_cycle
            session.commit()

        finally:
            session.close()

    def _cleanup_stale_matches(self, live_match_ids_this_cycle):
        session = get_db_session()
        try:
            all_status_rows = session.query(LiveMatchStatus).all()
            for row in all_status_rows:
                if row.match_id not in live_match_ids_this_cycle:
                    session.delete(row)
                    self.active_matches.pop(row.match_id, None)
            session.commit()
        finally:
            session.close()

    def _save_alert(self, session, alert, match_id):
        market = alert["market"]
        side = alert.get("side") or alert.get("player", "")

        if not self._should_fire_alert(match_id, market, side):
            return False

        db_alert = ValueBetAlert(
            match_id=match_id,
            market=market,
            side=side,
            blended_probability=alert.get("blended_probability") or alert.get("model_probability", 0.0),
            offered_odds=alert["offered_odds"],
            edge_pct=alert["edge_pct"],
            recommended_stake=alert["recommended_stake"],
            fired_at=datetime.utcnow(),
        )
        session.add(db_alert)
        print(f"[ALERT] {alert}")
        return True


def create_scheduler():
    manager = LiveMatchManager(bankroll=Config.STARTING_BANKROLL)
    scheduler = BackgroundScheduler()
    scheduler.add_job(manager.poll_cycle, "interval",
                       seconds=Config.POLL_INTERVAL_SECONDS, id="live_poll")
    scheduler.start()
    return scheduler
