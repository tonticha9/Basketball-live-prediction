"""
scheduler.py
Polling loop with automatic settlement AND database-backed alert
deduplication (survives process restarts, unlike in-memory dicts).
Also drops the heavy raw_odds_json snapshot to reduce memory pressure
on low-RAM instances.
"""

import requests
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from config import Config
from models import (
    get_db_session, LiveMatchSnapshot, ValueBetAlert, LiveMatchStatus,
    RecentAlertKey, get_or_create_paper_account, get_or_create_api_key_settings,
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
        self._standings_cache_time = {}

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
            print(f"[_fetch] Response was not valid JSON. Raw text: {resp.text[:200]}")
            return {"result": {}}
        if not isinstance(data, dict):
            print(f"[_fetch] Unexpected response type ({type(data).__name__}).")
            return {"result": {}}
        if data.get("success") == 0:
            print(f"[_fetch] API returned success=0.")
        return data

    def _should_fire_alert(self, session, match_id, market, side, cooldown_seconds=300) -> bool:
        """
        Database-backed deduplication. Survives process restarts since
        it's stored in Postgres, not Python memory.
        """
        key = f"{match_id}:{market}:{side}"
        now = datetime.utcnow()

        existing = session.query(RecentAlertKey).filter_by(alert_key=key).first()
        if existing:
            if (now - existing.last_fired_at).total_seconds() < cooldown_seconds:
                return False
            existing.last_fired_at = now
        else:
            session.add(RecentAlertKey(alert_key=key, last_fired_at=now))

        return True

    def _cleanup_old_dedup_keys(self, session, max_age_hours=6):
        """Prevents the dedup table from growing unbounded over time."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        session.query(RecentAlertKey).filter(RecentAlertKey.last_fired_at < cutoff).delete()

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

        all_events = [e for e in livescore_data.get("result", []) if isinstance(e, dict)]
        live_events = [e for e in all_events if AllSportsLivescoreParser.is_live(e)]
        finished_events = [e for e in all_events if str(e.get("event_status", "")).strip() == "Finished"]

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

        try:
            self._settle_finished_matches(finished_events)
        except Exception as e:
            print(f"[poll_cycle] Settlement error: {e}")

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
        now = datetime.utcnow()
        cache_time = self._standings_cache_time.get(league_key)

        # Refresh standings at most once per hour per league (reduces both
        # API calls and memory churn from repeatedly storing large payloads)
        if league_key not in self._standings_cache or (cache_time and (now - cache_time).total_seconds() > 3600):
            try:
                self._standings_cache[league_key] = self._fetch(
                    {"met": "Standings", "leagueId": league_key}, api_key
                )
                self._standings_cache_time[league_key] = now
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

        home_team = match_state.get("home_team")
        away_team = match_state.get("away_team")

        session = get_db_session()
        try:
            status_row = session.query(LiveMatchStatus).filter_by(match_id=match_id).first()
            if status_row is None:
                status_row = LiveMatchStatus(match_id=match_id, last_updated=datetime.utcnow())
                session.add(status_row)

            status_row.home_team = home_team
            status_row.away_team = away_team
            status_row.league_name = match_state.get("league_name")
            status_row.quarters_completed = match_state["quarters_completed"]
            status_row.current_total = match_state["current_total"]
            status_row.score_diff = match_state["score_diff"]
            status_row.last_updated = datetime.utcnow()
            has_alert_this_cycle = False

            # NOTE: raw_odds_json intentionally NOT stored here anymore —
            # the full odds payload (hundreds of bookmaker entries) was a
            # major source of memory pressure on the low-RAM instance.
            snapshot = LiveMatchSnapshot(
                match_id=match_id, league_name=match_state.get("league_name"),
                home_team=home_team, away_team=away_team,
                quarters_completed=match_state["quarters_completed"],
                minutes_elapsed_current_q=match_state["minutes_elapsed_current_q"],
                current_total=match_state["current_total"], score_diff=match_state["score_diff"],
                minutes_remaining=match_state["minutes_remaining"],
                raw_odds_json=None, polled_at=datetime.utcnow(),
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
                if alert and self._save_alert(session, alert, match_id, home_team, away_team,
                                               "full_game_ha", None):
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
                    if q_alert and self._save_alert(session, q_alert, match_id, home_team, away_team,
                                                     "quarter_ha", {"quarter_number": current_q}):
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
                    if p_alert and self._save_alert(
                        session, p_alert, match_id, home_team, away_team,
                        "player_points", {"player_name": player["player"], "threshold": m["threshold"]},
                    ):
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
                if oe_alert and self._save_alert(session, oe_alert, match_id, home_team, away_team,
                                                  "odd_even", None):
                    has_alert_this_cycle = True

            hsq_odds = AllSportsOddsParser.get_highest_scoring_quarter_odds(odds_data, match_id)
            if hsq_odds and match_state["completed_quarters"]:
                completed_totals = {
                    i + 1: q["home"] + q["away"]
                    for i, q in enumerate(match_state["completed_quarters"])
                }
                hsq_alert = match_bundle["hsq_orch"].evaluate(completed_totals, hsq_odds)
                if hsq_alert and self._save_alert(session, hsq_alert, match_id, home_team, away_team,
                                                   "hsq", None):
                    has_alert_this_cycle = True

            status_row.has_active_alert = has_alert_this_cycle
            self._cleanup_old_dedup_keys(session)
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

    def _save_alert(self, session, alert, match_id, home_team, away_team, market_type, settlement_meta):
        market = alert["market"]
        side = alert.get("side") or alert.get("player", "")

        if not self._should_fire_alert(session, match_id, market, side):
            return False

        db_alert = ValueBetAlert(
            match_id=match_id, market=market, side=side,
            blended_probability=alert.get("blended_probability") or alert.get("model_probability", 0.0),
            offered_odds=alert["offered_odds"], edge_pct=alert["edge_pct"],
            recommended_stake=alert["recommended_stake"], fired_at=datetime.utcnow(),
            home_team=home_team, away_team=away_team,
            market_type=market_type, settlement_meta=settlement_meta,
        )
        session.add(db_alert)
        print(f"[ALERT] {home_team} vs {away_team} — {alert}")
        return True

    def _settle_finished_matches(self, finished_events):
        if not finished_events:
            return

        session = get_db_session()
        try:
            for event in finished_events:
                match_id = str(event.get("event_key"))
                pending = (
                    session.query(ValueBetAlert)
                    .filter_by(match_id=match_id, settled_result=None)
                    .all()
                )
                if not pending:
                    continue

                final_result = str(event.get("event_final_result", "")).strip()
                try:
                    home_final_str, away_final_str = final_result.split("-")
                    home_final = int(home_final_str.strip())
                    away_final = int(away_final_str.strip())
                except (ValueError, AttributeError):
                    continue

                quarters = AllSportsLivescoreParser.extract_all_quarters_for_finished(event)
                player_stats = AllSportsPlayerStatsParser.get_all_players_live_stats(event)

                for alert in pending:
                    won = self._determine_result(alert, home_final, away_final, quarters, player_stats)
                    if won is None:
                        continue
                    self._apply_settlement(session, alert, won)

            session.commit()
        finally:
            session.close()

    def _determine_result(self, alert, home_final, away_final, quarters, player_stats):
        mtype = alert.market_type

        if mtype == "full_game_ha":
            if alert.side == "Home":
                return home_final > away_final
            elif alert.side == "Away":
                return away_final > home_final
            return None

        if mtype == "quarter_ha":
            meta = alert.settlement_meta or {}
            q_num = meta.get("quarter_number")
            if not q_num or q_num < 1 or q_num > 4:
                return None
            q = quarters[q_num - 1]
            if q is None:
                return None
            if alert.side == "Home":
                return q["home"] > q["away"]
            elif alert.side == "Away":
                return q["away"] > q["home"]
            return None

        if mtype == "odd_even":
            total = home_final + away_final
            is_even = (total % 2 == 0)
            if alert.side == "Even":
                return is_even
            elif alert.side == "Odd":
                return not is_even
            return None

        if mtype == "hsq":
            valid_quarters = [(i + 1, q["home"] + q["away"]) for i, q in enumerate(quarters) if q is not None]
            if not valid_quarters:
                return None
            best_q = max(valid_quarters, key=lambda x: x[1])[0]
            predicted_q = (alert.side or "").replace("Q", "")
            try:
                return int(predicted_q) == best_q
            except ValueError:
                return None

        if mtype == "player_points":
            meta = alert.settlement_meta or {}
            player_name = meta.get("player_name")
            threshold = meta.get("threshold")
            if not player_name or threshold is None:
                return None
            for p in player_stats:
                if p["player"].strip().lower() == player_name.strip().lower():
                    return p["points"] > threshold
            return None

        return None

    def _apply_settlement(self, session, alert, won: bool):
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)

        profit_loss = (
            alert.recommended_stake * (alert.offered_odds - 1)
            if won else -alert.recommended_stake
        )

        alert.settled_result = won
        alert.profit_loss = round(profit_loss, 2)

        account.current_balance = round(account.current_balance + profit_loss, 2)
        account.total_bets_placed += 1
        if won:
            account.total_wins += 1
        else:
            account.total_losses += 1
        account.updated_at = datetime.utcnow()

        print(f"[AUTO-SETTLE] Alert #{alert.id} ({alert.market}) -> "
              f"{'WON' if won else 'LOST'}, P/L: {profit_loss:.2f}")


def create_scheduler():
    manager = LiveMatchManager(bankroll=Config.STARTING_BANKROLL)
    scheduler = BackgroundScheduler()
    scheduler.add_job(manager.poll_cycle, "interval",
                       seconds=Config.POLL_INTERVAL_SECONDS, id="live_poll")
    scheduler.start()
    return scheduler
