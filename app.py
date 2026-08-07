"""
app.py
Main Flask app: dashboard, history (with charts), settings (with
admin controls for EV threshold, bankroll, poll interval).
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from collections import defaultdict

from config import Config
from models import (
    init_db, get_db_session,
    LiveMatchSnapshot, ValueBetAlert, LiveMatchStatus,
    PaperTradingAccount, get_or_create_paper_account,
    APIKeySettings, get_or_create_api_key_settings,
    AdminSettings, get_or_create_admin_settings,
    User, register_user_visit,
)

app = Flask(__name__)
init_db()


# =====================================================
# DASHBOARD
# =====================================================
@app.route("/")
def dashboard():
    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        overall_change = round(account.current_balance - account.starting_balance, 2)

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        todays_settled = (
            session.query(ValueBetAlert)
            .filter(ValueBetAlert.settled_result.isnot(None))
            .filter(ValueBetAlert.fired_at >= today_start)
            .all()
        )
        todays_change = round(sum(a.profit_loss or 0 for a in todays_settled), 2)

        recent_alerts = (
            session.query(ValueBetAlert)
            .order_by(ValueBetAlert.fired_at.desc())
            .limit(15)
            .all()
        )
        live_matches = (
            session.query(LiveMatchStatus)
            .order_by(LiveMatchStatus.last_updated.desc())
            .all()
        )

        return render_template(
            "dashboard.html",
            account=account, overall_change=overall_change,
            todays_change=todays_change, alerts=recent_alerts,
            live_matches=live_matches,
        )
    finally:
        session.close()


@app.route("/api/dashboard_data")
def api_dashboard_data():
    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        overall_change = round(account.current_balance - account.starting_balance, 2)

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        todays_settled = (
            session.query(ValueBetAlert)
            .filter(ValueBetAlert.settled_result.isnot(None))
            .filter(ValueBetAlert.fired_at >= today_start)
            .all()
        )
        todays_change = round(sum(a.profit_loss or 0 for a in todays_settled), 2)

        alerts = (
            session.query(ValueBetAlert)
            .order_by(ValueBetAlert.fired_at.desc())
            .limit(15)
            .all()
        )
        live_matches = (
            session.query(LiveMatchStatus)
            .order_by(LiveMatchStatus.last_updated.desc())
            .all()
        )

        win_rate = (
            round(account.total_wins / account.total_bets_placed * 100, 1)
            if account.total_bets_placed > 0 else 0
        )

        return jsonify({
            "balance": account.current_balance,
            "starting_balance": account.starting_balance,
            "overall_change": overall_change,
            "todays_change": todays_change,
            "total_bets": account.total_bets_placed,
            "win_rate": win_rate,
            "alerts": [{
                "id": a.id, "match_id": a.match_id,
                "home_team": a.home_team or "?", "away_team": a.away_team or "?",
                "market": a.market, "side": a.side, "market_type": a.market_type,
                "probability": a.blended_probability, "odds": a.offered_odds,
                "edge_pct": a.edge_pct, "stake": a.recommended_stake,
                "settled_result": a.settled_result, "profit_loss": a.profit_loss,
                "fired_at": a.fired_at.strftime("%Y-%m-%d %H:%M:%S"),
            } for a in alerts],
            "live_matches": [{
                "match_id": m.match_id, "home_team": m.home_team, "away_team": m.away_team,
                "league_name": m.league_name, "quarters_completed": m.quarters_completed,
                "current_total": m.current_total, "score_diff": m.score_diff,
                "has_active_alert": m.has_active_alert,
                "last_updated": m.last_updated.strftime("%Y-%m-%d %H:%M:%S"),
            } for m in live_matches],
        })
    finally:
        session.close()


# =====================================================
# HISTORY
# =====================================================
@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/api/history_data")
def api_history_data():
    days_back = int(request.args.get("days", 30))
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)

        settled = (
            session.query(ValueBetAlert)
            .filter(ValueBetAlert.settled_result.isnot(None))
            .filter(ValueBetAlert.fired_at >= cutoff)
            .order_by(ValueBetAlert.fired_at.asc())
            .all()
        )

        daily_groups = defaultdict(list)
        for a in settled:
            day_key = a.fired_at.strftime("%Y-%m-%d")
            daily_groups[day_key].append(a)

        daily_summary = []
        for day_key in sorted(daily_groups.keys(), reverse=True):
            bets = daily_groups[day_key]
            day_pl = round(sum(b.profit_loss or 0 for b in bets), 2)
            day_wins = sum(1 for b in bets if b.settled_result)
            day_win_rate = round(day_wins / len(bets) * 100, 1) if bets else 0

            daily_summary.append({
                "date": day_key, "n_bets": len(bets), "wins": day_wins,
                "losses": len(bets) - day_wins, "win_rate": day_win_rate,
                "profit_loss": day_pl,
                "bets": [{
                    "id": b.id, "home_team": b.home_team or "?",
                    "away_team": b.away_team or "?", "market": b.market,
                    "side": b.side, "odds": b.offered_odds, "stake": b.recommended_stake,
                    "won": b.settled_result, "profit_loss": b.profit_loss,
                    "time": b.fired_at.strftime("%H:%M"),
                } for b in bets],
            })

        total_staked = sum(b.recommended_stake or 0 for b in settled)
        total_pl = sum(b.profit_loss or 0 for b in settled)
        total_wins = sum(1 for b in settled if b.settled_result)
        overall_win_rate = round(total_wins / len(settled) * 100, 1) if settled else 0
        overall_roi = round((total_pl / total_staked) * 100, 2) if total_staked > 0 else 0

        return jsonify({
            "daily_summary": daily_summary,
            "overall": {
                "n_bets": len(settled), "wins": total_wins,
                "losses": len(settled) - total_wins,
                "win_rate": overall_win_rate,
                "total_profit_loss": round(total_pl, 2),
                "roi_pct": overall_roi,
                "current_balance": account.current_balance,
                "starting_balance": account.starting_balance,
            }
        })
    finally:
        session.close()


# =====================================================
# SETTINGS
# =====================================================
@app.route("/settings")
def settings_page():
    session = get_db_session()
    try:
        register_user_visit(session, request.remote_addr or "unknown")
        api_settings = get_or_create_api_key_settings(session)
        admin = get_or_create_admin_settings(session, Config)
        users = session.query(User).order_by(User.last_seen.desc()).limit(50).all()
        return render_template("settings.html",
                               api_settings=api_settings,
                               admin=admin,
                               users=users)
    finally:
        session.close()


@app.route("/api/settings/update_api_key", methods=["POST"])
def update_api_key():
    data = request.get_json(silent=True) or {}
    new_key = data.get("api_key", "").strip()
    expires_in_days = int(data.get("expires_in_days", 30))

    if not new_key:
        return jsonify({"error": "API key haiwezi kuwa tupu"}), 400

    session = get_db_session()
    try:
        settings = get_or_create_api_key_settings(session)
        settings.api_key = new_key
        settings.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        settings.is_active = True
        settings.updated_at = datetime.utcnow()
        session.commit()
        return jsonify({
            "success": True,
            "expires_at": settings.expires_at.strftime("%Y-%m-%d %H:%M"),
        })
    finally:
        session.close()


@app.route("/api/settings/clear_api_key", methods=["POST"])
def clear_api_key():
    session = get_db_session()
    try:
        settings = get_or_create_api_key_settings(session)
        settings.api_key = None
        settings.expires_at = None
        settings.is_active = False
        settings.updated_at = datetime.utcnow()
        session.commit()
        return jsonify({"success": True})
    finally:
        session.close()


@app.route("/api/settings/status")
def api_key_status():
    session = get_db_session()
    try:
        settings = get_or_create_api_key_settings(session)
        expired = (settings.expires_at is not None and
                   settings.expires_at < datetime.utcnow())
        return jsonify({
            "has_key": settings.api_key is not None,
            "is_active": settings.is_active and not expired,
            "expired": expired,
            "expires_at": settings.expires_at.strftime("%Y-%m-%d %H:%M")
            if settings.expires_at else None,
        })
    finally:
        session.close()


@app.route("/api/settings/update_admin", methods=["POST"])
def update_admin_settings():
    data = request.get_json(silent=True) or {}
    ev = data.get("ev_threshold")
    bankroll = data.get("bankroll")
    poll = data.get("poll_interval")

    if ev is None or bankroll is None or poll is None:
        return jsonify({"error": "Missing fields"}), 400

    try:
        ev = float(ev)
        bankroll = float(bankroll)
        poll = int(poll)
    except (ValueError, TypeError):
        return jsonify({"error": "Thamani si sahihi"}), 400

    if not (0.01 <= ev <= 0.20):
        return jsonify({"error": "EV threshold lazima iwe 0.01–0.20"}), 400
    if bankroll < 1000:
        return jsonify({"error": "Bankroll lazima iwe angalau 1000"}), 400
    if not (10 <= poll <= 120):
        return jsonify({"error": "Poll interval lazima iwe 10–120 sekunde"}), 400

    session = get_db_session()
    try:
        settings = get_or_create_admin_settings(session, Config)
        settings.ev_threshold = ev
        settings.bankroll = bankroll
        settings.poll_interval = poll
        settings.updated_at = datetime.utcnow()
        session.commit()
        return jsonify({
            "success": True,
            "ev_threshold": settings.ev_threshold,
            "bankroll": settings.bankroll,
            "poll_interval": settings.poll_interval,
        })
    finally:
        session.close()


@app.route("/api/settings/users")
def api_users_list():
    session = get_db_session()
    try:
        users = session.query(User).order_by(User.last_seen.desc()).limit(50).all()
        return jsonify([{
            "username": u.username,
            "first_seen": u.first_seen.strftime("%Y-%m-%d %H:%M"),
            "last_seen": u.last_seen.strftime("%Y-%m-%d %H:%M"),
        } for u in users])
    finally:
        session.close()


@app.route("/api/reset_paper_account", methods=["POST"])
def reset_paper_account():
    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        account.current_balance = account.starting_balance
        account.total_bets_placed = 0
        account.total_wins = 0
        account.total_losses = 0
        account.updated_at = datetime.utcnow()
        session.commit()
        return jsonify({"success": True})
    finally:
        session.close()


@app.route("/api/full_reset", methods=["POST"])
def full_reset():
    session = get_db_session()
    try:
        deleted_count = session.query(ValueBetAlert).delete()
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        account.current_balance = account.starting_balance
        account.total_bets_placed = 0
        account.total_wins = 0
        account.total_losses = 0
        account.updated_at = datetime.utcnow()
        session.commit()
        return jsonify({
            "success": True,
            "deleted_alerts": deleted_count,
            "new_balance": account.current_balance,
        })
    finally:
        session.close()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


from scheduler import create_scheduler
scheduler = create_scheduler()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
