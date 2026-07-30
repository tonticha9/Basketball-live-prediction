"""
app.py
Main Flask application: dashboard UI + paper trading + scheduler bootstrap.
No Telegram — dashboard-only alerts for now.
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime

from config import Config
from models import (
    init_db, get_db_session, LiveMatchSnapshot, ValueBetAlert,
    PaperTradingAccount, get_or_create_paper_account
)

app = Flask(__name__)

# Creates bball_ tables if they don't exist yet.
# Safe to run repeatedly — does not touch or affect existing tennis tables.
init_db()


# =========================================================
# DASHBOARD ROUTES
# =========================================================
@app.route("/")
def dashboard():
    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        recent_alerts = (
            session.query(ValueBetAlert)
            .order_by(ValueBetAlert.fired_at.desc())
            .limit(25)
            .all()
        )
        return render_template(
            "dashboard.html",
            account=account,
            alerts=recent_alerts,
        )
    finally:
        session.close()


@app.route("/api/alerts")
def api_alerts():
    """JSON feed the dashboard can poll to auto-refresh without full page reload."""
    session = get_db_session()
    try:
        alerts = (
            session.query(ValueBetAlert)
            .order_by(ValueBetAlert.fired_at.desc())
            .limit(25)
            .all()
        )
        return jsonify([{
            "id": a.id,
            "match_id": a.match_id,
            "market": a.market,
            "side": a.side,
            "probability": a.blended_probability,
            "odds": a.offered_odds,
            "edge_pct": a.edge_pct,
            "stake": a.recommended_stake,
            "settled_result": a.settled_result,
            "profit_loss": a.profit_loss,
            "fired_at": a.fired_at.strftime("%Y-%m-%d %H:%M:%S"),
        } for a in alerts])
    finally:
        session.close()


@app.route("/api/settle_bet/<int:alert_id>", methods=["POST"])
def settle_bet(alert_id):
    """Manually mark a paper-trading alert as won/lost — updates fake balance."""
    data = request.get_json(silent=True) or {}
    won = data.get("won", False)

    session = get_db_session()
    try:
        alert = session.query(ValueBetAlert).filter_by(id=alert_id).first()
        if alert is None:
            return jsonify({"error": "Alert not found"}), 404

        if alert.settled_result is not None:
            return jsonify({"error": "Alert already settled"}), 400

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

        session.commit()
        return jsonify({"success": True, "new_balance": account.current_balance})
    finally:
        session.close()


@app.route("/api/account_summary")
def account_summary():
    session = get_db_session()
    try:
        account = get_or_create_paper_account(session, Config.STARTING_BANKROLL)
        win_rate = (
            round(account.total_wins / account.total_bets_placed * 100, 1)
            if account.total_bets_placed > 0 else 0
        )
        roi_pct = round(
            (account.current_balance - account.starting_balance)
            / account.starting_balance * 100, 2
        ) if account.starting_balance > 0 else 0

        return jsonify({
            "current_balance": account.current_balance,
            "starting_balance": account.starting_balance,
            "total_bets": account.total_bets_placed,
            "total_wins": account.total_wins,
            "total_losses": account.total_losses,
            "win_rate": win_rate,
            "roi_pct": roi_pct,
        })
    finally:
        session.close()


@app.route("/api/reset_paper_account", methods=["POST"])
def reset_paper_account():
    """Resets the paper trading balance back to the starting bankroll.
    Does NOT delete past alert history — only resets the fake balance."""
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


@app.route("/health")
def health():
    """Simple healthcheck endpoint for Render."""
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


# =========================================================
# SCHEDULER BOOTSTRAP — starts live polling in the background
# =========================================================
from scheduler import create_scheduler
scheduler = create_scheduler()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
