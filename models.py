"""
models.py
SQLAlchemy models. All tables prefixed 'bball_'.
Includes AdminSettings for runtime-adjustable parameters.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from config import Config

Base = declarative_base()


class LiveMatchSnapshot(Base):
    __tablename__ = "bball_live_match_snapshots"
    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True, nullable=False)
    league_name = Column(String)
    home_team = Column(String)
    away_team = Column(String)
    quarters_completed = Column(Integer)
    minutes_elapsed_current_q = Column(Float)
    current_total = Column(Integer)
    score_diff = Column(Integer)
    minutes_remaining = Column(Float)
    raw_odds_json = Column(JSON)
    polled_at = Column(DateTime, nullable=False)


class ValueBetAlert(Base):
    __tablename__ = "bball_value_bet_alerts"
    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True, nullable=False)
    market = Column(String, nullable=False)
    side = Column(String)
    blended_probability = Column(Float)
    offered_odds = Column(Float)
    edge_pct = Column(Float)
    recommended_stake = Column(Float)
    fired_at = Column(DateTime, nullable=False)
    closing_odds = Column(Float, nullable=True)
    settled_result = Column(Boolean, nullable=True)
    profit_loss = Column(Float, nullable=True)
    clv_pct = Column(Float, nullable=True)
    home_team = Column(String, nullable=True)
    away_team = Column(String, nullable=True)
    market_type = Column(String, nullable=True)
    settlement_meta = Column(JSON, nullable=True)


class TeamStrengthCache(Base):
    __tablename__ = "bball_team_strength_cache"
    id = Column(Integer, primary_key=True)
    team_name = Column(String, unique=True, index=True)
    wins = Column(Integer)
    losses = Column(Integer)
    updated_at = Column(DateTime, nullable=False)


class PaperTradingAccount(Base):
    __tablename__ = "bball_paper_trading_account"
    id = Column(Integer, primary_key=True)
    current_balance = Column(Float, nullable=False)
    starting_balance = Column(Float, nullable=False)
    total_bets_placed = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)
    total_losses = Column(Integer, default=0)
    updated_at = Column(DateTime, nullable=False)


class User(Base):
    __tablename__ = "bball_users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)


class APIKeySettings(Base):
    __tablename__ = "bball_api_key_settings"
    id = Column(Integer, primary_key=True)
    provider_name = Column(String, nullable=False, default="AllSportsAPI")
    api_key = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, nullable=False)


class LiveMatchStatus(Base):
    __tablename__ = "bball_live_match_status"
    id = Column(Integer, primary_key=True)
    match_id = Column(String, unique=True, index=True, nullable=False)
    home_team = Column(String)
    away_team = Column(String)
    league_name = Column(String)
    quarters_completed = Column(Integer)
    current_total = Column(Integer)
    score_diff = Column(Integer)
    has_active_alert = Column(Boolean, default=False)
    last_updated = Column(DateTime, nullable=False)


class RecentAlertKey(Base):
    """Database-backed dedup — survives process restarts."""
    __tablename__ = "bball_recent_alert_keys"
    id = Column(Integer, primary_key=True)
    alert_key = Column(String, unique=True, index=True, nullable=False)
    last_fired_at = Column(DateTime, nullable=False)


class AdminSettings(Base):
    """
    Admin-adjustable runtime settings stored in DB.
    Changes take effect on the next poll cycle without redeploying.
    """
    __tablename__ = "bball_admin_settings"
    id = Column(Integer, primary_key=True)
    ev_threshold = Column(Float, nullable=False, default=0.04)
    bankroll = Column(Float, nullable=False, default=1000000.0)
    poll_interval = Column(Integer, nullable=False, default=10)
    updated_at = Column(DateTime, nullable=False)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_or_create_paper_account(session, starting_balance: float):
    account = session.query(PaperTradingAccount).first()
    if account is None:
        account = PaperTradingAccount(
            current_balance=starting_balance, starting_balance=starting_balance,
            updated_at=datetime.utcnow(),
        )
        session.add(account)
        session.commit()
    return account


def get_or_create_api_key_settings(session):
    settings = session.query(APIKeySettings).first()
    if settings is None:
        settings = APIKeySettings(
            provider_name="AllSportsAPI", api_key=None,
            expires_at=None, is_active=False, updated_at=datetime.utcnow(),
        )
        session.add(settings)
        session.commit()
    return settings


def get_or_create_admin_settings(session, config):
    settings = session.query(AdminSettings).first()
    if settings is None:
        settings = AdminSettings(
            ev_threshold=float(getattr(config, 'EV_ALERT_THRESHOLD', 0.04)),
            bankroll=float(getattr(config, 'STARTING_BANKROLL', 1000000.0)),
            poll_interval=int(getattr(config, 'POLL_INTERVAL_SECONDS', 10)),
            updated_at=datetime.utcnow(),
        )
        session.add(settings)
        session.commit()
    return settings


def register_user_visit(session, username: str):
    user = session.query(User).filter_by(username=username).first()
    if user is None:
        user = User(username=username, first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow())
        session.add(user)
    else:
        user.last_seen = datetime.utcnow()
    session.commit()
    return user


# =====================================================
# ENGINE & SESSION
# =====================================================
engine = create_engine(Config.DATABASE_URL, **Config.SQLALCHEMY_ENGINE_OPTIONS)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _run_safe_migrations():
    migration_statements = [
        "ALTER TABLE bball_value_bet_alerts ADD COLUMN IF NOT EXISTS home_team VARCHAR",
        "ALTER TABLE bball_value_bet_alerts ADD COLUMN IF NOT EXISTS away_team VARCHAR",
        "ALTER TABLE bball_value_bet_alerts ADD COLUMN IF NOT EXISTS market_type VARCHAR",
        "ALTER TABLE bball_value_bet_alerts ADD COLUMN IF NOT EXISTS settlement_meta JSON",
    ]
    with engine.connect() as conn:
        for stmt in migration_statements:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                print(f"[migration] Skipped: {stmt[:60]} — {e}")


def init_db():
    Base.metadata.create_all(bind=engine)
    _run_safe_migrations()


def get_db_session():
    return SessionLocal()
