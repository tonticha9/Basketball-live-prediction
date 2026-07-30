"""
models.py
SQLAlchemy models. All tables prefixed 'bball_' to safely coexist
with your existing tennis tables in the same Render Postgres database.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON
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


def register_user_visit(session, username: str):
    user = session.query(User).filter_by(username=username).first()
    if user is None:
        user = User(username=username, first_seen=datetime.utcnow(), last_seen=datetime.utcnow())
        session.add(user)
    else:
        user.last_seen = datetime.utcnow()
    session.commit()
    return user


# --- Engine & Session setup ---
engine = create_engine(Config.DATABASE_URL, **Config.SQLALCHEMY_ENGINE_OPTIONS)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Creates all bball_ tables if they don't exist yet. Safe to call
    repeatedly — won't touch or affect any existing tennis tables."""
    Base.metadata.create_all(bind=engine)


def get_db_session():
    return SessionLocal()
