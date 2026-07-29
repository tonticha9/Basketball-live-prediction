"""
models.py
SQLAlchemy models. All tables prefixed 'bball_' to safely coexist
with your existing tennis tables in the same Render Postgres database.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
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


# --- Engine & Session setup ---
engine = create_engine(Config.DATABASE_URL, **Config.SQLALCHEMY_ENGINE_OPTIONS)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Creates all bball_ tables if they don't exist yet. Safe to call
    repeatedly — won't touch or affect any existing tennis tables."""
    Base.metadata.create_all(bind=engine)


def get_db_session():
    return SessionLocal()
