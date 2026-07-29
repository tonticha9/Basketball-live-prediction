"""
config.py
Central configuration — reads from environment variables (set these
in Render's dashboard under your service's "Environment" tab).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "").replace(
        "postgres://", "postgresql://", 1
    )

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 5,
        "max_overflow": 2,
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    ALLSPORTS_API_KEY = os.environ.get("ALLSPORTS_API_KEY", "")
    ALLSPORTS_BASE_URL = "https://apiv2.allsportsapi.com/basketball/"

    STARTING_BANKROLL = float(os.environ.get("STARTING_BANKROLL", "1000.0"))
    EV_ALERT_THRESHOLD = float(os.environ.get("EV_ALERT_THRESHOLD", "0.03"))
    POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
