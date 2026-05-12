"""Application configuration.

On Streamlit Cloud, secrets defined in the dashboard are automatically
injected into os.environ before the app starts — so plain os.getenv()
is all that is needed.

For local development, values are loaded from .env via python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Load .env for local development ──────────────────────────────────────────
# This is a no-op when the file doesn't exist (e.g. on Streamlit Cloud).
# Must happen before any os.getenv() calls below.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # don't override values already in os.environ
except ImportError:
    pass

# ── Config values ─────────────────────────────────────────────────────────────
# Read directly from os.environ.  Streamlit Cloud injects secrets here
# automatically; local dev gets them from .env loaded above.

MP_API_KEY: str      = os.environ.get("MP_API_KEY", "")
BRENDA_EMAIL: str    = os.environ.get("BRENDA_EMAIL", "")
BRENDA_PASSWORD: str = os.environ.get("BRENDA_PASSWORD", "")
S2_API_KEY: str      = os.environ.get("S2_API_KEY", "")
OCP_DATA_DIR: str    = os.environ.get("OCP_DATA_DIR", str(Path("./data").resolve()))
CACHE_DB_PATH: str   = os.environ.get("CACHE_DB_PATH", str(Path("./query_cache.sqlite").resolve()))
DATABASE_URL: str    = os.environ.get("DATABASE_URL", "sqlite:///catalystiq.db")
LOG_LEVEL: str       = os.environ.get("LOG_LEVEL", "INFO")
