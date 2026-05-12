"""Application configuration.

On Streamlit Cloud, secrets are injected via st.secrets.
We copy them into os.environ at import time so all downstream
code that does `from config import X` gets the live values.

On local dev, values come from .env via python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Step 1: load .env for local dev ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ── Step 2: copy Streamlit secrets into os.environ ───────────────────────────
# This runs every time config is imported. On Streamlit Cloud st.secrets
# is available immediately. On local dev without secrets.toml it silently
# does nothing. Either way os.environ ends up with the right values.
try:
    import streamlit as st
    for _key, _val in st.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
except Exception:
    pass

# ── Step 3: read from os.environ ─────────────────────────────────────────────
MP_API_KEY: str      = os.environ.get("MP_API_KEY", "")
BRENDA_EMAIL: str    = os.environ.get("BRENDA_EMAIL", "")
BRENDA_PASSWORD: str = os.environ.get("BRENDA_PASSWORD", "")
S2_API_KEY: str      = os.environ.get("S2_API_KEY", "")
OCP_DATA_DIR: str    = os.environ.get("OCP_DATA_DIR", str(Path("./data").resolve()))
CACHE_DB_PATH: str   = os.environ.get("CACHE_DB_PATH", str(Path("./query_cache.sqlite").resolve()))
DATABASE_URL: str    = os.environ.get("DATABASE_URL", "sqlite:///catalystiq.db")
LOG_LEVEL: str       = os.environ.get("LOG_LEVEL", "INFO")
