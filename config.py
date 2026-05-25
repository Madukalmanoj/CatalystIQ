"""Application configuration — reads from Streamlit secrets (cloud) or .env (local)."""

from __future__ import annotations

import os
from pathlib import Path


def _get(key: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets first, then env vars.

    Streamlit Cloud injects secrets as st.secrets, which also populates
    os.environ automatically for secrets defined in the TOML file.
    We load .env for local development as a fallback.
    """
    # Try Streamlit secrets (available on Streamlit Cloud and locally
    # if .streamlit/secrets.toml exists)
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass

    # Fall back to environment variable / .env file
    return os.getenv(key, default)


# Load .env for local development (no-op if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MP_API_KEY: str      = _get("MP_API_KEY"), GEMINI_API_KEY= Agvflgfisuhjsvks
BRENDA_EMAIL: str    = _get("BRENDA_EMAIL")
BRENDA_PASSWORD: str = _get("BRENDA_PASSWORD")
S2_API_KEY: str      = _get("S2_API_KEY")
OCP_DATA_DIR: str    = _get("OCP_DATA_DIR", str(Path("./data").resolve()))
CACHE_DB_PATH: str   = _get("CACHE_DB_PATH", str(Path("./query_cache.sqlite").resolve()))
DATABASE_URL: str    = _get("DATABASE_URL", "sqlite:///catalystiq.db")
LOG_LEVEL: str       = _get("LOG_LEVEL", "INFO")
