"""Application configuration.

Priority order for each value:
  1. st.secrets  — Streamlit Cloud dashboard secrets (and local .streamlit/secrets.toml)
  2. os.environ  — environment variables / .env file loaded by dotenv

config values are intentionally read lazily via get() so they always
reflect the current runtime state, whether called from a Streamlit
context or a plain Python process (tests, CLI).
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env for local development — no-op if file absent or dotenv not installed
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """Return config value, checking st.secrets then os.environ."""
    # 1. Streamlit secrets (works on Cloud and locally with secrets.toml)
    try:
        import streamlit as st
        # st.secrets raises an exception if called outside a Streamlit session
        # (e.g. during pytest). The try/except handles that safely.
        val = st.secrets.get(key, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    except Exception:  # noqa: BLE001
        pass

    # 2. Environment variable / .env
    val = os.environ.get(key, default)
    return val if val is not None else default


# ── Public API ────────────────────────────────────────────────────────────────
# These are module-level properties so existing code (from config import X)
# keeps working. They call _get() at access time via a thin wrapper class.

class _LazyConfig:
    """Descriptor-based lazy config so values are read at access time, not import time."""

    __slots__ = ()

    @property
    def MP_API_KEY(self) -> str:
        return _get("MP_API_KEY", "")

    @property
    def BRENDA_EMAIL(self) -> str:
        return _get("BRENDA_EMAIL", "")

    @property
    def BRENDA_PASSWORD(self) -> str:
        return _get("BRENDA_PASSWORD", "")

    @property
    def S2_API_KEY(self) -> str:
        return _get("S2_API_KEY", "")

    @property
    def OCP_DATA_DIR(self) -> str:
        return _get("OCP_DATA_DIR", str(Path("./data").resolve()))

    @property
    def CACHE_DB_PATH(self) -> str:
        return _get("CACHE_DB_PATH", str(Path("./query_cache.sqlite").resolve()))

    @property
    def DATABASE_URL(self) -> str:
        return _get("DATABASE_URL", "sqlite:///catalystiq.db")

    @property
    def LOG_LEVEL(self) -> str:
        return _get("LOG_LEVEL", "INFO")


# Replace this module with the lazy config instance so attribute access
# (import config; config.MP_API_KEY) reads live values every time.
import sys as _sys
_sys.modules[__name__] = _LazyConfig()  # type: ignore[assignment]
