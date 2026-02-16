"""API module for multi-agent data assistant.

This module provides FastAPI endpoints for the analyst loop.
"""

from haikugraph.api.server import app, create_app

__all__ = ["app", "create_app"]
