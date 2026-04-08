"""
server/app.py — entry point for multi-mode deployment (openenv validate requirement).
Imports and re-exports the FastAPI app from the root app.py.
"""
import sys
import os

# Ensure root is on the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 — re-export for openenv server entry point
