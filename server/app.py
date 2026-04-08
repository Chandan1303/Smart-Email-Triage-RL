"""
server/app.py — entry point for multi-mode deployment (openenv validate requirement).
"""
import sys
import os

# Ensure root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
