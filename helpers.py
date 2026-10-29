"""
utils/helpers.py
Shared utility functions used across all modules.
"""

import os
import json
from datetime import datetime


# ── Path helpers ─────────────────────────────────────────────────────────────

# Project root is two levels up from this file (utils/helpers.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS_DIR    = os.path.join(PROJECT_ROOT, "logs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "output")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")


def ensure_dirs():
    """Create any missing runtime directories."""
    for path in [LOGS_DIR, REPORTS_DIR, MODELS_DIR]:
        os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """Return a readable timestamp string: 2024-05-16 14:30:00"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_log_path(filename: str) -> str:
    """Return the full path to a file inside the logs directory."""
    return os.path.join(LOGS_DIR, filename)


def get_report_path(filename: str) -> str:
    """Return the full path to a file inside the reports/output directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return os.path.join(REPORTS_DIR, filename)


# ── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(filepath: str) -> dict:
    """Load a JSON file; return empty dict if file missing or corrupt."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(filepath: str, data: dict) -> None:
    """Save a dictionary to a JSON file (pretty-printed)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# ── Display helpers ──────────────────────────────────────────────────────────

def print_banner():
    """Print a styled startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║      AI Smart Exam Monitoring & Proctoring System        ║
║      SP24-BAI-016 | SP24-BAI-017 | SP24-BAI-012          ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title: str):
    """Print a formatted section separator."""
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def print_warning(message: str):
    """Print a formatted warning message."""
    print(f"  ⚠  WARNING: {message}")


def print_success(message: str):
    """Print a formatted success message."""
    print(f"  ✔  {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"  ℹ  {message}")
