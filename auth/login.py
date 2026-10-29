"""
auth/login.py
─────────────────────────────────────────────────────────────
Student Authentication Module

Syllabus Coverage:
  • Regex             – Roll number format validation
  • Exception Handling – Invalid input, file errors
  • File Handling      – Saving student info to JSON
  • OOP               – StudentSession class
"""

import re
import json
import os
from datetime import datetime

# Import shared utilities using a relative path that works from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    get_log_path, save_json, load_json,
    print_section, print_success, print_warning, get_timestamp
)


# ── Regex Pattern ────────────────────────────────────────────────────────────
# Matches roll numbers like: SP24-BAI-016
ROLL_NUMBER_PATTERN = r"^SP\d{2}-BAI-\d{3}$"


# ── OOP: StudentSession Class ────────────────────────────────────────────────

class StudentSession:
    """
    Represents a single student's proctoring session.

    Attributes:
        name        (str)  : Student full name
        roll_number (str)  : Validated roll number
        session_id  (str)  : Unique session identifier
        start_time  (str)  : Session start timestamp
        warnings    (dict) : Detection counts per category
        risk_score  (int)  : Final computed cheating risk score
    """

    def __init__(self, name: str, roll_number: str):
        self.name         = name.strip().title()
        self.roll_number  = roll_number.strip().upper()
        self.session_id   = f"{self.roll_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time   = get_timestamp()
        self.end_time     = None
        self.risk_score   = 0

        # Mutable warning counts (using dict as required by syllabus)
        self.warnings = {
            "phone_detected":     0,
            "multiple_persons":   0,
            "looking_away":       0,
            "face_missing":       0,
            "total_alerts":       0,
        }

        # Timeline list for regression analysis
        self.timeline: list = []   # list of (second, cumulative_warnings)

    def add_warning(self, category: str, detail: str = ""):
        """
        Increment warning count for a given category and
        append an entry to the session timeline.
        """
        if category in self.warnings:
            self.warnings[category] += 1
            self.warnings["total_alerts"] += 1

        # Timeline entry for regression: (elapsed_seconds, cumulative_count)
        elapsed = self._get_elapsed_seconds()
        self.timeline.append({
            "second":     elapsed,
            "category":   category,
            "detail":     detail,
            "cumulative": self.warnings["total_alerts"],
            "timestamp":  get_timestamp(),
        })

    def _get_elapsed_seconds(self) -> int:
        """Return seconds elapsed since session start."""
        try:
            start = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
            delta = datetime.now() - start
            return int(delta.total_seconds())
        except Exception:
            return 0

    def to_dict(self) -> dict:
        """Serialize session data to a dictionary for JSON storage."""
        return {
            "session_id":   self.session_id,
            "name":         self.name,
            "roll_number":  self.roll_number,
            "start_time":   self.start_time,
            "end_time":     self.end_time,
            "warnings":     self.warnings,
            "risk_score":   self.risk_score,
            "timeline":     self.timeline,
        }

    def save(self):
        """Save this session to logs/sessions.json."""
        self.end_time = get_timestamp()
        filepath = get_log_path("sessions.json")

        # Load existing sessions
        all_sessions = load_json(filepath)
        all_sessions[self.session_id] = self.to_dict()

        save_json(filepath, all_sessions)
        print_success(f"Session saved → {filepath}")

    def __repr__(self):
        return (
            f"StudentSession("
            f"name={self.name!r}, "
            f"roll={self.roll_number!r}, "
            f"alerts={self.warnings['total_alerts']})"
        )


# ── Validation Functions ─────────────────────────────────────────────────────

def validate_roll_number(roll_number: str) -> bool:
    """
    Validate roll number against Regex pattern.
    Returns True if valid, False otherwise.

    Valid format: SP24-BAI-016
    """
    pattern = re.compile(ROLL_NUMBER_PATTERN)
    return bool(pattern.match(roll_number.strip().upper()))


def validate_name(name: str) -> bool:
    """
    Validate that the name:
      - is not empty
      - contains only letters and spaces
      - is at least 2 characters
    """
    name = name.strip()
    if len(name) < 2:
        return False
    # Allow letters and spaces only
    return bool(re.match(r"^[A-Za-z\s]+$", name))


# ── Login Function ────────────────────────────────────────────────────────────

def run_login() -> StudentSession:
    """
    Interactive login prompt.

    Uses:
      • Exception Handling  – catches unexpected errors
      • Regex               – validates roll number
      • File Handling       – saves student info to JSON

    Returns a validated StudentSession object.
    """
    print_section("STUDENT LOGIN")
    print("  Please enter your details to begin the exam.\n")

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            # ── Get student name ─────────────────────────────────────────
            name = input("  Enter your full name   : ").strip()

            if not validate_name(name):
                print_warning("Invalid name. Use letters and spaces only (min 2 chars).")
                attempts += 1
                continue

            # ── Get roll number ──────────────────────────────────────────
            roll = input("  Enter your roll number : ").strip()

            if not validate_roll_number(roll):
                print_warning(
                    f"Invalid roll number format.\n"
                    f"    Expected pattern: SP24-BAI-016\n"
                    f"    You entered     : {roll}"
                )
                attempts += 1
                continue

            # ── Create and save session ──────────────────────────────────
            session = StudentSession(name=name, roll_number=roll)

            # Save initial session entry to JSON
            session.save()

            print_success(f"Login successful! Welcome, {session.name}")
            print_success(f"Session ID: {session.session_id}")
            return session

        except KeyboardInterrupt:
            print("\n\n  Exam cancelled by user.")
            raise SystemExit(0)

        except Exception as e:
            # Exception Handling: catch all unexpected errors
            print_warning(f"Unexpected error during login: {e}")
            attempts += 1

    raise SystemExit("Too many failed login attempts. Exiting.")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    session = run_login()
    print(f"\n  Session object: {session}")
    print(f"  Session dict  : {session.to_dict()}")
