"""
analytics/score_calculator.py
─────────────────────────────────────────────────────────────
Cheating Risk Score Calculator

Syllabus Coverage:
  • Data Analysis      – Pandas for data manipulation
  • AI Mathematics     – Weighted formula
  • Data Structures    – Dict, List
  • OOP               – ScoreCalculator class
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_info, print_success, get_log_path, get_timestamp

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ── Risk Score Weights (Dict) ─────────────────────────────────────────────────
RISK_WEIGHTS: dict = {
    "phone_detected":   40,   # highest risk — direct evidence
    "multiple_persons": 50,   # very high — unauthorized help
    "looking_away":      5,   # moderate — ambiguous
    "face_missing":     10,   # significant — student may have left
}

RISK_THRESHOLDS: dict = {
    "safe":       (0,   20),
    "suspicious": (21,  50),
    "warning":    (51,  80),
    "high_risk":  (81, 200),
}


# ── OOP: ScoreCalculator Class ────────────────────────────────────────────────

class ScoreCalculator:
    """
    Calculates the Cheating Risk Score from session warning counts.

    Risk Formula:
        risk_score = (phone_count   × 40)
                   + (person_count  × 50)
                   + (lookaway_count × 5)
                   + (face_missing  × 10)

    Attributes:
        weights     : Dict of per-category risk weights
        score_log   : List of score snapshots over time
    """

    def __init__(self):
        self.weights: dict   = RISK_WEIGHTS.copy()
        self.score_log: list = []   # List of (timestamp, score) tuples

    def calculate(self, warnings: dict) -> int:
        """
        Calculate integer risk score from a warnings dictionary.

        Args:
            warnings: dict like {"phone_detected": 2, "looking_away": 5, ...}

        Returns:
            integer risk score (0–200+)
        """
        score = 0

        for category, count in warnings.items():
            weight  = self.weights.get(category, 0)
            score  += count * weight

        # Clamp to reasonable maximum
        score = min(score, 200)

        # Log this snapshot
        self.score_log.append({
            "timestamp": get_timestamp(),
            "score":     score,
            "warnings":  dict(warnings),
        })

        return score

    def get_risk_level(self, score: int) -> str:
        """
        Return a human-readable risk level string based on thresholds.

        Returns: "safe" | "suspicious" | "warning" | "high_risk"
        """
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= score <= high:
                return level
        return "high_risk"

    def get_risk_color(self, level: str) -> tuple:
        """
        Return BGR color for OpenCV overlay based on risk level.
        Returns a tuple (immutable — fixed colors).
        """
        colors: dict = {
            "safe":       (0, 220, 100),   # green
            "suspicious": (0, 200, 255),   # yellow
            "warning":    (0, 120, 255),   # orange
            "high_risk":  (0, 0, 220),     # red
        }
        return colors.get(level, (255, 255, 255))

    def generate_dataframe(self) -> "pd.DataFrame | None":
        """
        Build a Pandas DataFrame from the score log.
        Used for further Pandas analysis and regression.
        """
        if not PANDAS_AVAILABLE:
            print_info("Pandas not available; skipping DataFrame generation.")
            return None

        if not self.score_log:
            return pd.DataFrame(columns=["timestamp", "score"])

        df = pd.DataFrame(self.score_log)

        # Add derived columns
        df["risk_level"]   = df["score"].apply(self.get_risk_level)
        df["score_change"] = df["score"].diff().fillna(0)

        return df

    def get_statistics(self, warnings: dict) -> dict:
        """
        Return descriptive statistics about the session's risk.

        Uses Pandas (Numpy fallback) for mean, max, std.
        """
        score = self.calculate(warnings)
        level = self.get_risk_level(score)

        stats = {
            "final_score":      score,
            "risk_level":       level,
            "total_warnings":   warnings.get("total_alerts", 0),
            "breakdown":        {
                k: v for k, v in warnings.items() if k != "total_alerts"
            },
        }

        if PANDAS_AVAILABLE and self.score_log:
            df = pd.DataFrame(self.score_log)
            stats["score_mean"] = round(df["score"].mean(), 2)
            stats["score_max"]  = int(df["score"].max())
            stats["score_std"]  = round(df["score"].std(), 2)
        elif NUMPY_AVAILABLE and self.score_log:
            scores               = [s["score"] for s in self.score_log]
            stats["score_mean"]  = round(float(np.mean(scores)), 2)
            stats["score_max"]   = int(np.max(scores))
            stats["score_std"]   = round(float(np.std(scores)), 2)

        return stats


# ── Module-level convenience function ────────────────────────────────────────
def compute_risk_score(warnings: dict) -> dict:
    """Quick helper used by main.py."""
    calc = ScoreCalculator()
    return calc.get_statistics(warnings)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_warnings = {
        "phone_detected":   2,
        "multiple_persons": 1,
        "looking_away":     8,
        "face_missing":     2,
        "total_alerts":     13,
    }

    result = compute_risk_score(sample_warnings)
    print("\nRisk Analysis:")
    for k, v in result.items():
        print(f"  {k}: {v}")
