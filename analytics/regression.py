"""
analytics/regression.py
─────────────────────────────────────────────────────────────
Linear Regression Analysis of Cheating Behavior Over Time

Syllabus Coverage:
  • AI Mathematics     – Linear Regression, Chain Rule explanation
  • Data Science       – Scikit-Learn, Pandas, NumPy
  • OOP               – RegressionAnalyzer class
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_info, print_warning, print_success

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for file saving
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ── OOP: RegressionAnalyzer Class ────────────────────────────────────────────

class RegressionAnalyzer:
    """
    Applies Linear Regression to predict cheating probability over time.

    Uses:
      - x-axis: elapsed seconds during exam
      - y-axis: cumulative warning count

    Predicts how warnings will continue to grow.

    AI Mathematics Note:
      YOLO uses backpropagation to train its weights.
      Backpropagation relies on the CHAIN RULE:
          dL/dw = (dL/dy) × (dy/dw)
      where L = loss, y = output, w = weight.
      This chain of derivatives adjusts each weight to
      minimise detection error — the core of deep learning.
    """

    def __init__(self):
        self.model        = None
        self.is_trained   = False
        self.slope        = 0.0
        self.intercept    = 0.0
        self.r2_score     = 0.0
        self.mse          = 0.0
        self.X_train      = []
        self.y_train      = []

    def train(self, timeline: list) -> bool:
        """
        Train regression model on session timeline data.

        Args:
            timeline: list of dicts with "second" and "cumulative" keys

        Returns:
            True if training succeeded, False otherwise.
        """
        if not timeline or len(timeline) < 3:
            print_warning("Not enough data points for regression (need ≥ 3).")
            self._use_fallback(timeline)
            return False

        if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
            print_warning("Scikit-learn / NumPy not available. Using fallback.")
            self._use_fallback(timeline)
            return False

        try:
            # Extract features
            seconds     = [entry["second"]     for entry in timeline]
            cumulatives = [entry["cumulative"]  for entry in timeline]

            X = np.array(seconds).reshape(-1, 1)
            y = np.array(cumulatives)

            self.X_train = seconds
            self.y_train = cumulatives

            # ── Linear Regression ────────────────────────────────────
            self.model = LinearRegression()
            self.model.fit(X, y)

            self.slope     = float(self.model.coef_[0])
            self.intercept = float(self.model.intercept_)

            # Model evaluation
            y_pred        = self.model.predict(X)
            self.r2_score = float(r2_score(y, y_pred))
            self.mse      = float(mean_squared_error(y, y_pred))

            self.is_trained = True
            print_success(
                f"Regression trained: slope={self.slope:.4f}, "
                f"R²={self.r2_score:.4f}"
            )
            return True

        except Exception as e:
            print_warning(f"Regression training error: {e}")
            self._use_fallback(timeline)
            return False

    def _use_fallback(self, timeline: list):
        """Simple manual slope calculation when sklearn is unavailable."""
        if len(timeline) < 2:
            self.slope = 0.0
            return

        try:
            t1, c1 = timeline[0]["second"],  timeline[0]["cumulative"]
            t2, c2 = timeline[-1]["second"], timeline[-1]["cumulative"]
            dt     = t2 - t1
            self.slope     = (c2 - c1) / dt if dt > 0 else 0.0
            self.intercept = c1
            self.is_trained = True
        except (KeyError, ZeroDivisionError):
            self.slope     = 0.0
            self.intercept = 0.0

    def predict(self, future_second: int) -> float:
        """
        Predict cumulative warnings at a future time point.

        Args:
            future_second: seconds from exam start

        Returns:
            predicted cumulative warning count
        """
        if not self.is_trained:
            return 0.0

        if SKLEARN_AVAILABLE and self.model is not None:
            import numpy as np
            pred = self.model.predict([[future_second]])[0]
            return max(0.0, float(pred))
        else:
            # Manual: y = mx + b
            pred = self.slope * future_second + self.intercept
            return max(0.0, float(pred))

    def get_probability(self) -> float:
        """
        Convert slope into a 0–1 cheating probability estimate.
        Steeper slope = higher probability.
        """
        # Sigmoid-like normalisation: probability = 1 / (1 + e^(-k * slope))
        if not NUMPY_AVAILABLE:
            return min(max(self.slope / 2.0, 0.0), 1.0)

        k    = 3.0   # scaling factor
        prob = 1 / (1 + np.exp(-k * self.slope))
        return float(round(prob, 4))

    def save_graph(self, output_path: str) -> bool:
        """
        Save a Matplotlib regression scatter plot + trend line.

        Args:
            output_path: full path to save the PNG

        Returns:
            True if saved successfully.
        """
        if not MATPLOTLIB_AVAILABLE:
            print_warning("Matplotlib not available; skipping regression graph.")
            return False

        if not self.X_train:
            print_warning("No training data; cannot plot regression.")
            return False

        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#1A1A2E")
            ax.set_facecolor("#16213E")

            # Scatter: actual data points
            ax.scatter(
                self.X_train, self.y_train,
                color="#E94560", s=40, zorder=5, label="Actual warnings"
            )

            # Trend line
            if self.is_trained:
                x_line   = list(range(0, max(self.X_train) + 30, 5))
                y_line   = [self.predict(x) for x in x_line]
                ax.plot(x_line, y_line, color="#0F3460", linewidth=2.5,
                        linestyle="--", label=f"Trend (slope={self.slope:.3f})")

            ax.set_xlabel("Time (seconds)", color="white")
            ax.set_ylabel("Cumulative Warnings", color="white")
            ax.set_title("Warning Trend — Linear Regression", color="white", fontsize=13)
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1A1A2E", labelcolor="white")
            ax.spines[:].set_color("#333355")

            # Annotation: R² score
            ax.text(
                0.98, 0.05,
                f"R² = {self.r2_score:.3f}",
                transform=ax.transAxes,
                ha="right", color="#A0A0C0", fontsize=9
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
            print_success(f"Regression graph saved → {output_path}")
            return True

        except Exception as e:
            print_warning(f"Could not save regression graph: {e}")
            return False

    def get_summary(self) -> dict:
        """Return a summary dict of regression results."""
        return {
            "slope":           round(self.slope, 4),
            "intercept":       round(self.intercept, 4),
            "r2_score":        round(self.r2_score, 4),
            "mse":             round(self.mse, 4),
            "probability":     self.get_probability(),
            "interpretation":  self._interpret(),
        }

    def _interpret(self) -> str:
        """Human-readable interpretation of the regression slope."""
        if self.slope <= 0:
            return "No increasing cheating trend detected."
        elif self.slope < 0.05:
            return "Very low warning rate — likely safe."
        elif self.slope < 0.15:
            return "Moderate warning rate — student shows some suspicious behavior."
        elif self.slope < 0.30:
            return "High warning rate — consistent suspicious activity."
        else:
            return "Very high warning rate — strong evidence of cheating."


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    # Simulate a session timeline
    timeline = []
    cumulative = 0
    for second in range(0, 300, 10):
        if random.random() < 0.3:
            cumulative += 1
        timeline.append({"second": second, "cumulative": cumulative})

    analyzer = RegressionAnalyzer()
    analyzer.train(timeline)

    print("\nRegression Summary:")
    for k, v in analyzer.get_summary().items():
        print(f"  {k}: {v}")

    print(f"\n  Predicted warnings at 300s: {analyzer.predict(300):.1f}")
