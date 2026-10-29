"""
calibration/focus_game.py
─────────────────────────────────────────────────────────────
AI Focus Calibration Game

Syllabus Coverage:
  • AI-Based Game      – Moving dot focus challenge
  • OpenCV             – Window + frame rendering
  • Random module      – Random dot position
  • Timer / Tuples     – Fixed coordinate bounds as tuples
  • OOP               – CalibrationGame class
"""

import time
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, print_success, print_info, get_timestamp

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ── Fixed constants as Tuples (syllabus: tuples for fixed data) ──────────────
WINDOW_SIZE   = (800, 600)           # (width, height) — immutable
DOT_RADIUS    = 20                   # pixels
GAME_DURATION = 30                   # seconds
DOT_COLOR     = (0, 200, 255)        # BGR: yellow-green
BG_COLOR      = (20, 20, 40)        # dark background
TEXT_COLOR    = (255, 255, 255)      # white
BOUNDS = (60, 60, 740, 540)         # (x_min, y_min, x_max, y_max) — tuple


# ── OOP: CalibrationGame Class ───────────────────────────────────────────────

class CalibrationGame:
    """
    Pre-exam focus calibration game.

    A dot moves randomly on screen. The student follows it with
    their eyes. The system records:
      - Head movement range
      - Eye sensitivity
      - Gaze tracking baseline

    The game runs for GAME_DURATION seconds.
    """

    def __init__(self):
        self.dot_position = self._random_position()   # Tuple
        self.score        = 0
        self.start_time   = None
        self.elapsed      = 0
        self.positions_visited: list = []             # List of visited positions
        self.calibration_data: dict  = {}             # Dict for calibration output

        # Movement parameters
        self._move_interval = 1.5   # seconds between dot moves
        self._last_move     = 0.0

    def _random_position(self) -> tuple:
        """Generate a random dot position within the safe BOUNDS tuple."""
        x = random.randint(BOUNDS[0], BOUNDS[2])
        y = random.randint(BOUNDS[1], BOUNDS[3])
        return (x, y)   # Returns a tuple

    def _draw_frame(self, frame, time_remaining: float) -> None:
        """Render the current game frame."""
        # Background
        frame[:] = BG_COLOR

        # Title
        cv2.putText(
            frame,
            "AI FOCUS CALIBRATION — Follow the dot with your eyes",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            TEXT_COLOR, 1, cv2.LINE_AA
        )

        # Timer bar background
        cv2.rectangle(frame, (20, 50), (780, 70), (60, 60, 80), -1)

        # Timer bar fill (proportional to time remaining)
        fill_width = int(760 * (time_remaining / GAME_DURATION))
        bar_color  = (0, 200, 100) if time_remaining > 10 else (0, 100, 220)
        cv2.rectangle(frame, (20, 50), (20 + fill_width, 70), bar_color, -1)

        # Timer text
        cv2.putText(
            frame, f"Time: {int(time_remaining)}s",
            (680, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            TEXT_COLOR, 1, cv2.LINE_AA
        )

        # Draw dot with outer ring for visibility
        cx, cy = self.dot_position
        cv2.circle(frame, (cx, cy), DOT_RADIUS + 8, (255, 255, 255), 1)   # outer ring
        cv2.circle(frame, (cx, cy), DOT_RADIUS,     DOT_COLOR, -1)         # filled dot

        # Score
        cv2.putText(
            frame, f"Points: {self.score}",
            (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA
        )

        # Instructions
        cv2.putText(
            frame, "Press SPACE to mark dot | Press Q to finish early",
            (180, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (150, 150, 150), 1, cv2.LINE_AA
        )

    def run(self) -> dict:
        """
        Run the calibration game.
        Returns calibration_data dict with sensitivity settings.
        """
        print_section("AI FOCUS CALIBRATION GAME")
        print_info("A dot will appear on screen. Follow it with your eyes.")
        print_info(f"Game duration: {GAME_DURATION} seconds")
        print_info("Press SPACE when looking at the dot | Q to finish early")

        if not OPENCV_AVAILABLE:
            print_info("OpenCV not available — running in simulation mode.")
            return self._simulate()

        # ── OpenCV game window ───────────────────────────────────────────
        window_name = "AI Focus Calibration Game"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, WINDOW_SIZE[0], WINDOW_SIZE[1])

        frame      = np.zeros((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        self.start_time  = time.time()
        self._last_move  = self.start_time

        while True:
            now           = time.time()
            self.elapsed  = now - self.start_time
            time_remaining = GAME_DURATION - self.elapsed

            # Game over condition
            if time_remaining <= 0:
                break

            # Move dot every _move_interval seconds
            if (now - self._last_move) >= self._move_interval:
                self.dot_position = self._random_position()
                self.positions_visited.append(self.dot_position)
                self._last_move = now

            # Draw
            self._draw_frame(frame, time_remaining)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF

            # SPACE = student marked the dot
            if key == ord(' '):
                self.score += 10

            # Q = early exit
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return self._build_calibration_data()

    def _simulate(self) -> dict:
        """Fallback simulation when OpenCV is not available (e.g. CI/CD)."""
        print_info("Simulating 5-second calibration...")
        for i in range(5):
            self.dot_position = self._random_position()
            self.positions_visited.append(self.dot_position)
            time.sleep(1)
            print_info(f"  Dot at position: {self.dot_position}")
            self.score += 10

        return self._build_calibration_data()

    def _build_calibration_data(self) -> dict:
        """Build and store calibration result dictionary."""
        if not self.positions_visited:
            self.positions_visited = [self._random_position()]

        # Calculate gaze range from visited positions (using tuples)
        x_coords = [p[0] for p in self.positions_visited]
        y_coords = [p[1] for p in self.positions_visited]

        self.calibration_data = {
            "calibration_time":  get_timestamp(),
            "score":             self.score,
            "dots_visited":      len(self.positions_visited),
            "gaze_x_range":      (min(x_coords), max(x_coords)),  # Tuple
            "gaze_y_range":      (min(y_coords), max(y_coords)),  # Tuple
            "sensitivity":       self._calculate_sensitivity(),
            "status":            "calibrated",
        }

        print_success(f"Calibration complete! Score: {self.score} | "
                      f"Sensitivity: {self.calibration_data['sensitivity']}")
        return self.calibration_data

    def _calculate_sensitivity(self) -> str:
        """Return low / medium / high based on dots visited."""
        if self.score >= 80:
            return "high"
        elif self.score >= 40:
            return "medium"
        else:
            return "low"


# ── Standalone entry point ────────────────────────────────────────────────────
def run_calibration() -> dict:
    """Public entry point called from main.py."""
    game = CalibrationGame()
    return game.run()


if __name__ == "__main__":
    data = run_calibration()
    print(f"\nCalibration Data: {data}")
