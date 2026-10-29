"""
detection/eye_tracker.py
─────────────────────────────────────────────────────────────
Eye & Gaze Tracking Module

Syllabus Coverage:
  • OpenCV           – Haar cascade face/eye detection
  • OOP              – EyeTracker class
  • Data Structures  – Dict for gaze state, List for gaze history
  • Tuples           – Fixed screen zone boundaries
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_info, print_warning, print_success, get_timestamp

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ── Gaze zone thresholds as Tuples (immutable, fixed pixel ratios) ───────────
# (left_ratio, right_ratio) relative to frame width
GAZE_CENTER_ZONE  = (0.30, 0.70)   # 30–70% of frame width = center
LOOKAWAY_TIMEOUT  = 2.5             # seconds before triggering warning


# ── OOP: EyeTracker Class ─────────────────────────────────────────────────────

class EyeTracker:
    """
    Tracks student gaze direction and face presence using
    OpenCV Haar Cascade classifiers.

    Detects:
      - Face center position
      - Looking left / right / center
      - Face missing (student left seat)

    Attributes:
        face_cascade     : OpenCV Haar cascade for face detection
        eye_cascade      : OpenCV Haar cascade for eye detection
        gaze_history     : List of gaze states over time
        gaze_state       : Dict of current gaze readings
        lookaway_start   : Timestamp when student started looking away
    """

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade  = None
        self.is_ready     = False

        # Data Structures (syllabus)
        self.gaze_history: list = []     # List of timestamped gaze events
        self.gaze_state: dict   = {      # Dict for current gaze state
            "face_detected":    False,
            "eye_count":        0,
            "direction":        "unknown",
            "face_x_ratio":     0.5,
            "looking_away":     False,
            "face_missing":     False,
        }

        self.lookaway_start      = None
        self.face_missing_start  = None
        self.alert_counts: dict  = {
            "looking_away":   0,
            "face_missing":   0,
        }

        self._load_cascades()

    def _load_cascades(self):
        """Load Haar cascade XML files from OpenCV data directory."""
        if not OPENCV_AVAILABLE:
            print_warning("OpenCV not available. Eye tracker in simulation mode.")
            return

        try:
            cv2_data    = cv2.data.haarcascades
            face_path   = os.path.join(cv2_data, "haarcascade_frontalface_default.xml")
            eye_path    = os.path.join(cv2_data, "haarcascade_eye.xml")

            self.face_cascade = cv2.CascadeClassifier(face_path)
            self.eye_cascade  = cv2.CascadeClassifier(eye_path)
            self.is_ready     = True
            print_success("Eye tracker cascades loaded.")

        except Exception as e:
            print_warning(f"Could not load Haar cascades: {e}")
            self.is_ready = False

    def track(self, frame) -> dict:
        """
        Analyze a single frame for gaze direction.

        Args:
            frame: OpenCV BGR image

        Returns:
            gaze_result dict with state and any generated alerts
        """
        if not self.is_ready or not OPENCV_AVAILABLE:
            return self._simulate_gaze()

        result = {
            "face_detected":    False,
            "eye_count":        0,
            "direction":        "unknown",
            "looking_away":     False,
            "face_missing":     False,
            "alerts":           [],
            "timestamp":        get_timestamp(),
        }

        try:
            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Face detection ────────────────────────────────────────
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                result["face_missing"]  = True
                result["direction"]     = "face_missing"
                self._handle_face_missing(result)
            else:
                result["face_detected"] = True
                self.face_missing_start = None

                # Use largest face
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

                # ── Gaze direction from face center x ─────────────────
                face_center_x = x + w // 2
                face_x_ratio  = face_center_x / frame_w  # 0.0–1.0

                result["face_x_ratio"] = face_x_ratio
                result["direction"]    = self._classify_direction(face_x_ratio)

                # ── Eye detection within face ROI ──────────────────────
                face_roi = gray[y:y+h, x:x+w]
                eyes     = self.eye_cascade.detectMultiScale(
                    face_roi, scaleFactor=1.1, minNeighbors=3
                )
                result["eye_count"] = min(len(eyes), 2)

                # ── Look-away detection ────────────────────────────────
                if result["direction"] in ("looking_left", "looking_right"):
                    self._handle_lookaway(result)
                else:
                    self.lookaway_start = None  # reset timer

                # ── Draw on frame ──────────────────────────────────────
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    result["direction"],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 200), 2
                )

        except Exception as e:
            print_warning(f"Eye tracking error: {e}")

        # Update internal state and history
        self.gaze_state = result
        self.gaze_history.append(result)

        return result

    def _classify_direction(self, face_x_ratio: float) -> str:
        """
        Classify gaze direction based on face horizontal position.
        Uses the GAZE_CENTER_ZONE tuple for thresholds.
        """
        left_thresh, right_thresh = GAZE_CENTER_ZONE   # unpack tuple

        if face_x_ratio < left_thresh:
            return "looking_right"    # mirrored camera
        elif face_x_ratio > right_thresh:
            return "looking_left"
        else:
            return "looking_center"

    def _handle_lookaway(self, result: dict):
        """Start or check the lookaway timer."""
        now = time.time()
        if self.lookaway_start is None:
            self.lookaway_start = now

        elapsed = now - self.lookaway_start
        if elapsed >= LOOKAWAY_TIMEOUT:
            result["looking_away"] = True
            result["alerts"].append(
                f"Looking away for {elapsed:.1f}s ({result['direction']})"
            )
            self.alert_counts["looking_away"] += 1
            self.lookaway_start = now   # reset to avoid spamming

    def _handle_face_missing(self, result: dict):
        """Check if face has been missing too long."""
        now = time.time()
        if self.face_missing_start is None:
            self.face_missing_start = now

        elapsed = now - self.face_missing_start
        if elapsed >= LOOKAWAY_TIMEOUT:
            result["face_missing"] = True
            result["alerts"].append(f"Face missing for {elapsed:.1f}s")
            self.alert_counts["face_missing"] += 1
            self.face_missing_start = now

    def _simulate_gaze(self) -> dict:
        """Return dummy gaze data when OpenCV is unavailable."""
        import random
        directions = ["looking_center", "looking_center", "looking_left", "looking_right"]
        direction  = random.choice(directions)
        return {
            "face_detected":  True,
            "eye_count":      2,
            "direction":      direction,
            "looking_away":   direction != "looking_center",
            "face_missing":   False,
            "alerts":         [],
            "timestamp":      get_timestamp(),
        }

    def get_summary(self) -> dict:
        """Return eye tracking summary for the session."""
        return {
            "total_frames":         len(self.gaze_history),
            "alert_counts":         self.alert_counts,
            "gaze_history_sample":  self.gaze_history[-10:],   # last 10 entries
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = EyeTracker()

    if OPENCV_AVAILABLE:
        cap = cv2.VideoCapture(0)
        print_info("Eye tracker running — press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = tracker.track(frame)

            for alert in result.get("alerts", []):
                print_warning(alert)

            cv2.imshow("Eye Tracker Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("\nEye tracker summary:", tracker.get_summary()["alert_counts"])
