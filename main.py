"""
main.py
─────────────────────────────────────────────────────────────
AI Smart Exam Monitoring & Proctoring System
Entry Point — Connects ALL modules

Run:
    python main.py

Exam Flow:
    1. Login (auth/login.py)
    2. Calibration Game (calibration/focus_game.py)
    3. Camera feed starts
    4. YOLO Detection (detection/yolo_detector.py)
    5. Eye Tracking (detection/eye_tracker.py)
    6. Warning Logging (stored in StudentSession)
    7. Analytics (analytics/score_calculator.py + regression.py)
    8. PDF Report (reports/report_generator.py)

Authors:
    Hamza        — SP24-BAI-016
    Hashmah Haq  — SP24-BAI-017
    Faseeha Azam — SP24-BAI-012
"""

import os
import sys
import time

# ── Ensure project root is in Python path ────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Import shared utilities ───────────────────────────────────────────────────
from utils.helpers import (
    print_banner, print_section, print_info,
    print_warning, print_success, ensure_dirs,
    get_log_path, save_json
)

# ── Import all modules ────────────────────────────────────────────────────────
from auth.login               import run_login, StudentSession
from calibration.focus_game   import run_calibration
from detection.yolo_detector  import Detector
from detection.eye_tracker    import EyeTracker
from detection.face_detector  import FaceDetector
from analytics.score_calculator import ScoreCalculator
from analytics.regression       import RegressionAnalyzer
from reports.report_generator   import ReportGenerator

# ── Optional OpenCV ───────────────────────────────────────────────────────────
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print_warning("OpenCV not found. Running in no-camera simulation mode.")


# ══════════════════════════════════════════════════════════════════════════════
#  PROCTORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ProctoringEngine:
    """
    Central controller that coordinates all modules
    throughout a student's exam session.

    Attributes:
        session      : StudentSession — student info + warning log
        detector     : Detector      — YOLOv8 object detection
        eye_tracker  : EyeTracker    — OpenCV gaze tracking
        face_detector: FaceDetector  — quick face presence check
        calculator   : ScoreCalculator
        cap          : cv2.VideoCapture (or None)
        running      : bool — loop control flag
    """

    def __init__(self, session: StudentSession, calibration_data: dict):
        self.session          = session
        self.calibration_data = calibration_data
        self.running          = False

        # Initialise AI detection modules
        print_info("Initialising detection modules...")
        self.detector      = Detector()
        self.eye_tracker   = EyeTracker()
        self.face_detector = FaceDetector()
        self.calculator    = ScoreCalculator()

        # Camera
        self.cap = None

        # Frame processing counters
        self._frame_count      = 0
        self._yolo_every_n     = 5    # run YOLO every 5 frames (performance)
        self._eye_every_n      = 3    # run eye tracker every 3 frames
        self._last_hud_time    = 0.0

    def start_camera(self) -> bool:
        """Open the default webcam. Returns True if successful."""
        if not OPENCV_AVAILABLE:
            print_info("No OpenCV — camera skipped (simulation mode).")
            return False
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera (index 0).")
            print_success("Webcam opened successfully.")
            return True
        except Exception as e:
            print_warning(f"Camera error: {e}")
            return False

    def stop_camera(self):
        """Release the webcam safely."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if OPENCV_AVAILABLE:
            cv2.destroyAllWindows()

    def _draw_hud(self, frame, risk_score: int, risk_level: str):
        """
        Overlay HUD (Heads-Up Display) on the camera frame.
        Shows: student name, risk score, risk level, warning counts.
        """
        if not OPENCV_AVAILABLE:
            return frame

        h, w = frame.shape[:2]

        # Semi-transparent dark banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Student name
        cv2.putText(frame, f"{self.session.name}  |  {self.session.roll_number}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # Risk score (colored)
        level_colors = {
            "safe": (0, 200, 100), "suspicious": (0, 220, 255),
            "warning": (0, 140, 255), "high_risk": (0, 0, 220),
        }
        hud_color = level_colors.get(risk_level, (255, 255, 255))
        cv2.putText(frame, f"RISK: {risk_score}  [{risk_level.upper()}]",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

        # Warning counts (right side)
        w_text = (f"Phone:{self.session.warnings['phone_detected']}  "
                  f"Persons:{self.session.warnings['multiple_persons']}  "
                  f"Gaze:{self.session.warnings['looking_away']}")
        cv2.putText(frame, w_text, (w - 400, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Timer
        elapsed = int(time.time() - self._start_time)
        cv2.putText(frame, f"Time: {elapsed}s",
                    (w - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Alert flash
        if self.session.warnings["total_alerts"] > 0:
            alert_text = f"! {self.session.warnings['total_alerts']} ALERTS !"
            cv2.putText(frame, alert_text, (w // 2 - 80, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run_proctoring_loop(self, duration_seconds: int = 3600):
        """
        Main proctoring loop.

        Captures frames from webcam, runs detections,
        logs warnings to the StudentSession, and shows
        the annotated feed.

        Args:
            duration_seconds: Max exam duration (default 1 hour).
                              Set to small value for testing.
        """
        self.running     = True
        self._start_time = time.time()

        has_camera = self.start_camera()

        print_section("PROCTORING STARTED")
        print_info(f"Monitoring {self.session.name} for up to {duration_seconds}s")
        print_info("Press Q in the camera window (or Ctrl+C) to end the exam.")

        # ── Periodic score update (every N seconds) ────────────────────
        _score_interval = 10   # recalculate risk every 10 s
        _last_score_t   = self._start_time
        risk_score      = 0
        risk_level      = "safe"

        try:
            while self.running:
                elapsed = time.time() - self._start_time

                # ── Time limit reached ─────────────────────────────────
                if elapsed >= duration_seconds:
                    print_info("Exam duration reached. Ending session.")
                    break

                # ── Camera frame ───────────────────────────────────────
                if has_camera and self.cap:
                    ret, frame = self.cap.read()
                    if not ret:
                        print_warning("Camera frame lost. Retrying...")
                        time.sleep(0.1)
                        continue
                else:
                    # Simulation mode: sleep briefly
                    time.sleep(0.5)
                    self._frame_count += 1
                    self._simulate_warnings()
                    continue

                self._frame_count += 1

                # ── YOLO detection (every N frames) ────────────────────
                if self._frame_count % self._yolo_every_n == 0:
                    yolo_result = self.detector.detect(frame)

                    if yolo_result["phone_detected"]:
                        self.session.add_warning("phone_detected",
                                                  "Phone visible in frame")
                        print_warning("Phone detected!")

                    if yolo_result["multiple_persons"]:
                        self.session.add_warning("multiple_persons",
                                                  f"{yolo_result['person_count']} persons")
                        print_warning(f"Multiple persons: {yolo_result['person_count']}")

                    # Use annotated frame from YOLO if available
                    annotated = yolo_result.get("annotated_frame", frame)
                else:
                    annotated = frame

                # ── Eye / gaze tracking (every N frames) ───────────────
                if self._frame_count % self._eye_every_n == 0:
                    gaze_result = self.eye_tracker.track(annotated)

                    if gaze_result.get("looking_away"):
                        self.session.add_warning("looking_away",
                                                  gaze_result.get("direction", ""))
                        print_warning(f"Gaze alert: {gaze_result.get('direction')}")

                    if gaze_result.get("face_missing"):
                        self.session.add_warning("face_missing", "No face detected")
                        print_warning("Face not detected in frame!")

                # ── Recalculate risk score periodically ────────────────
                if (time.time() - _last_score_t) >= _score_interval:
                    stats       = self.calculator.get_statistics(self.session.warnings)
                    risk_score  = stats["final_score"]
                    risk_level  = stats["risk_level"]
                    _last_score_t = time.time()

                    # Also log score snapshot to timeline
                    self.calculator.score_log.append({
                        "timestamp": self.session.start_time,
                        "score":     risk_score,
                        "warnings":  dict(self.session.warnings),
                    })

                # ── Draw HUD and show frame ─────────────────────────────
                display = self._draw_hud(annotated, risk_score, risk_level)
                cv2.imshow("AI Proctoring System — Press Q to End", display)

                # Q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print_info("Exam ended by user (Q pressed).")
                    break

        except KeyboardInterrupt:
            print_info("\nExam interrupted by keyboard.")

        finally:
            self.running = False
            self.stop_camera()
            print_success("Proctoring loop ended.")

    def _simulate_warnings(self):
        """Generate random warnings during simulation mode (no camera)."""
        import random
        if self._frame_count % 20 == 0 and random.random() < 0.2:
            self.session.add_warning("looking_away", "simulated")

    def finalize_session(self) -> dict:
        """
        Post-exam: calculate final score, run regression, save session.
        Returns a dict of all analytics results.
        """
        print_section("FINALIZING SESSION")

        # ── Final risk score ───────────────────────────────────────────
        calc  = ScoreCalculator()
        stats = calc.get_statistics(self.session.warnings)
        self.session.risk_score = stats["final_score"]

        print_info(f"Final Risk Score : {stats['final_score']}")
        print_info(f"Risk Level       : {stats['risk_level'].upper()}")

        # ── Regression analysis ────────────────────────────────────────
        reg_analyzer = RegressionAnalyzer()
        reg_analyzer.train(self.session.timeline)
        reg_summary = reg_analyzer.get_summary()

        reg_graph_path = reg_analyzer.save_graph(
            get_log_path("regression_graph.png").replace(
                "logs/", "reports/output/"
            )
        )

        print_info(f"Regression slope : {reg_summary['slope']}")
        print_info(f"Cheating prob    : {reg_summary['probability']*100:.1f}%")
        print_info(f"Interpretation   : {reg_summary['interpretation']}")

        # ── Save session to JSON ────────────────────────────────────────
        self.session.save()

        # ── Generate reports ────────────────────────────────────────────
        gen    = ReportGenerator(self.session.to_dict(), stats, reg_summary)
        output = gen.generate_all()

        print_section("EXAM COMPLETE")
        print_success(f"Student  : {self.session.name}")
        print_success(f"Roll No  : {self.session.roll_number}")
        print_success(f"Score    : {stats['final_score']} ({stats['risk_level'].upper()})")
        print_success(f"PDF      : {output.get('pdf', 'not generated')}")

        return {
            "session":    self.session.to_dict(),
            "stats":      stats,
            "regression": reg_summary,
            "output":     output,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Full system entry point.

    Steps:
      1. Ensure runtime directories exist
      2. Print startup banner
      3. Student login (Regex validation)
      4. Calibration game
      5. Proctoring loop (YOLO + Eye tracking)
      6. Finalize: score + regression + PDF report
    """

    # ── Step 1: Setup ──────────────────────────────────────────────────
    ensure_dirs()
    print_banner()

    # ── Step 2: Student Login ──────────────────────────────────────────
    session = run_login()

    # ── Step 3: Calibration Game ───────────────────────────────────────
    print()
    print_info("Starting calibration game in 3 seconds...")
    time.sleep(3)
    calibration_data = run_calibration()

    # ── Step 4: Start Proctoring ───────────────────────────────────────
    print()
    print_info("Calibration complete. Starting exam monitoring...")
    time.sleep(1)

    engine = ProctoringEngine(session, calibration_data)

    # ── Duration: change to 3600 for a real 1-hour exam ───────────────
    # Set to 60 seconds for quick demo/testing
    EXAM_DURATION = int(os.environ.get("EXAM_DURATION", 60))
    engine.run_proctoring_loop(duration_seconds=EXAM_DURATION)

    # ── Step 5: Finalize and generate reports ──────────────────────────
    results = engine.finalize_session()

    print_section("THANK YOU")
    print_info("All reports saved to reports/output/")
    print_info("Session data saved to logs/sessions.json")

    return results


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
