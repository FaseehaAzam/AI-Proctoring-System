"""
detection/yolo_detector.py
─────────────────────────────────────────────────────────────
YOLOv8 Object Detection Module

Syllabus Coverage:
  • AI / Deep Learning  – YOLOv8 real-time detection
  • OOP                – Detector class
  • Exception Handling  – Camera errors, model load errors
  • Data Structures     – Dict for detection counts, List for alerts
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_info, print_warning, print_success, get_timestamp, MODELS_DIR

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ── YOLO class labels (COCO dataset indices) ─────────────────────────────────
# 0 = person, 67 = cell phone
PERSON_CLASS     = 0
CELL_PHONE_CLASS = 67

# Alert thresholds
PHONE_CONFIDENCE_THRESHOLD  = 0.45
PERSON_CONFIDENCE_THRESHOLD = 0.50


# ── OOP: Detector Class ───────────────────────────────────────────────────────

class Detector:
    """
    AI-powered object detector using YOLOv8.

    Detects:
      - Mobile phones (triggers cheating warning)
      - Multiple persons (triggers cheating alert)

    Attributes:
        model         : Loaded YOLOv8 model
        detection_log : List of all detection events
        counts        : Dict of cumulative detection counts
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name   = model_name
        self.model        = None
        self.is_loaded    = False

        # Data Structures (syllabus)
        self.detection_log: list = []                # List of detection events
        self.counts: dict        = {                 # Dict for counts
            "phone_detected":   0,
            "multiple_persons": 0,
            "total_detections": 0,
        }

        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model; auto-downloads on first run."""
        if not YOLO_AVAILABLE:
            print_warning("ultralytics not installed. Running in simulation mode.")
            return

        try:
            model_path = os.path.join(MODELS_DIR, self.model_name)
            print_info(f"Loading YOLOv8 model: {self.model_name}")

            # YOLO auto-downloads the model if not present locally
            self.model     = YOLO(model_path if os.path.exists(model_path) else self.model_name)
            self.is_loaded = True
            print_success("YOLOv8 model loaded successfully.")

        except Exception as e:
            print_warning(f"Could not load YOLO model: {e}")
            self.is_loaded = False

    def detect(self, frame) -> dict:
        """
        Run detection on a single video frame.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            result dict:
              {
                "phone_detected":     bool,
                "person_count":       int,
                "multiple_persons":   bool,
                "alerts":             list[str],
                "timestamp":          str,
                "annotated_frame":    frame (with bounding boxes drawn)
              }
        """
        result = {
            "phone_detected":   False,
            "person_count":     0,
            "multiple_persons": False,
            "alerts":           [],
            "timestamp":        get_timestamp(),
            "annotated_frame":  frame,
        }

        if not self.is_loaded or not YOLO_AVAILABLE:
            return self._simulate_detection(result)

        try:
            # ── Run YOLO inference ────────────────────────────────────
            predictions = self.model(frame, verbose=False)

            for pred in predictions:
                boxes = pred.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    class_id    = int(box.cls[0])
                    confidence  = float(box.conf[0])

                    # ── Phone detection ───────────────────────────────
                    if (class_id == CELL_PHONE_CLASS and
                            confidence >= PHONE_CONFIDENCE_THRESHOLD):
                        result["phone_detected"] = True
                        alert = f"Phone detected (conf: {confidence:.2f})"
                        result["alerts"].append(alert)
                        self.counts["phone_detected"] += 1

                    # ── Person counting ───────────────────────────────
                    if (class_id == PERSON_CLASS and
                            confidence >= PERSON_CONFIDENCE_THRESHOLD):
                        result["person_count"] += 1

            # ── Multiple persons check ────────────────────────────────
            if result["person_count"] > 1:
                result["multiple_persons"] = True
                alert = f"Multiple persons detected: {result['person_count']}"
                result["alerts"].append(alert)
                self.counts["multiple_persons"] += 1

            # ── Draw bounding boxes ───────────────────────────────────
            if OPENCV_AVAILABLE:
                annotated = predictions[0].plot() if predictions else frame
                result["annotated_frame"] = annotated

        except Exception as e:
            print_warning(f"Detection error: {e}")

        # Update totals and log
        if result["alerts"]:
            self.counts["total_detections"] += 1
            self._log_detection(result)

        return result

    def _simulate_detection(self, result: dict) -> dict:
        """Simulate detection output when YOLO is unavailable."""
        import random
        if random.random() < 0.05:   # 5% chance of simulated phone detection
            result["phone_detected"] = True
            result["alerts"].append("Phone detected (simulated)")
        return result

    def _log_detection(self, result: dict):
        """Append detection event to internal log list."""
        entry = {
            "timestamp":        result["timestamp"],
            "phone_detected":   result["phone_detected"],
            "person_count":     result["person_count"],
            "multiple_persons": result["multiple_persons"],
            "alerts":           result["alerts"],
        }
        self.detection_log.append(entry)

    def get_summary(self) -> dict:
        """Return a summary dict of all detections this session."""
        return {
            "counts":           self.counts,
            "total_events":     len(self.detection_log),
            "detection_log":    self.detection_log,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = Detector()

    if OPENCV_AVAILABLE:
        cap = cv2.VideoCapture(0)
        print_info("Running detection on webcam — press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print_warning("Camera not accessible.")
                break

            result = detector.detect(frame)

            for alert in result["alerts"]:
                print_warning(alert)

            cv2.imshow("YOLO Detection Test", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("\nDetection summary:", detector.get_summary()["counts"])
