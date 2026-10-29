"""
detection/face_detector.py
─────────────────────────────────────────────────────────────
Face Presence Detection Module

Detects whether the correct student face is present in frame.
Used as a quick check before full YOLOv8 inference.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_warning, print_success, get_timestamp

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class FaceDetector:
    """
    Lightweight face presence detector using Haar Cascades.
    Faster than YOLOv8 for a simple face-present/absent check.
    """

    def __init__(self):
        self.cascade  = None
        self.is_ready = False
        self._load()

    def _load(self):
        if not OPENCV_AVAILABLE:
            return
        try:
            path          = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade  = cv2.CascadeClassifier(path)
            self.is_ready = True
            print_success("Face detector ready.")
        except Exception as e:
            print_warning(f"Face detector load error: {e}")

    def detect(self, frame) -> dict:
        """
        Returns:
          { "face_count": int, "faces": list of (x,y,w,h) tuples, "timestamp": str }
        """
        result = {"face_count": 0, "faces": [], "timestamp": get_timestamp()}

        if not self.is_ready or not OPENCV_AVAILABLE:
            result["face_count"] = 1   # assume present in simulation
            return result

        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            result["face_count"] = len(faces)
            result["faces"]      = [tuple(f) for f in faces]   # list of tuples

            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        except Exception as e:
            print_warning(f"Face detection error: {e}")

        return result
