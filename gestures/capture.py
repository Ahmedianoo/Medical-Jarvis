"""
File: capture.py
----------------
Goal:
    Handle all webcam-related operations.
    This file is responsible ONLY for initializing the camera and
    capturing video frames in real time.

Pipeline Stage:
    Phase 1 — Video Input & Calibration

Inputs:
    - Camera index (default = 0 for internal webcam)

Outputs:
    - Video frame (BGR image) captured from the webcam

Notes:
    - No image processing, hand detection, or gesture logic is done here.
    - This separation allows the capture module to be reused or replaced
      (e.g., video file instead of webcam) without affecting other stages.
"""


