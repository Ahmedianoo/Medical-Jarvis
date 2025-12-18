"""
File: recognize_gesture.py
--------------------------
Goal:
    Recognize static and dynamic hand gestures based on
    extracted hand features and motion information.

Pipeline Stage:
    Phase 4 — Gesture Recognition

Inputs:
    - features: Dictionary containing hand features
      (e.g., finger count, centroid movement)
    - motion data (dx, dy) for dynamic gestures

Outputs:
    - gesture_label: String representing the recognized gesture
      (e.g., 'PALM', 'FIST', 'SWIPE_LEFT', 'PINCH')

Approach:
    - Rule-based decision logic
    - Thresholding on finger count and hand motion

Notes:
    - This module does NOT interact with the UI or filesystem.
    - It strictly converts features into a semantic gesture label.
"""
