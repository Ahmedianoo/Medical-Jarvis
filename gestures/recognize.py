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
import cv2
import numpy as np

def is_hand(features, min_area=30000, max_area=90000, min_circularity=0.08, max_circularity=0.75):
    """
    Returns True if a valid hand is detected
    """
    circularity = features['circularity']
    area = features['area']

    return ( min_area < features['area'] < max_area  and min_circularity < features['circularity'] < max_circularity)

