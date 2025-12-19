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
        returns True if a valid hand is detected
    """
    circularity = features['circularity']
    area = features['area']

    return ( min_area < features['area'] < max_area  and min_circularity < features['circularity'] < max_circularity)


def get_static_gesture(features):
    v_shapes = features['convexity_defect']
    circularity = features['circularity']
    area = features['area']
    aspect_ratio = features['aspect_ratio']

    if v_shapes == 0 and circularity > 0.15: #closed hand
        return 'FIST'
    if v_shapes >= 3 and area > 55000:      #open hand
        return 'PALM'
    if v_shapes == 1 and aspect_ratio > 0.4: #for cursor, victory sign
        return 'POINT'
    return 'UNKOWN'
    

