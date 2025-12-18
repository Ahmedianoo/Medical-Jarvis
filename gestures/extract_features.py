"""
File: extract_features.py
-------------------------
Goal:
    Extract geometric features from the detected hand contour
    that will later be used for gesture recognition.

Pipeline Stage:
    Phase 3 — Feature Extraction

Inputs:
    - hand_contour: Contour corresponding to the detected hand

Outputs:
    - features (dict) containing:
        • finger_count: Estimated number of visible fingers
        • defects: Convexity defects of the hand contour

Techniques Used:
    - Convex hull computation
    - Convexity defect analysis
    - Angle and depth calculations between fingers

Notes:
    - This module only extracts features.
    - No gesture classification or command mapping is done here.
"""
