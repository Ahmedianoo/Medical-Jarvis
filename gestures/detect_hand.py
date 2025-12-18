"""
File: detect_hand.py
--------------------
Goal:
    Detect the user's hand region in a video frame using
    color-based skin segmentation and contour detection.

Pipeline Stage:
    Phase 2 — Hand Detection

Inputs:
    - frame: A single BGR image captured from the webcam

Outputs:
    - mask: Binary image representing detected skin regions
    - hand_contour: The largest contour assumed to be the hand
      (None if no hand is detected)

Methods Used:
    - Color space conversion (BGR → YCrCb)
    - Skin color thresholding
    - Morphological operations (erosion & dilation)
    - Contour extraction

Notes:
    - This module does NOT analyze gestures or finger counts.
    - It only isolates the hand from the background.
"""
