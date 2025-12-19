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

import cv2
import numpy as np


def extract_features(contour):


    """
        cv2.moments() calculates spatial moments of a contour 
        moments are weighted (unweighted here because the img is binary) averages of pixel intensities, we can use them to get area, centroid, ... .
        m00 = sum of all pixels inside contour = area
        m10 = sum of x-coordinates of all pixels
        m01 = sum of y-coordinates of all pixels
    """
    moment = cv2.moments(contour)
    if moment['m00'] == 0: # to avoid dividing by zeros
        cx = 0
        cy = 0
    else:
        cx = int(moment['m10'] // moment['m00'])
        cy = int(moment['m01'] // moment['m00'])



    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
    else:
        defects = None
    finger_count = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            depth = d / 256

            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c_len = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos((b**2 + c_len**2 - a**2) / (2 * b * c_len)) * 180 / np.pi

            if angle < 90 and depth > 20:
                finger_count += 1
        # if finger_count > 0:
        #     finger_count += 1  # approximate thumb

    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
    aspect_ratio = w / h

    features = {'centroid': (cx, cy), 'hull': hull, 'defects': defects, 'finger_count': finger_count, 'bounding_box': (x, y, w, h), 'aspect_ratio': aspect_ratio
                , 'area': area, 'circularity': circularity }

    return features


