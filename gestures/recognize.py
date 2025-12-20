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


def is_hand(features, min_area=45000, max_area=90000, min_circularity=0.08, max_circularity=0.75):
    """
        returns True if a valid hand is detected
        the hand should be close enough to the laptop camera, but also not so close so that it takes the whole frame
    """


    return ( min_area < features['area'] < max_area  and min_circularity < features['circularity'] < max_circularity)


def get_static_gesture(features):
    """
        before calling this funtction, is_hand function must be called first to make sure if this is a hand or not
        by doing some extra if conditions and adjusting the ranges, more static gestures could be detected
    """
    v_shapes = features['convexity_defect']
    circularity = features['circularity']
    area = features['area']
    aspect_ratio = features['aspect_ratio']

    if v_shapes == 0 and circularity > 0.3 and aspect_ratio > 0.6: #closed hand
        return 'FIST'
    if v_shapes >= 3 and area > 55000:      #open hand
        return 'PALM'
    if v_shapes == 1 and 0.4 < aspect_ratio < 0.7: #for cursor, victory sign
        return 'POINT'
    return 'UNKOWN'
    

def detect_motion(x1, y1, x2, y2, start, end, min_distance=400, dominant_axis_ration = 2):
    """
        this function will take an initial and final centroid, this delay of frames will be handled by the top level
        also it will get the static hand gesture at the beginning and end of the motion, this also should be handled by the end of level
        and the distance threshold to consider it as motion

        and it will return the motion direction along with the type of this motion,
            if the motion started with palm and ended with palm to the right, it is different from doing the motion in the same direction but with a fist

        also this function does not handle the cursor tracking, this is handles by the top level, were we have the cntroids of the hand and know the gesture 

        the caling of this function with the right parameters is the responsibilty of the top level   
    """
    dx = x1 - x2
    dy = y1 - y2
    distance = np.sqrt(dx ** 2 + dy ** 2)

    motion = "UNKNOWN_MOTION"
    direction = "UNKNOWN_DIRECTION"
    if distance >= min_distance:
        if start == 'PALM' and end == 'PALM':
            motion = "PALM_PALM"
            
        if start == 'PALM' and end == 'FIST':
            motion = "PALM_FIST"

        if start == 'FIST' and end == 'PALM':
            motion = "FIST_PALM"

        if start == 'FIST' and end == 'FIST':
            motion = "FIST_FIST"      

        # we have to make sure it is not a random motion, it has to be in x or y direction, diagonal direction is not allowed here
        if abs(dx) > dominant_axis_ration * abs(dy):
            direction = 'RIGHT' if dx > 0 else 'LEFT'
        elif abs(dy) > dominant_axis_ration * abs(dx):     
            direction = 'DOWN' if dy > 0 else 'UP'



    return motion, direction   


