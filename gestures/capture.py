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



import cv2
import numpy as np


def initialize_camera(index=0):
    """
        0 here referes to default webcam
        1 - external webcam
        2 - another camer
        also a video can be added
    """

    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open camera")
    
    return cap


def get_frame(cap):



    """
        this capture a single image from camera
        ret - boolen tells if it is catpured correctly or not
        frame - the actual image captured
    """    
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        return None

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1) 
    return frame




