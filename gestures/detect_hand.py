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
import cv2
import numpy as np

def detect_hand(frame):

    """
        Convert to YCrCb (better for skin detection)
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    
    
    """
        Skin color range (should be reviewed)
    """
    lower = np.array([0, 131, 70])
    upper = np.array([255, 180, 145])

    """
        mask according to these thresholds
    """
    mask = cv2.inRange(ycrcb, lower, upper)

    # Now we clean the image (like preprocessing)


    """
        Gaussian
        mask - image to perform filter on
        window size - image to perform filter on
        sigmaX - Standard deviation in X direction. 0 → calculated from kernel size automatically
    """
    mask = cv2.GaussianBlur(mask, (3, 3), 0)


    """
        mask - image to perform filter on
        window size - Structuring element. None → uses 3x3 default square
        Number of times erosion is applied. Higher → more shrinking
    """
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)


    """
        using canny to detect edges for the aim of seperating hand and face or any object with similar color of hand if they overlapped
    """
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(frame_gray, 50, 100)
    # invert_edges = cv2.bitwise_not(edges)
    """
     anding mask with mask, only apply the AND where invert_edges != 0. else pixel = 0
    """
    # seperated_mask = cv2.bitwise_and(mask, mask, mask=invert_edges) 
    # seperated_mask = cv2.erode(seperated_mask, (8, 8), iterations=2)
    

    """
        get contours
        cv2.RETR_EXTERNAL - Purpose: Determines which contours are retrieved.
        cv2.CHAIN_APPROX_SIMPLE - Purpose: Determines how the contour points are stored.
        CHAIN_APPROX_SIMPLE → compresses horizontal, vertical, and diagonal segments → stores only endpoints.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask, None
    # Largest contour → likely the hand
    # candidates = []

    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)
    #     circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
    #     aspect_ratio = w / h

    #     if area < 3000:
    #         continue
    #     if circularity > 0.6:
    #         continue
        
    #     if 0.8 < aspect_ratio < 1.3:
    #         continue
    #     cy = y + h // 2
    #     if cy < frame.shape[0] // 3:  # ignore top 1/3
    #         continue

    #     candidates.append(contour)
    #seperate large parts with rectangles    


    # if not candidates:
    #     return mask, None
    
    c = max(contours, key=cv2.contourArea)
    return mask, c



