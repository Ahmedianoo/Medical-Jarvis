"""
SEGMENTATION PIPELINE — GUIDE

GOAL:
- Detect and separate RBCs, WBCs, and platelets from the preprocessed image.
- Handle overlapping RBCs using watershed.
- Produce labeled masks for each cell type.
- Output clean images for counting and feature extraction.

WHAT TO SEE AFTER THIS STEP:
- Each RBC, WBC, and platelet clearly segmented.
- Overlapping RBCs separated into individual cells.
- Minimal background noise in the masks.
- Small platelets visible, WBCs fully captured, RBC edges distinct.

STEPS TO FOLLOW:
1. RBC Segmentation:
   - Threshold V or H/S channels to isolate red/pink RBCs.
   - Morphological operations to clean noise (opening/closing).
   - Distance transform to prepare for watershed.
   - Apply watershed to separate overlapping RBCs.
   - Label each RBC.

2. WBC Segmentation:
   - Threshold H/S or processed L channel to isolate purple WBCs.
   - Morphology to clean edges and remove small noise.
   - Label WBC(s).

3. Platelet Segmentation:
   - Threshold purple hue/intensity.
   - Keep only small-area contours (smaller than RBCs).
   - Remove isolated noise if needed.
   - Label platelets.

4. (Optional) Post-processing:
   - Morphological cleanup for small artifacts.
   - Combine masks if needed for visualization.

NOTES:
- Segmentation relies heavily on correct preprocessing (denoised + CLAHE).
- Parameter tuning (thresholds, kernel sizes) is image-dependent.
- Output masks will be used for counting and feature extraction in the next steps.


color picker may be useful for threshold selection: https://redketchup.io/color-picker
hue colors in opencv range from 0-179 as it uses 8 bits representation. 
 - so if you selected a color in color selector divide the hue value by 2 to get the corresponding opencv hue value.
"""
import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
from scipy import ndimage
import os
from utils.helpers import show_images
from preprocess import preprocess_img
from skimage.filters import frangi



def get_hessian_response(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fr = frangi(
        gray,
        sigmas=range(1, 6),   
        alpha=0.2,
        beta=0.2,
        gamma=5
    )

    fr = cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX)
    fr = np.uint8(fr)
   #  fr = cv2.equalizeHist(fr)
    return fr



def thresholding_RBC(preprocessed_img, lower_red1=np.array([0, 10, 45]), upper_red1= np.array([10, 255, 255]),
                     lower_red2=np.array([160, 10, 45]), upper_red2= np.array([179, 255, 255])):
   # pass preprocessed image to this function 
   # it returns a binary mask for RBCs based on HSV thresholding
   hsv_processed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2HSV)
      
   # the default values for thresholding ranges for the three channels H, S, V are tried and tested on image 00003 

   mask1 = cv2.inRange(hsv_processed_img, lower_red1, upper_red1)
   mask2 = cv2.inRange(hsv_processed_img, lower_red2, upper_red2)

   RBC_mask = mask1 + mask2
   return RBC_mask

def segment_RBC(img):
   #this function takes the img, and return the result which is the image with the RBCs surrounded with redlines, and the watershed result
   preprocessed_img = preprocess_img(img)
   thresholding_RBC_mask = thresholding_RBC(preprocessed_img)

   kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   img_opening = cv2.morphologyEx(thresholding_RBC_mask, cv2.MORPH_OPEN, kernel_opening) #for removing small noise
   kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
   img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing) #for closing small holes in RBCs
   distance = cv2.distanceTransform(img_closing, cv2.DIST_L2, 5)

   _, labels = cv2.connectedComponents(img_closing)

   sure_fg = np.zeros_like(distance, dtype=np.uint8)
   for i in range(1, labels.max()+1):
      component = distance * (labels == i)  # extract distance values of this RBC
      t = 0.3 * component.max()           # threshold for this RBC
      sure_fg[component > t] = 255

   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   sure_bg = cv2.dilate(img_closing, kernel, iterations=3)

   unknown = cv2.subtract(sure_bg, sure_fg)

   _, markers = cv2.connectedComponents(sure_fg)
   markers = markers + 1
   markers[unknown == 255] = 0



   hessian_resp = get_hessian_response(preprocessed_img)
   hessian_color  = cv2.cvtColor(hessian_resp, cv2.COLOR_GRAY2BGR)

   segmented = cv2.watershed(hessian_color, markers)
   result = preprocessed_img.copy()
   result[segmented == -1] = [255, 0, 0]


   distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
   distance_norm = np.uint8(distance_norm)
   sure_fg_vis = sure_fg.copy()
   sure_bg_vis = sure_bg.copy()

   return result, segmented
