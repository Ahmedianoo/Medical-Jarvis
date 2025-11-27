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
import os
from utils.helpers import show_images
from preprocess import preprocess_img

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
    pass



# this part is for test till the work is done i put it here if you want to see how it is working, the flow i mean, delete it if you want
# img = cv2.imread('../data/input/JPEGImages/BloodImage_00003.jpg')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# preprocessed_img = preprocess_img(img)
# thresholding_RBC_mask = thresholding_RBC(preprocessed_img)
# show_images([img_rgb, preprocessed_img, thresholding_RBC_mask], ['Original Image', 'Preprocessed Image', 'thresholding_RBC_mask'], [None, None, 'gray'])

