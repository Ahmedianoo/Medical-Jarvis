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



def apply_hessian_frangi(img):
    # frangi uses hessian but makes sure the detected part is a part of a structure

    # it operates only on single channel imgs
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #the frangi filter uses the hessian matrix of the img  
    img_frangi = frangi(img_gray, sigmas=range(1, 8), alpha=0.2, beta=0.2, gamma=5)
    """
    sigmas: controls the scale of structures detected, each sigma in the range is applied, it actually expresses gaussian blur
      low sigma: detects small thin edges, small detail, small blood cells, noise.
      medium sigma: detects normal cell boundaries.
      high sigma: detects larger, thicker shapes, thicker edges.

    alpha:
      low alpha: more sensitive to curved round structures and more strict in rejecting noise 

    beta:
      low beta: highlights strong edges more, rejects weak edges,  
      high beta: includes more noise and soft boundaries, can detect faint edges

    gamma:
      low: includes more noise, keeps low contrast regions
      high: suppresses noise, keeps only strong contrast regions, 
    """

    img_frangi = cv2.normalize(img_frangi, None, 0, 255, cv2.NORM_MINMAX)
    img_frangi = np.uint8(img_frangi)
    """
      frangi outputs a float image => 0 -> 1
      watershed requires 8-bit 0 -> 255
    """

    return img_frangi



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

   #preprocessing
   preprocessed_img = preprocess_img(img, clahe_cliplimit=3, clahe_tileGridSize=(4, 4))

   #apllying thersholding 
   thresholding_RBC_mask = thresholding_RBC(preprocessed_img)

   #opening and closing
   kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   img_opening = cv2.morphologyEx(thresholding_RBC_mask, cv2.MORPH_OPEN, kernel_opening) #for removing small noise
   kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
   img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing) #for closing small holes in RBCs


   distance = cv2.distanceTransform(img_closing, cv2.DIST_L2, 5)
   # distance_smooth = cv2.GaussianBlur(distance, (5,5), 0)

   """
      how far this pixel is from the nearest background pixel
      the center of the RBC has the maximum distance value, the edges of the RBC have distance close to 0

      cv2.DIST_L2 = Euclidean distance
      third paramater = size of the neighborhood used to approximate the distance
   """

  
   num_labels, labels = cv2.connectedComponents(img_closing)
   """
      dividing the components of the mask to apply local thresholding to get better results for the watershed
      it scans the binary mask and assigns a unique label to each connected region

      num_labels includes the background
   """


   sure_foreground = np.zeros_like(distance, dtype=np.uint8)
   for i in range(1, labels.max()+1):
      component = distance * (labels == i)  # extract distance values of this RBC - component, that have this certain label with this certain dist
      t = 0.325 * component.max()           # threshold for this RBC - component
      sure_foreground[component > t] = 255


   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
   sure_bg = cv2.dilate(img_closing, kernel, iterations=3)

   unknown = cv2.subtract(sure_bg, sure_foreground)

   _, markers = cv2.connectedComponents(sure_foreground)
   markers = markers + 1
   markers[unknown == 255] = 0

   """
      sure foreground to make get the centers of the RBCs
      sure background to include the background with us
      unknown will be everything except the forground
      the watershed uses this unknown region to grow in
      and it needs those values: unknown: 0 ,background: 1, cells: 2, 3, 4, ...
   """


   """
      think of making xor between the hessian response and the mask before getting the distance transform to isolate components more effectively
   """
   hessian_response = apply_hessian_frangi(preprocessed_img)

   #the watershed works on 3 channels imgs
   hessian_color  = cv2.cvtColor(hessian_response, cv2.COLOR_GRAY2BGR)

   segmented = cv2.watershed(hessian_color, markers)

   """
      watershed uses the markers to grow, and uses the hessian borders as stopping borders
      it does return labels of the pixels with those values
      boundry of watershed: -1, ,background: 1, cells: 2, 3, 4, ...
   """

   result = preprocessed_img.copy()
   result[segmented == -1] = [255, 0, 0]


   #the none parameter is optional, we could put an image instead of it to return the value to it 
   # distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
   # distance_norm = np.uint8(distance_norm)
   # show_images([result, segmented], ['result', 'segmented'], [None, None])

   return result, segmented


img = cv2.imread('../data/input/JPEGImages/BloodImage_00014.jpg')
segment_RBC(img)