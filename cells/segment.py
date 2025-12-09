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
from skimage.feature import peak_local_max
import os
from utils.helpers import show_images
from preprocess import preprocess_img


from skimage.filters import frangi
from skimage.color import rgb2gray


def get_hessian_response(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fr = frangi(
        gray,
        sigmas=range(2, 6),   
        alpha=0.5,
        beta=0.9,
        gamma=15
    )

    fr = cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX)
    fr = np.uint8(fr)
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
    pass



# this part is for test till the work is done i put it here if you want to see how it is working, the flow i mean, delete it if you want
img = cv2.imread('../data/input/JPEGImages/BloodImage_00002.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
preprocessed_img = preprocess_img(img)
thresholding_RBC_mask = thresholding_RBC(preprocessed_img)

kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_opening = cv2.morphologyEx(thresholding_RBC_mask, cv2.MORPH_OPEN, kernel_opening) #for removing small noise

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# dilated = cv2.dilate(img_opening, kernel, iterations=1)


kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel_closing) #for closing small holes in RBCs


# hessian_resp = get_hessian_response(preprocessed_img)
# hessian_on_mask = cv2.bitwise_and(hessian_resp, hessian_resp, mask=img_closing)


# Distance transform
distance = cv2.distanceTransform(img_closing, cv2.DIST_L2, 5)
# distance = distance * (1 + (hessian_on_mask / 255.0)) 


# Sure foreground (local maxima)
_, sure_fg = cv2.threshold(distance, 0.2* distance.max(), 255, 0)
sure_fg = np.uint8(sure_fg)



# Sure background
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
sure_bg = cv2.dilate(img_closing, kernel, iterations=3)



# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)


# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(img_closing, cv2.MORPH_GRADIENT, kernel_grad)
hessian_resp = get_hessian_response(preprocessed_img)
grad_hessian = cv2.addWeighted(grad.astype(np.float32), 1.0,
                              hessian_resp.astype(np.float32), 1.0, 0)
grad_hessian = np.uint8(grad_hessian)


# Watershed
# segmented = cv2.watershed(cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR), markers)

grad_hessian_color = cv2.cvtColor(grad_hessian, cv2.COLOR_GRAY2BGR)

segmented = cv2.watershed(grad_hessian_color, markers)

result = preprocessed_img.copy()
result[segmented == -1] = [255, 0, 0]   # red boundaries

distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
distance_norm = np.uint8(distance_norm)
sure_fg_vis = sure_fg.copy()
sure_bg_vis = sure_bg.copy()



show_images(
    [distance_norm, sure_fg_vis, sure_bg_vis, unknown, hessian_resp],
    ['Distance Transform', 'Sure FG (markers)', 'Sure BG', 'Unknown', 'hessian_resp'],
    ['gray', 'gray', 'gray', 'gray', 'gray']
)
show_images([segmented, result, hessian_resp, img_rgb, preprocessed_img], ['segmented', 'result', 'hessian_resp', 'img_rgb', 'preprocessed_img'], [None, 'gray', 'gray', None, None])


show_images([img_rgb, preprocessed_img, thresholding_RBC_mask, img_opening, img_closing], 
            ['Original Image', 'Preprocessed Image', 'thresholding_RBC_mask', 'opening', 'closing'], 
            [None, None, 'gray', 'gray', 'gray'])

