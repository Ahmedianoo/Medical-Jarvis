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


Explanantion of cv2.connectedComponentsWithStats(mask):
- White pixels touching each other (8-neighbourhood) form a component (a blob).
- Give a label to each blob, and Measure its area (how many white pixels),
  then we can delete all blobs that are “too big” (WBCs) or “too small” (noise) and keep the “medium–small” ones (platelets).
- Returns:
   - num_labels: number of connected components (blobs) found (including background)
   - labels: an image where each pixel has a label corresponding to its blob
   - stats: for each label, gives [x, y, width, height, area]
   - centroids: (x, y) coordinates of the center of each blob
"""
import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
from scipy import ndimage
import os
from utils.helpers import show_images
from skimage.measure import regionprops, label
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
   """
      pass preprocessed image to this function 
      it returns a binary mask for RBCs based on HSV thresholding
   """

   hsv_processed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2HSV)
      
   # the default values for thresholding ranges for the three channels H, S, V are tried and tested on image 00003 

   mask1 = cv2.inRange(hsv_processed_img, lower_red1, upper_red1)
   mask2 = cv2.inRange(hsv_processed_img, lower_red2, upper_red2)

   RBC_mask = mask1 + mask2
   return RBC_mask


def segment_RBC(img):
    
   """
      this function takes the img, and return the result which is the image with the RBCs surrounded with redlines, and the watershed result
      it basically starts with preprocessing and the thresholding with hue to extract RBCc
      and then it performs the distance transform then applying watershed
      the watershed uses the markers from the distance transform and the stopping boundries from the hessian filter
   """

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


def label_RBC(img):
   """
      this function takes the img and then it apply the segmentation (segment_RBC function)
      after that we get the labels of the watershed image -> remove noise, merge broken cells
      then return the image with the boxes and the updated and enhanced labels
   """
   
   #segmentation
   cells_boundries, watershed_img = segment_RBC(img)

   """
      getting the RBCs mask by putting each unknown or background in the watershed will be zero and other regions remain the same
   """
   RBCs_mask = np.zeros_like(watershed_img, dtype = np.uint8)
   RBCs_mask = np.where(watershed_img > 1, watershed_img, 0)

   """
      labeling the different cells in the mask
   """
   RBCs_labels = label(RBCs_mask)

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   resulted_boxes = img_rgb.copy()

   
   min_area = 1200   # min area a red blood cell could have
   max_distance = 63 # distance between centers to merges small fragments
   noise_area = 120  # all labels with area below this one is removed

   """
      regionprops, returns the labels but with some features. like area and centroids
   """
   cells = regionprops(RBCs_labels)

   """
      removing the noise before starting the merge process, so the noise is not merged
      the RBCs_labels must be updated each time we make modifications to the labels
      and the cells(region props) has to be updated as well
   """
   for cell in cells:
      if cell.area < noise_area:
         RBCs_labels[RBCs_labels == cell.label] = 0  

   cells = regionprops(RBCs_labels) 


   """
      here we are trying to merge small detected fragments to create the original cell
      first we get the area of the current cell, if it is below the threshold, it could be merged but we must check if there is near cells first
      and the cells must be updated each time a merge operation happen that's why we break many loops
   """
   while True:
      updated = False
      cells = regionprops(RBCs_labels)
      for cell in cells:
         cx_current, cy_current = cell.centroid
         if cell.area < min_area: 

            for other_cell in cells:

               if cell.label == other_cell.label:
                  continue

               cx_other, cy_other = other_cell.centroid
               distance = np.sqrt((cx_current - cx_other)**2 + (cy_current-cy_other)**2)
               if distance < max_distance:
                  RBCs_labels[RBCs_labels == cell.label] = other_cell.label 
                  updated = True
                  break

            if updated:
               break
      if not updated:
         break
   """
      getting the resulted boxes for the most updated cells and drawing green rectangle with text "RBC" above this rectangle
   """
   # for cell in cells:
   #    y0, x0, y1, x1 = cell.bbox
   #    cv2.rectangle(resulted_boxes, (x0, y0), (x1, y1), (0, 255, 255), 1)
   #    text_pos = (x0, max(y0 - 5, 0))
   #    cv2.putText(resulted_boxes, "RBC", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

   return resulted_boxes, RBCs_labels

# img = cv2.imread('../data/input/JPEGImages/BloodImage_00014.jpg')
# resulted_boxes, RBCs_labels = segment_label_RBC(img)
# show_images([resulted_boxes], ['resulted_boxes'], [None])


def segmenting_purple_cells(preprocessed_img):
   hsv_preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2HSV)
   # Platelets range from 250 to 320 degrees in hue channel
   # cv2 only range from 0 to 180, so purple range is from 115 to 130 in cv2 hue scale
   lower_purple = np.array([115, 40, 40])
   upper_purple = np.array([130, 255, 255])

   # inRange, checks (lower_purple <= pixel <= upper_purple) for each pixel
   # if true, pixel value is set to 255, else 0
   mask = cv2.inRange(hsv_preprocessed_img, lower_purple, upper_purple)

   kernel = np.ones((3, 3), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # remove any small white noise
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # close small black holes in the detected purple cells

   result = cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=mask)
   return result, mask


def platelet(preprocessed_img, min_area=20, max_area=500):
   _, purple_mask = segmenting_purple_cells(preprocessed_img)
   num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(purple_mask)
   plat_mask = np.zeros_like(purple_mask)

   # Index 0 is background label (not needed), so we start at index 1
   for i in range(1, num_labels):
      area = stats[i, cv2.CC_STAT_AREA]
      if min_area < area < max_area:
         plat_mask[labels == i] = 255

   return plat_mask


def wbc(preprocessed_img, min_area=800, max_area=None):
   _, purple_mask = segmenting_purple_cells(preprocessed_img)
   num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(purple_mask)

   wbc_mask = np.zeros_like(purple_mask)

   # Index 0 is background label (not needed), so we start at index 1
   for i in range(1, num_labels):
      area = stats[i, cv2.CC_STAT_AREA]
      if area >= min_area and (max_area is None or area <= max_area):
         wbc_mask[labels == i] = 255

   return wbc_mask


def label_WBC(img):
   """
   Segments WBCs, labels them, and draws bounding boxes.
   WBCs are typically identified by their large purple nucleus.
   """
   preprocessed_img = preprocess_img(img, clahe_cliplimit=3, clahe_tileGridSize=(4, 4))

   wbc_mask_binary = wbc(preprocessed_img)
   wbc_labels = label(wbc_mask_binary)

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
   resulted_boxes = img_rgb.copy()

   cells = regionprops(wbc_labels)
   for cell in cells:
      y0, x0, y1, x1 = cell.bbox
      cv2.rectangle(resulted_boxes, (x0, y0), (x1, y1), (0, 0, 255), 2)
      text_pos = (x0, max(y0 - 5, 0))
      cv2.putText(resulted_boxes, "WBC", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
   return resulted_boxes, wbc_labels

def label_Platelets(img):
   """
   Segments Platelets, labels them, and draws bounding boxes.
   Platelets are small purple spots.
   """
   preprocessed_img = preprocess_img(img, clahe_cliplimit=3, clahe_tileGridSize=(4, 4))

   platelet_mask_binary = platelet(preprocessed_img)

   platelet_labels = label(platelet_mask_binary)

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   resulted_boxes = img_rgb.copy()

   cells = regionprops(platelet_labels)

   for cell in cells:
      y0, x0, y1, x1 = cell.bbox

      cv2.rectangle(resulted_boxes, (x0, y0), (x1, y1), (255, 0,0), 1)
      text_pos = (x0, max(y0 - 5, 0))
      cv2.putText(resulted_boxes, "Platelet", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0,0), 1, cv2.LINE_AA)
   return resulted_boxes, platelet_labels

def label_all_cells(img):
   """
   labels all cells, and draws bounding boxes.
   """
   _, rbc_labels = label_RBC(img)

   preprocessed_img = preprocess_img(img, clahe_cliplimit=3, clahe_tileGridSize=(4, 4))
   wbc_mask_binary = wbc(preprocessed_img)
   platelet_mask_binary = platelet(preprocessed_img)

   wbc_labels = label(wbc_mask_binary)
   platelet_labels = label(platelet_mask_binary)

   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   final_result = img_rgb.copy()

   # --- Draw RBCs (Green) ---
   for cell in regionprops(rbc_labels):
      y0, x0, y1, x1 = cell.bbox
      cv2.rectangle(final_result, (x0, y0), (x1, y1), (0, 255, 0), 1)
      cv2.putText(final_result, "RBC", (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

   # --- Draw WBCs (Blue) ---
   for cell in regionprops(wbc_labels):
      y0, x0, y1, x1 = cell.bbox
      cv2.rectangle(final_result, (x0, y0), (x1, y1), (0, 0, 255), 2)
      cv2.putText(final_result, "WBC", (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

   # --- Draw Platelets (Red) ---
   for cell in regionprops(platelet_labels):
      y0, x0, y1, x1 = cell.bbox
      cv2.rectangle(final_result, (x0, y0), (x1, y1), (255, 0, 0), 1)
      cv2.putText(final_result, "Platelet", (x0, max(y0 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

   return final_result




img = cv2.imread('../data/input/JPEGImages/BloodImage_00003.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
preprocessed_img = preprocess_img(img)
plat_mask = platelet(preprocessed_img)
wbc_mask = wbc(preprocessed_img)
show_images([img_rgb, preprocessed_img, plat_mask, wbc_mask], ['Original Image', 'Preprocessed Image', 'Platelet Mask', 'wbc mask'], [None, None, 'gray', 'gray'])

# img = cv2.imread('../data/input/JPEGImages/BloodImage_00014.jpg')
# resulted_boxes, RBCs_labels = segment_label_RBC(img)
# show_images([resulted_boxes], ['resulted_boxes'], [None])

img = cv2.imread('../data/input/JPEGImages/BloodImage_00003.jpg')
#wbc_result_img, wbc_labels = label_WBC(img)
#platelet_result_img, platelet_labels = label_Platelets(img)
all_cells_result_img = label_all_cells(img)
show_images([all_cells_result_img], ['All Cells Detection'])

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
preprocessed_img = preprocess_img(img)
_, purple_mask = segmenting_purple_cells(preprocessed_img)
wbc_binary = wbc(purple_mask)
platelet_binary = platelet(purple_mask)

show_images([purple_mask, wbc_binary, platelet_binary], ['Purple Mask (All)', 'WBC Binary Mask', 'Platelet Binary Mask'])