"""
===============================================================

1. WHAT WE ARE DOING
We are processing real blood smear microscope images that contain:
- RBCs (light red, circular, often overlapping)
- WBCs (large purple cells)
- Platelets (very small purple dots)


3. PREPROCESSING (WHAT IT DOES)
This step cleans and enhances the image so segmentation works correctly.
We will:
- Reduce noise
- Enhance contrast (cells become clearer)
- Normalize color so thresholding works
- Remove background artifacts

Prepare the image for segmentation
Goal: Make RBC/WBC/platelet boundaries clearer.
1. Smoother background: Noise (dots, grain) reduces.
2. RBC edges become easier to see: Not sharp black edges, but clear boundaries.
3. WBC (purple cell) stands out strongly: After preprocessing, WBC should look: bright purple, clear borders, easy to isolate
4. Platelets (tiny purple dots) remain visible: Even though noise is reduced, platelets should NOT disappear.
5. Colors become more uniform: Images taken under different microscope lighting should become more similar.


4. SEGMENTATION (WHAT IT DOES)
This step actually finds the cells.
We will:
- Threshold colors to isolate RBCs, WBCs, platelets
- Use morphology to clean the masks
- Use distance transform + watershed to separate overlapping RBCs

Goal: Produce clean masks for each cell and prepare for counting.

the dataset we will use is: https://www.kaggle.com/datasets/surajiiitm/bccd-dataset
nice paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05074-2

cv2 is the standard for medical image preprocessing because: Speed, Better support for HSV, filtering, morphology, More control + simpler API
scikit-image is still used, but for specific tasks: watershed implementation, region properties extraction, label images ,morphology with more options

===============================================================

"""
import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
import os
from cells.utils.helpers import show_images

def convert_bgr2rgb(img): # the default reading format in cv2 is BGR so we need to convert it to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convert_rgb2hsv(img): # HSV is better when dealing with blood cells, and using the V channel for preprocessing
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
"""
V (Value) = brightness/intensity → most stable for detecting cell shapes regardless of staining color.

H (Hue) = actual color → RBCs/WBCs vary with stain, lighting, and overlap → unstable.

S (Saturation) = color vividness → also variable, noisy, small platelets may disappear.
"""

def apply_gaussian_blur(img, kernel_size=(5,5), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def apply_median_blur(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)

"""
Gaussian first
Removes small high-frequency noise (sensor grain), Smooths intensity gently, Produces a nice base for median

Median second
Removes isolated dots (salt-and-pepper noise), Preserves edges better than Gaussian,
If you did median first, the Gaussian could blur the edges again, reducing platelet visibility
"""

def convert_rgb2Lab(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def apply_CLAHE(img, cliplimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

"""
Global HE enhances contrast of the whole image
Uneven illumination in blood smears → edges of small/overlapping cells may still be lost
CLAHE = local contrast → edges preserved, platelets remain visible
Low clip limit = good enhancement, but if too week cell edges are lost
High clip limit = agressive enhancement



LAB L channel = better for adaptive contrast / uneven illumination.
L lightness, A green-red, B blue-yellow
"""



def preprocess_img(img, gaussian_kernel_size=(5,5), gaussian_sigama=0, meidan_kernel_size=3, clahe_cliplimit=2.0, clahe_tileGridSize=(8,8)):
    rgb_img = convert_bgr2rgb(img)
    hsv_img = convert_rgb2hsv(rgb_img)

    V_channel = hsv_img[:, :, 2]
    gaussian_blur_V_channel = apply_gaussian_blur(V_channel, kernel_size=gaussian_kernel_size, sigma=gaussian_sigama)
    median_blur_V_channel = apply_median_blur(gaussian_blur_V_channel, kernel_size=meidan_kernel_size)

    processed_hsv_img = hsv_img.copy()
    processed_hsv_img[:, :, 2] = median_blur_V_channel
    rgb_after_hsv_processing = cv2.cvtColor(processed_hsv_img, cv2.COLOR_HSV2RGB)

    lab_img = convert_rgb2Lab(rgb_after_hsv_processing)
    L_channel = lab_img[:, :, 0] 
    clahe_L_channel = apply_CLAHE(L_channel, cliplimit=clahe_cliplimit, tileGridSize=clahe_tileGridSize)

    processed_lab_img = lab_img.copy()
    processed_lab_img[:, :, 0] = clahe_L_channel
    processed_rgb_img = cv2.cvtColor(processed_lab_img, cv2.COLOR_LAB2RGB)

    return processed_rgb_img


# img = cv2.imread('../data/input/JPEGImages/BloodImage_00003.jpg')
# processed_img = preprocess_img(img)
# show_images([convert_bgr2rgb(img), processed_img], titles=['Original Image', 'Preprocessed Image'], figsize=(12,6))


#-----------------------testing the preprocessing step-----------------------#
# rgb_img = convert_bgr2rgb(img)
# hsv_img = convert_rgb2hsv(rgb_img)

# H_channel = hsv_img[:, :, 0]
# S_channel = hsv_img[:, :, 1]
# V_channel = hsv_img[:, :, 2]
# gaussian_blur_V_channel = apply_gaussian_blur(V_channel, kernel_size=(5,5), sigma=0)
# median_blur_V_channel = apply_median_blur(gaussian_blur_V_channel, kernel_size=3)

# processed_hsv_img = hsv_img.copy()
# processed_hsv_img[:, :, 2] = median_blur_V_channel
# rgb_after_hsv_processing = cv2.cvtColor(processed_hsv_img, cv2.COLOR_HSV2RGB)


# lab_img = convert_rgb2Lab(rgb_after_hsv_processing)
# L_channel = lab_img[:, :, 0] 
# A_channel = lab_img[:, :, 1]
# B_channel = lab_img[:, :, 2]
# clahe_L_channel = apply_CLAHE(L_channel, cliplimit=2.0, tileGridSize=(8,8))

# processed_lab_img = lab_img.copy()
# processed_lab_img[:, :, 0] = clahe_L_channel
# processed_rgb_img = cv2.cvtColor(processed_lab_img, cv2.COLOR_LAB2RGB)



# imgs_to_show_1 = [rgb_img, hsv_img, H_channel, S_channel, V_channel]
# titles_1 = ['Original RGB Image', 'HSV Image', 'H Channel', 'S Channel', 'V Channel']
# cmaps_1 = [None, None, 'hsv', 'gray', 'gray']
# show_images(imgs_to_show_1, titles=titles_1, cmaps=cmaps_1, figsize=(20,8))

# imgs_to_show_2 = [rgb_img, rgb_after_hsv_processing,  V_channel, gaussian_blur_V_channel, median_blur_V_channel]
# titles_2 = ['Original RGB Image', 'rgb_after_hsv_processing','Original V Channel',  'After Gaussian Blur', 'After Median Blur']
# cmaps_2 = [None, None, 'gray', 'gray', 'gray']
# show_images(imgs_to_show_2, titles=titles_2, cmaps=cmaps_2, figsize=(20,8))


# imgs_to_show_3 = [rgb_after_hsv_processing, L_channel, clahe_L_channel, A_channel, B_channel]
# titles_3 = ['RGB after HSV Processing', 'Original L Channel', 'CLAHE L Channel', 'A Channel', 'B Channel']
# cmaps_3 = [None, 'gray', 'gray', 'gray', 'gray']
# show_images(imgs_to_show_3, titles=titles_3, cmaps=cmaps_3, figsize=(20,8))

# imgs_to_show_4 = [rgb_img, rgb_after_hsv_processing, processed_rgb_img]
# titles_4 = ['Original RGB Image', 'After HSV V Channel Processing', 'After CLAHE on L Channel']
# cmaps_4 = [None, None, None]
# show_images(imgs_to_show_4, titles=titles_4, cmaps=cmaps_4, figsize=(16,6))






