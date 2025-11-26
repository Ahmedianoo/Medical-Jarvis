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


"""
import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
import os
from utils.helpers import show_images
