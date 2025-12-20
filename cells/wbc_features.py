import cv2
import numpy as np

from typing import List, Dict, Optional

from pprint import pprint
from preprocess import preprocess_img
from segment import wbc

class WBCClassifier:
    """    
    This classifier distinguishes between five types of white blood cells:
         Lymphocyte - Neutrophil - Monocyte - Eosinophil - Basophil
    
    The classification is based on morphological and color features extracted
    from the nucleus and cytoplasm regions of each cell.
    """
    
    def __init__(self):
        """
        Initialize the classifier with iteratively determined threshold values.
        Note: These thresholds were determined after many attempts that lastes nearly two days just to adjust thresholds
        """
        self.thresholds = {
            'lymphocyte_hue_max': 82,
            'lymphocyte_circularity_min': 0.60,
            
            'neutrophil_hue_min': 100,
            'neutrophil_texture_min': 85,
            
            'monocyte_solidity_max': 0.72,
            'monocyte_area_min': 10000,
            
            'eosinophil_hue_min': 82,
            'eosinophil_hue_max': 100
        }
    
    def extract_individual_wbcs(self, original_img: np.ndarray, wbc_mask: np.ndarray) -> List[Dict]:
        """
        Separate and extract individual white blood cell regions from the image.
        
        Args:
            original_img: Original preprocessed image in RGB format
            wbc_mask: Binary mask indicating WBC nucleus locations
            
        Returns:
            List of dictionaries, each containing:
                - image: Cropped cell region
                - nucleus_mask: Binary mask for the nucleus
                - centroid: Center coordinates of the cell
                - bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wbc_mask)
        wbc_regions = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Add padding around the cell to capture surrounding cytoplasm
            padding = 22
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(original_img.shape[1], x + w + padding), min(original_img.shape[0], y + h + padding)
            
            wbc_regions.append({
                'image': original_img[y1:y2, x1:x2],
                'nucleus_mask': (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255,
                'centroid': centroids[i],
                'bbox': (x1, y1, x2, y2)
            })
        return wbc_regions

    def extract_nucleus_features(self, nucleus_mask: np.ndarray) -> Optional[Dict]:
        """
        Extract morphological features from the nucleus region.
        
        Args:
            nucleus_mask: Binary mask of the nucleus
            
        Returns:
            Dictionary containing:
                - area: Nucleus area in pixels
                - circularity: Measure of how circular the nucleus is (0-1)
                - solidity: Ratio of contour area to convex hull area
                - num_lobes: Number of nuclear lobes detected
            Returns None if no valid contour is found.
        """
        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Circularity: 1.0 for perfect circle, lower for irregular shapes
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Solidity: Ratio of actual area to convex hull area
        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        # Estimate nuclear lobes by eroding the mask and counting components
        eroded = cv2.erode(nucleus_mask, np.ones((3, 3), np.uint8), iterations=3)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(eroded)
        num_lobes = max(1, num_labels - 1)
        
        return {
            'area': area, 
            'circularity': circularity, 
            'solidity': solidity, 
            'num_lobes': num_lobes
        }

    def extract_cytoplasm_features(self, cell_img: np.ndarray, nucleus_mask: np.ndarray) -> Optional[Dict]:
        """
        Extract features from the cytoplasm region surrounding the nucleus.
        
        Args:
            cell_img: Cropped image of the cell in RGB format
            nucleus_mask: Binary mask of the nucleus
            
        Returns:
            Dictionary containing:
                - mean_hue: Average hue value in cytoplasm (HSV color space)
                - texture_variance: Variance of Laplacian (measures granularity)
                - cn_ratio: Cytoplasm-to-nucleus area ratio
            Returns None if cytoplasm cannot be extracted.
        """
        if cell_img.size == 0 or nucleus_mask.size == 0: 
            return None
        
        # Create binary mask for entire cell using Otsu's method
        gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
        _, cell_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Subtract nucleus from cell mask to get cytoplasm
        nucleus_dilated = cv2.dilate(nucleus_mask, np.ones((3, 3), np.uint8), iterations=1)
        cytoplasm_mask = cv2.bitwise_and(cell_mask, cv2.bitwise_not(nucleus_dilated))
        
        if np.sum(cytoplasm_mask) == 0: 
            return None
        
        # Extract hue from HSV color space
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_RGB2HSV)
        mean_hue = np.mean(hsv[cytoplasm_mask > 0][:, 0])
        
        # Calculate texture using Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian[cytoplasm_mask > 0])
        
        # Calculate cytoplasm to nucleus ratio
        cn_ratio = np.sum(cytoplasm_mask > 0) / np.sum(nucleus_mask > 0)
        
        return {
            'mean_hue': mean_hue, 
            'texture_variance': texture_variance, 
            'cn_ratio': cn_ratio
        }

    def classify_wbc(self, nf: Optional[Dict], cf: Optional[Dict]) -> str:
        """
        Classify a white blood cell based on its nucleus and cytoplasm features.
        
        Classification strategy:
        1. Lymphocytes are identified by high circularity and low hue
        2. Neutrophils are the most common, identified by higher hue or multi-lobed nuclei
        3. Monocytes have large, irregular nuclei with low solidity
        4. Eosinophils fall in a specific hue range (orange-pink cytoplasm)
        5. Basophils are assigned to remaining cases
        
        Args:
            nf: Nucleus features dictionary
            cf: Cytoplasm features dictionary
            
        Returns:
            String indicating cell type: 'LYMPHOCYTE', 'NEUTROPHIL', 'MONOCYTE', 'EOSINOPHIL', or 'BASOPHIL'
        """
        if not nf or not cf: 
            return "BASOPHIL"
        
        t = self.thresholds
        
        # Lymphocytes have round, compact nuclei with low hue values
        if nf['circularity'] > t['lymphocyte_circularity_min'] and cf['mean_hue'] < t['lymphocyte_hue_max']:
            return 'LYMPHOCYTE'

        # Neutrophils typically have higher hue values or multiple lobes with granular texture
        if cf['mean_hue'] >= t['neutrophil_hue_min'] or (nf['num_lobes'] >= 2 and cf['texture_variance'] > t['neutrophil_texture_min']):
            return 'NEUTROPHIL'

        # Monocytes are large cells with irregular, kidney-shaped nuclei
        if nf['area'] > t['monocyte_area_min'] and nf['solidity'] < t['monocyte_solidity_max']:
            return 'MONOCYTE'

        # Eosinophils have characteristic orange-pink cytoplasm (specific hue range)
        if t['eosinophil_hue_min'] <= cf['mean_hue'] < t['eosinophil_hue_max']:
            return 'EOSINOPHIL'
            
        # Default classification for rare or ambiguous cases
        return 'BASOPHIL'

    def classify_all_wbcs(self, original_img: np.ndarray, wbc_mask: np.ndarray) -> List[Dict]:
        """
        This function is given the preprocessed_img and mask of wbc
        Complete classification pipeline for all white blood cells in an image.
        
        Args:
            original_img: Preprocessed image in RGB format
            wbc_mask: Binary mask indicating WBC nucleus locations
            
        Returns:
            List of dictionaries containing classification results for each cell:
                - id: Cell index
                - type: Classified cell type
                - centroid: Center coordinates
                - bbox: Bounding box
                - nucleus_features: Extracted nucleus features
                - cytoplasm_features: Extracted cytoplasm features
        """
        wbc_regions = self.extract_individual_wbcs(original_img, wbc_mask)
        results = []
        
        for idx, wbc in enumerate(wbc_regions):
            nf = self.extract_nucleus_features(wbc['nucleus_mask'])
            cf = self.extract_cytoplasm_features(wbc['image'], wbc['nucleus_mask'])
            
            results.append({
                'id': idx, 
                'type': self.classify_wbc(nf, cf),
                'centroid': wbc['centroid'], 
                'bbox': wbc['bbox'],
                'nucleus_features': nf,
                'cytoplasm_features': cf
            })
        return results


classifier = WBCClassifier()

# To be compared with 
# https://www.kaggle.com/code/mohamedkamal77/cnn-train-0-99-val-0-98-test-0-986

# img = cv2.imread("../data/input/JPEGImages/BloodImage_00015.jpg")
# pre = preprocess_img(img)
# wbc_mask = wbc(pre)
# res = classifier.classify_all_wbcs(pre, wbc_mask)
# pprint(res)