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
import pandas as pd
from segment import *
def extract_filtered_rbc_features(img, labels, min_area=1500, max_area=10000, remove_borders=True):
    """
    Extracts specific parameters and filters RBCs by size and position.

    Args:
        img: Original image (for intensity calculations)
        labels: Labeled mask from the segmentation step
        min_area: Minimum pixel area to be considered a valid single RBC (filters fragments)
        max_area: Maximum pixel area (filters overlapping clumps)
        remove_borders: If True, excludes cells touching the image edge (partial cells)
    """
    properties = regionprops(labels)
    rbc_data = []

    # Get image dimensions to check for border contact
    img_h, img_w = img.shape[:2]

    for prop in properties:
        
        # --- 1. BORDER FILTERING ---
        # If a cell touches the edge of the image, its area is incomplete.
        # We check if the bounding box coordinates touch 0 or the image width/height.
        if remove_borders:
            y0, x0, y1, x1 = prop.bbox
            if y0 == 0 or x0 == 0 or y1 == img_h or x1 == img_w:
                continue

        # --- 2. SIZE FILTERING ---
        # Filter out:
        # - Small fragments/noise (Area < min_area)
        # - Large overlapping clumps/doublets (Area > max_area)
        if prop.area < min_area or prop.area > max_area:
            continue

        # --- 3. PARAMETER CALCULATION ---
        # Circularity Index (Perfect circle = 1.0)
        circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
        
        # Aspect Ratio (Elongation)
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
        
        # Mean Intensity
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mean_intensity = np.mean(gray_img[labels == prop.label])
        else:
            mean_intensity = prop.mean_intensity

        rbc_data.append({
            "label_id": prop.label,
            "Area": prop.area,
            "Circularity": round(circularity, 3),
            "Aspect Ratio": round(aspect_ratio, 3),
            "Mean Intensity": round(mean_intensity, 2)
        })

    return pd.DataFrame(rbc_data)

def visualize_filtered_rbcs(img, original_labels, filtered_df):
    """
    Draws bounding boxes ONLY around the RBCs that passed the filtering criteria
    present in the filtered_df.
    """
    # Create a copy of the image to draw on
    viz_img = img.copy()

    # If dataframe is empty, return the plain image
    if filtered_df.empty:
        print("No valid cells to visualize.")
        return viz_img

    # Get a set of the valid label IDs for fast lookup
    valid_ids = set(filtered_df['label_id'].values)

    # Iterate through ALL regions found in the initial segmentation
    for prop in regionprops(original_labels):
        # check if this specific cell's ID exists in our filtered list
        if prop.label in valid_ids:
            y0, x0, y1, x1 = prop.bbox
            
            # Draw a distinct green box for valid, counted RBCs
            cv2.rectangle(viz_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Optional: put the area size above it to verify visually
            text = f"RBC {prop.area}"
            cv2.putText(viz_img, text, (x0, max(y0 - 5, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return viz_img

def compute_diagnostic_parameters(df_rbc, img_shape, wbc_count=0, platelet_count=0):
    """
    Computes the summary diagnostic parameters as defined in the provided document table.

    Args:
        df_rbc (pd.DataFrame): The dataframe containing valid RBC features.
        img_shape (tuple): The shape of the original image (height, width, channels).
        wbc_count (int): Total count of WBCs (from label_WBC).
        platelet_count (int): Total count of Platelets (from label_Platelets).
        
    Returns:
        dict: A dictionary of the calculated parameters.
    """
    img_h, img_w = img_shape[:2]
    total_img_area = img_h * img_w

    # 1. Cell Counts
    rbc_count = len(df_rbc)

    # 2. Average Cell Size (Mean area of RBCs)
    # We use the 'Area' column from our extracted features
    avg_size = df_rbc['Area'].mean() if not df_rbc.empty else 0

    # 3. Circularity Index
    # (Global average of the RBC circularity)
    avg_circularity = df_rbc['Circularity'].mean() if not df_rbc.empty else 0

    # 4. Aspect Ratio
    # (Global average of elongation)
    avg_aspect_ratio = df_rbc['Aspect Ratio'].mean() if not df_rbc.empty else 0

    # 5. Cell Density (Distribution of cells per unit area)
    # Calculated as: Total RBCs / Total Image Pixels (Scaled to cells per 10k pixels for readability)
    cell_density = (rbc_count / total_img_area) * 10000

    # 6. Overlap Ratio / Confluency
    # Definition: The degree to which cells cover the slide. 
    # Calculated as: (Sum of all Cell Areas) / Total Image Area
    # This represents the % of the image covered by cells (Crowding Factor).
    total_cell_area = df_rbc['Area'].sum() if not df_rbc.empty else 0
    overlap_ratio = (total_cell_area / total_img_area) * 100

    return {
        "RBC Count": rbc_count,
        "WBC Count": wbc_count,
        "Platelet Count": platelet_count,
        "Avg Cell Size (px)": round(avg_size, 2),
        "Circularity Index": round(avg_circularity, 3),
        "Aspect Ratio": round(avg_aspect_ratio, 3),
        "Cell Density": f"{round(cell_density, 2)} per 10k px",
        "Overlap Ratio (Crowding)": f"{round(overlap_ratio, 2)} %"
    }

# 1. Load Image
img_path = '../data/input/JPEGImages/BloodImage_00003.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not read image at {img_path}")
else:
    # 2. Initial Segmentation (Gets everything, including clumps and border cells)
    # rbc_labels_all contains every segmented object
    initial_result_img, rbc_labels_all = label_RBC(img)
    print(rbc_labels_all)
    # 3. Filtering & Feature Extraction
    # We apply strict filters here. Overlapping cells (large area) and border cells are dropped.
    all_regions = regionprops(rbc_labels_all)
    raw_count = len(all_regions)
    print(f"Total Detected Objects (Raw): {raw_count}")
    df_rbc_filtered = extract_filtered_rbc_features(
        img, 
        rbc_labels_all, 
        remove_borders=True # Removes incomplete cells at edges
    )

    # 4. Visualize the Result
    # This creates an image showing ONLY the cells that survived step 3
    final_filtered_image = visualize_filtered_rbcs(img, rbc_labels_all, df_rbc_filtered)

    # 5. Display Results
    normal_rbc_count = len(df_rbc_filtered)
    print(f"Total Counted (Filtered) RBCs: {normal_rbc_count}")

    # Compare initial segmentation vs final filtered result
    show_images(
        [initial_result_img, final_filtered_image],
        ['Initial Segmentation (All Objects)', f'Final Filtered RBCs (Count: {normal_rbc_count})']
    )

    _, wbc_labels = label_WBC(img)
    wbc_count = len(np.unique(wbc_labels)) - 1 # Subtract background
    # Platelets (Assume you have this function)
    _, platelet_labels = label_Platelets(img)
    platelet_count = len(np.unique(platelet_labels)) - 1

    params = compute_diagnostic_parameters(
        df_rbc_filtered, 
        img.shape, 
        wbc_count=wbc_count, 
        platelet_count=platelet_count
    )
    # 5. Print the "Parameter Extraction" Table
    print("\n" + "="*50)
    print("      3. PARAMETER EXTRACTION REPORT")
    print("="*50)
    print(f"{'Parameter':<30} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Cell Count (RBC)':<30} | {params['RBC Count']}")
    print(f"{'Cell Count (WBC)':<30} | {params['WBC Count']}")
    print(f"{'Cell Count (Platelets)':<30} | {params['Platelet Count']}")
    print("-" * 50)
    print(f"{'Average Cell Size':<30} | {params['Avg Cell Size (px)']}")
    print(f"{'Circularity Index':<30} | {params['Circularity Index']}")
    print(f"{'Aspect Ratio':<30} | {params['Aspect Ratio']}")
    print(f"{'Cell Density':<30} | {params['Cell Density']}")
    print(f"{'Overlap Ratio':<30} | {params['Overlap Ratio (Crowding)']}")
    print("="*50 + "\n")