"""
===============================================================
 PROJECT: Overlapping Blood Cell Separation & Counting
 SUMMARY FOR TEAMMATES
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

===============================================================
"""






