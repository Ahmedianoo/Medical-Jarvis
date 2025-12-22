import sys
from pprint import pprint
import streamlit as st
import pandas as pd

# Add the base project directory to path
project_root = r"C:\Users\Dell\Desktop\Coding\jarvis-optimizers"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports From Cells Processing Module
import cells.utils
import streamlit.components.v1 as components
from cells.utils.helpers import *
from cells import segment, preprocess, wbc_features
from cells.segment import wbc, label_WBC
from cells.wbc_features import WBCClassifier
from cells.preprocess import preprocess_img
from cells.knn_classifier import KNNWBCClassifier

from tabs.Home import Home


def WBCs():
    if st.session_state.get("opencv_img") is not None:
        opencv_img = st.session_state["opencv_img"]
        
        # Create two main columns: Left for Image, Right for Data Tables
        col_left, spacer, col_right = st.columns([2, 0.25, 2])

        with col_left:
            WBCs_Boxed, _ = label_WBC(opencv_img)
            st.image(WBCs_Boxed, caption="Detected WBCs (Bounding Boxes)", use_container_width=True)
            
            # Contextual help for the user
            st.info("The image highlights detected White Blood Cells. The analysis on the right compares morphological features across two different classification models.")

        with col_right:
            st.header("Standard Classifier")
            wbcClass = WBCClassifier()
            preprocessed_wbc = preprocess_img(opencv_img)
            wbc_mask_normal = wbc(preprocessed_wbc)
            features_dict_normal = wbcClass.classify_all_wbcs(preprocessed_wbc, wbc_mask_normal)

            if features_dict_normal:
                df_normal = format_wbc_dataframe(features_dict_normal)
                st.subheader("Standard Feature Analysis")
                st.dataframe(df_normal, use_container_width=True)
                
                csv_normal = df_normal.to_csv().encode("utf-8")
                st.download_button("Download Standard CSV", data=csv_normal, file_name="standard_wbc.csv", key="btn_standard")
            else:
                st.warning("No WBCs detected by Standard Classifier.")

            st.divider() # Visual separation between tables

            st.header("KNN Classifier")
            knnClass = KNNWBCClassifier()
            preprocessed_knn = preprocess_img(opencv_img)
            wbc_mask_knn = wbc(preprocessed_knn)
            features_dict_knn = knnClass.classify_all_wbcs(preprocessed_knn, wbc_mask_knn)

            if features_dict_knn:
                df_knn = format_wbc_dataframe(features_dict_knn)
                st.subheader("KNN Feature Analysis")
                st.dataframe(df_knn, use_container_width=True)
                
                csv_knn = df_knn.to_csv().encode("utf-8")
                st.download_button("Download KNN CSV", data=csv_knn, file_name="knn_wbc.csv", key="btn_knn")
            else:
                st.warning("No WBCs detected by KNN Classifier.")

    else:
        st.warning("Please upload an image on the Home tab first.")

def format_wbc_dataframe(features_list):
    table_data = []
    for cell in features_list:
        row = {"Type": cell.get("type", "Unknown")}
        for k, v in cell.get("cytoplasm_features", {}).items():
            row[f"Cytoplasm: {k}"] = float(v)
        for k, v in cell.get("nucleus_features", {}).items():
            row[f"Nucleus: {k}"] = float(v)
        table_data.append(row)

    df = pd.DataFrame(table_data)
    if not df.empty:
        # Round and Transpose for the vertical view 
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        
        df_vertical = df.transpose()
        df_vertical.columns = [f"Cell {i+1}: {t}" for i, t in enumerate(df["Type"])]
        return df_vertical.astype(str)
    return pd.DataFrame()