import sys
from pprint import pprint
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import cv2
import subprocess
import os
import time

# --- 1. Page Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="Medical Jarvis")

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

from tabs.Home import Home


def WBCs():
    if st.session_state["opencv_img"] is not None:
        opencv_img = st.session_state["opencv_img"]
        WBCs_Boxed, _ = label_WBC(opencv_img)
        col_img, col_table = st.columns([1, 1])

        with col_img:
            st.image(WBCs_Boxed, caption="WBCs Boxed Image", width=700)

        wbcClass = WBCClassifier()
        preprocessed = preprocess_img(opencv_img)
        wbc_mask = wbc(preprocessed)
        features_dict = wbcClass.classify_all_wbcs(preprocessed, wbc_mask)

        table_data = []
        for cell in features_dict:
            row = {"Type": cell["type"]}
            for k, v in cell.get("cytoplasm_features", {}).items():
                row[f"Cytoplasm: {k}"] = float(v)
            for k, v in cell.get("nucleus_features", {}).items():
                row[f"Nucleus: {k}"] = float(v)
            table_data.append(row)

        df = pd.DataFrame(table_data)
        if not df.empty:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].round(2)

            with col_table:
                st.subheader("WBC Feature Analysis")
                df_vertical = df.transpose()
                df_vertical.columns = [
                    f"Cell {i+1}: {t}" for i, t in enumerate(df["Type"])
                ]
                st.dataframe(df_vertical.astype(str), use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download WBC CSV", data=csv, file_name="wbc_analysis.csv"
                )
        else:
            st.warning("No WBCs detected.")
    else:
        st.warning("Please upload an image on the Home tab first.")