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
from cells.segment import label_Platelets
from cells.wbc_features import WBCClassifier
from cells.preprocess import preprocess_img
from cells.features import extract_platelet_features

from tabs.Home import Home



def Platelets():
    if st.session_state["opencv_img"] is not None:
        opencv_img = st.session_state["opencv_img"]
        platelete_Boxed, platelete_labels = label_Platelets(opencv_img)
        col_img, col_table = st.columns([1, 1])

        with col_img:
            st.image(platelete_Boxed, caption="Platelets Boxed Image", width=700)

        df, _ = extract_platelet_features(None, platelete_labels)

        if df is not None and not df.empty:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].round(2)
            with col_table:
                st.subheader("Platelet Feature Comparison")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Platelets CSV", data=csv, file_name="platelets.csv"
                )
        else:
            st.warning("No platelets detected.")
    else:
        st.warning("Please upload an image on the Home tab first.")