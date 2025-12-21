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

# Add the base project directory to path
project_root = r"C:\Users\Dell\Desktop\Coding\jarvis-optimizers"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports From Cells Processing Module
import cells.utils
import streamlit.components.v1 as components
from cells.utils.helpers import *
from cells import segment, preprocess, wbc_features
from cells.segment import label_RBC
from cells.features import extract_platelet_features, GUI_extract_RBC_features

def RBCs():
    if st.session_state["opencv_img"] is not None:
        opencv_img = st.session_state["opencv_img"]
        col_img, col_table = st.columns([1, 1])

        RBCs_Boxed, _ = label_RBC(opencv_img)
        df_rbc_all_cells, raw_rbc_count, dict_avg_stats = GUI_extract_RBC_features(
            opencv_img
        )

        with col_img:
            st.image(RBCs_Boxed, caption="RBCs Boxed Image", width=700)

        dict_avg_stats["Cell Count"] = raw_rbc_count

        with col_table:
            # st.subheader("RBC Summary Statistics")
            chart_options = [
                "Statistics",
                "Area",
                "Circularity",
                "Aspect Ratio",
                "Correlations",
            ]
            
            st.segmented_control(
                    "Analysis View",
                    options=chart_options,
                    key="active_chart_tab",          # Streamlit tracks the value here
                    default=st.session_state.active_chart_tab,

                )
            # if selected_chart and selected_chart != st.session_state.active_chart_tab:
            #     st.session_state.active_chart_tab = selected_chart
            #     st.rerun()

            tab = st.session_state.active_chart_tab
            if tab == "Statistics":
                df_stats = pd.DataFrame(
                    list(dict_avg_stats.items()), columns=["Feature", "Value"]
                )
                st.table(df_stats)
                csv = df_rbc_all_cells.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="rbc_data.csv")

            elif tab == "Area":
                fig_area = px.histogram(
                    df_rbc_all_cells,
                    x="Area",
                    title="RBC Area Distribution",
                    color_discrete_sequence=["#ff4b4b"],
                )
                st.plotly_chart(fig_area, use_container_width=True)

            elif tab == "Circularity":
                fig_circ = px.histogram(
                    df_rbc_all_cells,
                    x="Circularity",
                    title="RBC Circularity",
                    color_discrete_sequence=["#2ecc71"],
                )
                st.plotly_chart(fig_circ, use_container_width=True)

            elif tab == "Aspect Ratio":
                fig_aspect = px.histogram(
                    df_rbc_all_cells,
                    x="Aspect Ratio",
                    title="RBC Aspect Ratio",
                    color_discrete_sequence=["#9b59b6"],
                )
                st.plotly_chart(fig_aspect, use_container_width=True)

            elif tab == "Correlations":
                fig_scatter = px.scatter(
                    df_rbc_all_cells,
                    x="Area",
                    y="Circularity",
                    title="Area vs Circularity",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Please upload an image on the Home tab first.")
