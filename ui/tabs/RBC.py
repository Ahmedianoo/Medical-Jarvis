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
        df_rbc_all_cells, raw_rbc_count, dict_avg_stats = GUI_extract_RBC_features(opencv_img)

        with col_img:
            st.image(RBCs_Boxed, caption="RBCs Boxed Image", use_container_width=True)

        print(dict_avg_stats)
        dict_avg_stats["cell_count"] = raw_rbc_count

        with col_table:
            chart_options = ["Statistics", "Area", "Circularity", "Aspect Ratio", "Correlations"]
            
            st.segmented_control(
                "Analysis View",
                options=chart_options,
                key="active_chart_tab",
                default="Statistics"
            )

            tab = st.session_state.active_chart_tab

            if tab == "Statistics":
                df_stats = pd.DataFrame(list(dict_avg_stats.items()), columns=["Feature", "Value"])
                st.table(df_stats)
                csv = df_rbc_all_cells.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="rbc_data.csv")
                st.info("**Summary:** Overall health metrics of the sample. High counts indicate Polycythemia; low counts indicate Anemia.")

            elif tab == "Area":
                fig_area = px.histogram(df_rbc_all_cells, x="Area", title="RBC Area Distribution", color_discrete_sequence=["#ff4b4b"])
                st.plotly_chart(fig_area, use_container_width=True)
                
                with st.expander("Clinical Interpretation: Area"):
                    st.write("**Good:** A tight cluster around the mean (80-100 fL equivalent).")
                    st.write("**Bad (Microcytosis):** Shift to the left. Suggests Iron Deficiency Anemia.")
                    st.write("**Bad (Macrocytosis):** Shift to the right. Suggests Vitamin B12/Folate deficiency.")

            elif tab == "Circularity":
                fig_circ = px.histogram(df_rbc_all_cells, x="Circularity", title="RBC Circularity", color_discrete_sequence=["#2ecc71"])
                st.plotly_chart(fig_circ, use_container_width=True)
                
                with st.expander("Clinical Interpretation: Circularity"):
                    st.write("**Good:** Values near 1.0 indicate healthy, flexible biconcave discs.")
                    st.write("**Bad:** Very high circularity with low area may indicate **Spherocytosis** (fragile cells). Low circularity indicates irregular shapes like **Echinocytes** (burr cells) or **Acanthocytes**.")

            elif tab == "Aspect Ratio":
                fig_aspect = px.histogram(df_rbc_all_cells, x="Aspect Ratio", title="RBC Aspect Ratio", color_discrete_sequence=["#9b59b6"])
                st.plotly_chart(fig_aspect, use_container_width=True)
                
                with st.expander("Clinical Interpretation: Aspect Ratio"):
                    st.write("**Good:** Ratio near 1.0 (Round cells).")
                    st.write("**Bad (Elongation):** High aspect ratio suggests **Sickle Cells** (Drepanocytes) or **Elliptocytes**, which often cause blood flow blockages.")

            elif tab == "Correlations":
                fig_scatter = px.scatter(df_rbc_all_cells, x="Area", y="Circularity", title="Area vs Circularity")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                with st.expander("Clinical Interpretation: Correlations"):
                    st.write("**Interpretation:** This chart helps find 'Outliers'.")
                    st.write("**Good:** A dense cloud of points in the center.")
                    st.write("**Bad:** Scattered points indicate **Anisocytosis** (varying sizes) and **Poikilocytosis** (varying shapes), common in severe blood disorders.")
    else:
        st.warning("Please upload an image on the Home tab first.")