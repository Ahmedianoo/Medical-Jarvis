import streamlit as st
import plotly.express as px
import numpy as np
import cv2
import sys
import time

project_root = r"C:\Users\Dell\Desktop\Coding\jarvis-optimizers"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import cells.utils
import streamlit.components.v1 as components
from cells.utils.helpers import *
from cells import segment, preprocess, wbc_features
from cells.segment import label_all_cells, label_RBC, wbc, label_WBC, label_Platelets
from cells.wbc_features import WBCClassifier
from cells.preprocess import preprocess_img
from cells.features import extract_platelet_features, GUI_extract_RBC_features


def Home():
    with st.form(key="upload_form"):
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        col_submit, spacer, col_reset = st.columns([1, 5, 1])

        with col_submit:
            submit = st.form_submit_button(
                ":material/upload: Submit",
                use_container_width=True,
                type="primary"
            )
        
        with col_reset:
            reset = st.form_submit_button(
                ":material/refresh: Reset",
                use_container_width=True,
                type="secondary"
            )

    if reset:
        st.session_state["opencv_img"] = None
        st.rerun()

    if submit:
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if opencv_img is not None:
                st.success("Submitted Image Successfully")
                st.session_state["opencv_img"] = opencv_img
                after_labeling = label_all_cells(opencv_img)

                first_co, cent_co, last_co = st.columns([1, 6, 1])
                with cent_co:
                    st.image(after_labeling, caption="Processed Image", width=500)
        else:
            st.warning("No image uploaded.")