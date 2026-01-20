import sys
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import base64
import streamlit.components.v1 as components

# --- 1. Page Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="Medical Jarvis")

# Add the base project directory to path
project_root = r"C:\Users\ahmed\OneDrive\Desktop\Semester work\image\Project\jarvis-optimizers"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports From Cells Processing Module
#import cells.utils
import streamlit.components.v1 as components
#from cells.utils.helpers import *
#from cells import segment, preprocess, wbc_features
#from cells.segment import label_all_cells, label_RBC, wbc, label_WBC, label_Platelets
#from cells.wbc_features import WBCClassifier
#from cells.preprocess import preprocess_img
#rom cells.features import extract_platelet_features, GUI_extract_RBC_features

# Import Tabs
from tabs.Home import Home
from tabs.RBC import RBCs
from tabs.WBC import WBCs
from tabs.Platelet import Platelets

from tabs.Camera import toggle_camera
from tabs.Camera import read_gesture_state


st.markdown(
    """
    <style>
        /* Target the main app container */
        .stApp {
            background: #F8FAFC !important;
            background-attachment: fixed !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_base64_font(font_path):
    with open(font_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


font_path = r"C:\Users\ahmed\OneDrive\Desktop\Semester work\image\Project\jarvis-optimizers\ui\fonts\Aquire-BW0ox.otf"
font_base64 = get_base64_font(font_path)


# --- 2. Initialize Session State ---
tabs_options = ["Home", "Red Blood Cells", "White Blood Cells", "Platelets"]

if "active_chart_tab" not in st.session_state:
    st.session_state.active_chart_tab = "Statistics"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"
if "opencv_img" not in st.session_state:
    st.session_state.opencv_img = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "camera_process" not in st.session_state:
    st.session_state.camera_process = None
if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = {
        "gesture": "UNKNOWN",
        "motion": "UNKNOWN",
        "raw": "No data",
    }
if "previous_gesture" not in st.session_state:
    st.session_state.previous_gesture = {
        "gesture": "UNKNOWN",
        "motion": "UNKNOWN",
        "raw": "No data",
    }


# --- 4. Header with Camera Button ---
col_title, col_gesture, col_button = st.columns([2, 2, 1], vertical_alignment="center")

with col_title:
    font_html = f"""
    <style>
    @font-face {{
        font-family: 'JarvisFont';
        src: url(data:font/opentype;base64,{font_base64}) format('opentype');
        font-weight: normal;
        font-style: normal;
    }}

    .medical-title {{
        font-family: 'JarvisFont', Arial, sans-serif;
        color: #0F172A;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }}
    
    .medical-subtitle {{
        font-family: sans-serif;
        color: #64748B;
        font-size: 13px;
        margin-top: 4px;
        margin-bottom: 0;
        padding: 0;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        line-height: 1.2;
    }}
    </style>

    <div style="padding: 0; margin: 0;">
        <h1 class="medical-title">
            Medical <span style="color: #2563EB;">Jarvis</span>
        </h1>
        <p class="medical-subtitle">Just A Rather Very Intelligent Service</p>
    </div>
    """

    components.html(font_html, height=100)

with col_gesture:
    if st.session_state.camera_active:
        st.markdown(
            f"""
        <div style='display: flex; gap: 10px; padding: 12px; background-color: #0F172A ; border-radius: 5px;  align-items: center;'>
            <div style='flex: 1; border-right: 1px solid #444; padding-right: 10px;'>
                <p style='margin: 0; font-size: 15px; color: #fff; '>Current</p>
                <p style='margin: 2px 0; font-size: 13px; color: #00ff00;'><b>G:</b> {st.session_state.current_gesture['gesture']}</p>
                <p style='margin: 0; font-size: 13px; color: #ff00ff;'><b>M:</b> {st.session_state.current_gesture['motion']}</p>
            </div>
            <div style='flex: 1; padding-left: 5px;'>
                <p style='margin: 0; font-size: 15px; color: #fff;'>Previous</p>
                <p style='margin: 2px 0; font-size: 12px; color:  #00ff00;'><b>G:</b> {st.session_state.previous_gesture['gesture']}</p>
                <p style='margin: 0; font-size: 12px; color: #aa00aa;'><b>M:</b> {st.session_state.previous_gesture['motion']}</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


with col_button:
    col_camera_left, col_camera_right = st.columns([2, 1], vertical_alignment="center")
    icon = (
        ":material/videocam_off:"
        if st.session_state.camera_active
        else ":material/videocam:"
    )
    button_type = "secondary" if st.session_state.camera_active else "primary"

    with col_camera_right:
        # Logic for state
        is_active = st.session_state.camera_active

        # 1. Configuration based on state
        icon = ":material/videocam_off:" if is_active else ":material/videocam:"

        bg_color = "#28a745" if is_active else "#FF4B4B"
        text_color = "white"

        # 2. Apply CSS
        with stylable_container(
            key="camera_button_style",
            css_styles=f"""
                /* 1. Force the outer container to match your desired size */
                div[data-testid="stButton"] {{
                    width: 100px !important;
                    height: 50px !important;
                }}

                /* 2. Target the button and force it to fill that container */
                div[data-testid="stButton"] > button {{
                    background-color: {bg_color} !important;
                    color: {text_color} !important;
                    width: 100% !important;
                    height: 100% !important;
                    border: none !important;
                    border-radius: 8px !important;
                    transition: all 0.3s ease;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }}
                
                /* 3. Scale the icon */
                span[aria-label="videocam icon"] span {{
                    font-size: 2.2rem !important; /* Slightly reduced to fit 50px height */
                    line-height: 1 !important;
                }}

                button:hover {{
                    opacity: 0.9 !important;
                    transform: scale(1.02);
                }}
            """,
        ):
            # Keep use_container_width=True so it fills the 100px container we defined above
            if st.button(icon, use_container_width=True):
                toggle_camera()
                st.rerun()


# Read gestures BEFORE rendering the segmented control
@st.fragment(run_every=1.0)
def background_gesture_listener():
    if st.session_state.camera_active:
        if read_gesture_state():
            st.rerun()


background_gesture_listener()

selected_tab = st.segmented_control(
    "Navigation",
    options=tabs_options,
    selection_mode="single",
    default=st.session_state.active_tab,
)

# Sync manual clicks on segmented control back to session state
if selected_tab and selected_tab != st.session_state.active_tab:
    st.session_state.active_tab = selected_tab
    st.rerun()


st.divider()

# =========================================================
# ================= Tab Content Logic =====================
# =========================================================

# HOME TAB
if st.session_state.active_tab == "Home":
    Home()

# RED BLOOD CELLS TAB
elif st.session_state.active_tab == "Red Blood Cells":
    RBCs()

# WHITE BLOOD CELLS TAB
elif st.session_state.active_tab == "White Blood Cells":
    WBCs()

# PLATELETS TAB
elif st.session_state.active_tab == "Platelets":
    Platelets()

# =========================================================
