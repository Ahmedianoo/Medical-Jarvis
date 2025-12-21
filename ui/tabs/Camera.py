import sys
import streamlit as st
import subprocess
import os
import time

PROJECT_ROOT = r"C:\Users\Dell\Desktop\Coding\jarvis-optimizers"
TABS_OPTIONS = ["Home", "Red Blood Cells", "White Blood Cells", "Platelets"]
CHART_OPTIONS = ["Statistics", "Area", "Circularity", "Aspect Ratio", "Correlations",]


def toggle_camera():
    """Toggle camera script on/off"""
    if st.session_state.camera_active:
        # Close camera
        if st.session_state.camera_process is not None:
            st.session_state.camera_process.terminate()
            st.session_state.camera_process = None
        st.session_state.camera_active = False
    else:
        # Start camera script
        camera_script_path = os.path.join(PROJECT_ROOT, "gestures", "main.py")

        # Check if script exists
        if os.path.exists(camera_script_path):
            try:
                # Run the camera script as a subprocess
                process = subprocess.Popen([sys.executable, camera_script_path])
                st.session_state.camera_process = process
                st.session_state.camera_active = True
            except Exception as e:
                st.error(f"Failed to start camera: {e}")
        else:
            st.error(f"Camera script not found at: {camera_script_path}")


def read_gesture_state():
    isTabChanged = False
    gesture_file = os.path.join(PROJECT_ROOT, "gestures", "gesture_command.txt")

    try:
        if os.path.exists(gesture_file):
            with open(gesture_file, "r") as f:
                content = f.read().strip()
                if not content:
                    return False

                parts = content.split(",")
                if len(parts) == 2:
                    new_gesture = {
                        "gesture": parts[0],
                        "motion": parts[1],
                        "raw": content,
                    }

                    # Check for state change
                    if new_gesture["raw"] != st.session_state.current_gesture["raw"]:
                        
                        # Transition logic
                        prev_motion = st.session_state.current_gesture["motion"]
                        prev_gesture = st.session_state.current_gesture["gesture"]
                        
                        # =====================================
                        # SWIPE RIGHT
                        # =====================================
                        if (
                            prev_motion == "PALM_PALM_RIGHT"
                            and new_gesture["motion"] == "UNKNOWN"
                        ):
                            current_idx = TABS_OPTIONS.index(
                                st.session_state.active_tab
                            )
                            next_idx = (current_idx + 1) % len(TABS_OPTIONS)

                            # Update the actual state
                            st.session_state.active_tab = TABS_OPTIONS[next_idx]
                            isTabChanged = True
                            
                        # ===========================================
                        # SWIPE LEFT
                        # ===========================================
                        if (
                            prev_motion == "PALM_PALM_LEFT"
                            and new_gesture["motion"] == "UNKNOWN"
                        ):
                            current_idx = TABS_OPTIONS.index(
                                st.session_state.active_tab
                            )
                            next_idx = (current_idx - 1) % len(TABS_OPTIONS)

                            # Update the actual state
                            st.session_state.active_tab = TABS_OPTIONS[next_idx]
                            isTabChanged = True
                        
                        # ===========================================
                        # CLOSE CAMERA
                        # ===========================================
                        if (
                            prev_gesture == "UNKNOWN"
                            and new_gesture["gesture"] == "POINT"
                        ):
                            if st.session_state.camera_process is not None:
                                st.session_state.camera_process.terminate()
                                st.session_state.camera_process = None
                            
                            # 2. Reset the state flags
                            st.session_state.camera_active = False
                            
                            # 3. Trigger a rerun so the UI button updates to "Open"
                            isTabChanged = True
                            
                        # ===========================================
                        # RESET
                        # ===========================================
                        if (
                            prev_gesture == "FIST"
                            and new_gesture["gesture"] == "PALM"
                        ):
                            # 1. Clear the image data
                            st.session_state["opencv_img"] = None
                            
                            # 2. Force navigation back to the Home tab
                            st.session_state.active_tab = "Home"
                            
                            # 3. Signal a rerun to refresh the UI and reset the navigation bar
                            isTabChanged = True
                        
                        # ===========================================
                        # SWIPE CHART RIGHT
                        # ===========================================
                        if (
                            prev_motion == "FIST_FIST_RIGHT"
                            and new_gesture["motion"] == "UNKNOWN"
                            and st.session_state.active_tab == "Red Blood Cells"
                        ):
                            current_idx = CHART_OPTIONS.index(
                                st.session_state.active_chart_tab
                            )
                            next_idx = (current_idx + 1) % len(CHART_OPTIONS)

                            # Update the actual state
                            st.session_state.active_chart_tab = CHART_OPTIONS[next_idx]
                            isTabChanged = True
                        
                        # ===========================================
                        # SWIPE CHART LEFT
                        # ===========================================
                        if (
                            prev_motion == "FIST_FIST_LEFT"
                            and new_gesture["motion"] == "UNKNOWN"
                            and st.session_state.active_tab == "Red Blood Cells"
                        ):
                            current_idx = CHART_OPTIONS.index(
                                st.session_state.active_chart_tab
                            )
                            next_idx = (current_idx - 1) % len(CHART_OPTIONS)

                            # Update the actual state
                            st.session_state.active_chart_tab = CHART_OPTIONS[next_idx]
                            isTabChanged = True
                            
                        
                        # Sync state
                        st.session_state.previous_gesture = (
                            st.session_state.current_gesture.copy()
                        )
                        st.session_state.current_gesture = new_gesture
    except Exception as e:
        print(f"Error reading gesture: {e}")
    return isTabChanged
