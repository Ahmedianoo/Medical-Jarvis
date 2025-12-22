import sys
import streamlit as st
import subprocess
import os
import time

PROJECT_ROOT = r"C:\Users\Dell\Desktop\Coding\jarvis-optimizers"
TABS_OPTIONS = ["Home", "Red Blood Cells", "White Blood Cells", "Platelets"]
CHART_OPTIONS = ["Statistics", "Area", "Circularity", "Aspect Ratio", "Correlations"]


def toggle_camera():
    """
    Toggle camera script on/off
    """
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
    """
    Map Gesture To Action - DEBUG VERSION
    """
    isTabChanged = False
    gesture_file = os.path.join(PROJECT_ROOT, "gestures", "gesture_command.txt")

    try:
        # DEBUG: Check if file exists
        if not os.path.exists(gesture_file):
            print(f"DEBUG: Gesture file does not exist: {gesture_file}")
            return False
        
        # DEBUG: Check file modification time
        file_mod_time = os.path.getmtime(gesture_file)
        current_time = time.time()
        time_diff = current_time - file_mod_time
        print(f"DEBUG: File last modified {time_diff:.2f} seconds ago")
        
        with open(gesture_file, "r") as f:
            content = f.read().strip()
            
            # DEBUG: Print raw content
            print(f"DEBUG: Raw file content: '{content}'")
            
            if not content:
                print("DEBUG: File is empty")
                return False

            parts = content.split(",")
            print(f"DEBUG: Split parts: {parts}, length: {len(parts)}")
            
            if len(parts) == 2:
                new_gesture = {
                    "gesture": parts[0],
                    "motion": parts[1],
                    "raw": content,
                }
                
                # DEBUG: Print current vs new
                print(f"DEBUG: Current gesture: {st.session_state.current_gesture}")
                print(f"DEBUG: New gesture: {new_gesture}")

                # Check for state change
                if new_gesture["raw"] != st.session_state.current_gesture["raw"]:
                    print("DEBUG: ✓ Gesture changed detected!")
                    
                    # SAVE CURRENT STATE FIRST
                    prev_motion = st.session_state.current_gesture["motion"]
                    prev_gesture = st.session_state.current_gesture["gesture"]
                    
                    print(f"DEBUG: Transition from ({prev_gesture}, {prev_motion}) -> ({new_gesture['gesture']}, {new_gesture['motion']})")
                    
                    # UPDATE STATE IMMEDIATELY
                    st.session_state.previous_gesture = st.session_state.current_gesture.copy()
                    st.session_state.current_gesture = new_gesture
                    
                    # NOW CHECK TRANSITIONS
                    # =====================================
                    # SWIPE RIGHT
                    # =====================================
                    if prev_motion == "PALM_PALM_RIGHT" and new_gesture["motion"] == "UNKNOWN":
                        print("DEBUG: ✓ SWIPE RIGHT detected")
                        current_idx = TABS_OPTIONS.index(st.session_state.active_tab)
                        next_idx = (current_idx + 1) % len(TABS_OPTIONS)
                        print(f"DEBUG: Switching from {TABS_OPTIONS[current_idx]} to {TABS_OPTIONS[next_idx]}")
                        st.session_state.active_tab = TABS_OPTIONS[next_idx]
                        isTabChanged = True
                        
                    # ===========================================
                    # SWIPE LEFT
                    # ===========================================
                    elif prev_motion == "PALM_PALM_LEFT" and new_gesture["motion"] == "UNKNOWN":
                        print("DEBUG: ✓ SWIPE LEFT detected")
                        current_idx = TABS_OPTIONS.index(st.session_state.active_tab)
                        next_idx = (current_idx - 1) % len(TABS_OPTIONS)
                        print(f"DEBUG: Switching from {TABS_OPTIONS[current_idx]} to {TABS_OPTIONS[next_idx]}")
                        st.session_state.active_tab = TABS_OPTIONS[next_idx]
                        isTabChanged = True
                    
                    # ===========================================
                    # CLOSE CAMERA
                    # ===========================================
                    elif prev_gesture == "UNKNOWN" and new_gesture["gesture"] == "POINT":
                        print("DEBUG: ✓ CLOSE CAMERA detected")
                        if st.session_state.camera_process is not None:
                            st.session_state.camera_process.terminate()
                            st.session_state.camera_process = None
                        st.session_state.camera_active = False
                        isTabChanged = True
                        
                    # ===========================================
                    # RESET
                    # ===========================================
                    elif prev_gesture == "FIST" and new_gesture["gesture"] == "PALM":
                        print("DEBUG: ✓ RESET detected")
                        st.session_state["opencv_img"] = None
                        st.session_state.active_tab = "Home"
                        isTabChanged = True
                    
                    # ===========================================
                    # SWIPE CHART RIGHT
                    # ===========================================
                    elif (prev_motion == "FIST_FIST_RIGHT" and 
                          new_gesture["motion"] == "UNKNOWN" and 
                          st.session_state.active_tab == "Red Blood Cells"):
                        print("DEBUG: ✓ SWIPE CHART RIGHT detected")
                        current_idx = CHART_OPTIONS.index(st.session_state.active_chart_tab)
                        next_idx = (current_idx + 1) % len(CHART_OPTIONS)
                        st.session_state.active_chart_tab = CHART_OPTIONS[next_idx]
                        isTabChanged = True
                    
                    # ===========================================
                    # SWIPE CHART LEFT
                    # ===========================================
                    elif (prev_motion == "FIST_FIST_LEFT" and 
                          new_gesture["motion"] == "UNKNOWN" and 
                          st.session_state.active_tab == "Red Blood Cells"):
                        print("DEBUG: ✓ SWIPE CHART LEFT detected")
                        current_idx = CHART_OPTIONS.index(st.session_state.active_chart_tab)
                        next_idx = (current_idx - 1) % len(CHART_OPTIONS)
                        st.session_state.active_chart_tab = CHART_OPTIONS[next_idx]
                        isTabChanged = True
                    else:
                        print("DEBUG: ✗ No matching transition pattern found")
                        
                    print(f"DEBUG: isTabChanged = {isTabChanged}")
                else:
                    print("DEBUG: No change - same gesture as before")
            else:
                print(f"DEBUG: Invalid format - expected 2 parts, got {len(parts)}")
                
    except Exception as e:
        print(f"ERROR reading gesture: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"DEBUG: Returning isTabChanged = {isTabChanged}")
    return isTabChanged