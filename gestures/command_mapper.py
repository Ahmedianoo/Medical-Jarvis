"""
File: command_mapper.py
-----------------------
Goal:
    Map recognized gestures to application-level commands
    and communicate them to the UI layer.

Pipeline Stage:
    Phase 5 — Command Mapping

Inputs:
    - gesture_label: String output from the gesture recognition module

Outputs:
    - command: String command written to a shared text file
      (e.g., 'NEXT', 'PROCESS', 'RESET')

Mechanism:
    - Dictionary-based mapping
    - File-based inter-process communication

Notes:
    - Writing commands to a file allows loose coupling
      between the gesture system and the Streamlit UI.
"""
