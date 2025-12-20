"""
File: main.py
-------------
Goal:
    Serve as the main execution pipeline for the gesture
    recognition system during development and testing.

Pipeline Stage:
    Integration / Debugging

Inputs:
    - Webcam frames via capture.py
    - Hand detection and feature extraction modules

Outputs:
    - Visual debugging output (frames with annotations)
    - Console logs (optional)

Notes:
    - This file is mainly for testing and demonstration.
    - The Streamlit application can replace this file
      in the final system.
"""



#  detect_hand test
# import cv2
# from capture import initialize_camera, get_frame
# from detect_hand import detect_hand

# def main():
#     cap = initialize_camera()

#     while True:
#         frame = get_frame(cap)
#         if frame is None:
#             break

#         mask, contour = detect_hand(frame)

#         if contour is not None:
#             cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

#         cv2.imshow("Hand Detection", frame)
#         cv2.imshow("Skin Mask", mask)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


#testing 
# import cv2
# from capture import initialize_camera, get_frame
# from detect_hand import detect_hand
# from extract_features import extract_features
# import numpy as np

# # Initialize camera
# cap = initialize_camera(0)

# while True:
#     frame = get_frame(cap)
#     if frame is None:
#         continue

#     # Detect hand
#     mask, contour = detect_hand(frame)

#     if contour is not None:
#         # Extract features
#         features = extract_features(contour)

#         # Draw bounding box
#         x, y, w, h = features['bounding_box']
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Draw hand contour
#         cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

#         # Draw centroid
#         cx, cy = features['centroid']
#         cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

#         # Draw convexity defects as red points
#         if features['defects'] is not None:
#             for i in range(features['defects'].shape[0]):
#                 s, e, f, d = features['defects'][i, 0]
#                 far = tuple(contour[f][0])
#                 depth = d / 256
#                 a = np.linalg.norm(np.array(contour[e][0]) - np.array(contour[s][0]))
#                 b = np.linalg.norm(np.array(far) - np.array(contour[s][0]))
#                 c_len = np.linalg.norm(np.array(contour[e][0]) - np.array(far))
#                 angle = np.arccos((b**2 + c_len**2 - a**2) / (2 * b * c_len)) * 180 / np.pi
#                 if angle < 90 and depth > 80:
#                     cv2.circle(frame, far, 5, (0, 0, 255), -1)

#         # Display finger count
#         cv2.putText(
#             frame,
#             f"Fingers: {features['convexity_defect']}",
#             (10, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 255),
#             2,
#         )

#     # Show frame and mask
#     cv2.imshow("Hand Detection", frame)
#     cv2.imshow("Skin Mask", mask)

#     # Exit on ESC
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

#testing hand static features
# import cv2
# from capture import initialize_camera, get_frame
# from detect_hand import detect_hand
# from extract_features import extract_features
# import numpy as np

# # ===============================
# # HAND PRESENCE GATE & STATIC GESTURE
# # ===============================
# def is_hand(features, min_area=50000, max_area=100000, min_circularity=0.08, max_circularity=0.75):
#     """
#     returns True if a valid hand is detected
#     """
#     return (min_area < features['area'] < max_area and
#             min_circularity < features['circularity'] < max_circularity)

# def get_static_gesture(features):
#     v_shapes = features['convexity_defect']
#     circularity = features['circularity']
#     area = features['area']
#     aspect_ratio = features['aspect_ratio'] #0.9, 40

#     if v_shapes == 0 and circularity > 0.25 and aspect_ratio > 0.6:  # closed hand
#         return 'FIST'
#     if v_shapes >= 3 and area > 55000 :       # open hand and aspect_ratio < 1.0
#         return 'PALM'
#     if v_shapes == 1 and 0.4 < aspect_ratio < 0.7: # for cursor, victory sign
#         return 'POINT'
#     # if v_shapes == 0 and aspect_ratio > 0.75 and circularity < 0.25: 
#     #     return 'PROCESS'
#     return 'UNKNOWN'

# def draw_hand_gate_debug(frame, features):
#     hand_flag = is_hand(features)
#     color = (0, 255, 0) if hand_flag else (0, 0, 255)

#     cv2.putText(frame, f"HAND: {hand_flag}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.putText(frame, f"Area: {int(features['area'])}", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.putText(frame, f"Circularity: {features['circularity']:.2f}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.putText(frame, f"Aspect Ratio: {features['aspect_ratio']:.2f}", (10, 120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# # ===============================
# # INITIALIZE CAMERA
# # ===============================
# cap = initialize_camera(0)

# while True:
#     frame = get_frame(cap)
#     if frame is None:
#         continue

#     mask, contour = detect_hand(frame)

#     if contour is not None:
#         features = extract_features(contour)

#         # ---- HAND GATE DEBUG ----
#         draw_hand_gate_debug(frame, features)

#         # Draw contour always (for debugging)
#         cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

#         # Draw centroid
#         cx, cy = features['centroid']
#         cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

#         # Draw bounding box ONLY if hand is valid
#         if is_hand(features):
#             x, y, w, h = features['bounding_box']
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             # Recognize static gesture
#             gesture = get_static_gesture(features)
#             cv2.putText(frame, f"Gesture: {gesture}", (10, 210),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#         else:
#             cv2.putText(frame, "IGNORED OBJECT", (10, 150),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Draw convexity defects (visual aid only)
#         if features['defects'] is not None:
#             for i in range(features['defects'].shape[0]):
#                 s, e, f, d = features['defects'][i, 0]
#                 far = tuple(contour[f][0])
#                 depth = d / 256
#                 a = np.linalg.norm(np.array(contour[e][0]) - np.array(contour[s][0]))
#                 b = np.linalg.norm(np.array(far) - np.array(contour[s][0]))
#                 c_len = np.linalg.norm(np.array(contour[e][0]) - np.array(far))
#                 angle = np.arccos((b**2 + c_len**2 - a**2) / (2 * b * c_len)) * 180 / np.pi
#                 if angle < 90 and depth > 80:
#                     cv2.circle(frame, far, 5, (0, 0, 255), -1)

#         # Finger count (debug only)
#         cv2.putText(
#             frame,
#             f"Defects: {features['convexity_defect']}",
#             (10, 180),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 0),
#             2,
#         )

#     cv2.imshow("Hand Detection", frame)
#     cv2.imshow("Skin Mask", mask)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

#testing dynamic gestures
# import cv2
# import numpy as np
# from capture import initialize_camera, get_frame
# from detect_hand import detect_hand
# from extract_features import extract_features
# from collections import deque

# # ===============================
# # PARAMETERS
# # ===============================
# FRAME_WINDOW = 20     # Number of frames to compute motion
# MIN_DISTANCE = 100      # Minimum centroid displacement to consider a motion
# DOMINANT_AXIS_RATIO = 1.4 # Ratio to exclude diagonal movements

# # ===============================
# # FUNCTIONS (your own)
# # ===============================
# def is_hand(features, min_area=40000, max_area=100000, min_circularity=0.08, max_circularity=0.75):
#     return (min_area < features['area'] < max_area and
#             min_circularity < features['circularity'] < max_circularity)

# def get_static_gesture(features):
#     v_shapes = features['convexity_defect']
#     circularity = features['circularity']
#     area = features['area']
#     aspect_ratio = features['aspect_ratio']

#     if v_shapes == 0  and aspect_ratio > 0.6:  # closed hand and circularity > 0.1
#         return 'FIST'
#     if v_shapes >= 3 and area > 45000:       # open hand
#         return 'PALM'
#     if v_shapes == 1 and 0.4 < aspect_ratio < 0.7: # for cursor, victory sign
#         return 'POINT'
#     return 'UNKNOWN'

# def detect_motion(x1, y1, x2, y2, start, end, min_distance=400, dominant_axis_ration=2):
#     dx = x2 - x1
#     dy = y2 - y1
#     distance = np.sqrt(dx ** 2 + dy ** 2)

#     motion = "UNKNOWN_MOTION"
#     direction = "UNKNOWN_DIRECTION"
#     if distance >= min_distance:
#         if start == 'PALM' and end == 'PALM':
#             motion = "PALM_PALM"
#         elif start == 'PALM' and end == 'FIST':
#             motion = "PALM_FIST"
#         elif start == 'FIST' and end == 'PALM':
#             motion = "FIST_PALM"
#         elif start == 'FIST' and end == 'FIST':
#             motion = "FIST_FIST"

#         # Exclude diagonal
#         if abs(dx) > dominant_axis_ration * abs(dy):
#             direction = 'RIGHT' if dx > 0 else 'LEFT'
#         elif abs(dy) > dominant_axis_ration * abs(dx):
#             direction = 'DOWN' if dy > 0 else 'UP'

#     return motion, direction

# # ===============================
# # CAMERA INITIALIZATION
# # ===============================
# cap = initialize_camera(0)

# # Buffer to store (centroid + gesture)
# motion_buffer = deque(maxlen=FRAME_WINDOW)

# while True:
#     frame = get_frame(cap)
#     if frame is None:
#         continue

#     mask, contour = detect_hand(frame)

#     if contour is not None:
#         features = extract_features(contour)
#         cx, cy = features['centroid']

#         # ---- HAND GATE DEBUG ----
#         hand_flag = is_hand(features)
#         color = (0, 255, 0) if hand_flag else (0, 0, 255)
#         cv2.putText(frame, f"HAND: {hand_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(frame, f"Area: {int(features['area'])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.putText(frame, f"Circularity: {features['circularity']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.putText(frame, f"Aspect Ratio: {features['aspect_ratio']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Draw contour
#         cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

#         # Draw centroid
#         cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
#         cv2.putText(frame, f"Centroid: ({int(cx)}, {int(cy)})", (10, 150),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         if hand_flag:
#             x, y, w, h = features['bounding_box']
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             gesture = get_static_gesture(features)
#             cv2.putText(frame, f"Gesture: {gesture}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#             # Store in motion buffer
#             motion_buffer.append((cx, cy, gesture))

#             # Compute motion if buffer full
#             if len(motion_buffer) == FRAME_WINDOW:
#                 x1, y1, g1 = motion_buffer[0]
#                 x2, y2, g2 = motion_buffer[-1]
#                 motion, direction = detect_motion(x1, y1, x2, y2, g1, g2, MIN_DISTANCE, DOMINANT_AXIS_RATIO)

#                 # Show motion info on frame
#                 cv2.putText(frame, f"Motion: {motion} Dir: {direction}", (10, 250),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

#         else:
#             cv2.putText(frame, "IGNORED OBJECT", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     cv2.imshow("Hand Detection", frame)
#     cv2.imshow("Skin Mask", mask)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()
