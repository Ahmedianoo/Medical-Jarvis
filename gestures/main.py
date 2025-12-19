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
import cv2
from capture import initialize_camera, get_frame
from detect_hand import detect_hand
from extract_features import extract_features
import numpy as np

# ===============================
# HAND PRESENCE GATE & STATIC GESTURE
# ===============================
def is_hand(features, min_area=30000, max_area=90000, min_circularity=0.08, max_circularity=0.75):
    """
    returns True if a valid hand is detected
    """
    return (min_area < features['area'] < max_area and
            min_circularity < features['circularity'] < max_circularity)

def get_static_gesture(features):
    v_shapes = features['convexity_defect']
    circularity = features['circularity']
    area = features['area']
    aspect_ratio = features['aspect_ratio']

    if v_shapes == 0 and circularity > 0.15:  # closed hand
        return 'FIST'
    if v_shapes >= 3 and area > 55000:       # open hand
        return 'PALM'
    if v_shapes == 1 and aspect_ratio > 0.4: # for cursor, victory sign
        return 'POINT'
    return 'UNKNOWN'

def draw_hand_gate_debug(frame, features):
    hand_flag = is_hand(features)
    color = (0, 255, 0) if hand_flag else (0, 0, 255)

    cv2.putText(frame, f"HAND: {hand_flag}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"Area: {int(features['area'])}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"Circularity: {features['circularity']:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"Aspect Ratio: {features['aspect_ratio']:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ===============================
# INITIALIZE CAMERA
# ===============================
cap = initialize_camera(0)

while True:
    frame = get_frame(cap)
    if frame is None:
        continue

    mask, contour = detect_hand(frame)

    if contour is not None:
        features = extract_features(contour)

        # ---- HAND GATE DEBUG ----
        draw_hand_gate_debug(frame, features)

        # Draw contour always (for debugging)
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # Draw centroid
        cx, cy = features['centroid']
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # Draw bounding box ONLY if hand is valid
        if is_hand(features):
            x, y, w, h = features['bounding_box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize static gesture
            gesture = get_static_gesture(features)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "IGNORED OBJECT", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw convexity defects (visual aid only)
        if features['defects'] is not None:
            for i in range(features['defects'].shape[0]):
                s, e, f, d = features['defects'][i, 0]
                far = tuple(contour[f][0])
                depth = d / 256
                a = np.linalg.norm(np.array(contour[e][0]) - np.array(contour[s][0]))
                b = np.linalg.norm(np.array(far) - np.array(contour[s][0]))
                c_len = np.linalg.norm(np.array(contour[e][0]) - np.array(far))
                angle = np.arccos((b**2 + c_len**2 - a**2) / (2 * b * c_len)) * 180 / np.pi
                if angle < 90 and depth > 80:
                    cv2.circle(frame, far, 5, (0, 0, 255), -1)

        # Finger count (debug only)
        cv2.putText(
            frame,
            f"Defects: {features['convexity_defect']}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Skin Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


