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
import cv2
from capture import initialize_camera, get_frame
from detect_hand import detect_hand
from extract_features import extract_features
import numpy as np

# Initialize camera
cap = initialize_camera(0)

while True:
    frame = get_frame(cap)
    if frame is None:
        continue

    # Detect hand
    mask, contour = detect_hand(frame)

    if contour is not None:
        # Extract features
        features = extract_features(contour)

        # Draw bounding box
        x, y, w, h = features['bounding_box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw hand contour
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # Draw centroid
        cx, cy = features['centroid']
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # Draw convexity defects as red points
        if features['defects'] is not None:
            for i in range(features['defects'].shape[0]):
                s, e, f, d = features['defects'][i, 0]
                far = tuple(contour[f][0])
                depth = d / 256
                a = np.linalg.norm(np.array(contour[e][0]) - np.array(contour[s][0]))
                b = np.linalg.norm(np.array(far) - np.array(contour[s][0]))
                c_len = np.linalg.norm(np.array(contour[e][0]) - np.array(far))
                angle = np.arccos((b**2 + c_len**2 - a**2) / (2 * b * c_len)) * 180 / np.pi
                if angle < 90 and depth > 20:
                    cv2.circle(frame, far, 5, (0, 0, 255), -1)

        # Display finger count
        cv2.putText(
            frame,
            f"Fingers: {features['finger_count']}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    # Show frame and mask
    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Skin Mask", mask)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
