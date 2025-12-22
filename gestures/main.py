import cv2
import os
from collections import deque
from capture import initialize_camera, get_frame
from detect_hand import detect_hand
from extract_features import extract_features
from recognize import is_hand, get_static_gesture, detect_motion

# ===============================
# PATH (WRITE FILE NEXT TO SCRIPT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_FILE = os.path.join(BASE_DIR, "gesture_command.txt")

# ===============================
# PARAMETERS
# ===============================
FRAME_WINDOW = 20
MIN_DISTANCE = 100
DOMINANT_AXIS_RATIO = 1.4

# ===============================
# CAMERA
# ===============================
cap = initialize_camera(0)

# Buffer to store (centroid + gesture)
motion_buffer = deque(maxlen=FRAME_WINDOW)

last_written = ""
motion_output = "UNKNOWN"  # persist last valid motion

print("Writing gesture file to:")
print(GESTURE_FILE)

while True:
    frame = get_frame(cap)
    if frame is None:
        continue

    mask, contour = detect_hand(frame)

    gesture_output = "UNKNOWN"

    if contour is not None:
        features = extract_features(contour)
        cx, cy = features['centroid']

        # ---- HAND GATE DEBUG ----
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

        # Draw contour
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # Draw centroid
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Centroid: ({int(cx)}, {int(cy)})", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if hand_flag:
            x, y, w, h = features['bounding_box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Static gesture
            gesture_output = get_static_gesture(features)
            cv2.putText(frame, f"Gesture: {gesture_output}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Store motion data
            motion_buffer.append((cx, cy, gesture_output))

            # Detect motion when buffer is full
            if len(motion_buffer) == FRAME_WINDOW:
                x1, y1, g1 = motion_buffer[0]
                x2, y2, g2 = motion_buffer[-1]

                motion, direction = detect_motion(
                    x1, y1, x2, y2,
                    g1, g2,
                    MIN_DISTANCE,
                    DOMINANT_AXIS_RATIO
                )

                if motion != "UNKNOWN_MOTION" and direction != "UNKNOWN_DIRECTION":
                    motion_output = f"{motion}_{direction}"
                    motion_buffer.clear()

            cv2.putText(frame, f"Motion: {motion_output}", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        else:
            cv2.putText(frame, "IGNORED OBJECT", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            motion_output = "UNKNOWN"

    else:
        motion_output = "UNKNOWN"

    # ===============================
    # FILE WRITE (ATOMIC UPDATE)
    # ===============================
    combined_output = f"{gesture_output},{motion_output}"

    if combined_output != last_written:
        with open(GESTURE_FILE, "w") as f:
            f.write(combined_output)
        last_written = combined_output

    # ===============================
    # DISPLAY
    # ===============================
    cv2.imshow("Hand Detection", frame)
    # cv2.imshow("Skin Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
