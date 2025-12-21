import cv2
from collections import deque
from capture import initialize_camera, get_frame
from detect_hand import detect_hand
from extract_features import extract_features
from recognize import is_hand, get_static_gesture, detect_motion

FRAME_WINDOW = 20
MIN_DISTANCE = 100
DOMINANT_AXIS_RATIO = 1.4
GESTURE_FILE = "gesture_command.txt"


cap = initialize_camera(0)

# buffer to store (centroid + gesture)
motion_buffer = deque(maxlen=FRAME_WINDOW)
last_written = ""
motion_output = "UNKNOWN"  # persist last valid motion

while True:
    frame = get_frame(cap)
    if frame is None:
        continue

    mask, contour = detect_hand(frame)

    gesture_output = "UNKNOWN"

    if contour is not None:
        features = extract_features(contour)
        cx, cy = features['centroid']

        
        hand_flag = is_hand(features)
        color = (0, 255, 0) if hand_flag else (0, 0, 255)
        cv2.putText(frame, f"HAND: {hand_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Area: {int(features['area'])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Circularity: {features['circularity']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Aspect Ratio: {features['aspect_ratio']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw contour
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # draw centroid
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Centroid: ({int(cx)}, {int(cy)})", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if hand_flag:
            x, y, w, h = features['bounding_box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # stable gesture
            gesture_output = get_static_gesture(features)
            cv2.putText(frame, f"Gesture: {gesture_output}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # store in motion buffer
            motion_buffer.append((cx, cy, gesture_output))

            # sompute motion if buffer full
            if len(motion_buffer) == FRAME_WINDOW:
                x1, y1, g1 = motion_buffer[0]
                x2, y2, g2 = motion_buffer[-1]
                motion, direction = detect_motion(x1, y1, x2, y2, g1, g2, MIN_DISTANCE, DOMINANT_AXIS_RATIO)

                if motion != "UNKNOWN_MOTION" and direction != "UNKNOWN_DIRECTION":
                    motion_output = f"{motion}_{direction}"
                    motion_buffer.clear()  # reset buffer after atomic motion

            # show motion info on frame
            cv2.putText(frame, f"Motion: {motion_output}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        else:
            cv2.putText(frame, "IGNORED OBJECT", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # gesture unknown = reset motion
            motion_output = "UNKNOWN"

    else:
        # no contour = gesture unknown = reset motion
        motion_output = "UNKNOWN"


    combined_output = f"{gesture_output},{motion_output}"
    if combined_output != last_written:
        with open(GESTURE_FILE, "w") as f:
            f.write(combined_output)
        last_written = combined_output
        print(f"✓ WROTE TO FILE: {combined_output}")  # Debug output
        print(f"  File location: {GESTURE_FILE}")  # Show file path

    # ===============================
    # CROP AND RESIZE FRAME (NEW)
    # ===============================
    height, width = frame.shape[:2]
    
    # Crop center area (square focused on hand)
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to small window
    display_frame = cv2.resize(cropped_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Show frames
    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Skin Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # escape
        break

cap.release()
cv2.destroyAllWindows()