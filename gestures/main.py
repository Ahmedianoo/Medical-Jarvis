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
