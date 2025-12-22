import cv2
import numpy as np
import os
import time
import json
import gc
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detect_hand import detect_hand
from extract_features import extract_features
from recognize import is_hand

# ===============================
# CONFIGURATION
# ===============================
TEST_IMAGE_DIR = "Hands/Hands"  # Change if needed
RESULTS_DIR = "test_results"
GROUND_TRUTH_FILE = "ground_truth.json"
MODEL_PATH = "hand_landmarker.task"  # Download from MediaPipe

# ===============================
# MEDIAPIPE SETUP
# ===============================
print("=" * 60)
print("INITIALIZING MEDIAPIPE...")
print("=" * 60)

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"✗ ERROR: Model file not found: {MODEL_PATH}")
    print("\n📥 Download the model file using this command:")
    print("\nwget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print("\nOr manually download from:")
    print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print("\nSave it in the same directory as this script.")
    exit(1)

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(options)
print("✓ MediaPipe initialized successfully\n")

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_image_files(image_dir):
    """Return all valid image files in a directory"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"✗ ERROR: Folder does not exist: {image_dir}")
        return []
    files = [f for f in image_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    print(f"Found {len(files)} image(s) in {image_dir}")
    return files


def create_results_folder(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    return folder


def draw_landmarks_on_image(image, detection_result):
    """Draw hand landmarks on image using the new API"""
    annotated_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Draw landmarks for each detected hand
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17)  # Palm
            ]
            
            # Draw connections
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]
                    
                    start_x = int(start_point.x * width)
                    start_y = int(start_point.y * height)
                    end_x = int(end_point.x * width)
                    end_y = int(end_point.y * height)
                    
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
    
    return annotated_image

# ===============================
# HAND DETECTION TESTER
# ===============================
class HandDetectionTester:
    def __init__(self, image_dir, results_dir):
        self.image_dir = Path(image_dir)
        self.results_dir = create_results_folder(results_dir)
        self.results = {"custom_system": [], "mediapipe": [], "comparison": []}

    def test_custom_system(self, image_path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"✗ Failed to load image: {image_path.name}")
            return None

        start_time = time.time()
        mask, contour = detect_hand(frame)
        processing_time = time.time() - start_time

        hand_detected = False
        confidence = 0.0
        features = None

        if contour is not None:
            features = extract_features(contour)
            hand_detected = is_hand(features)
            if hand_detected:
                area_score = min(features["area"] / 50000, 1.0)
                circ_score = 1.0 - abs(features["circularity"] - 0.7) / 0.7
                aspect_score = 1.0 - abs(features["aspect_ratio"] - 0.7) / 0.7
                confidence = (area_score + circ_score + aspect_score) / 3.0

        result = {
            "detected": hand_detected,
            "confidence": confidence,
            "processing_time": processing_time,
            "features": features,
            "contour": contour,
            "mask": mask,
        }
        
        # Clean up frame from memory
        del frame
        
        return result

    def test_mediapipe(self, image_path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"✗ Failed to load image: {image_path.name}")
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        start_time = time.time()
        results = hand_detector.detect(mp_image)
        processing_time = time.time() - start_time

        hand_detected = results.hand_landmarks is not None and len(results.hand_landmarks) > 0
        confidence = 0.0
        num_hands = 0

        if hand_detected:
            num_hands = len(results.hand_landmarks)
            # Calculate average confidence from handedness scores
            if results.handedness:
                # results.handedness is a list of lists of Classification objects
                confidence = sum(h[0].score for h in results.handedness) / len(results.handedness)

        result = {
            "detected": hand_detected,
            "confidence": confidence,
            "num_hands": num_hands,
            "processing_time": processing_time,
            "detection_result": results,
        }
        
        # Clean up frames from memory
        del frame, rgb_frame, mp_image
        
        return result

    def create_comparison_visualization(self, image_path, custom_result, mp_result):
        frame = cv2.imread(str(image_path))
        h, w = frame.shape[:2]
        output = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Custom system
        left_frame = frame.copy()
        if custom_result["detected"] and custom_result["contour"] is not None:
            cv2.drawContours(left_frame, [custom_result["contour"]], -1, (0, 255, 0), 2)
            if custom_result["features"]:
                cx, cy = custom_result["features"]["centroid"]
                cv2.circle(left_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                x, y, w_box, h_box = custom_result["features"]["bounding_box"]
                cv2.rectangle(left_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        status = "DETECTED" if custom_result["detected"] else "NOT DETECTED"
        color = (0, 255, 0) if custom_result["detected"] else (0, 0, 255)
        cv2.putText(left_frame, f"Custom: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(left_frame, f"Conf: {custom_result['confidence']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(left_frame, f"Time: {custom_result['processing_time']*1000:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # MediaPipe
        right_frame = frame.copy()
        if mp_result["detected"] and mp_result["detection_result"]:
            right_frame = draw_landmarks_on_image(right_frame, mp_result["detection_result"])

        status = "DETECTED" if mp_result["detected"] else "NOT DETECTED"
        color = (0, 255, 0) if mp_result["detected"] else (0, 0, 255)
        cv2.putText(right_frame, f"MediaPipe: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(right_frame, f"Conf: {mp_result['confidence']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(right_frame, f"Hands: {mp_result['num_hands']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(right_frame, f"Time: {mp_result['processing_time']*1000:.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output[:, :w] = left_frame
        output[:, w:] = right_frame
        
        # Clean up temporary frames
        del frame, left_frame, right_frame
        
        return output

    def run_tests(self):
        # Updated pattern to match Hand_0000002.jpg format
        image_files = list(self.image_dir.glob("Hand_*.jpg")) + \
                      list(self.image_dir.glob("Hand_*.png")) + \
                      list(self.image_dir.glob("Hand_*.jpeg"))
        if not image_files:
            print("✗ No images found. Place images in the folder and try again.")
            return

        # Sort files to ensure consistent ordering and limit to first 300
        image_files = sorted(image_files)
        print(f"\n{'='*60}")
        print(f"Testing first {len(image_files)} images")
        print(f"{'='*60}\n")

        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] TESTING: {image_path.name}")
            custom_result = self.test_custom_system(image_path)
            mp_result = self.test_mediapipe(image_path)

            if custom_result is None or mp_result is None:
                print("⚠ Skipping this image due to loading error.")
                continue

            self.results["custom_system"].append({
                "image": image_path.name,
                "detected": custom_result["detected"],
                "confidence": custom_result["confidence"],
                "time_ms": custom_result["processing_time"]*1000,
            })
            self.results["mediapipe"].append({
                "image": image_path.name,
                "detected": mp_result["detected"],
                "confidence": mp_result["confidence"],
                "num_hands": mp_result["num_hands"],
                "time_ms": mp_result["processing_time"]*1000,
            })
            agreement = custom_result["detected"] == mp_result["detected"]
            self.results["comparison"].append({
                "image": image_path.name,
                "agreement": agreement,
                "custom_detected": custom_result["detected"],
                "mediapipe_detected": mp_result["detected"],
            })

            # Save comparison image
            vis = self.create_comparison_visualization(image_path, custom_result, mp_result)
            vis_path = self.results_dir / f"comparison_{image_path.stem}.jpg"
            cv2.imwrite(str(vis_path), vis)

            # Save mask for custom system
            if custom_result["mask"] is not None:
                mask_path = self.results_dir / f"mask_{image_path.stem}.jpg"
                cv2.imwrite(str(mask_path), custom_result["mask"])

            print(f"✓ Image processed and saved: {vis_path.name}")
            
            # Clean up memory every 50 images
            if i % 50 == 0:
                del vis
                gc.collect()
                print(f"  [Memory cleanup at image {i}]")

    def generate_report(self):
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        total = len(self.results["comparison"])
        if total == 0:
            print("✗ No results to report")
            return
        agreements = sum(1 for r in self.results["comparison"] if r["agreement"])
        agreement_rate = agreements / total * 100
        print(f"Total Images Tested: {total}")
        print(f"Agreement Rate: {agreement_rate:.1f}% ({agreements}/{total})")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    tester = HandDetectionTester(TEST_IMAGE_DIR, RESULTS_DIR)
    tester.run_tests()
    tester.generate_report()