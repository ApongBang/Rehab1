# DTW (Dynamic time wrapping) of two videos 
# Saves an excel with respective data  

import cv2
import mediapipe as mp
import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Key joints to track
key_joints = [15, 16, 11, 12, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7]  # Hands, elbows, hips, face landmarks

# Function to extract pose landmarks

def extract_key_landmarks(landmarks, key_joints):
    if not landmarks:
        return None
    return np.array([[landmarks.landmark[j].x, landmarks.landmark[j].y, landmarks.landmark[j].z] for j in key_joints]).flatten()

# Normalize landmarks relative to the hip

def normalize_landmarks(landmarks, reference_joint=0):
    if landmarks is None:
        return None
    ref = np.array([landmarks[reference_joint]])
    return (landmarks - ref) / np.linalg.norm(ref)

# Function to calculate DTW similarity between two pose sequences

def calculate_dtw_similarity(sequence1, sequence2):
    if len(sequence1) == 0 or len(sequence2) == 0:
        return float('inf')

    distance, _ = fastdtw(sequence1, sequence2, dist=euclidean)
    return distance

# Video input
cap1 = cv2.VideoCapture("output.mp4")
cap2 = cv2.VideoCapture("Rehab_Vid2.mp4")

pose_sequence1 = []
pose_sequence2 = []

# Data to save DTW distances
frame_distances = []

# Mediapipe pose detection
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap1.isOpened() or not cap2.isOpened():
        print("Cannot open camera")
        exit()

    frame_count = 0
    start_time = time.time()

    while True:
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        if not ret1 or not ret2:
            print("Cannot receive frame")
            break

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        results1 = holistic.process(img1_rgb)
        results2 = holistic.process(img2_rgb)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            img1,
            results1.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        mp_drawing.draw_landmarks(
            img2,
            results2.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Extract and normalize pose landmarks
        pose1 = extract_key_landmarks(results1.pose_landmarks, key_joints)
        pose2 = extract_key_landmarks(results2.pose_landmarks, key_joints)

        if pose1 is not None and pose2 is not None:
            pose1 = normalize_landmarks(pose1)
            pose2 = normalize_landmarks(pose2)

            pose_sequence1.append(pose1)
            pose_sequence2.append(pose2)

        # Calculate DTW similarity every 30 frames
        if frame_count % 30 == 0 and len(pose_sequence1) > 30 and len(pose_sequence2) > 30:
            dtw_distance = calculate_dtw_similarity(pose_sequence1[-30:], pose_sequence2[-30:])
            print(f"DTW Distance at frame {frame_count}: {dtw_distance:.2f}")
            
            # Save frame and distance to the list
            frame_distances.append((frame_count, dtw_distance))

        frame_count += 1

        # Show videos
        cv2.imshow("Video 1", img1)
        cv2.imshow("Video 2", img2)

        if cv2.waitKey(5) == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Save distances to an Excel file
df = pd.DataFrame(frame_distances, columns=['Frame', 'DTW Distance'])
df.to_excel('dtw_distances.xlsx', index=False)
print("Saved DTW distances to dtw_distances.xlsx")
