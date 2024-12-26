# 影片加判斷

import cv2
import mediapipe as mp
import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
import os

# Mediapipe setup
mp_holistic = mp.solutions.holistic

# Key joints to track (focus on hands and elbows for similarity, hips and eyes for reference)
hands_elbows = [15, 16, 11, 12]  # Hands and elbows
hips_eyes = [23, 24, 1, 4]       # Hips and eyes for reference

# Function to extract pose landmarks
def extract_key_landmarks(landmarks, key_joints):
    if landmarks is None:
        return None
    return np.array([[landmarks[j].x, landmarks[j].y, landmarks[j].z] for j in key_joints]).flatten()

# Normalize landmarks relative to the hips and eyes
def normalize_landmarks_with_reference(landmarks, reference_joints):
    if landmarks is None:
        return None
    ref_points = np.mean(np.array([[landmarks[j].x, landmarks[j].y, landmarks[j].z] for j in reference_joints]), axis=0)
    landmarks_array = np.array([[landmarks[j].x, landmarks[j].y, landmarks[j].z] for j in range(len(landmarks))])
    normalized_landmarks = landmarks_array - ref_points
    return normalized_landmarks.flatten() / np.linalg.norm(ref_points)

# Function to calculate DTW similarity between two pose sequences
def calculate_dtw_similarity(sequence1, sequence2):
    if len(sequence1) == 0 or len(sequence2) == 0:
        return float('inf')

    distance, _ = fastdtw(sequence1, sequence2, dist=euclidean)
    return distance

# Function to capture screenshots for significant distances
def capture_screenshot(dtw_distance, frame_count, img1, img2, screenshot_folder, error_Frame):
    if dtw_distance > 45.0:  # Threshold for significant difference
        frame_number_to_capture = frame_count
        error_Frame.append(frame_count)
        screenshot_name1 = os.path.join(screenshot_folder, f"standard_{frame_number_to_capture}.png")
        screenshot_name2 = os.path.join(screenshot_folder, f"user_{frame_number_to_capture}.png")
        cv2.imwrite(screenshot_name1, img1)
        cv2.imwrite(screenshot_name2, img2)
        print(f"Captured screenshot at frame {frame_number_to_capture} for DTW distance {dtw_distance:.2f}")

# Video input
cap1 = cv2.VideoCapture("output2.mp4")
cap2 = cv2.VideoCapture("Rehab_Vid2.mp4")

pose_sequence1 = []
pose_sequence2 = []
Error_Frame = []

# Data to save DTW distances
frame_distances = []

# Create screenshot folder
screenshot_folder = "screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

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

        # Draw pose landmarks on both videos
        if results1.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results2.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Combine both videos side by side
        combined_frame = np.hstack((img1, img2))

        # Extract and normalize pose landmarks
        if results1.pose_landmarks and results2.pose_landmarks:
            pose1 = extract_key_landmarks(results1.pose_landmarks.landmark, hands_elbows)
            pose2 = extract_key_landmarks(results2.pose_landmarks.landmark, hands_elbows)

            reference1 = extract_key_landmarks(results1.pose_landmarks.landmark, hips_eyes)
            reference2 = extract_key_landmarks(results2.pose_landmarks.landmark, hips_eyes)

            if pose1 is not None and pose2 is not None and reference1 is not None and reference2 is not None:
                pose1 = normalize_landmarks_with_reference(results1.pose_landmarks.landmark, hips_eyes)
                pose2 = normalize_landmarks_with_reference(results2.pose_landmarks.landmark, hips_eyes)

                pose_sequence1.append(pose1)
                pose_sequence2.append(pose2)

        # Calculate DTW similarity every 30 frames
        if frame_count % 30 == 0 and len(pose_sequence1) > 30 and len(pose_sequence2) > 30:
            dtw_distance = calculate_dtw_similarity(pose_sequence1[-30:], pose_sequence2[-30:])
            print(f"DTW Distance at frame {frame_count}: {dtw_distance:.2f}")
            
            # Save frame and distance to the list
            frame_distances.append((frame_count, dtw_distance))

            # Capture screenshots for significant DTW distances
            capture_screenshot(dtw_distance, frame_count, img1, img2, screenshot_folder, Error_Frame)

        frame_count += 1

        # Display combined frame
        cv2.imshow("Combined Video", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Save distances to an Excel file
df = pd.DataFrame(frame_distances, columns=['Frame', 'DTW Distance'])
df.to_excel('dtw_distances.xlsx', index=False)
print("Saved DTW distances to dtw_distances.xlsx")
print(f"Error frames: {Error_Frame}")
