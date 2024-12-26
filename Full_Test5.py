# Compares similarity of two videos with Kmean


import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Focus on key joints for simplicity
key_joints = [15, 16, 11, 12, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7]  # Hands, elbows, hips, and face landmarks
output_list = []
screenshot_folder = "screenshots"
os.makedirs(screenshot_folder, exist_ok=True)
screenshot_count = 0

# Function to extract pose landmarks for clustering
def extract_key_landmarks(landmarks, key_joints):
    if not landmarks:
        return None
    return np.array([[landmarks.landmark[j].x, landmarks.landmark[j].y, landmarks.landmark[j].z] for j in key_joints]).flatten()

# Function to perform pose sequence clustering
def pose_clustering(pose_sequence1, pose_sequence2, n_clusters=5):
    """
    Cluster pose sequences and measure similarity based on cluster assignments.

    Args:
        pose_sequence1: List of pose vectors from video 1.
        pose_sequence2: List of pose vectors from video 2.
        n_clusters: Number of clusters for k-means.

    Returns:
        float: Overall similarity score between two sequences.
    """
    combined_sequence = np.vstack((pose_sequence1, pose_sequence2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_sequence)

    # Get cluster assignments for both sequences
    labels1 = kmeans.predict(pose_sequence1)
    labels2 = kmeans.predict(pose_sequence2)

    # Measure similarity based on cluster assignments
    _, distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, pose_sequence2)
    similarity_score = 1 - (np.mean(distances) / np.linalg.norm(combined_sequence))
    return similarity_score

# Function to compare frames and evaluate similarity
def evaluate_pose_similarity(cluster_similarity, frame_count, img1, img2, screenshot_folder, screenshot_count):
    if cluster_similarity < 0.7:  # Threshold for significant difference
        frame_number_to_capture = frame_count
        screenshot_name1 = os.path.join(screenshot_folder, f"standard_{frame_number_to_capture}.png")
        screenshot_name2 = os.path.join(screenshot_folder, f"user_{frame_number_to_capture}.png")
        cv2.imwrite(screenshot_name1, img1)
        cv2.imwrite(screenshot_name2, img2)
        screenshot_count += 1
        print(f"Captured screenshot at frame {frame_number_to_capture}")
    return screenshot_count

cap1 = cv2.VideoCapture("output.mp4") 
cap2 = cv2.VideoCapture("Rehab_Vid2.mp4")  

pose_sequence1 = []
pose_sequence2 = []

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

        img1 = cv2.resize(img1, (520, 600))
        img2 = cv2.resize(img2, (520, 600))

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

        # Extract pose landmarks for clustering
        pose1 = extract_key_landmarks(results1.pose_landmarks, key_joints)
        pose2 = extract_key_landmarks(results2.pose_landmarks, key_joints)

        if pose1 is not None and pose2 is not None:
            pose_sequence1.append(pose1)
            pose_sequence2.append(pose2)

        # Perform clustering and similarity evaluation every 10 frames
        if len(pose_sequence1) > 10 and len(pose_sequence2) > 10:
            similarity_score = pose_clustering(np.array(pose_sequence1[-10:]), np.array(pose_sequence2[-10:]))
            print(f"Similarity score at frame {frame_count}: {similarity_score:.2f}")

            screenshot_count = evaluate_pose_similarity(similarity_score, frame_count, img1, img2, screenshot_folder, screenshot_count)

        frame_count += 1

        # Display FPS
        frame_time = time.time() - start_time
        if frame_time > 0:
            fps = frame_count / frame_time
            cv2.putText(img1, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img2, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show videos
        cv2.imshow("Video 1", img1)
        cv2.imshow("Video 2", img2)

        if cv2.waitKey(5) == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()