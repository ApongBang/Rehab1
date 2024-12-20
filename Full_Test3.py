import cv2
import mediapipe as mp
import numpy as np
import statistics
import os
import requests
import json
import time

mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法

joint_list = [[16, 14, 12],[15, 13, 11],[14, 12, 24],[13, 11, 23]]
#([Right Wrist, Right Elbow, Right Shoulder],[Left Wrist, Left Elbow, Left Shoulder],[Right Elbow, Right Shoulder, Right Hip],[Left Elbow, Left Shoulder, Left Hip])
# For calculatioins of body part angles 
joint_list2 = [[24, 12, 11],[23, 11, 12],[12, 24, 23],[11, 23, 24]]
#([[Right Hip, Right Shoulder, Left Shoulder],[Left Hip, Left Shoulder, Right Shoulder],[Right Shoulder, Right Hip, Left Hip],[Left Shoulder, Left Hip, Right Hip]])



def CalculateAngles(results, img, joint_list, start_time, frame_count, img_scale=(520, 600)):
    """
    Calculate angles between joints using MediaPipe pose landmarks and draw them on the image.

    Args:
        results (object): MediaPipe results containing pose landmarks.
        img (numpy.ndarray): The image frame to annotate.
        joint_list (list): List of joint triplets (landmark indices) to calculate angles.
        start_time (float): The starting time to calculate elapsed time.
        frame_count (int): Current frame count.
        img_scale (tuple): Tuple specifying the image scale for coordinate transformation.

    Returns:
        int: Updated frame count.
    """
    if results.pose_landmarks:
        # Access the pose landmarks
        RHL = results.pose_landmarks
        
        # Calculate and display elapsed time
        frame_time = time.time() - start_time
        cv2.putText(img, f"Time: {frame_time:.2f} seconds", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Increment the frame count
        frame_count += 1

        # Calculate angles for each joint triplet
        for joint in joint_list:
            # Extract joint coordinates
            a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
            b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
            c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
            
            # Calculate the angle in degrees
            radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians_fingers * 180.0 / np.pi)  # Convert radians to degrees

            # Ensure angle is within [0, 180] degrees
            if angle > 180.0:
                angle = 360 - angle

            # Annotate the image with the calculated angle
            cv2.putText(img, str(round(angle, 2)), tuple(np.multiply(b, img_scale).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame_count





cap1 = cv2.VideoCapture("Rehab_Vid1.mp4") 
cap2 = cv2.VideoCapture("Rehab_Vid2.mp4")  

# mediapipe 啟用偵測全身
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap1.isOpened() or not cap2.isOpened():
        print("Cannot open camera")
        exit()

    start_time = time.time()
    frame_count = 0
    
    while True:

        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Cannot receive frame")
            break
        
        img1 = cv2.resize(img1,(520,600))
        img2 = cv2.resize(img2, (520, 600))
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        
        results1 = holistic.process(img1_rgb)
        results2 = holistic.process(img2_rgb)              # 開始偵測全身
        
    
        # 身體偵測，繪製身體骨架
        mp_drawing.draw_landmarks(
            img2,
            results2.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        
        mp_drawing.draw_landmarks(
            img1,
            results1.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        
        frame_count = CalculateAngles(results1, img1, joint_list, start_time, frame_count)
        frame_time = time.time() - start_time
        frame_count = CalculateAngles(results2, img2, joint_list, start_time, frame_count)
        
        # if results1.pose_landmarks:
        #     RHL = results1.pose_landmarks
        #     frame_time = time.time() - start_time
        #     cv2.putText(img1, f"Time: {frame_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     frame_count += 1
        #     # 計算角度
        #     for joint in joint_list:
                
        #         a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
        #         b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
        #         c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
        #         # 計算弧度
        #         radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        #         angle1 = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

        #         if angle1 > 180.0:
        #             angle1 = 360 - angle1
        #         cv2.putText(img1, str(round(angle1, 2)), tuple(np.multiply(b, [520,600]).astype(int)),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)



    
        frame_time = time.time() - start_time

        #print(f" frame_time data type is :{type(frame_time)}") 
        #frame_time is float 

        cv2.putText(img2, f"Time: {frame_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img1, f"Time: {frame_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        
        if frame_time > 0:
            fps = frame_count / frame_time
            cv2.putText(img2, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("oxxostudio1", img1)
        cv2.imshow('oxxostudio2', img2)
        
        if cv2.waitKey(5) == ord('q'):
            break    # 按下 q 鍵停止
