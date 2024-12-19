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
