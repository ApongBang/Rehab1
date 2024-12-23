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

print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {mp.__version__}")
print(f"NumPy Version: {np.__version__}")

frame_number_to_capture = -1  # 初始化為負值，表示還沒有要捕捉的幀
screenshot_folder = "screenshots"  # 創建一個文件夾以保存截圖
os.makedirs(screenshot_folder, exist_ok=True) # 如果不存在，則創建文件夾
screenshot_count = 0

joint_list = [[16, 14, 12],[15, 13, 11],[14, 12, 24],[13, 11, 23]]
joint_list2 = [[24, 12, 11],[23, 11, 12],[12, 24, 23],[11, 23, 24]]
output_list = []
target_frames = []

#預設角度
#[23, 24, 12] [23, 24, 26]
#[26, 28, 32] 次等
cap1 = cv2.VideoCapture(0)  
cap2 = cv2.VideoCapture("Rehab_Vid1.mp4")  

#cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 624)
#cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#set frame sizes first




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
    
        img1 = cv2.resize(img1, (624, 720))
        img2 = cv2.resize(img2, (624, 720))
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        
        results1 = holistic.process(img1_rgb)              # 開始偵測全身
        results2 = holistic.process(img2_rgb)              # 開始偵測全身
        
    
        # 身體偵測，繪製身體骨架
        mp_drawing.draw_landmarks(img1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(img2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          
        if results1.pose_landmarks:
            RHL = results1.pose_landmarks
            frame_time = time.time() - start_time
            cv2.putText(img1, f"Time: {frame_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_count += 1
            # 計算角度
            for joint in joint_list:
                
                a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                # 計算弧度
                radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle1 = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

                if angle1 > 180.0:
                    angle1 = 360 - angle1
                cv2.putText(img1, str(round(angle1, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #     if results1.pose_landmarks:
    #         RHL = results1.pose_landmarks
            
    #         # 計算角度
    #         for joint in joint_list2:
                
    #             a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
    #             b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
    #             c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
    #             # 計算角度
    #             radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    #             angle3 = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

    #             if angle3 > 180.0:
    #                 angle3 = 360 - angle3

    #             cv2.putText(img1, str(round(angle3, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #     if results2.pose_landmarks:
    #         RHL = results2.pose_landmarks
    #         frame_count += 1
    #         # 計算角度
    #         for joint in joint_list:
                
    #             a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
    #             b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
    #             c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
    #             # 計算弧度
    #             radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    #             angle2 = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

    #             if angle2 > 180.0:
    #                 angle2 = 360 - angle2

    #             cv2.putText(img2, str(round(angle2, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #     if results2.pose_landmarks:
    #         RHL = results2.pose_landmarks
    #         # 計算角度
    #         for joint in joint_list2:
                
    #             a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
    #             b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
    #             c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
    #             # 計算弧度
    #             radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    #             angle4 = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

    #             if angle4 > 180.0:
    #                 angle4 = 360 - angle4

    #             cv2.putText(img2, str(round(angle4, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    #         #計算誤差率(解釋)   
            dd1 = angle1-angle2
            mis1 = dd1*1.6/180
            dd2 = angle3-angle4
            mis2 = dd2*0.4/180
            mis3 = (mis1+mis2)*100
            if mis3  > 50:
                frame_number_to_capture = frame_count  #要捕捉的幀數
                screenshot_name = os.path.join(screenshot_folder, f"standard_{frame_number_to_capture}.png")
                screenshot_name2 = os.path.join(screenshot_folder, f"user_{frame_number_to_capture}.png")
                cv2.imwrite(screenshot_name, img1)  # 保存截圖
                cv2.imwrite(screenshot_name2, img2)
                screenshot_count += 1
                print(f"在幀 {frame_number_to_capture} 時捕捉截圖")
            if screenshot_count >= 5:
               break
            if mis3  > 60:
                frame_number_to_capture = frame_count  #要捕捉的幀數
                screenshot_namemax = os.path.join(screenshot_folder, f"standard_{frame_number_to_capture}.png")
                
            output_list.append(mis3)
        mean = statistics.mean(output_list)
    #     # 顯示兩個影片結果
    #     # 顯示兩個影片結果
        
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)

        frame_time = time.time() - start_time
        cv2.putText(img2, f"Time: {frame_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        if frame_time > 0:
            fps = frame_count / frame_time
            cv2.putText(img1, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img2, f"FPS: {round(fps, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('oxxostudio1', img1)
        cv2.imshow('oxxostudio2', img2)

        if cv2.waitKey(5) == ord('q'):
            break    # 按下 q 鍵停止
    # if mean > 0:
    #     print("錯誤率",mean,"%")
    #     error_message = f"錯誤率 {mean}%"
    # if mean < 0:
    #     mean = 0-mean
    #     print("錯誤率",mean,"%")
    #     error_message = f"錯誤率 {mean}%"
    #     # 將錯誤率寫入記事本檔案
    # desktop_path = os.path.expanduser("~/Desktop")
    # output_file_path = os.path.join(desktop_path, "error_rate.txt")
    # with open(output_file_path, "a") as file:
    #     file.write(error_message +"\n" )
    
    #測試


# # IFTTT Webhooks URL網址+要使
#         webhook_url = "https://maker.ifttt.com/trigger/ERROR_TEST/json/with/key/bjmff5PKrmSMBskiEjTSZC"

#         mean = round(mean, 3)
# # mean2 = round(mean2, 3)

#         data = { '動作錯誤率(%)' : {
#                 "整體錯誤率": str(mean),
#                 "檢查": str(mean),
#                  "photo": screenshot_namemax
#             }}
#         json_data = json.dumps(data)

# # 發送 HTTP POST 請求，將 JSON 數據傳輸到 Webhook
#         response = requests.post(webhook_url, data=json_data, headers={'Content-Type': 'application/json'})


#         if response.status_code == 200:
#             print("成功發送至IFTTT Webhooks :)")
#         else:
#             print("發送失敗")


# cap1.release()
# cap2.release()

# cv2.destroyAllWindows()

