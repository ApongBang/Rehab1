import cv2
import time
import numpy as np

def save_video(output_file, duration=15, fps=30):
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return

    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print(f"Recording... Saving video to {output_file}")
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame")
            break

        # Flip the frame horizontally if needed
        frame = cv2.flip(frame, 1)

        # Write the frame to the video file
        out.write(frame)

        # Display the frame
        cv2.imshow("Recording Video", frame)

        # Break the loop after the specified duration
        if time.time() - start_time >= duration:
            print("Recording complete")
            break

        # Allow user to interrupt with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording interrupted by user")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def test_camera():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return

    print("Camera will close in 10 Seconds")


    Start_Time = np.round(time.perf_counter())
    print(f"Start_Time is : {Start_Time}")
    End_Time = np.round(Start_Time+5,0)
    print(f"Estimited End time is : {End_Time}")



    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        Current_Time = np.round(time.perf_counter())
        print(f"Current_Time:{Current_Time}")
        # If frame is read correctly, ret is True
        if (ret==0):
            print("Error: Cannot read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("User Camera", frame)

        #Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera test...")
            break

        if (Current_Time==End_Time):
            break


        


    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the camera test
if __name__ == "__main__":
    #test_camera()
    save_video("output.mp4",duration =15, fps=30)




# Save a 15-second video
#save_video("output.mp4", duration=15, fps=30)