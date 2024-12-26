# Display one video and record the camera 

import cv2
import time

def save_and_display_video(video_file, output_file, duration, fps=30):
    # Open the video file (Rehab_Vid1.mp4)
    cap1 = cv2.VideoCapture(video_file)
    if not cap1.isOpened():
        print("Error: Cannot open the video file.")
        return

    # Open the default camera (0)
    cap2 = cv2.VideoCapture(0)
    if not cap2.isOpened():
        print("Error: Cannot access the camera.")
        cap1.release()
        return

    # Get the width and height of the frames from the camera
    frame_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print(f"Recording... Saving video to {output_file}")
    start_time = time.time()

    while True:
        # Read frames from the video file
        ret1, img1 = cap1.read()

        # Read frames from the camera
        ret2, frame = cap2.read()

        if not ret1:
            print("Warning: End of video file reached.")
            break

        if not ret2:
            print("Error: Failed to read frame from the camera.")
            break

        # Resize the video file frame to match display requirements
        img1 = cv2.resize(img1, (520, 600))

        # Write the camera frame to the video file
        out.write(frame)

        # Display the video file and camera feed simultaneously
        cv2.imshow("Rehab_Vid1", img1)  # Display the video file
        cv2.imshow("Recording Video", frame)  # Display the live camera feed

        # Break the loop after the specified duration
        if time.time() - start_time >= duration:
            print("Recording complete")
            break

        # Allow user to interrupt with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording interrupted by user")
            break

    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    save_and_display_video("Rehab_Vid1.mp4", "output2.mp4", duration=30, fps=30)
