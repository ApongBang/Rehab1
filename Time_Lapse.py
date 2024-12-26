# 

import cv2
import os

# Path to input video file
video_file = r"E:\醫工\大四\微處理機\彭祥恩期末報告\彭祥恩_微處理機_校正後.mp4"  # Replace with the path to your video file

# Path to save the output time-lapse video
output_file = r"E:\醫工\大四\微處理機\彭祥恩期末報告\Output.mp4"  # Replace with the desired output path

# Desired output video duration and FPS
output_duration = 15  # seconds
output_fps = 30  # frames per second
total_output_frames = output_duration * output_fps

# Open the video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

# Get video properties
original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_duration = total_frames / original_fps if original_fps > 0 else 0  # seconds

if original_fps == 0 or total_frames == 0:
    print("Error: Failed to retrieve video properties.")
    cap.release()
    exit()

print(f"Original FPS: {original_fps}")
print(f"Total Frames: {total_frames}")
print(f"Original Duration: {original_duration:.2f} seconds")

# Calculate the frame skip interval to achieve the target total frames
frame_skip = max(1, int(total_frames / total_output_frames))
print(f"Frame skip: {frame_skip}")

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("Error: Video dimensions could not be retrieved.")
    cap.release()
    exit()

# Define the codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for compatibility
out = cv2.VideoWriter(output_file, fourcc, output_fps, (width, height))

frame_count = 0
frames_written = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write only the required frames to meet the output frame limit
    if frame_count % frame_skip == 0 and frames_written < total_output_frames:
        out.write(frame)
        frames_written += 1
        print(f"Writing frame {frame_count}")

    if frames_written >= total_output_frames:
        break

    frame_count += 1

print(f"Time-lapse video saved as {output_file}")
print(f"Total frames written: {frames_written}")

# Release resources
cap.release()
out.release()

# Test the output video
cap_out = cv2.VideoCapture(output_file)
if cap_out.isOpened():
    print("Output video successfully created and can be opened.")
else:
    print("Error: Output video cannot be opened.")
cap_out.release()
