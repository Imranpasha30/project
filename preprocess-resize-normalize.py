import cv2
import numpy as np
import os
from tqdm import tqdm

# Parameters
num_frames = 30  # Number of frames to capture from each video
frame_height = 64  # Height of the frame
frame_width = 64  # Width of the frame
channels = 1  # Number of channels (1 for grayscale)

# Paths
source_folder = 'D:\project\dummy'
processed_data = []

# Get the list of video files
video_files = [f for f in sorted(os.listdir(source_folder)) if f.endswith(('.mp4', '.avi', '.mov'))]

# Initialize the progress bar
pbar = tqdm(total=len(video_files), desc='Processing Videos', unit='video')

# Process each video
for video_name in video_files:
    video_path = os.path.join(source_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break  # If there are no frames left
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)

        # If the video is shorter than required frames, repeat the last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])

        processed_data.append(np.array(frames).reshape(num_frames, frame_height, frame_width, channels))

    finally:
        cap.release()

    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

# Convert the list to a numpy array
processed_data = np.array(processed_data)

# Now 'processed_data' is ready to be used as input for your neural network
