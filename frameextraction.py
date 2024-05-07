import cv2
import os

# Define the directory containing video files and the output folder
video_directory = 'D:\project\\non-violent\cam2'
output_folder = 'D:\project\\non_violent_frames'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all video files in the directory
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

# Initialize the total number of videos
total_videos = len(video_files)

# Process each video file
for index, video_file in enumerate(video_files):
    # Construct the full path to the video file
    video_path = os.path.join(video_directory, video_file)

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    # Set the frame rate (e.g., 10 frames per second)
    frame_rate = 10
    prev_frame_time = 0

    # Initialize the frame count
    frame_count = 0

    # Loop through the video frames
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If the frame was read successfully
            if ret:
                # Calculate the time in milliseconds for the current frame
                current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Check if the current frame time has passed the time for the next frame
                if (current_frame_time - prev_frame_time) >= (1000 / frame_rate):
                    # Save the frame to the output folder
                    frame_filename = f'{video_file[:-4]}_frame{frame_count:06d}.jpg'
                    cv2.imwrite(os.path.join(output_folder, frame_filename), frame)

                    # Increment the frame count
                    frame_count += 1

                    # Update the previous frame time
                    prev_frame_time = current_frame_time
            else:
                break
    except Exception as e:
        print(f"Error processing video {video_file}: {e}")

    # When done with the video, release the video capture object
    cap.release()

    # Print progress
    print(f"Processed video {index + 1}/{total_videos}: {video_file}")

print("All videos have been processed.")