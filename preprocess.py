import os
import cv2

# Function to resize and convert video format and rename files
def resize_and_convert_video(video_path, output_folder, target_width, target_height, index):
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_folder, f"{index:04d}nonfight.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    # Resize frames and write to new video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    print(f"Video resized and saved to {output_video_path}")

# Directory containing video files
videos_folder = 'D:\project\dummy'

# Output folder for resized videos
output_folder = 'D:\project\\train\\nonvoilance'

# Target size for resizing
target_width = 640
target_height = 480

# Initialize index for renaming
index =801

# Iterate through video files, resize/convert format, and rename
for filename in os.listdir(videos_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(videos_folder, filename)
        resize_and_convert_video(video_path, output_folder, target_width, target_height, index)
        index += 1
