import cv2
import os

# Function to label a video clip and save labels to a text file
def label_video_and_save(video_path, output_folder, frame_rate=10):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    # Create a list to store labels
    labels = []
    frame_counter = 0

    while True:
        ret, frame = cap.read()

        # Check if there are no more frames
        if not ret:
            break

        # Check if the frame rate needs to be adjusted
        if frame_counter % (30 // frame_rate) != 0:
            frame_counter += 1
            continue

        # Display frame and get user input for labeling
        cv2.imshow('Frame', frame)
        label = input(f"Frame {frame_counter}: Enter label (1 for violence, 0 for non-violence, 's' to skip): ")

        # Check if the user wants to skip the frame
        if label.lower() == 's':
            frame_counter += 1
            continue

        # Validate the label input
        if label in ['0', '1']:
            labels.append(label)
        else:
            print("Invalid label. Please enter 1 for violence or 0 for non-violence.")
            continue

        frame_counter += 1

        # Press 'q' to quit labeling
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close video capture and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Save labels to a text file in the output folder
    video_name = os.path.basename(video_path).split('.')[0]
    labels_file_path = os.path.join(output_folder, f"{video_name}_labels.txt")
    with open(labels_file_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Labeled data saved in {labels_file_path}")

# Function to label all videos in a folder
def label_videos_in_folder(folder_path, output_folder, frame_rate=10):
    # Iterate over all video files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".avi") or filename.endswith(".mp4"):  # Add other video formats if needed
            video_path = os.path.join(folder_path, filename)
            label_video_and_save(video_path, output_folder, frame_rate)

# Example usage:

videos_folder = "D:/project/fight"  # Path to the folder containing videos
output_folder = "D:/project/labeled_data"  # Output folder for labeled data
label_videos_in_folder(videos_folder, output_folder, frame_rate=10)
