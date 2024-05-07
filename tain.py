import cv2
import os

# Function to label a video clip and save it to an output folder
def label_video_and_save(video_path, output_folder, frame_rate=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

        # Save labels to a text file after each frame is labeled
        save_labels(output_folder, video_path, labels)

        frame_counter += 1

        # Press 'q' to quit labeling
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close video capture and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Save labels to a text file after all frames are labeled
    save_labels(output_folder, video_path, labels)

def save_labels(output_folder, video_path, labels):
    # Save labels to a text file in the output folder
    video_name = os.path.basename(video_path).split('.')[0]
    labels_file_path = os.path.join(output_folder, f"{video_name}_labels.txt")
    with open(labels_file_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Labeled data saved in {labels_file_path}")

# Example usage:
video_path = "D:/project/fight/0_DzLlklZa0_3.avi"
output_folder = "D:/project/labeled_data"
label_video_and_save(video_path, output_folder, frame_rate=10)
