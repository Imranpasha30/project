import cv2
import os

# Define the source and destination folders
source_folder = 'D:\project\\nonvoilent123'
destination_folder = 'D:\project\dummy'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(('.mp4', '.avi', '.mov')):
        # Read the video
        cap = cv2.VideoCapture(os.path.join(source_folder, filename))
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
        out = cv2.VideoWriter(os.path.join(destination_folder, filename), fourcc, fps, (frame_width, frame_height),
                              isColor=False)

        # Convert each frame to grayscale and write it to the new video
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out.write(gray_frame)
            else:
                break

        # Release everything when done
        cap.release()
        out.release()
    else:
        continue

print("Conversion to grayscale completed!")
