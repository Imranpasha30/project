import cv2
import numpy as np
import os
from tqdm import tqdm

# Parameters
frame_height = 64  # Height of the frame
frame_width = 64  # Width of the frame

# Paths
source_folder = 'D:\project\\voilframe'
output_folder = 'D:\project\outputfilr11'  # Output folder path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of image files
image_files = [f for f in sorted(os.listdir(source_folder)) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Initialize the progress bar
pbar = tqdm(total=len(image_files), desc='Resizing Images', unit='image')

# Initialize the counter
processed_count = 0

# Process each image
for image_name in image_files:
    image_path = os.path.join(source_folder, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    if image is None:
        print(f"Failed to load image: {image_name}")
        continue
    resized_image = cv2.resize(image, (frame_height, frame_width))  # Resize the image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, resized_image)  # Save the resized image
    processed_count += 1  # Increment the processed file count
    pbar.update(1)  # Update the progress bar

# Close the progress bar
pbar.close()

# Print the total number of processed files
print(f"Total number of processed files: {processed_count}")
