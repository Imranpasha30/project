import cv2
import os
import numpy as np

# Set the directory where your images are stored
image_directory = 'D:\project\outputfilr11'

# Get a list of all files in the directory
all_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
total_images = len(all_files)
print(f"Total images to process: {total_images}")

# Initialize a counter for processed images
processed_images = 0

# Process each file in the directory
for filename in all_files:
    file_path = os.path.join(image_directory, filename)

    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the percentage of black pixels
    black_pixels = np.sum(image == 0)
    total_pixels = image.shape[0] * image.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100

    # Check if the image is 50% or more black
    if black_percentage >= 20:
        os.remove(file_path)
        print(f"Deleted {filename} because it was {black_percentage:.2f}% black.")

    # Update the processed images counter
    processed_images += 1
    print(f"Processed {processed_images}/{total_images} images.")

print("Processing complete.")
