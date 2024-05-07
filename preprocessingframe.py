import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy.signal import wiener
from skimage import exposure

# Set the input and output folder paths
input_folder = 'D:\project\\voilframe'
output_folder = 'D:\\project\\outputfilr11'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of files in the input folder
files = os.listdir(input_folder)
total_files = len(files)  # Count the total number of files

# Initialize counters and other variables
processed_files = 0
file_counter = 0

# Define preprocessing functions
def resize_image(image, size=(640, 480)):
    return cv2.resize(image, size)

def normalize_image(image):
    # Convert image to float32 and normalize the range to [0, 1]
    return (image / 255.0).astype(np.float32)

def augment_image(image):
    # Ensure the image is in 8-bit format if it's not already
    if image.dtype != np.uint8:
        image = (image * 255).astype('uint8')

    # Apply random transformations (rotation, zoom, flip, etc.)
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)

    # Randomly rotate the image
    angle = np.random.uniform(-10, 10)  # Rotation angle range
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Randomly adjust the brightness
    value = np.random.uniform(0.9, 1.1)  # Brightness range
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Randomly zoom the image
    zoom_factor = np.random.uniform(0.9, 1.1)  # Zoom range
    h, w = image.shape[:2]
    h_taken = int(zoom_factor * h)
    w_taken = int(zoom_factor * w)
    h_start = (h - h_taken) // 2
    w_start = (w - w_taken) // 2
    image = image[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    image = cv2.resize(image, (w, h))
    return image

# Process each frame in the input folder
for filename in files:
    # Load the frame
    image = cv2.imread(os.path.join(input_folder, filename))

    # Preprocessing steps
    image = resize_image(image)
    image = normalize_image(image)
    image = augment_image(image)

    # Save the processed frame to the output folder with a new name
    new_filename = f'frame{file_counter:06d}.jpg'
    cv2.imwrite(os.path.join(output_folder, new_filename), image)

    # Increment counters
    file_counter += 1
    processed_files += 1
    print(f'Processed {processed_files}/{total_files} files.')

# Function to load images and labels
def load_images_and_labels(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    images = []
    labels = []

    for image_file in image_files:
        # Load the image
        img = cv2.imread(os.path.join(folder_path, image_file))

        # Assign the label "nonviolence" to each image
        label = "nonviolence"

        # Append the image and label to their respective lists
        images.append(img)
        labels.append(label)

    return images, labels

# Load images and labels
images, labels = load_images_and_labels(output_folder)

# Placeholder for PCA application - this part needs to be adjusted as PCA cannot be applied directly to image data


# Apply PCA for dimensionality reduction


print('All files have been processed and preprocessed for model training.')
