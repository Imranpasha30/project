import os
import numpy as np
import cv2
import gc  # Import garbage collector interface
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import tensorflow as tf
from google.colab import drive

drive.mount('/content/gdrive')

# Ensure that TensorFlow uses the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Parameters
num_frames = 30
target_height, target_width, channels = 224, 224, 3
batch_size = 2

# Function to calculate optical flow (no resizing needed here)
def calculate_optical_flow(prev_frame, next_frame):
    # Ensure both frames are grayscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) if len(next_frame.shape) == 3 else next_frame

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle

# Function to load video data and labels from a single folder
def load_video_data(folder_path, label):
    video_data = []
    labels = []
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_name in tqdm(video_files, desc='Preprocessing videos'):
        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        prev_frame = None

        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                magnitude, angle = calculate_optical_flow(prev_frame, frame)
                optical_flow_frame = np.dstack((magnitude, angle))
                frames.append(optical_flow_frame)
            prev_frame = frame

        while len(frames) < num_frames:
            frames.append(np.zeros((target_height, target_width, 2)))

        video_data.append(frames)
        labels.append(label)
        cap.release()

    return np.array(video_data), np.array(labels)

# Function to load video data and labels from multiple folders
def load_data_from_folders(parent_folder, label):
    all_video_data = []
    all_labels = []

    # Get a list of all subfolders in the parent folder
    subfolders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    # Loop through each subfolder and load the data
    for folder in tqdm(subfolders, desc='Loading folders'):
        video_data, labels = load_video_data(folder, label)
        all_video_data.append(video_data)
        all_labels.append(labels)
        del video_data, labels  # Delete variables to free memory
        gc.collect()  # Call garbage collector

    # Combine all data and labels from each folder
    X = np.vstack(all_video_data)
    y = np.hstack(all_labels)

    return X, y

# Update the paths to the parent folders containing the subfolders of videos
violence_parent_folder = '/content/gdrive/MyDrive/Colab Notebooks/SAAI project/newdata/voilaceparentfolder'
nonviolence_parent_folder = '/content/gdrive/MyDrive/Colab Notebooks/SAAI project/newdata/nonvoilanceparent'

# Load data from the parent folders
violence_data, violence_labels = load_data_from_folders(violence_parent_folder, 1)
nonviolence_data, nonviolence_labels = load_data_from_folders(nonviolence_parent_folder, 0)

# Combine and shuffle the data
X = np.vstack((violence_data, nonviolence_data))
y = np.hstack((violence_labels, nonviolence_labels))

# Rest of your code for model creation, training, and evaluation remains the same...
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% for testing

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define the VGG16 base model
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(target_height, target_width, channels))
for layer in vgg_base.layers:
    layer.trainable = False  # Set VGG16 layers to non-trainable

# Define the 3D CNN and LSTM model with added Dropout layers
print("Creating the model...")
model = Sequential()
model.add(TimeDistributed(vgg_base, input_shape=(num_frames, target_height, target_width, channels)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64))
model.add(Dropout(0.5))  # Dropout layer added
model.add(Dense(2, activation='softmax'))  # Change to softmax for one-hot encoded labels
print("model created")

# Compile the model with a reduced learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for saving the model and reducing learning rate on plateau
checkpoint = ModelCheckpoint('E:\\saai\\model_best.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Train the model with callbacks
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {test_acc}')


# Make sure to save and release memory after training
model_save_path = '/content/drive/MyDrive/Colab Notebooks/SAAI project'
model.save(model_save_path)
print(f'Model saved at: {model_save_path}')
tf.keras.backend.clear_session()  # Clear TensorFlow session to free memory
