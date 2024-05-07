import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, TimeDistributed, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
num_frames = 30
frame_height = 64
frame_width = 64
channels = 1  # Grayscale

# Paths to your data folders
violence_folder = 'D:\\project\\train\\voilance'
nonviolence_folder = 'D:\\project\\train\\nonvoilance'

# Function to preprocess videos
def preprocess_videos(video_folder, label):
    processed_videos = []
    labels = []
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_name in video_files:
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)

        while len(frames) < num_frames:
            frames.append(frames[-1])

        processed_videos.append(np.array(frames).reshape(num_frames, frame_height, frame_width, channels))
        labels.append(label)

        cap.release()

    return np.array(processed_videos), np.array(labels)

# Preprocess the videos and assign labels
violence_data, violence_labels = preprocess_videos(violence_folder, 1)
nonviolence_data, nonviolence_labels = preprocess_videos(nonviolence_folder, 0)

# Combine the data and labels
X = np.concatenate((violence_data, nonviolence_data), axis=0)
y = np.concatenate((violence_labels, nonviolence_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the 3D CNN and LSTM model with added Dropout layers
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(num_frames, frame_height, frame_width, channels)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))  # Dropout layer added
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64))
model.add(Dropout(0.5))  # Another Dropout layer added
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a reduced learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for saving the model and reducing learning rate on plateau
checkpoint = ModelCheckpoint('model_best.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Train the model with callbacks
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=16, callbacks=[checkpoint, reduce_lr])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
model_save_path = 'D:\\project\\trained_model.h5'
model.save(model_save_path)

# Print the path where the model is saved
print(f'Model saved at: {model_save_path}')
