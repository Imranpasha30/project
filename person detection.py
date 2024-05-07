import cv2
import numpy as np
from keras.models import load_model
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses most warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppresses deprecated function warnings

# Load the pre-trained model
model = load_model('D:\project\modelnew.h5')

# Parameters
num_frames = 30  # Assuming this is the correct number of frames as per your training configuration
frame_height = 64
frame_width = 64
channels = 1  # Grayscale
video_path = 'D:\project\sam.mp4'
# Function to preprocess frames
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.reshape(1, frame_height, frame_width, channels)

# Function to predict violence in a video
def predict_violence(video_path, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []
    frame_count = 0

    # Get the video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Skip frames if needed
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)

        if len(frames) == num_frames:
            video_clip = np.array(frames).reshape(1, num_frames, frame_height, frame_width, channels)
            prediction = model.predict(video_clip)
            predictions.append(prediction[0][0])
            frames.pop(0)  # Remove the oldest frame

            print(f'Prediction: {prediction[0][0]}')  # Debugging line

            if predictions[-1] > 0.5:
                label = 'Violence'
                color = (0, 0, 255)
            else:
                label = 'Normal'
                color = (0, 255, 0)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Video', frame)

        # Wait for a period that matches the video frame rate
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function with the video path and skip frames
predict_violence(video_path, skip_frames=2)

