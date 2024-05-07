import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('D:\\project\\trained_model.h5')

# Parameters
num_frames = 30
frame_height = 64
frame_width = 64
channels = 1  # Grayscale

# Function to preprocess frames
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.reshape(1, frame_height, frame_width, channels)

# Function to predict violence in a video
def predict_violence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)

        # Ensure we always have 'num_frames' to predict on
        if len(frames) == num_frames:
            video_clip = np.array(frames).reshape(1, num_frames, frame_height, frame_width, channels)
            prediction = model.predict(video_clip)
            predictions.append(prediction[0][0])

            # Clear the frames list for the next set of predictions
            frames.pop(0)

        # Annotate the frame based on the prediction
        if predictions and predictions[-1] > 0.5:  # Threshold can be adjusted
            # Highlight frame with red box and label for violence
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
            cv2.putText(frame, 'Violence', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Highlight frame with green box and label for normal
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(frame, 'Normal', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Path to the video file
video_path = 'D:\project\\2 - sêÆsêåF+çtÜäµò¦µì«T¢å\\val\Fight\\0Ow4cotKOuw_3.avi'  # Adjust the path as needed
predict_violence(video_path)
