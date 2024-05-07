import asyncio
import cv2
import numpy as np
from keras.models import load_model
import os
import tensorflow as tf
from telegram import Bot
import geopy.geocoders
from datetime import datetime
import io
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load the pre-trained model
model = load_model('D:\\project\\model_best.h5')


# Define the send_message function
async def send_message(token, chat_id, frame, location=None):
    try:
        now = datetime.now()
        formatted_now = now.strftime(" DATE: %m/%d/%Y, TIME: %I:%M:%S %p")

        bot = Bot(token=token)

        if location:
            lat, lng = location.latitude, location.longitude
            location_url = f"https://www.google.com/maps/place/{lat},{lng}"
            finalmessage = f"VIOLENCE DETECTED !!! \n Location: {location_url} \n Time: {formatted_now}"
        else:
            finalmessage = f"VIOLENCE DETECTED !!! \n Time: {formatted_now}"

        print("Message is going to be sent.")

        # Send the message
        await bot.send_message(chat_id=chat_id, text=finalmessage)

        # Convert the frame to a byte buffer for sending as a photo
        frame_buffer = io.BytesIO()
        frame_image = Image.fromarray(frame)
        frame_image.save(frame_buffer, format='JPEG')
        frame_buffer.seek(0)

        # Send the photo
        await bot.send_photo(chat_id=chat_id, photo=frame_buffer)

        print("Message sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Parameters
num_frames = 30
frame_height = 64
frame_width = 64
channels = 1  # Grayscale
video_path = 'D:\\project\\0006.mp4'


# Function to preprocess frames

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (frame_height, frame_width))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.reshape(1, frame_height, frame_width, channels)


# Function to predict violence in a video
async def predict_violence(video_path, skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []
    frame_count = 0

    # Get the video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize geocoder
    locator = geopy.geocoders.Nominatim(user_agent="myGeocoder")

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

            if predictions[-1] > 0.5:
                label = 'Violence'
                color = (0, 0, 255)
                # Convert the current frame to an image object and send an alert
                location = None
                try:
                    location = locator.geocode("me")
                except Exception as e:
                    print(f"Failed to retrieve location information: {e}")

                # Send message with or without location
                await send_message('7178217143:AAHdiwlUSda_M1ZG8VK3NOJhotGHFA2npe4', '-1002029882964', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), location)
            else:
                label = 'Normal'
                color = (0, 255, 0)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Video', frame)

        # Wait for a period that matches the video frame rate
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


async def main():
    # Call the function with the video path and skip frames
    await predict_violence(video_path, skip_frames=2)


# If running the script directly, run the main function
if __name__ == "__main__":
    asyncio.run(main())
