from pytube import YouTube

# Function to download a YouTube video
def download_video(video_url, save_path):
    try:
        # Create a YouTube object with the URL
        yt = YouTube(video_url)
        
        # Get the highest resolution stream available
        video_stream = yt.streams.get_highest_resolution()
        
        # Download the video to the specified path
        video_stream.download(save_path)
        print("Download completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the URL of the YouTube video you want to download
video_url = 'https://youtu.be/0BIaDVnYp2A?si=1yK3ThJ1tY6FRFq2'

# Replace with the path where you want to save the downloaded video
save_path = '/D:\\project'

# Call the function to download the video
download_video(video_url, save_path)
