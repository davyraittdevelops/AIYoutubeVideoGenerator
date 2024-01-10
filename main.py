import pickle
from moviepy.editor import ImageClip, AudioFileClip
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from openai import OpenAI
from pathlib import Path
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import random
import requests
import time
import pysrt

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def generate_book_summary(book_title, api_key, max_retries=3, delay=2):
    """
    Generate a book summary using OpenAI's GPT-4 Turbo model with a retry mechanism.

    :param book_title: Title of the book to summarize.
    :param api_key: OpenAI API key.
    :param max_retries: Maximum number of retries for the API call.
    :param delay: Delay in seconds between retries.
    :return: Summary of the book.
    """
    url = "https://api.openai.com/v1/chat/completions"
    prompt = f"Summarize the book, make the summary around 525 words and keep the words and text simple '{book_title}'."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                full_summary = response.json()['choices'][0]['message']['content']
                return full_summary
            else:
                attempts += 1
                time.sleep(delay)  # Wait for a bit before retrying
        except requests.RequestException as e:
            attempts += 1
            time.sleep(delay)  # Wait for a bit before retrying

    return "Max retries reached. Unable to generate summary.", []

def text_to_speech(text, book_title, api_key, max_retries=3, delay=2):
    """
    Convert text to speech using OpenAI's API with a retry mechanism.

    :param text: Text to convert to speech.
    :param book_title: Title of the book (used for naming the directory).
    :param api_key: OpenAI API key.
    :param max_retries: Maximum number of retries for the API call.
    :param delay: Delay in seconds between retries.
    :return: None.
    """
    client = OpenAI(api_key=api_key)

    # Sanitize the book title to create a valid directory name
    safe_book_title = book_title.replace(" ", "_").replace('"', '').replace(':', '').lower()
    directory_path = Path(__file__).parent / f"Videos/{safe_book_title}"

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    # Set the file path for the audio file
    speech_file_path = directory_path / "voice.mp3"

    # List of available voices
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    # Randomly select a voice
    selected_voice = random.choice(voices)

    attempts = 0
    while attempts < max_retries:
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=selected_voice,
                input=text
            )

            response.stream_to_file(speech_file_path)
            print(f"Audio saved as {speech_file_path} using voice: {selected_voice}")
            return
        except Exception as e:
            print(f"Attempt {attempts + 1}: Error occurred - {e}")
            attempts += 1
            time.sleep(delay)  # Wait before retrying

    print("Max retries reached. Unable to generate speech.")

def generate_book_thumbnail(book_title, api_key, max_retries=3, delay=2):
    """
    Generate a book thumbnail using OpenAI's DALL-E 3 model with a retry mechanism.

    :param book_title: Title of the book (used for thumbnail content).
    :param api_key: OpenAI API key.
    :param max_retries: Maximum number of retries for generating the thumbnail.
    :param delay: Delay in seconds between retries.
    :return: None.
    """
    client = OpenAI(api_key=api_key)
    prompt = f"Create a thumbnail for my YouTube video. The video is a book summary about the book '{book_title}'."

    attempts = 0
    while attempts < max_retries:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",
                quality="standard",
                n=1
            )

            image_url = response.data[0].url

            # Save the image
            image_response = requests.get(image_url)
            safe_book_title = book_title.replace(" ", "_").replace('"', '').replace(':', '').lower()
            image_path = Path(__file__).parent / f"Videos/{safe_book_title}/image.png"
            with open(image_path, 'wb') as file:
                file.write(image_response.content)
            print(f"Thumbnail saved as {image_path}")
            return
        except Exception as e:
            print(f"Attempt {attempts + 1}: Error occurred - {e}")
            attempts += 1
            time.sleep(delay)  # Wait before retrying

    print("Max retries reached. Unable to generate thumbnail.")

def generate_subtitles(api_key, audio_file_path, max_retries=3, delay=2):
    """
    Generate subtitles using OpenAI's Whisper model with a retry mechanism.

    :param api_key: OpenAI API key.
    :param audio_file_path: Path to the audio file.
    :param max_retries: Maximum number of retries for generating subtitles.
    :param delay: Delay in seconds between retries.
    :return: Path to the SRT file.
    """
    client = OpenAI(api_key=api_key)

    attempts = 0
    while attempts < max_retries:
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt"
                )

            # Assuming response contains the SRT content directly
            srt_content = response

            # Save the SRT content to a file
            srt_file_path = audio_file_path.replace('.mp3', '.srt')
            with open(srt_file_path, "w") as srt_file:
                srt_file.write(srt_content)

            return srt_file_path
        except Exception as e:
            print(f"Attempt {attempts + 1}: Error occurred - {e}")
            attempts += 1
            time.sleep(delay)  # Wait before retrying

    print("Max retries reached. Unable to generate subtitles.")
    return None

def time_to_seconds(time_obj):
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000

def create_subtitle_clips(subtitles, videosize, fontsize=50, font='Arial', color='white', debug=False):
    subtitle_clips = []
    video_width, video_height = videosize

    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        text_clip = TextClip(subtitle.text, fontsize=fontsize, font=font, color=color, bg_color='black', size=(video_width*3/4, None), method='caption').set_start(start_time).set_duration(duration)
        subtitle_x_position = 'center'
        subtitle_y_position = video_height * 4 / 5
        text_position = (subtitle_x_position, subtitle_y_position)
        subtitle_clips.append(text_clip.set_position(text_position))

    return subtitle_clips


def service_youtube():
    """Create a YouTube service instance with proper token handling."""
    creds = None
    token_file = 'youtube_token.pickle'
    client_secret_file = 'client_secret.json'
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]

    # Load existing tokens
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    return build('youtube', 'v3', credentials=creds)



def upload_video_to_youtube(service, file_path, title, description, category_id, privacy_status):
    """
    Uploads a video to YouTube.

    :param service: Authorized YouTube service instance.
    :param file_path: Path to the video file to upload.
    :param title: Title of the video.
    :param description: Description of the video.
    :param category_id: Category ID of the video.
    :param book_title: Title of the book for additional keyword.
    :param privacy_status: Privacy status of the video (e.g., 'public', 'private', 'unlisted').
    """
    # Default keywords along with the book title as an additional keyword
    keywords = [
        'Book Summary', 'Self Help', 'Personal Growth', 'Motivation',
        'Life Improvement', 'Success', 'Mindfulness', 'Happiness',
        'Personal Development', 'Inspiration', 'Productivity', 'Well-being',
        'Life Hacks', 'Mental Health', 'Goal Setting', 'Self-Care',
        'Positive Thinking', 'Life Coaching', 'Self-Improvement', 'Empowerment',
        title  # Including the book title as a keyword
    ]

    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': keywords,
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    # Call the API's videos.insert method to create and upload the video.
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype='video/*')
    insert_request = service.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = insert_request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    print(f"Video ID '{response['id']}' was successfully uploaded.")


def process_books(api_key, file_path):
    # Initialize Google Drive service
    youtube_service = service_youtube()

    with open(file_path, 'r+') as file:
        lines = file.readlines()

        # Create a set to keep track of processed book titles
        processed_books = set()

        # Flag to indicate if a book has been processed in this run
        book_processed = False

        for i, line in enumerate(lines):
            # Check if the book is already marked as processed
            if line.startswith("PROCESSED:"):
                processed_books.add(line.strip().replace("PROCESSED:", "").strip())
                continue

            book_title = line.strip()

            # Skip if the book title has already been processed
            if book_title in processed_books:
                continue

            # Process the first unprocessed book title
            if not book_processed:
                print(f"Processing: {book_title}")

                # Generate summary
                full_summary = generate_book_summary(book_title, api_key)
                print(full_summary)
                full_summary += 'Thank you for watching! Please subscribe to the channel and feel free to leave any requests in the comments.'

                # Text to Speech
                text_to_speech(full_summary, book_title, api_key)

                # Generate book thumbnail
                generate_book_thumbnail(book_title, api_key)

                # Paths for image and audio files
                safe_book_title = book_title.replace(" ", "_").replace('"', '').replace(':', '').lower()
                directory_path = Path(__file__).parent / f"Videos/{safe_book_title}"
                image_path = directory_path / "image.png"
                audio_path = directory_path / "voice.mp3"
                srt_path = generate_subtitles(api_key, str(audio_path))

                # Determine the duration of the audio file
                audio_clip = AudioFileClip(str(audio_path))
                duration = audio_clip.duration

                # Output path for the video
                output_path = directory_path / f"Book_Summary_{safe_book_title}.mp4"
                output_video_path = directory_path / f"Book_Summary_{safe_book_title}_subbed.mp4"

                # Create video
                video_title = f"Book Summary: {book_title}"
                summary_lines = full_summary.split('\n')
                video_description = "\n".join(summary_lines[:2]) if len(summary_lines) >= 2 else full_summary
                create_video_from_image_and_audio(str(image_path), str(audio_path), str(output_path), duration)

                video = VideoFileClip(str(output_path))
                try:
                    subtitles = pysrt.open(str(srt_path))
                except UnicodeDecodeError:
                    subtitles = open_srt_with_encoding(str(srt_path), encoding='windows-1252')

                subtitle_clips = create_subtitle_clips(subtitles, video.size)
                final_video = CompositeVideoClip([video] + subtitle_clips)
                final_video.write_videofile(str(output_video_path))

                # Upload the video to Google Drive
                upload_video_to_youtube(youtube_service, str(output_video_path), video_title, video_description, '22',
                                         'public')

                # Mark the book as processed
                lines[i] = f"PROCESSED: {book_title}\n"

                break  # Stop after processing one book

        # Write the updated lines back to the file
        file.seek(0)
        file.writelines(lines)
        file.truncate()


def open_srt_with_encoding(file_path, encoding='windows-1252'):
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        content = f.read()
    return pysrt.from_string(content)

def service_google_drive():
    """Create a Google Drive service instance with proper token handling."""
    creds = None
    token_file = 'token.json'
    client_secret_file = 'client_secret.json'

    # Load existing tokens
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing access token: {e}")
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def upload_file_to_drive(service, file_path, name=None, description=None, folder_id=None):
    """
    Uploads a file to Google Drive with metadata.

    :param service: Authorized Google Drive service instance.
    :param file_path: Path to the file to upload.
    :param name: Name of the file (title) for metadata.
    :param description: Description of the file for metadata.
    :param folder_id: ID of the folder to upload the file into. (optional)
    """
    print('Using name: ', name)
    print('Using description : ', description)
    file_metadata = {
        'name': name or os.path.basename(file_path),  # Use the specified name or the file's base name
        'description': description or "Uploaded by my Python script",
        'mimeType': 'video/mp4'
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"File ID: {file.get('id')}")


def create_video_from_image_and_audio(image_path, audio_path, output_path, duration):
    video_clip = ImageClip(image_path, duration=duration)
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)

    final_video = CompositeVideoClip([video_clip])
    final_video.write_videofile(output_path, fps=24)

def main(api_key, books_file_path):
    while True:
        process_books(api_key, books_file_path)

        # Randomly choose a delay between 20 to 24 hours
        delay_hours = random.uniform(20, 24)
        delay_seconds = delay_hours * 3600

        print(f"Waiting for {delay_hours:.2f} hours before processing the next book.")
        time.sleep(delay_seconds)

api_key = 'placeholder'
google_api_key = 'placeholder'
books_file_path = "books.txt"
main(api_key, books_file_path);
