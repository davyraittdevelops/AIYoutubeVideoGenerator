# AIYoutubeVideoGenerator
Hobby project, to see if I can generate those TTS videos you always see everywhere. 

This repository contains a comprehensive script designed to automate the process of creating book summary videos for YouTube. It begins by generating a summary of a book using OpenAI's GPT-4 Turbo model and then converts this summary into speech. A thumbnail for the video is created using DALL-E 3, and subtitles are generated using the Whisper model. The script then combines these elements into a video, which is subsequently uploaded to YouTube using Google's YouTube API. Additionally, the script handles OAuth tokens for Google services and uploads the video to Google Drive, ensuring a streamlined process from summary generation to video publication.
