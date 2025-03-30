import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips
import speech_recognition as sr

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "message": "No video file uploaded"}), 400
    
    video_file = request.files['video']
    time_offset = float(request.form.get('timeOffset', 0))
    
    # Save the uploaded video
    video_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp4")
    video_file.save(video_filename)
    
    # Process the video: Generate transcript and video with subtitles
    transcript, transcript_filename = generate_transcript(video_filename)
    video_with_subtitles_filename = generate_video_with_subtitles(video_filename, transcript, time_offset)
    
    # Provide both download links
    return jsonify({
        "success": True,
        "videoURL": f'/download/{video_with_subtitles_filename}',
        "transcriptURL": f'/download/{transcript_filename}'
    })


def generate_transcript(video_filename):
    # Use speech_recognition library to transcribe the audio
    recognizer = sr.Recognizer()
    audio_filename = video_filename.replace('.mp4', '.wav')

    # Extract audio from video
    clip = VideoFileClip(video_filename)
    clip.audio.write_audiofile(audio_filename)
    
    # Transcribe the audio
    transcript = []
    with sr.AudioFile(audio_filename) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            transcript.append(f"0.0s - {text}")
        except Exception as e:
            transcript.append(f"Error: {str(e)}")

    transcript_filename = f"transcript_{uuid.uuid4().hex}.txt"
    transcript_path = os.path.join(OUTPUT_FOLDER, transcript_filename)
    
    # Write transcript to a file
    with open(transcript_path, 'w') as f:
        for entry in transcript:
            f.write(f"{entry}\n")
    
    return transcript, transcript_filename


def generate_video_with_subtitles(video_filename, transcript, time_offset):
    # Generate video with subtitles using moviepy
    clip = VideoFileClip(video_filename)
    video_duration = clip.duration

    # Create subtitle clips for each line in the transcript
    subtitle_clips = []
    for line in transcript:
        timestamp, text = line.split(' - ', 1)
        timestamp = float(timestamp[:-1]) + time_offset  # Adjust with time offset
        
        # Create TextClip for the subtitle
        subtitle = TextClip(text, fontsize=24, color='white', bg_color='black', size=clip.size)
        subtitle = subtitle.set_position(('center', 'bottom')).set_duration(video_duration - timestamp).set_start(timestamp)
        subtitle_clips.append(subtitle)
    
    # Combine the video with the subtitles
    final_clip = clip
    final_clip = concatenate_videoclips([final_clip.set_duration(video_duration)]).fx(vfx.composite, *subtitle_clips)

    # Save the final video
    video_with_subtitles_filename = f"video_with_subtitles_{uuid.uuid4().hex}.mp4"
    video_with_subtitles_path = os.path.join(OUTPUT_FOLDER, video_with_subtitles_filename)
    final_clip.write_videofile(video_with_subtitles_path, codec='libx264')

    return video_with_subtitles_filename


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)