import os
os.environ["AUDIO_BACKEND"] = "soundfile"
import tempfile
import subprocess
import requests
import gradio as gr
import speech_recognition as sr
import numpy as np
import librosa
import torch
import shutil
import logging
import yt_dlp
import imageio_ffmpeg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load once
recognizer = sr.Recognizer()

from transformers import pipeline
classifier = pipeline("audio-classification", model="ylacombe/accent-classifier")

def download_direct_video(url):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
                'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
            if not os.path.exists(downloaded_file):
                return None, "Downloaded file not found"
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                shutil.copy(downloaded_file, tmpfile.name)
                return tmpfile.name, None
    except Exception as e:
        return None, f"Download failed: {str(e)}"

def extract_audio(video_path):
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            wav_path = audio_tmp.name
        cmd = [
            ffmpeg_path,
            "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-hide_banner", "-loglevel", "error", wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, result.stderr.strip()
        if not os.path.exists(wav_path):
            return None, "Audio extraction failed"
        return wav_path, None
    except Exception as e:
        return None, f"Audio extraction failed: {str(e)}"

def safe_load_audio(path):
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype='float32')
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio, 16000
    except:
        audio, sr = librosa.load(path, sr=16000, mono=True)
        return audio, sr

def classify_accent(audio_path):
    try:
        audio, sr = safe_load_audio(audio_path)
        if len(audio) < 48000:
            return None, "Audio too short (min 3s)"
        audio = (audio * 32767).astype(np.int16)
        results = classifier(audio, top_k=5)
        processed = [{
            "label": r["label"],
            "score": round(r["score"] * 100, 1)
        } for r in results]
        primary = max(processed, key=lambda x: x["score"])
        return {
            "accent": primary["label"],
            "confidence": primary["score"],
            "details": processed
        }, None
    except Exception as e:
        return None, f"Classification failed: {str(e)}"

def transcribe_audio(audio_path):
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data), None
    except Exception as e:
        return None, f"Transcription failed: {str(e)}"

def generate_explanation(results, transcript):
    if not results:
        return "No classification results available"
    conf_str = ", ".join(f"{r['label']}: {r['score']}%" for r in results["details"])
    explanation = (
        f"**{results['accent']} accent detected with {results['confidence']}% confidence.**\n"
        f"Top predictions: {conf_str}\n\n"
    )
    if transcript:
        snippet = transcript[:200] + "..." if len(transcript) > 200 else transcript
        explanation += f"**Transcript snippet:**\n\"{snippet}\"\n\n"
    explanation += (
        "**Interpretation guide:**\n"
        "- Scores above 70% indicate strong accent characteristics\n"
        "- Scores between 40-70% suggest mixed accent features\n"
        "- Scores below 40% may indicate minimal accent influence"
    )
    return explanation

def process_video(url):
    video_path, err = download_direct_video(url)
    if err:
        return f"Video download error: {err}"
    audio_path, err = extract_audio(video_path)
    if err:
        return f"Audio extraction error: {err}"
    results, err = classify_accent(audio_path)
    if err:
        return f"Accent classification error: {err}"
    transcript, err = transcribe_audio(audio_path)
    if err:
        return f"Transcription error: {err}"
    return generate_explanation(results, transcript)

# Gradio UI
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Text(label="Paste video URL (e.g., Loom/YouTube/MP4)"),
    outputs=gr.Markdown(label="Accent Analysis Report"),
    title="🎙️ AI Accent Analyzer",
    description="Upload a video URL to analyze speaker's accent and get transcription.",
)

demo.launch()
