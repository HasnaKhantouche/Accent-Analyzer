import os
os.environ["AUDIO_BACKEND"] = "soundfile"
import re
import tempfile
import subprocess
import requests
import streamlit as st
import speech_recognition as sr
import numpy as np
import librosa
import torch
import time
import shutil
import logging
import yt_dlp

# imageio-ffmpeg provides its own FFmpeg binary at install time
import imageio_ffmpeg

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_accent_model():
    try:
        from transformers import pipeline
        classifier = pipeline(
            "audio-classification",
            model="ylacombe/accent-classifier"
        )
        st.success("Loaded accent classification model")
        return classifier
    except ImportError:
        st.error("Transformers library not installed properly")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

@st.cache_resource
def load_speech_recognizer():
    return sr.Recognizer()

def download_direct_video(url):
    """
    Handles Loom and direct MP4 URLs using yt-dlp.
    Returns (temp_file_path, error_message, video_title)
    """
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
                video_title = info.get('title', 'Loom Video')
                downloaded_file = ydl.prepare_filename(info)

            # Find the downloaded file
            if not os.path.exists(downloaded_file):
                return None, "Downloaded file not found", None

            # Copy to a permanent temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                shutil.copy(downloaded_file, tmpfile.name)
                return tmpfile.name, None, video_title

    except Exception as e:
        return None, f"Download failed: {str(e)}", None
        
def extract_audio(video_path):
    """
    Use the FFmpeg binary provided by imageio-ffmpeg to extract a mono, 16kHz WAV from the downloaded MP4.
    Returns (wav_path, None) on success, or (None, error_message) on failure.
    """
    try:
        # 1) Get the ffmpeg executable from imageio-ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if not ffmpeg_path:
            return None, "Audio extraction failed: imageio-ffmpeg could not locate ffmpeg"

        # 2) Create a temp file for WAV output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            wav_path = audio_tmp.name

        # 3) Build and run the ffmpeg command
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output if it exists
            "-i", video_path,
            "-vn",  # Drop video track
            "-acodec", "pcm_s16le",  # Encode as 16-bit PCM
            "-ar", "16000",  # Resample to 16 kHz
            "-ac", "1",  # Mono audio
            "-hide_banner",  # Suppress banner text
            "-loglevel", "error",  # Only show errors
            wav_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown FFmpeg error"
            if "moov atom not found" in error_msg:
                return None, "Video file is incomplete or corrupted"
            elif "Invalid data found" in error_msg:
                return None, "Invalid video file format"
            else:
                return None, f"FFmpeg error: {error_msg}"

        # 4) Verify output
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return None, "Audio extraction failed: output file is empty"

        return wav_path, None

    except subprocess.TimeoutExpired:
        return None, "Audio extraction timed out after 2 minutes"
    except Exception as e:
        return None, f"Audio extraction failed: {str(e)}"

def safe_load_audio(path):
    """Enhanced universal audio loader with better error handling"""
    try:
        # Method 1: Try soundfile directly (best option)
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio, 16000
        
    except Exception as e:
        logger.warning(f"SoundFile failed ({str(e)}), trying fallbacks...")
        try:
            # Method 2: Try librosa with modern parameters
            audio, sr = librosa.load(path, sr=16000, mono=True)
            return audio, sr
        except Exception as e:
            logger.warning(f"Librosa modern failed ({str(e)}), trying legacy...")
            try:
                # Method 3: Try librosa legacy
                audio, sr = librosa.load(path, sr=16000, mono=True, backend='soundfile')
                return audio, sr
            except Exception as e:
                logger.warning(f"Librosa legacy failed ({str(e)}), trying pydub...")
                try:
                    # Method 4: Final fallback using pydub
                    from pydub import AudioSegment
                    audio = (AudioSegment.from_file(path)
                             .set_frame_rate(16000)
                             .set_channels(1))
                    samples = np.array(audio.get_array_of_samples())
                    return samples.astype(np.float32) / 32768.0, 16000
                except Exception as e:
                    raise ValueError(f"All audio loading methods failed: {str(e)}")

def classify_accent(audio_path, classifier):
    """Enhanced classification with better error handling"""
    try:
        audio, sr = safe_load_audio(audio_path)
        
        # Enhanced audio validation
        if len(audio) < 48000:  # 3 seconds at 16kHz
            return None, "Audio too short (minimum 3 seconds required)"
            
        # Normalization for classifier (safer implementation)
        if isinstance(audio[0], np.float32):
            audio = (audio * 32767).astype(np.int16)
        elif isinstance(audio[0], np.float64):
            audio = (audio * 32767).astype(np.int16)
            
        # Add length check after normalization
        if len(audio) == 0:
            return None, "Empty audio after processing"
            
        results = classifier(audio, top_k=5)

        processed = []
        total_score = sum(r["score"] for r in results)
        for r in results:
            processed.append({
                "label": r["label"],
                "score": round((r["score"] / total_score) * 100, 1)
            })

        primary = max(processed, key=lambda x: x["score"])
        return {
            "accent": primary["label"],
            "confidence": primary["score"],
            "details": sorted(processed, key=lambda x: -x["score"])
        }, None

    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        return None, f"Classification error: {str(e)}"

def transcribe_audio(audio_path, recognizer):
    """
    Use Google's Speech Recognition via SpeechRecognition library to transcribe.
    Returns (transcript, None) or (None, error_msg).
    """
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text, None
    except sr.UnknownValueError:
        return None, "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return None, f"Could not request results: {str(e)}"
    except Exception as e:
        return None, f"Transcription failed: {str(e)}"

def generate_explanation(results, transcript):
    """
    Build a markdown explanation string from the classification results.
    """
    if not results:
        return "No classification results available"

    conf_str = ", ".join(
        f"{r['label']}: {r['score']:.1f}%"
        for r in results["details"]
    )
    explanation = (
        f"**{results['accent']} accent detected with {results['confidence']}% confidence.**\n\n"
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

def main():
    st.set_page_config(
        page_title="AI Accent Analyzer",
        layout="wide",
        page_icon="🎙️"
    )
    st.title("🎙️ AI-Powered Accent Classification")
    st.markdown("""
        *Analyze English accents from direct video URLs (MP4 or Loom)*
    """)

    url = st.text_input(
        "Enter video URL:",
        "",
        placeholder="https://example.com/video.mp4 or Loom video URL"
    )

    if st.button("Analyze Accent"):
        if not url:
            st.warning("Please enter a video URL")
            return

        video_title = "Unknown"
        file_path = None
        audio_path = None

        # Normal Direct URL flow
        with st.spinner("🔄 Downloading video..."):
            file_path, error, video_title = download_direct_video(url)
            if error:
                st.error(f"Download error: {error}")
                return

            st.success(f"Downloaded: {video_title}")
            logger.info(f"Downloaded file: {file_path} ({os.path.getsize(file_path)/1024:.1f} KB)")

            with st.spinner("🔊 Extracting audio..."):
                audio_path, error = extract_audio(file_path)
                if error:
                    st.error(f"Audio error: {error}")
                    return

        try:
            # Load models
            classifier = load_accent_model()
            recognizer = load_speech_recognizer()
            if not classifier:
                st.error("No accent classification model available")
                return

            # Accent classification
            with st.spinner("🧠 Analyzing accent..."):
                results, error = classify_accent(audio_path, classifier)
                if error:
                    st.error(f"Analysis error: {error}")
                    return

            # Transcription
            with st.spinner("📝 Transcribing content..."):
                transcript, trans_error = transcribe_audio(audio_path, recognizer)
                if trans_error:
                    st.warning(f"Transcription warning: {trans_error}")
                    transcript = None

            # Display results
            st.success("✅ Analysis Complete!")
            st.subheader(f"Results for: {video_title}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Primary Result")
                st.metric("Detected Accent", results["accent"])
                st.metric("Confidence Score", f"{results['confidence']}%")
                st.audio(audio_path)

            with col2:
                st.subheader("Detailed Breakdown")
                for r in results["details"]:
                    label = r["label"]
                    st.progress(int(r["score"]), text=f"{label}: {r['score']:.1f}%")

            st.subheader("Analysis Report")
            explanation = generate_explanation(results, transcript)
            st.markdown(explanation)

            if transcript:
                with st.expander("Full Transcript"):
                    st.write(transcript)

        finally:
            # Always clean up temporary files if they exist
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            logger.info("Cleaned up temporary files")

    st.markdown("---")
    st.info("""
        **💡 About this AI tool**: 
        - Supports direct video URLs (MP4 or Loom)
        - Classifies accents using state-of-the-art deep learning
        - No personal data is stored
    """)
    st.markdown("""
            **Sample Video to Test**:
            - American Accent (Direct URL):   http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4
        """)

if __name__ == "__main__":
    main()