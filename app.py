import streamlit as st
import librosa
import numpy as np
import requests
import tempfile
import os

# Define key names
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler profiles for major and minor
major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Normalize profiles
major_profile = major_profile / np.sum(major_profile)
minor_profile = minor_profile / np.sum(minor_profile)

# Function to estimate key
def estimate_key(chroma):
    correlations = []
    for i in range(12):
        rotated_major = np.roll(major_profile, i)
        rotated_minor = np.roll(minor_profile, i)
        corr_major = np.corrcoef(chroma, rotated_major)[0, 1]
        corr_minor = np.corrcoef(chroma, rotated_minor)[0, 1]
        correlations.append((corr_major, 'major', i))
        correlations.append((corr_minor, 'minor', i))
    
    best_corr = max(correlations, key=lambda x: x[0])
    key_index = best_corr[2]
    scale = best_corr[1]
    confidence = best_corr[0]
    key_name = keys[key_index]
    return f"{key_name} {scale}", confidence

# To adapt to genres, we can use different profiles. For simplicity, using standard Krumhansl.
# Note: Accuracy of this algorithm is typically around 70-80% on standard datasets. Achieving 94% may require machine learning models trained on large datasets or commercial tools like Mixed In Key.

st.title("Music Tonality Detector")

# User inputs for Telegram
bot_token = st.text_input("Telegram Bot Token", type="password")
chat_id = st.text_input("Telegram Chat ID")

uploaded_file = st.file_uploader("Upload a song file", type=["mp3", "wav", "ogg"])

if uploaded_file is not None and bot_token and chat_id:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name

    try:
        # Load audio
        y, sr = librosa.load(tmp_filename, sr=None)

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Normalize chroma
        chroma_mean = chroma_mean / np.sum(chroma_mean)

        # Estimate key
        detected_key, confidence = estimate_key(chroma_mean)

        # Generate detailed report
        report = f"""
Song Analysis Report
--------------------
File: {uploaded_file.name}
Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds
Sample Rate: {sr} Hz

Detected Tonality: {detected_key}
Confidence: {confidence:.4f} (correlation coefficient)

Note: This detection uses the Krumhansl-Schmuckler algorithm, which adapts to various musical structures but may vary in accuracy across genres. For higher precision, consider training a custom ML model.

Chroma Distribution (Pitch Class Strengths):
"""
        for i, val in enumerate(chroma_mean):
            report += f"{keys[i]}: {val:.4f}\n"

        st.text_area("Analysis Report", report, height=400)

        # Send to Telegram
        if st.button("Send Report to Telegram"):
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                "chat_id": chat_id,
                "text": report
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                st.success("Report sent to Telegram successfully!")
            else:
                st.error(f"Failed to send report: {response.text}")

    finally:
        # Clean up temporary file
        os.unlink(tmp_filename)
else:
    if not bot_token or not chat_id:
        st.warning("Please provide Telegram Bot Token and Chat ID to proceed.")
