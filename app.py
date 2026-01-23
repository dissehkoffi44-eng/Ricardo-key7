import streamlit as st
import librosa
import numpy as np
import requests
import tempfile
import os
from pydub import AudioSegment
import io
import wave
from scipy.signal import butter, lfilter
from collections import Counter

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
st.set_page_config(page_title="Music Key & Camelot Detector", page_icon="🎵", layout="wide")

NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enhanced profiles for ensemble voting (superior selectivity)
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    },
    "aarden": {
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 18.6691],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702699, 14.5212, 0.737619, 19.8145, 5.84214, 2.68046, 2.51091, 9.84455]
    },
    "sapp": {
        "major": [2, 0, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1],
        "minor": [2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0, 1]
    }
}

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

def butter_lowpass(y, sr, cutoff=180, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, y)

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=8.0)
    nyq = 0.5 * sr
    low = 60 / nyq
    high = 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    y_bass = butter_lowpass(y, sr, cutoff=150)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    profile_scores = {f"{NOTES_LIST[i]} {mode}": [] for i in range(12) for mode in ["major", "minor"]}
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                
                if mode == "minor":
                    dom_idx, leading_tone = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.42 and cv[leading_tone] > 0.32:
                        score *= 1.15
                
                if bv[i] > 0.58:
                    score += bv[i] * 0.38
                
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.48:
                    score += 0.12
                
                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.48:
                    score += 0.08
                
                if cv[i] > 0.50:
                    score += 0.45
                
                key_name = f"{NOTES_LIST[i]} {mode}"
                profile_scores[key_name].append(score)
    
    avg_scores = {k: np.mean(v) for k, v in profile_scores.items() if v}
    if not avg_scores:
        return {"key": "Unknown", "score": 0}
    
    best_key = max(avg_scores, key=avg_scores.get)
    best_score = avg_scores[best_key]
    
    candidates = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_key, top_score = candidates[0]
    if len(candidates) >= 2:
        second_key, second_score = candidates[1]
        top_i = NOTES_LIST.index(top_key.split()[0])
        second_i = NOTES_LIST.index(second_key.split()[0])
        dist = min(abs(top_i - second_i), 12 - abs(top_i - second_i))
        if dist in [3, 4, 9] and (second_score / top_score > 0.85):
            top_bv = bv[top_i]
            second_bv = bv[second_i]
            if top_bv < second_bv - 0.05:
                best_key = second_key
                best_score = second_score

    return {"key": best_key, "score": best_score}

def generate_piano_chord_audio(key_str, sr=22050, duration=2.0):
    root_note, mode = key_str.split()
    notes_freq = {
        'C':261.63, 'C#':277.18, 'D':293.66, 'D#':311.13, 'E':329.63,
        'F':349.23, 'F#':369.99, 'G':392.00, 'G#':415.30, 'A':440.00,
        'A#':466.16, 'B':493.88
    }
    
    intervals = [0, 4, 7] if mode == 'major' else [0, 3, 7]
    root_freq = notes_freq[root_note]
    freqs = [root_freq * (2 ** (i / 12)) for i in intervals]
    
    t = np.linspace(0, duration, int(sr * duration), False)
    
    attack, decay, sustain, release = 0.01, 0.2, 0.6, duration - 0.21
    env = np.zeros_like(t)
    atk_end = int(attack * sr)
    dec_end = int((attack + decay) * sr)
    rel_start = int((duration - release) * sr)
    
    env[:atk_end] = np.linspace(0, 1, atk_end)
    env[atk_end:dec_end] = np.linspace(1, sustain, dec_end - atk_end)
    env[dec_end:rel_start] = sustain
    env[rel_start:] = np.linspace(sustain, 0, len(env) - rel_start)
    
    y = np.zeros_like(t)
    for f in freqs:
        for harm in range(1, 6):
            amp = 1.0 / harm
            y += amp * np.sin(2 * np.pi * f * harm * t)
    
    y *= env
    y = 0.5 * y / np.abs(y).max()
    y_int = (y * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int.tobytes())
    buf.seek(0)
    return buf.read(), y

def simulate_ear_perception(chord_y, song_y, sr, chroma_song):
    stft_chord = np.abs(librosa.stft(chord_y))
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.mean(stft_chord, axis=1)
    
    peak_idxs = np.argsort(mag)[-12:]
    chord_freqs = freqs[peak_idxs]
    
    roughness = 0.0
    for i in range(len(chord_freqs)):
        for j in range(i+1, len(chord_freqs)):
            df = abs(chord_freqs[i] - chord_freqs[j])
            if 15 < df < 250:
                cbw = 0.25 * (chord_freqs[i] + chord_freqs[j]) / 2
                roughness += (mag[peak_idxs[i]] * mag[peak_idxs[j]]) * (df / cbw) ** 2
    
    consonance = 1 / (1 + roughness + 1e-6)
    
    chroma_chord = librosa.feature.chroma_stft(y=chord_y, sr=sr)
    chroma_chord_avg = np.mean(chroma_chord, axis=1)
    
    similarity = np.corrcoef(chroma_song, chroma_chord_avg)[0, 1]
    return 0.60 * similarity + 0.40 * consonance

def process_audio(file_bytes, file_name, sr_target=22050):
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == '.m4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            y = samples / 32768.0
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)
    except Exception as e:
        st.error(f"Erreur chargement {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 10:
        st.warning(f"{file_name} trop court ({duration}s) → imprécis.")

    y_filt = apply_sniper_filters(y, sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr), axis=1)
    bass_vector = get_bass_priority(y, sr)

    res = solve_key_sniper(chroma_avg, bass_vector)
    initial_key = res['key']
    initial_score = res['score']

    # Affinement sélectif avec simulation perceptive
    candidates = [initial_key]
    rel_mode = 'minor' if 'major' in initial_key else 'major'
    rel_offset = -3 if rel_mode == 'major' else 3
    rel_idx = (NOTES_LIST.index(initial_key.split()[0]) + rel_offset) % 12
    rel_key = f"{NOTES_LIST[rel_idx]} {rel_mode}"
    candidates.append(rel_key)
    candidates = list(set(candidates))

    perception_scores = {}
    best_perceptual_score = -1
    best_key = initial_key
    best_audio_bytes = None

    for cand_key in candidates:
        audio_bytes, chord_y = generate_piano_chord_audio(cand_key, sr=sr)
        perceptual_score = simulate_ear_perception(chord_y, y, sr, chroma_avg)
        perception_scores[cand_key] = perceptual_score
        
        if perceptual_score > best_perceptual_score + 0.05:  # Seuil pour plus de sélectivité
            best_perceptual_score = perceptual_score
            best_key = cand_key
            best_audio_bytes = audio_bytes

    # Confiance finale combinée (plus sélective)
    final_conf = min((initial_score + best_perceptual_score) / 2, 0.99)
    final_camelot = CAMELOT_MAP.get(best_key, "??")
    perception_adjusted = best_key != initial_key

    report = f"""Analyse terminée
──────────────────────────────
Fichier       : {file_name}
Durée         : {int(duration // 60):02d}:{int(duration % 60):02d}
Fréquence     : {sr} Hz

Tonalité      : {best_key}
Camelot       : {final_camelot}
Confiance     : {final_conf:.4f} (ensemble profils + perception)

Scores perception (accords simulés) :
""" + "\n".join(f"  {k:<10} : {v:.4f}" for k,v in sorted(perception_scores.items(), key=lambda x: x[1], reverse=True))

    report += "\n\nChroma moyen :\n" + "\n".join(f"  {k:<3} : {v:.4f}" for k,v in zip(NOTES_LIST, chroma_avg))

    return {
        "key": best_key,
        "camelot": final_camelot,
        "conf": final_conf,
        "audio_bytes": best_audio_bytes,
        "report": report,
        "perception_adjusted": perception_adjusted
    }

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────
st.title("🎵 Music Key & Camelot Detector")
st.markdown("Détection avancée de tonalité + Camelot — multi-profils & simulation perceptive pour précision supérieure")

# Secrets Telegram
try:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id   = st.secrets["TELEGRAM_CHAT_ID"]
    secrets_ok = True
except KeyError:
    bot_token = chat_id = None
    secrets_ok = False

if not secrets_ok:
    st.warning("Telegram non configuré → rapports non envoyés automatiquement")
    st.info("Ajoutez dans secrets.toml ou Streamlit Cloud :\n"
            "TELEGRAM_BOT_TOKEN = \"...\"\nTELEGRAM_CHAT_ID = \"...\"")


# Upload multiple files
uploaded_files = st.file_uploader(
    "Déposez un ou plusieurs fichiers audio",
    type=["mp3", "wav", "ogg", "flac", "m4a"],
    accept_multiple_files=True
)

if uploaded_files:
    total = len(uploaded_files)
    progress_global = st.progress(0)
    status_global = st.empty()

    results_container = st.container()

    for idx, file in enumerate(uploaded_files, 1):
        progress_global.progress((idx-1)/total)
        status_global.markdown(f"**Traitement {idx}/{total} :** {file.name}")

        with st.status(f"Analyse → {file.name}", expanded=(idx==1)) as status:
            status.write("Chargement & traitement audio...")
            file_bytes = file.getvalue()

            data = process_audio(file_bytes, file.name)

            if data:
                status.update(label="Terminé ✓", state="complete", expanded=False)

                # Affichage résultat
                with results_container:
                    st.markdown(f"### {file.name}")
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown(f"**Tonalité :** {data['key']}")
                        st.markdown(f"**Camelot :** <span style='font-size:2.2em; color:#f59e0b; font-weight:bold;'>{data['camelot']}</span>", unsafe_allow_html=True)
                        if data['perception_adjusted']:
                            st.markdown("<small style='color:#3b82f6;'>👂 Ajusté par simulation perceptive</small>", unsafe_allow_html=True)
                    with col2:
                        st.metric("Confiance", f"{data['conf']:.3f}")

                    st.audio(data['audio_bytes'], format='audio/wav')

                    st.text_area("Rapport détaillé", data['report'], height=420)

                    # Envoi Telegram individuel
                    if secrets_ok:
                        if st.button("Envoyer sur Telegram", key=f"send_{idx}_{hash(file.name)}"):
                            with st.spinner("Envoi..."):
                                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                                payload = {
                                    "chat_id": chat_id,
                                    "text": f"🎵 {file.name}\n\n{data['report']}",
                                    "parse_mode": "
