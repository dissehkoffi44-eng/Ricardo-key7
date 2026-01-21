import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment
import wave  # Pour génération WAV

# --- FORCE FFMEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

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
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    .warning-ambiguous {
        background: rgba(245, 158, 11, 0.15); color: #fbbf24;
        padding: 12px; border-radius: 10px; border: 1px solid #f59e0b;
        margin: 10px 0; font-family: 'JetBrains Mono', monospace;
    }
    .perception-alert {
        background: rgba(59, 130, 246, 0.15); color: #60a5fa;
        padding: 15px; border-radius: 15px; border: 1px solid #3b82f6;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    low = 80 / nyq
    high = 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector):
    best_overall_score = -1
    best_key = "Unknown"
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    candidates = []
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                
                # Bonus mineur (dominante + sensible)
                if mode == "minor":
                    dom_idx, leading_tone = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.45 and cv[leading_tone] > 0.35:
                        score *= 1.18
                
                # Bonus basse renforcé
                if bv[i] > 0.65:
                    score += bv[i] * 0.35
                
                # Bonus quinte
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.5:
                    score += 0.1
                
                # Bonus tierce
                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.5:
                    score += 0.1
                
                # Bonus tonique explicite
                if cv[i] > 0.55:
                    score += 0.25
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_key = f"{NOTES_LIST[i]} {mode}"
                
                candidates.append((f"{NOTES_LIST[i]} {mode}", score, i, bv[i]))

    # Post-processing léger pour les confusions à 4 demi-tons
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_key, top_score, top_i, top_bv = candidates[0]
        
        if len(candidates) >= 2:
            second_key, second_score, second_i, second_bv = candidates[1]
            dist = min(abs(top_i - second_i), 12 - abs(top_i - second_i))
            
            if dist == 4 and (second_score / top_score > 0.78):  # très proche
                if top_bv > second_bv + 0.12:
                    # basse confirme le premier → on garde
                    pass
                else:
                    # basse préfère le second ou neutre → on signale ambiguïté
                    best_key = top_key  # on garde quand même le meilleur score
                    # mais on pourra afficher un warning dans l'UI

    return {"key": best_key, "score": best_overall_score}

# Fonction pour générer un accord de piano simulé réaliste
def generate_piano_chord_audio(key_str, sr=22050, duration=2.0):
    root_note, mode = key_str.split()
    # Fréquences de base pour notes (A4=440Hz)
    notes_freq = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
        'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
        'A#': 466.16, 'B': 493.88
    }
    
    # Intervalles pour l'accord (sans octave pour simplicité)
    if mode == 'major':
        intervals = [0, 4, 7]  # tonique, tierce maj, quinte
    else:
        intervals = [0, 3, 7]  # tonique, tierce min, quinte
    
    # Génération des fréquences
    root_freq = notes_freq[root_note]
    freqs = [root_freq * (2 ** (i / 12)) for i in intervals]
    
    # Temps
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Enveloppe ADSR simple pour piano-like (Attack, Decay, Sustain, Release)
    attack = 0.01
    decay = 0.2
    sustain = 0.6
    release = duration - (attack + decay)
    
    env = np.zeros_like(t)
    atk_end = int(attack * sr)
    dec_end = int((attack + decay) * sr)
    rel_start = int((duration - release) * sr)
    
    env[:atk_end] = np.linspace(0, 1, atk_end)
    env[atk_end:dec_end] = np.linspace(1, sustain, dec_end - atk_end)
    env[dec_end:rel_start] = sustain
    env[rel_start:] = np.linspace(sustain, 0, len(env) - rel_start)
    
    # Génération du son avec harmoniques pour réalisme (piano-like)
    y = np.zeros_like(t)
    for f in freqs:
        # Fundamental + 4 harmoniques décroissantes
        for harm in range(1, 6):  # Augmenté à 5 pour plus de richesse
            amp = 1.0 / harm  # Amplitude diminue avec harmonique
            y += amp * np.sin(2 * np.pi * f * harm * t)
    
    y *= env  # Appliquer enveloppe
    y = 0.5 * y / np.abs(y).max()  # Normaliser
    
    # Convertir en int16
    y_int = (y * 32767).astype(np.int16)
    
    # Sauvegarder en BytesIO WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int.tobytes())
    
    buf.seek(0)
    return buf.read(), y  # Retourner bytes + signal audio brut pour analyse perceptuelle

# NOUVEAU : Simulation de la perception auditive humaine (consonance + similarité)
def simulate_ear_perception(chord_y, song_y, sr, chroma_song):
    """
    Simule l'oreille : Calcule un score de 'consonance perçue' et similarité.
    - Roughness (simplifiée) : Somme des battements entre fréquences (modèle basique Plomp-Levelt).
    - Similarité chroma : Corrélation entre chroma de l'accord et du morceau.
    Score haut = "sonne bien" (consonant et similaire).
    """
    # 1. Extraire fréquences dominantes de l'accord (via STFT)
    stft_chord = np.abs(librosa.stft(chord_y))
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.mean(stft_chord, axis=1)  # Magnitude moyenne
    
    # Sélectionner pics fréquentiels (top 10 pour simplicité)
    peak_idxs = np.argsort(mag)[-10:]
    chord_freqs = freqs[peak_idxs]
    
    # 2. Calculer roughness (dissonance perçue) : Somme des battements
    roughness = 0.0
    for i in range(len(chord_freqs)):
        for j in range(i+1, len(chord_freqs)):
            df = abs(chord_freqs[i] - chord_freqs[j])
            if 20 < df < 200:  # Battements perceptibles dans 20-200 Hz
                roughness += (mag[peak_idxs[i]] * mag[peak_idxs[j]]) * (df / 200) ** 2  # Formule simplifiée Plomp-Levelt
    
    consonance = 1 / (1 + roughness + 1e-6)  # Inverser pour consonance (0-1)
    
    # 3. Chroma de l'accord
    chroma_chord = librosa.feature.chroma_stft(y=chord_y, sr=sr)
    chroma_chord_avg = np.mean(chroma_chord, axis=1)
    
    # 4. Similarité avec le chroma du morceau
    similarity = np.corrcoef(chroma_song, chroma_chord_avg)[0, 1]
    
    # Score total perceptif (pondéré : 60% similarité, 40% consonance)
    perceptual_score = 0.6 * similarity + 0.4 * consonance
    return perceptual_score

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            y = samples / (2**15)
            sr = audio.frame_rate
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier {file_name}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    step, timeline, votes = 6, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 2))
    total_segments = len(segments)
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            prog_internal = int((idx / total_segments) * 100)
            _progress_callback(prog_internal, f"Scan chirurgical : {start}s / {int(duration)}s")

        idx_start, idx_end = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        res = solve_key_sniper(c_avg, b_seg)
        
        # Seuil relevé pour plus de fiabilité
        if res['score'] < 0.88: continue
        
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes:
        return None

    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / max(1, sum(votes.values()))) > 0.25
    target_key = most_common[1][0] if mod_detected else None
    
    # Détection d'ambiguïté F <-> Ab (ou autres à 4 demi-tons)
    ambiguous = False
    ambiguous_key = None
    if len(most_common) >= 2:
        n1 = most_common[0][0].split()[0]
        n2 = most_common[1][0].split()[0]
        idx1 = NOTES_LIST.index(n1)
        idx2 = NOTES_LIST.index(n2)
        dist = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
        if dist == 4 and most_common[1][1] / most_common[0][1] > 0.75:
            ambiguous = True
            ambiguous_key = most_common[1][0]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)

    # NOUVEAU : Simulation perceptuelle pour vérification/choix
    candidates = [final_key]  # Candidat principal
    if ambiguous and ambiguous_key:
        candidates.append(ambiguous_key)  # Ajouter l'ambigu
    # Ajouter la relative (ex: si major → minor, et vice-versa)
    rel_mode = 'minor' if 'major' in final_key else 'major'
    rel_key = final_key.replace('major', 'minor') if 'major' in final_key else final_key.replace('minor', 'major')
    rel_idx = (NOTES_LIST.index(final_key.split()[0]) - 3) % 12 if rel_mode == 'major' else (NOTES_LIST.index(final_key.split()[0]) + 3) % 12
    rel_key = f"{NOTES_LIST[rel_idx]} {rel_mode}"
    candidates.append(rel_key)
    candidates = list(set(candidates))  # Uniques

    best_perceptual_score = -1
    best_key = final_key
    best_audio_bytes = None
    perception_adjusted = False

    for cand_key in candidates:
        audio_bytes, chord_y = generate_piano_chord_audio(cand_key, sr=sr)
        perceptual_score = simulate_ear_perception(chord_y, y_filt, sr, chroma_avg)
        if perceptual_score > best_perceptual_score:
            best_perceptual_score = perceptual_score
            best_key = cand_key
            best_audio_bytes = audio_bytes
            if cand_key != final_key:
                perception_adjusted = True

    # Mise à jour si ajusté par perception
    if perception_adjusted:
        final_key = best_key
        final_conf = min(final_conf, 90)  # Baisser un peu la conf si ajusté

    res_obj = {
        "key": final_key,
        "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99),
        "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline,
        "chroma": chroma_avg.tolist(),
        "modulation": mod_detected,
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name,
        "ambiguous": ambiguous,  # nouveau flag pour l'UI
        "audio_bytes": best_audio_bytes,  # Bytes pour st.audio
        "perception_adjusted": perception_adjusted  # Flag pour UI
    }

    # Envoi Telegram (avec info perception)
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            df_tl = pd.DataFrame(timeline)
            fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
            img_tl = fig_tl.to_image(format="png", width=1000, height=500)
            fig_rd = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_rd.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            img_rd = fig_rd.to_image(format="png", width=600, height=600)
            caption = (f" 🎯 *RCDJ228 MUSIC SNIPER - RAPPORT*\n━━━━━━━━━━━━\n"
                       f" 📂 *FICHIER:* `{file_name}`\n"
                       f" 🎵 *TONALITÉ:* `{final_key.upper()}`\n"
                       f" 🧭 *CAMELOT:* `{res_obj['camelot']}`\n"
                       f" 📈 *CONFIANCE:* `{res_obj['conf']}%`\n"
                       f" 🥁 *TEMPO:* `{res_obj['tempo']} BPM`\n"
                       f" 🎸 *ACCORD:* `{res_obj['tuning']} Hz`\n"
                       f"{' ⚠️ *MODULATION:* ' + target_key.upper() if mod_detected else ' ✅ *STABILITÉ:* OK'}\n"
                       f"{' ⚠️ *AMBIGUÏTÉ POSSIBLE (4 demi-tons)*' if ambiguous else ''}\n"
                       f"{' 👂 *AJUSTÉ PAR PERCEPTION AUDITIVE*' if perception_adjusted else ''}\n━━━━━━━━━━━━")
            files = {'p1': ('timeline.png', img_tl, 'image/png'), 'p2': ('radar.png', img_rd, 'image/png')}
            media = [{'type': 'photo', 'media': 'attach://p1', 'caption': caption, 'parse_mode': 'Markdown'}, {'type': 'photo', 'media': 'attach://p2'}]
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup", data={'chat_id': CHAT_ID, 'media': json.dumps(media)}, files=files, timeout=15)
        except:
            pass

    del y, y_filt
    gc.collect()
    return res_obj

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }}; """

# --- INTERFACE PRINCIPALE ---
st.title("🎯 RCDJ228 MUSIC SNIPER")

uploaded_files = st.file_uploader("📂 Déposez vos fichiers audio", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    global_progress_placeholder = st.empty()
    total_files = len(uploaded_files)
    results_container = st.container()
    
    for i, f in enumerate(reversed(uploaded_files)):
        global_progress_placeholder.markdown(f"""
            <div style="padding:15px; border-radius:15px; background-color:rgba(16, 185, 129, 0.1); border:1px solid #10b981; margin-bottom:20px;">
                <h3 style="margin:0; color:#10b981;">🔍 ANALYSE EN COURS : {i+1} / {total_files}</h3>
                <p style="margin:5px 0 0 0; opacity:0.8;">Fichier actuel : {f.name}</p>
            </div>
            """, unsafe_allow_html=True)

        with st.status(f"🎯 Sniper scan : `{f.name}`", expanded=True) as status:
            inner_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(val, msg):
                inner_bar.progress(val)
                status_text.code(msg)

            audio_bytes = f.getvalue()
            data = process_audio_precision(audio_bytes, f.name, _progress_callback=update_progress)
            status.update(label=f"✅ {f.name} analysé", state="complete", expanded=False)

        if data:
            with results_container:
                st.markdown(f"<div class='file-header'>📂 ANALYSE TERMINÉE : {data['name']}</div>", unsafe_allow_html=True)
                
                color = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"
                st.markdown(f"""
                    <div class="report-card" style="background:{color};">
                        <h1 style="font-size:5.5em; margin:10px 0; font-weight:900;">{data['key'].upper()}</h1>
                        <p style="font-size:1.5em; opacity:0.9;">CAMELOT: <b>{data['camelot']}</b> | CONFIANCE: <b>{data['conf']}%</b></p>
                        {f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : {data['target_key'].upper()} ({data['target_camelot']})</div>" if data['modulation'] else ""}
                        {f"<div class='warning-ambiguous'>⚠️ AMBIGUÏTÉ POSSIBLE (peut-être tonalité à 4 demi-tons)</div>" if data.get('ambiguous', False) else ""}
                        {f"<div class='perception-alert'>👂 AJUSTÉ PAR SIMULATION PERCEPTIVE</div>" if data.get('perception_adjusted', False) else ""}
                    </div> """, unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                with m1: 
                    st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                with m2: 
                    st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{i}_{hash(data['name'])}"
                    components.html(f"""<button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER L'ACCORD SIMPLE</button>
                                    <script>{get_chord_js(btn_id, data['key'])}</script>""", height=110)

                # Lecteur audio pour accord piano réaliste (vérification finale)
                st.markdown("<div style='text-align:center; margin-top:10px; font-weight:bold; color:#10b981;'>🎹 VÉRIFICATION FINALE : ACCORD PIANO SIMULÉ</div>", unsafe_allow_html=True)
                st.audio(data['audio_bytes'], format='audio/wav')

                c1, c2 = st.columns([2, 1])
                with c1: 
                    fig_tl = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                    fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig_tl, use_container_width=True, key=f"tl_plot_{i}_{hash(f.name)}")
                with c2: 
                    fig_rd = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                    fig_rd.update_layout(template="plotly_dark", height=300, polar=dict(radialaxis=dict(visible=False)), margin=dict(l=30, r=30, t=20, b=20))
                    st.plotly_chart(fig_rd, use_container_width=True, key=f"rd_plot_{i}_{hash(f.name)}")
                
                st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

    global_progress_placeholder.success(f"🏁 Mission terminée : {total_files} fichiers analysés avec succès !")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")
    if st.button("🗑️ Vider la file d'analyse"):
        st.cache_data.clear()
        st.rerun()
