import streamlit as st
import librosa
import numpy as np
import requests
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydub import AudioSegment
import io
from collections import Counter
from scipy.signal import butter, lfilter
import gc

# Configuration page
st.set_page_config(page_title="Music Key Expert • Improved", page_icon="🎵", layout="wide")

# FFMPEG path (optionnel)
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# ────────────────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────────────────
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "aarden": {  # garde les valeurs longues pour plus de précision
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 18.6691],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702699, 14.5212, 0.737619, 19.8145, 5.84214, 2.68046, 2.51091, 9.84455]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

CAMELOT_MAP = {  # inchangé
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

WEIGHTS = {
    "profiles_global": 0.65,
    "segments": 0.25,
    "cens": 0.10           # nouveau poids pour chroma_cens
}

# ────────────────────────────────────────────────
# FONCTIONS TECHNIQUES
# ────────────────────────────────────────────────

def butter_lowpass(y, sr, cutoff=150):
    nyq = 0.5 * sr
    b, a = butter(4, cutoff / nyq, btype='low')
    return lfilter(b, a, y)

def apply_precision_filters(y, sr):
    # HPSS → on garde seulement la partie harmonique → très important !
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 4.5))  # margin ajustable
    nyq = 0.5 * sr
    b, a = butter(4, [60/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def send_telegram_auto(msg, bot_token, chat_id):
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        try:
            requests.post(url, data=payload, timeout=10)
        except:
            pass

def vote_profiles(chroma_vector, bass_vector, cens_vector=None):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-8)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-8)
    
    if cens_vector is not None:
        cens_norm = (cens_vector - cens_vector.min()) / (cens_vector.max() - cens_vector.min() + 1e-8)
    else:
        cens_norm = np.zeros_like(cv)

    scores = {f"{n} {m}": 0.0 for n in NOTES_LIST for m in ["major", "minor"]}
    
    for p_data in PROFILES.values():
        for mode in ["major", "minor"]:
            for i in range(12):
                corr_cqt = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                corr_cens = np.corrcoef(cens_norm, np.roll(p_data[mode], i))[0, 1] if cens_vector is not None else 0
                
                bonus = (bv[i] * 0.45) + (cv[i] * 0.35) + (cv[(i+7)%12] * 0.15) + (cens_norm[i] * 0.20)
                
                # vote combiné CQT + CENS
                combined_corr = 0.75 * corr_cqt + 0.25 * corr_cens
                scores[f"{NOTES_LIST[i]} {mode}"] += (combined_corr + bonus) / len(PROFILES)
    
    return scores

def process_audio(file_bytes, file_name, sr_target=22050):
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == '.m4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            y = samples / (2**(8 * audio.sample_width - 1))
            sr = audio.frame_rate
            if sr != sr_target:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
                sr = sr_target
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=sr_target, mono=True)

    except Exception as e:
        return {"error": f"Erreur décodage ({ext}): {str(e)}"}

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_precision_filters(y, sr)   # ← HPSS ici !

    # Analyse globale
    chroma_glob = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    chroma_cens_glob = np.mean(librosa.feature.chroma_cens(y=y_filt, sr=sr, tuning=tuning), axis=1)
    bass_glob = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y, sr), sr=sr), axis=1)
    global_scores = vote_profiles(chroma_glob, bass_glob, chroma_cens_glob)

    # Segments
    seg_size, overlap = 12, 6
    step = seg_size - overlap
    segment_votes = Counter()
    segment_timeline = []           # pour la timeline
    valid_count = 0

    for start_s in range(0, int(duration) - seg_size, step):
        y_seg = y_filt[int(start_s * sr): int((start_s + seg_size) * sr)]
        if np.max(np.abs(y_seg)) < 0.02: 
            continue
        
        c_seg = np.mean(librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning), axis=1)
        cens_seg = np.mean(librosa.feature.chroma_cens(y=y_seg, sr=sr, tuning=tuning), axis=1)
        b_seg = np.mean(librosa.feature.chroma_cqt(y=butter_lowpass(y_seg, sr), sr=sr), axis=1)
        
        seg_scores = vote_profiles(c_seg, b_seg, cens_seg)
        best_k = max(seg_scores, key=seg_scores.get)
        
        if seg_scores[best_k] >= 0.72:  # seuil un peu abaissé
            weight = 1.35 if 0.25 < (start_s / duration) < 0.75 else 1.0
            segment_votes[best_k] += seg_scores[best_k] * weight
            segment_timeline.append((start_s + seg_size/2, best_k))   # milieu du segment
            valid_count += 1

    # Détection modulation simple
    modulation_detected = None
    if len(segment_timeline) >= 5:
        mid = len(segment_timeline) // 2
        first_keys = [k for _, k in segment_timeline[:mid]]
        second_keys = [k for _, k in segment_timeline[mid:]]
        first = Counter(first_keys).most_common(1)
        second = Counter(second_keys).most_common(1)
        if first and second and first[0][0] != second[0][0]:
            modulation_detected = second[0][0]

    # Score final pondéré
    if segment_votes:
        total_v = sum(segment_votes.values())
        segment_votes_norm = {k: v / total_v for k, v in segment_votes.items()}
    else:
        segment_votes_norm = {}

    final_results = Counter()
    for key in global_scores:
        final_results[key] = (
            global_scores[key] * WEIGHTS["profiles_global"] +
            segment_votes_norm.get(key, 0) * WEIGHTS["segments"] +
            0.0  # cens déjà inclus dans vote_profiles
        )

    best_key, best_score = final_results.most_common(1)[0]
    
    # Pour top 3 + export
    top_keys = final_results.most_common(3)
    max_score = best_score if best_score > 0 else 1e-6
    
    return {
        "name": file_name,
        "key": best_key,
        "camelot": CAMELOT_MAP.get(best_key, "??"),
        "conf": best_score,
        "top3": [(k, s / max_score) for k, s in top_keys],   # normalisé 0-1 pour barres
        "valid_seg": valid_count,
        "duration": duration,
        "tuning": tuning,
        "modulation": modulation_detected,
        "chroma_global": chroma_glob,                     # pour heatmap debug
        "timeline": segment_timeline                      # pour graphique
    }

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────

st.title("🎵 Music Key Expert • v2")

global_progress = st.progress(0)
global_status = st.empty()

debug_mode = st.checkbox("Mode debug (heatmap + timeline détaillée)", value=False)

bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN")
chat_id = st.secrets.get("TELEGRAM_CHAT_ID")

uploaded_files = st.file_uploader(
    "Audios (FLAC, MP3, WAV, M4A)", 
    type=["flac", "mp3", "wav", "m4a"], 
    accept_multiple_files=True
)

results_list = []

if uploaded_files:
    n_files = len(uploaded_files)
    global_progress.progress(0)
    global_status.text(f"0 / {n_files} traités (0%)")

    for i, file in enumerate(uploaded_files, 1):
        percent = (i - 1) / n_files
        global_progress.progress(percent)
        global_status.text(f"{i-1}/{n_files} ({percent:.0%})")

        with st.spinner(f"Analyse {file.name} ({i}/{n_files})"):
            data = process_audio(file.getvalue(), file.name)
            gc.collect()

        if "error" not in data:
            results_list.append(data)

            mod_text = f"\n⚠️ *Modulation →* `{data['modulation']}`" if data.get('modulation') else ""
            report = (
                f"🎵 *{file.name}*\n"
                f"**{data['key']}** | Camelot **{data['camelot']}**\n"
                f"Conf **{data['conf']*100:.1f}%** | Seg {data['valid_seg']}{mod_text}"
            )
            send_telegram_auto(report, bot_token, chat_id)

            # Affichage résultat
            st.markdown("---")
            cols = st.columns([3, 1.2, 1.8, 1])
            with cols[0]:
                st.markdown(f"**{data['name']}**  ·  {data['duration']:.1f}s  ·  tuning {data['tuning']:+.2f}¢")
            with cols[1]:
                st.markdown(f"<h2 style='color:#f59e0b; text-align:center; margin:0;'>{data['camelot']}</h2>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"**{data['key']}**")
                if data.get('modulation'):
                    mod_cam = CAMELOT_MAP.get(data['modulation'], "??")
                    st.markdown(f"<span style='color:#ef4444;'>→ {data['modulation']} ({mod_cam})</span>", unsafe_allow_html=True)
            with cols[3]:
                st.metric("Confiance", f"{data['conf']*100:.1f}%")

            # Top 3
            st.caption("Top 3 clés probables")
            for key, rel_conf in data["top3"]:
                st.progress(rel_conf)
                st.caption(f"{key} — {rel_conf*100:.0f}% (relatif)")

            # Debug : heatmap chroma global
            if debug_mode:
                fig_hm = px.imshow(
                    data["chroma_global"].reshape(1, -1),
                    x=NOTES_LIST,
                    color_continuous_scale='Viridis',
                    title="Chroma global (CQT)"
                )
                fig_hm.update_layout(height=180, margin=dict(l=10,r=10,b=30,t=50))
                st.plotly_chart(fig_hm, use_container_width=True)

            # Timeline simple
            if debug_mode and data["timeline"]:
                times, keys = zip(*data["timeline"])
                key_to_num = {k: i for i, k in enumerate(NOTES_LIST)}
                num_keys = [key_to_num.get(k.split()[0], 0) for k in keys]

                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=times, y=num_keys,
                    mode='lines+markers',
                    line=dict(color='royalblue'),
                    marker=dict(size=8),
                    text=keys,
                    hovertemplate="%{text}<br>Time: %{x:.1f}s"
                ))
                fig_tl.update_yaxes(tickvals=list(range(12)), ticktext=NOTES_LIST, title="Note racine")
                fig_tl.update_layout(title="Évolution des clés détectées", height=300, margin=dict(l=40,r=20,b=40,t=50))
                st.plotly_chart(fig_tl, use_container_width=True)

        percent = i / n_files
        global_progress.progress(percent)
        global_status.success(f"{i}/{n_files} ({percent:.0%})")

    global_progress.progress(1.0)
    global_status.success(f"Terminé — {n_files} fichier(s) analysé(s)")

    # ─── Export CSV ───────────────────────────────
    if results_list and st.button("Exporter tous les résultats (CSV)"):
        df = pd.DataFrame([
            {
                "Fichier": r["name"],
                "Key": r["key"],
                "Camelot": r["camelot"],
                "Confiance": round(r["conf"]*100, 1),
                "Top2": r["top3"][1][0] if len(r["top3"])>1 else "",
                "Top3": r["top3"][2][0] if len(r["top3"])>2 else "",
                "Modulation": r.get("modulation", ""),
                "Segments_valides": r["valid_seg"],
                "Tuning_cents": round(r["tuning"], 2),
                "Durée_s": round(r["duration"], 1)
            }
            for r in results_list
        ])
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Télécharger résultats.csv",
            data=csv,
            file_name="music_keys_results.csv",
            mime="text/csv"
        )
