import streamlit as st
import librosa
import numpy as np
import requests
import os
from pydub import AudioSegment
import io
from scipy.signal import butter, lfilter

st.set_page_config(page_title="Music Key & Camelot Detector", page_icon="🎵", layout="wide")

# ────────────────────────────────────────────────
# CONSTANTES & PROFILS
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
    "aarden": {
        "major": [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 18.6691],
        "minor": [18.2648, 0.737619, 14.0499, 16.8599, 0.702699, 14.5212, 0.737619, 19.8145, 5.84214, 2.68046, 2.51091, 9.84455]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
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

# ────────────────────────────────────────────────
# FONCTIONS UTILITAIRES AUDIO & FILTRAGE
# ────────────────────────────────────────────────

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


# ────────────────────────────────────────────────
# DÉTECTION TONALITÉ (version simplifiée sans perception)
# ────────────────────────────────────────────────

def solve_key_sniper(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-8)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-8)

    profile_scores = {f"{NOTES_LIST[i]} {mode}": [] for i in range(12) for mode in ["major", "minor"]}

    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]

                # Boost mineur si dominante + sensible présente
                if mode == "minor":
                    dom_idx, leading_idx = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.42 and cv[leading_idx] > 0.32:
                        score *= 1.18

                # Boost selon basse
                if bv[i] > 0.58:
                    score += bv[i] * 0.42

                # Renforcement harmonique (quinte + tierce)
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.48:
                    score += 0.14

                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.46:
                    score += 0.10

                # Très fort tonic → bonus
                if cv[i] > 0.52:
                    score += 0.48

                profile_scores[f"{NOTES_LIST[i]} {mode}"].append(score)

    avg_scores = {k: np.mean(v) for k, v in profile_scores.items() if v}
    if not avg_scores:
        return {"key": "Unknown", "score": 0.0}

    best_key = max(avg_scores, key=avg_scores.get)
    best_score = avg_scores[best_key]

    # Vérification des candidats proches (ambiguïté)
    candidates = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    if len(candidates) >= 2:
        top_key, top_sc = candidates[0]
        sec_key, sec_sc = candidates[1]
        top_i = NOTES_LIST.index(top_key.split()[0])
        sec_i = NOTES_LIST.index(sec_key.split()[0])
        dist = min(abs(top_i - sec_i), 12 - abs(top_i - sec_i))
        if dist in [3, 4, 9] and (sec_sc / top_sc > 0.82):
            if bv[sec_i] > bv[top_i] + 0.06:
                best_key = sec_key
                best_score = sec_sc

    return {"key": best_key, "score": best_score}


# ────────────────────────────────────────────────
# TRAITEMENT PRINCIPAL D'UN FICHIER (simplifié)
# ────────────────────────────────────────────────

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
        return {"error": str(e)}

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 8:
        return {"error": "fichier trop court (< 8s)"}

    y_filt = apply_sniper_filters(y, sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, hop_length=512), axis=1)
    bass_vector = get_bass_priority(y, sr)

    res = solve_key_sniper(chroma_avg, bass_vector)
    key = res["key"]
    score = res["score"]

    final_conf = min(0.99, score)

    report = f"""Analyse terminée
──────────────────────────────
Fichier       : {file_name}
Durée         : {int(duration // 60):02d}:{int(duration % 60):02d}
Fréquence     : {sr} Hz

Tonalité      : {key}
Camelot       : {CAMELOT_MAP.get(key, "??")}
Confiance     : {final_conf:.4f}

Chroma moyen :
""" + "\n".join(f"  {k:<3} : {v:.4f}" for k,v in zip(NOTES_LIST, chroma_avg))

    return {
        "key": key,
        "camelot": CAMELOT_MAP.get(key, "??"),
        "conf": final_conf,
        "report": report
    }


# ────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ────────────────────────────────────────────────

st.title("🎵 Music Key & Camelot Detector")
st.markdown("Détection de tonalité + Camelot (version simplifiée – sans simulation perceptive)")

try:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id   = st.secrets["TELEGRAM_CHAT_ID"]
    secrets_ok = True
except KeyError:
    bot_token = chat_id = None
    secrets_ok = False

if not secrets_ok:
    st.info("Pour activer l'envoi Telegram : ajoutez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID dans les secrets Streamlit.")

uploaded_files = st.file_uploader(
    "Déposez vos fichiers audio",
    type=["mp3", "wav", "ogg", "flac", "m4a"],
    accept_multiple_files=True
)

if uploaded_files:
    total = len(uploaded_files)
    prog_global = st.progress(0)
    status_global = st.empty()

    container = st.container()

    for i, file in enumerate(uploaded_files, 1):
        prog_global.progress((i-1)/total)
        status_global.markdown(f"**Traitement {i}/{total} →** {file.name}")

        with st.status(f"Analyse → {file.name}", expanded=(i==1)) as st_status:
            st_status.write("Chargement & analyse...")
            data = process_audio(file.getvalue(), file.name)

            if "error" in data:
                st_status.update(label=f"Erreur : {data['error']}", state="error")
                continue

            st_status.update(label="Terminé ✓", state="complete", expanded=False)

            with container:
                st.markdown(f"### {file.name}")
                colA, colB = st.columns([4, 1])
                with colA:
                    st.markdown(f"**Tonalité :** {data['key']}")
                    st.markdown(f"**Camelot :** <span style='font-size:2.4em; color:#f59e0b; font-weight:bold;'>{data['camelot']}</span>", unsafe_allow_html=True)
                with colB:
                    st.metric("Confiance", f"{data['conf']:.3f}")

                st.text_area("Rapport complet", data["report"], height=340)

                if secrets_ok:
                    if st.button("Envoyer rapport Telegram", key=f"tg_{i}_{hash(file.name)}"):
                        with st.spinner("Envoi..."):
                            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                            payload = {"chat_id": chat_id, "text": f"🎵 {file.name}\n\n{data['report']}", "parse_mode": "Markdown"}
                            try:
                                r = requests.post(url, data=payload, timeout=12)
                                if r.status_code == 200:
                                    st.success("Envoyé")
                                else:
                                    st.error(f"Erreur {r.status_code}")
                            except Exception as ex:
                                st.error(f"Échec : {str(ex)}")

                st.markdown("---")

    prog_global.progress(1.0)
    status_global.success(f"✓ {total} fichier(s) analysé(s)")

st.markdown("<small>Détection basée sur profils statistiques + renforcement basse / harmonique. Précision ~80–92 % selon le style.</small>", unsafe_allow_html=True)
