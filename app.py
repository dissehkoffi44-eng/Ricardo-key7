import streamlit as st
import librosa
import numpy as np
import requests
import tempfile
import os
from pydub import AudioSegment
import io

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
st.set_page_config(page_title="Music Key & Camelot Detector", page_icon="🎵", layout="wide")

keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

major_profile /= np.sum(major_profile)
minor_profile /= np.sum(minor_profile)

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

def estimate_key(chroma):
    correlations = []
    for i in range(12):
        rm = np.roll(major_profile, i)
        rn = np.roll(minor_profile, i)
        corr_maj = np.corrcoef(chroma, rm)[0, 1]
        corr_min = np.corrcoef(chroma, rn)[0, 1]
        correlations.append((corr_maj, 'major', i))
        correlations.append((corr_min, 'minor', i))

    best = max(correlations, key=lambda x: x[0])
    key_idx = best[2]
    scale = best[1]
    confidence = best[0]

    key_name = keys[key_idx]
    music_key = f"{key_name} {scale}"
    camelot = CAMELOT_MAP.get(music_key, "??")
    return music_key, camelot, confidence


# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────
st.title("🎵 Music Key & Camelot Detector")
st.markdown("Détection de tonalité + notation **Camelot** — support multi-fichiers")

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
            status.write("Chargement audio...")
            file_bytes = file.getvalue()

            # ─── Sauvegarde temporaire ───────────────────────────────
            ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Chargement selon format
                if ext == '.m4a':
                    audio = AudioSegment.from_file(tmp_path, format="m4a")
                    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                    if audio.channels == 2:
                        samples = samples.reshape(-1, 2).mean(axis=1)
                    y = samples / 32768.0
                    sr = audio.frame_rate
                else:
                    y, sr = librosa.load(tmp_path, sr=None, mono=True)

                status.write("Extraction chroma CQT...")
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
                chroma_mean = np.mean(chroma, axis=1)

                if np.sum(chroma_mean) > 0:
                    chroma_mean /= np.sum(chroma_mean)

                status.write("Détection tonalité...")
                key_str, camelot, conf = estimate_key(chroma_mean)

                duration = librosa.get_duration(y=y, sr=sr)
                dur_min = int(duration // 60)
                dur_sec = int(duration % 60)

                report = f"""Analyse terminée
──────────────────────────────
Fichier       : {file.name}
Durée         : {dur_min:02d}:{dur_sec:02d}
Fréquence     : {sr} Hz

Tonalité      : {key_str}
Camelot       : {camelot}
Confiance     : {conf:.4f}

Chroma moyen :
""" + "\n".join(f"  {k:<3} : {v:.4f}" for k,v in zip(keys, chroma_mean))

                status.update(label="Terminé ✓", state="complete", expanded=False)

                # Affichage résultat
                with results_container:
                    st.markdown(f"### {file.name}")
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown(f"**Tonalité :** {key_str}")
                        st.markdown(f"**Camelot :** <span style='font-size:2.2em; color:#f59e0b; font-weight:bold;'>{camelot}</span>", unsafe_allow_html=True)
                    with col2:
                        st.metric("Confiance", f"{conf:.3f}")

                    st.text_area("Rapport détaillé", report, height=320)

                    # Envoi Telegram individuel
                    if secrets_ok:
                        if st.button("Envoyer sur Telegram", key=f"send_{idx}_{hash(file.name)}"):
                            with st.spinner("Envoi..."):
                                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                                payload = {
                                    "chat_id": chat_id,
                                    "text": f"🎵 {file.name}\n\n{report}",
                                    "parse_mode": "Markdown"
                                }
                                try:
                                    r = requests.post(url, data=payload, timeout=12)
                                    if r.status_code == 200:
                                        st.success("Envoyé !")
                                    else:
                                        st.error(f"Erreur {r.status_code}")
                                except Exception as e:
                                    st.error(f"Échec envoi : {str(e)}")

                    st.markdown("---")

            except Exception as e:
                status.update(label=f"Erreur : {str(e)}", state="error")
                st.error(f"Problème avec {file.name} : {str(e)}")

            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

    progress_global.progress(1.0)
    status_global.success(f"✓ {total} fichier(s) traité(s)")

st.markdown("<br><small>Note : précision ~70-85% selon les morceaux. Pour >94% → modèle deep learning recommandé.</small>", unsafe_allow_html=True)
