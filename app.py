import streamlit as st
import librosa
import numpy as np
import requests
import tempfile
import os
from pydub import AudioSegment

# Définir les noms des tonalités
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils Krumhansl-Schmuckler (major et minor)
major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Normalisation des profils
major_profile = major_profile / np.sum(major_profile)
minor_profile = minor_profile / np.sum(minor_profile)

# Mapping vers la notation Camelot
CAMELOT_MAP = {
    'C major': '8B',   'C# major': '3B',  'D major': '10B',  'D# major': '5B',
    'E major': '12B',  'F major': '7B',   'F# major': '2B',  'G major': '9B',
    'G# major': '4B',  'A major': '11B',  'A# major': '6B',  'B major': '1B',
    'C minor': '5A',   'C# minor': '12A', 'D minor': '7A',   'D# minor': '2A',
    'E minor': '9A',   'F minor': '4A',   'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A',  'A minor': '8A',   'A# minor': '3A',  'B minor': '10A'
}

def estimate_key(chroma):
    correlations = []
    for i in range(12):
        rotated_major = np.roll(major_profile, i)
        rotated_minor = np.roll(minor_profile, i)
        corr_major = np.corrcoef(chroma, rotated_major)[0, 1]
        corr_minor = np.corrcoef(chroma, rotated_minor)[0, 1]
        correlations.append((corr_major, 'major', i))
        correlations.append((corr_minor, 'minor', i))
    
    best = max(correlations, key=lambda x: x[0])
    key_index = best[2]
    scale = best[1]
    confidence = best[0]
    
    key_name = keys[key_index]
    music_key = f"{key_name} {scale}"
    camelot = CAMELOT_MAP.get(music_key, "Inconnu")
    
    return music_key, camelot, confidence


st.title("Music Key & Camelot Detector")

# ────────────────────────────────────────────────
# Gestion des secrets Telegram (recommandé pour le déploiement)
# ────────────────────────────────────────────────
try:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    secrets_configured = True
except KeyError:
    bot_token = None
    chat_id = None
    secrets_configured = False

if not secrets_configured:
    st.warning("Les identifiants Telegram ne sont pas configurés dans les secrets Streamlit.")
    st.info("Pour envoyer les rapports sur Telegram, ajoutez dans .streamlit/secrets.toml ou dans les settings Streamlit Cloud :\n"
            "TELEGRAM_BOT_TOKEN = \"votre_token\"\n"
            "TELEGRAM_CHAT_ID = \"votre_chat_id\"")


uploaded_file = st.file_uploader("Déposez un fichier audio", type=["mp3", "wav", "ogg", "flac", "m4a"])

if uploaded_file is not None:
    # Sauvegarde temporaire sécurisée
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name

    try:
        with st.spinner("Analyse en cours (chroma + détection de tonalité)..."):
            # Chargement audio avec gestion m4a via pydub
            ext = os.path.splitext(tmp_filename)[1].lower()
            if ext == '.m4a':
                audio = AudioSegment.from_file(tmp_filename, format="m4a")
                samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                y = samples / (2**15)  # Normalisation pour 16-bit
                sr = audio.frame_rate
            else:
                y, sr = librosa.load(tmp_filename, sr=None, mono=True)

            # Extraction des features chroma (CQT = bonne qualité)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            chroma_mean = np.mean(chroma, axis=1)

            # Normalisation L1
            if np.sum(chroma_mean) > 0:
                chroma_mean = chroma_mean / np.sum(chroma_mean)
            else:
                chroma_mean = np.zeros_like(chroma_mean)

            # Détection
            music_key, camelot_key, confidence = estimate_key(chroma_mean)

        # ────────────────────────────────────────────────
        # Rapport détaillé
        # ────────────────────────────────────────────────
        duration_sec = librosa.get_duration(y=y, sr=sr)
        report = f"""🎵 Rapport d'analyse tonalité
──────────────────────────────
Fichier : {uploaded_file.name}
Durée   : {duration_sec//60:.0f} min {duration_sec%60:.0f} s
Fréq.   : {sr} Hz

Tonalité détectée : {music_key}
Camelot           : {camelot_key}
Confiance         : {confidence:.4f}

Note : cet algorithme (Krumhansl-Schmuckler + chroma CQT) donne généralement 70–82 % de précision globale selon les datasets. La précision ≥94 % demande souvent un modèle entraîné (deep learning).

Répartition des classes de hauteur (chroma moyenné) :
"""
        for i, val in enumerate(chroma_mean):
            report += f"  {keys[i]:<3} : {val:>6.4f}\n"

        st.subheader("Résultat")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Tonalité :** {music_key}")
            st.markdown(f"**Camelot :** <span style='font-size:1.6em; color:#e67e22; font-weight:bold;'>{camelot_key}</span>", unsafe_allow_html=True)
        with col2:
            st.metric("Confiance", f"{confidence:.3f}")

        st.text_area("Rapport complet", report, height=380)

        # ────────────────────────────────────────────────
        # Envoi Telegram
        # ────────────────────────────────────────────────
        if secrets_configured and st.button("📤 Envoyer le rapport sur Telegram"):
            with st.spinner("Envoi en cours..."):
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": report,
                    "parse_mode": "Markdown"
                }
                try:
                    r = requests.post(url, data=payload, timeout=10)
                    if r.status_code == 200:
                        st.success("Message envoyé avec succès !")
                    else:
                        st.error(f"Échec envoi → {r.status_code} - {r.text[:200]}")
                except Exception as e:
                    st.error(f"Erreur réseau : {str(e)}")

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {str(e)}")
    
    finally:
        # Nettoyage impératif du fichier temporaire
        if os.path.exists(tmp_filename):
            try:
                os.unlink(tmp_filename)
            except:
                pass
