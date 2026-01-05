import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests
import gc
from scipy.signal import butter, lfilter

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 Key7 Ultimate PRO", page_icon="🎧", layout="wide")

# --- CONSTANTES HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

# Profils optimisés (Krumhansl-Schmuckler lissés)
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: 0.3s; }
    .metric-container:hover { border-color: #6366F1; transform: translateY(-3px); }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; border: 1px solid rgba(255,255,255,0.1); }
    .solid-note-box { background: rgba(99, 102, 241, 0.1); border: 1px dashed #6366F1; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS LOGIQUES ---

def apply_bandpass_filter(y, sr):
    nyq = 0.5 * sr
    low, high = 50 / nyq, 1500 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode == 'minor' else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def solve_key(chroma_avg):
    best_score, best_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, best_key = score, f"{NOTES_LIST[i]} {mode}"
    return best_key, best_score

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note, mode = parts[0], parts[1].lower()
    unique_id = f"play_{note}_{mode}_{key_suffix}".replace("#", "s").replace(" ", "")
    return components.html(f"""
    <button id="{unique_id}" style="background:#6366F1;color:white;border:none;border-radius:20px;padding:10px 20px;cursor:pointer;font-weight:bold;width:100%;">▶ TEST {note} {mode[:3].upper()}</button>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    let ctx = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if(!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach((inter, i) => {{
            const osc = ctx.createOscillator(); const g = ctx.createGain();
            osc.type = 'triangle'; osc.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, inter/12), now + (i*0.02));
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.3, now+0.05); g.gain.exponentialRampToValueAtTime(0.01, now+2);
            osc.connect(g); g.connect(ctx.destination); osc.start(now); osc.stop(now+2);
        }});
    }};
    </script>""", height=50)

# --- COEUR DE L'ANALYSE ---

@st.cache_data(show_spinner=False)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    y_filt = apply_bandpass_filter(y_harm, sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Analyse Stabilité (Segments pondérés)
    step, timeline = 8, []
    votes = Counter()
    for start in range(0, int(duration) - step, step):
        y_seg = y_filt[int(start*sr):int((start+step)*sr)]
        rms = np.mean(librosa.feature.rms(y=y_seg))
        if rms < 0.01: continue
        
        chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning)
        key, score = solve_key(np.mean(chroma, axis=1))
        weight = int(score * 100) + int(rms * 500)
        votes[key] += weight
        timeline.append({"Temps": start, "Note": key, "Conf": round(score*100, 1)})

    if not timeline: return None
    df_tl = pd.DataFrame(timeline)
    note_solide = votes.most_common(1)[0][0]

    # 2. Analyse de Résolution (Fin)
    y_end = y_harm[int(max(0, duration-6)*sr):]
    key_fin, score_fin = solve_key(np.mean(librosa.feature.chroma_cens(y=y_end, sr=sr, tuning=tuning), axis=1))

    # 3. Arbitrage
    final_decision = note_solide
    is_res = False
    if score_fin > 0.80 and key_fin in [v[0] for v in votes.most_common(3)]:
        final_decision = key_fin
        is_res = True

    conf_finale = int(df_tl[df_tl['Note'] == final_decision]['Conf'].mean())
    if is_res: conf_finale = min(conf_finale + 5, 100)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bg = "linear-gradient(135deg, #1D976C, #93F9B9)" if conf_finale > 82 else "linear-gradient(135deg, #2193B0, #6DD5ED)"
    
    fig = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark")
    fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    res = {
        "file_name": file_name, "tempo": int(float(tempo)),
        "rec": {"note": final_decision, "conf": conf_finale, "bg": bg},
        "note_solide": note_solide, "is_res": is_res, "timeline": timeline,
        "plot_bytes": fig.to_image(format="png", width=800, height=400)
    }
    del y, y_harm; gc.collect()
    return res

# --- INTERFACE ---
st.title("🎧 RCDJ228 Key7 Ultimate PRO")

files = st.file_uploader("📂 AUDIO FILES", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    for f in files:
        fid = f"{f.name}_{f.size}"
        data = get_full_analysis(f.read(), f.name)
        
        if data:
            with st.expander(f"📊 {data['file_name']}", expanded=True):
                st.markdown(f"""
                    <div class="final-decision-box" style="background:{data['rec']['bg']};">
                        <h1 style="font-size:4.5em; margin:0; font-weight:900;">{data['rec']['note']}</h1>
                        <h2 style="margin:0;">CAMELOT: {get_camelot_pro(data['rec']['note'])} • CERTITUDE: {data['rec']['conf']}%</h2>
                    </div>
                    <div class="solid-note-box">
                        💎 STABILITÉ : <b>{data['note_solide']}</b> | RÉSOLUTION FINALE : <b>{'OUI' if data['is_res'] else 'NON'}</b>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f'<div class="metric-container">BPM<br><span class="value-custom">{data["tempo"]}</span></div>', unsafe_allow_html=True)
                with c2: get_sine_witness(data['rec']['note'], fid)
                with c3:
                    if st.button(f"🚀 TELEGRAM REPORT", key=f"tg_{fid}"):
                        total_seg = len(data['timeline'])
                        main_count = sum(1 for s in data['timeline'] if s['Note'] == data['rec']['note'])
                        stability = int((main_count / total_seg) * 100) if total_seg > 0 else 0
                        
                        cap = (
                            f"✨ *RCDJ228 KEY7 ULTIMATE PRO*\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📂 *FICHIER :* `{data['file_name']}`\n"
                            f"⏱ *TEMPO :* `{data['tempo']} BPM`\n\n"
                            f"🎹 *RÉSULTAT HARMONIQUE*\n"
                            f"├─ Clé : `{data['rec']['note']}`\n"
                            f"├─ Camelot : `{get_camelot_pro(data['rec']['note'])}` \n"
                            f"└─ Certitude : `{data['rec']['conf']}%` {'✅' if data['rec']['conf'] > 85 else '⚠️'}\n\n"
                            f"📊 *ANALYSE EXPERTE*\n"
                            f"├─ Taux Stabilité : `{stability}%`\n"
                            f"├─ Résolution Finale : `{'OUI' if data['is_res'] else 'NON'}`\n"
                            f"└─ Statut : `{'PRO MIX READY' if stability > 70 else 'COMPLEX HARMONY'}`\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"🎧 *Generated by RCDJ228 AI*"
                        )
                        
                        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                                             files={'photo': data['plot_bytes']}, 
                                             data={'chat_id': CHAT_ID, 'caption': cap, 'parse_mode': 'Markdown'})
                        if resp.status_code == 200: st.toast("🚀 Rapport expert envoyé !")
                        else: st.error("Erreur d'envoi Telegram.")

                st.plotly_chart(px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", template="plotly_dark", title="Stabilité Harmonique (Chronologie)"), use_container_width=True)
        else:
            st.error(f"Erreur d'analyse sur {f.name}.")

if st.sidebar.button("🧹 CLEAR CACHE"):
    st.cache_data.clear()
    st.rerun()
