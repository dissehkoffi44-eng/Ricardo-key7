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

# --- CONFIGURATION & CSS ---
st.set_page_config(page_title="KEY ULTIMATE HARMONIC 2", page_icon="🎧", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .final-decision-box { 
        padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;
        color: white; box-shadow: 0 12px 24px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.1);
    }
    .stFileUploader { border: 2px dashed #6366F1; padding: 20px; border-radius: 15px; background: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION API & CONSTANTES ---
TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- FONCTIONS UTILITAIRES ---
def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        else: return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null; let oscillators = []; let gainNode = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            gainNode = audioCtx.createGain();
            gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.1, audioCtx.currentTime + 0.1);
            gainNode.connect(audioCtx.destination);
            const isMinor = '{mode}' === 'minor' || '{mode}' === 'dorian';
            const intervals = isMinor ? [0, 3, 7] : [0, 4, 7];
            intervals.forEach(interval => {{
                let osc = audioCtx.createOscillator();
                let freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                osc.connect(gainNode);
                osc.start();
                oscillators.push(osc);
            }});
        }} else {{
            if(gainNode) {{
                gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.1);
                setTimeout(() => {{ oscillators.forEach(o => o.stop()); oscillators = []; }}, 100);
            }}
            this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=40)

def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        response = requests.post(url, files=files, data=data, timeout=30).json()
        return response.get("ok", False)
    except: return False

# --- MOTEUR ANALYSE OPTIMISÉ ---
def analyze_segment(y, sr, tuning=0.0):
    nyq = 0.5 * sr
    low, high = 60 / nyq, 1000 / nyq
    b, a = butter(4, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    rms = np.sqrt(np.mean(y_filtered**2))
    if rms < 0.01: return None, 0.0
    chroma = librosa.feature.chroma_cens(y=y_filtered, sr=sr, hop_length=1024, n_chroma=12, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score

@st.cache_data(show_spinner="Analyse Harmonique Profonde...", max_entries=20)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.hpss(y)[0]
    duration = librosa.get_duration(y=y, sr=sr)
    timeline_data, votes = [], []
    
    progress_bar = st.progress(0)
    step = 10
    for i, start_t in enumerate(range(0, int(duration) - step, step)):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})
        progress_bar.progress(min(start_t / duration, 1.0))
    progress_bar.empty()

    if not votes: return {"file_name": file_name, "recommended": {"note": "N/A", "conf": 0, "label": "ERREUR", "bg": "red"}}

    df_tl = pd.DataFrame(timeline_data)
    counts = Counter(votes)
    top_votes = counts.most_common(5)
    
    # --- LOGIQUE DE CADENCE (V -> I) POUR FIABILITÉ ---
    n1 = top_votes[0][0]
    final_note = n1
    cadence_bonus = 0
    cadence_detected = False

    if len(top_votes) > 1:
        for i in range(min(len(top_votes), 4)):
            for j in range(i + 1, min(len(top_votes), 5)):
                candidate_a = top_votes[i][0]
                candidate_b = top_votes[j][0]
                
                note_a, mode_a = candidate_a.split(" ")
                note_b, mode_b = candidate_b.split(" ")
                
                idx_a, idx_b = NOTES_LIST.index(note_a), NOTES_LIST.index(note_b)
                
                # Vérification de la quinte (Dominante -> Tonique)
                # Si B est la tonique (I) et A est la dominante (V) -> intervalle de 7 demi-tons
                if (idx_a - idx_b) % 12 == 7:
                    # Cadence parfaite détectée. La Tonique (B) est la vraie tonalité.
                    final_note = candidate_b
                    cadence_bonus = 20
                    cadence_detected = True
                    break
                # Inversement
                elif (idx_b - idx_a) % 12 == 7:
                    final_note = candidate_a
                    cadence_bonus = 20
                    cadence_detected = True
                    break
            if cadence_detected: break

    # --- CALCUL DES SCORES FINAUX ---
    purity = (counts[final_note] / len(votes)) * 100
    avg_conf_final = df_tl[df_tl['Note'] == final_note]['Confiance'].mean()
    
    # Bonus Camelot existant
    musical_bonus = cadence_bonus
    n2 = top_votes[1][0] if len(top_votes) > 1 else final_note
    c1_code, c2_code = get_camelot_pro(final_note), get_camelot_pro(n2)
    if c1_code != "??" and c2_code != "??" and final_note != n2:
        val1, mod1 = int(c1_code[:-1]), c1_code[-1]
        val2, mod2 = int(c2_code[:-1]), c2_code[-1]
        if mod1 == mod2 and (abs(val1 - val2) == 1 or abs(val1 - val2) == 11): musical_bonus += 10
        elif val1 == val2 and mod1 != mod2: musical_bonus += 10

    musical_score = min(int((purity * 0.4) + (avg_conf_final * 0.4) + musical_bonus), 100)
    
    if musical_score > 85: label, bg = "NOTE INDISCUTABLE", "linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"
    elif musical_score > 65: label, bg = "NOTE TRÈS FIABLE", "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)"
    else: label, bg = "ANALYSE COMPLEXE", "linear-gradient(135deg, #f83600 0%, #f9d423 100%)"

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Note solide (stabilité pure)
    df_tl['is_stable'] = df_tl['Note'] == df_tl['Note'].shift(1)
    stability_scores = {n: (df_tl[df_tl['Note']==n].shape[0]*0.5 + df_tl[df_tl['Note']==n]['Confiance'].mean()*0.5) for n in df_tl['Note'].unique()}
    note_solide = max(stability_scores, key=stability_scores.get) if stability_scores else "N/A"

    return {
        "file_name": file_name,
        "recommended": {"note": final_note, "conf": musical_score, "label": label, "bg": bg},
        "note_solide": note_solide, "solid_conf": int(df_tl[df_tl['Note'] == note_solide]['Confiance'].mean()) if note_solide != "N/A" else 0,
        "vote": final_note, "vote_conf": int(purity),
        "n1": final_note, "c1": int(purity), "n2": n2, "c2": int((counts[n2]/len(votes))*100),
        "tempo": int(float(tempo)), "energy": int(np.clip(musical_score/10, 1, 10)),
        "timeline": timeline_data
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 KEY ULTIMATE HARMONIC 2</h1>", unsafe_allow_html=True)

if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []

with st.sidebar:
    st.header("⚙️ MAINTENANCE")
    if st.button("🧹 VIDER TOUT"):
        st.session_state.processed_files, st.session_state.order_list = {}, []
        st.cache_data.clear()
        gc.collect()
        st.rerun()

files = st.file_uploader("📂 DEPOSEZ VOS TRACKS", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    if files:
        for f in files:
            fid = f"{f.name}_{f.size}"
            if fid not in st.session_state.processed_files:
                f_bytes = f.read()
                res = get_full_analysis(f_bytes, f.name)
                if res["recommended"]["note"] != "N/A":
                    tg_cap = f"🎵 *FICHIER* : {res['file_name']}\n🔥 *KEY* : {res['recommended']['note']} ({get_camelot_pro(res['recommended']['note'])})\n🎯 Confiance : {res['recommended']['conf']}%"
                    upload_to_telegram(io.BytesIO(f_bytes), f.name, tg_cap)
                    st.session_state.processed_files[fid] = res
                    st.session_state.order_list.insert(0, fid)

        for fid in st.session_state.order_list:
            res = st.session_state.processed_files[fid]
            with st.expander(f"🎵 {res['file_name']}", expanded=True):
                st.markdown(f'<div class="final-decision-box" style="background: {res["recommended"]["bg"]};"><div style="text-transform:uppercase; letter-spacing:2px;">{res["recommended"]["label"]}</div><div style="font-size:4.5em; font-weight:900;">{res["recommended"]["note"]}</div><div style="font-size:1.8em;">{get_camelot_pro(res["recommended"]["note"])} • {res["recommended"]["conf"]}%</div></div>', unsafe_allow_html=True)
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.markdown(f'<div class="metric-container"><div class="label-custom">TONALITÉ</div><div class="value-custom">{res["vote"]}</div><div>{res["vote_conf"]}% presence</div></div>', unsafe_allow_html=True); get_sine_witness(res["vote"], f"v_{fid}")
                with c2: st.markdown(f'<div class="metric-container" style="border:2px solid #FFD700;"><div class="label-custom">💎 SOLIDE</div><div class="value-custom">{res["note_solide"]}</div><div>{res["solid_conf"]}%</div></div>', unsafe_allow_html=True); get_sine_witness(res["note_solide"], f"s_{fid}")
                with c3: st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="metric-container"><div class="label-custom">STABILITÉ 1&2</div><div>🥇 {res["n1"]}</div><div>🥈 {res["n2"]}</div></div>', unsafe_allow_html=True)
                with c5: st.markdown(f'<div class="metric-container"><div class="label-custom">ÉNERGIE</div><div class="value-custom">{res["energy"]}/10</div></div>', unsafe_allow_html=True)
                st.plotly_chart(px.scatter(pd.DataFrame(res['timeline']), x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.processed_files:
        df_hist = pd.DataFrame([{"Fichier": r["file_name"], "Note": r["recommended"]["note"], "Camelot": get_camelot_pro(r["recommended"]["note"]), "Confiance": f"{r['recommended']['conf']}%"} for r in st.session_state.processed_files.values()])
        st.dataframe(df_hist, use_container_width=True)
