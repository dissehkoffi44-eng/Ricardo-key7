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

# --- CONFIGURATION SÉCURISÉE & SECRETS ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 key7 PRO", page_icon="🎧", layout="wide")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #888; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 45px; border-radius: 20px; text-align: center; margin: 10px 0; 
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
    }
    .solid-note-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed #6366F1;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS LOGIQUES ---

def apply_bandpass_filter(y, sr, lowcut=100, highcut=3000):
    nyq = 0.5 * sr
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def validate_coherence(chroma_avg, proposed_key):
    try:
        parts = proposed_key.split(" ")
        note_name, mode = parts[0], parts[1].lower()
        idx = NOTES_LIST.index(note_name)
        theoretical_profile = np.roll(PROFILES[mode], idx)
        return np.corrcoef(chroma_avg, theoretical_profile)[0, 1]
    except: return 0

def detect_final_chord(y_harm, sr):
    try:
        duration = librosa.get_duration(y=y_harm, sr=sr)
        end_zone_start = max(0, duration - 10)
        y_end_zone = y_harm[int(end_zone_start * sr):]
        
        rms_end = librosa.feature.rms(y=y_end_zone)[0]
        if len(rms_end) == 0: return None, 0
        threshold = np.max(rms_end) * 0.15 
        
        valid_indices = np.where(rms_end > threshold)[0]
        if len(valid_indices) == 0: return None, 0
        
        last_valid_idx = valid_indices[-1]
        hop_length = 512
        sample_start = int(max(0, (last_valid_idx * hop_length) - (sr * 2.0)))
        sample_end = int(last_valid_idx * hop_length)
        y_final_point = y_end_zone[sample_start:sample_end]

        chroma_end = librosa.feature.chroma_cens(y=y_final_point, sr=sr)
        avg_end = np.mean(chroma_end, axis=1)
        
        best_score, best_key = -1, ""
        for mode, profile in PROFILES.items():
            for i in range(12):
                score = np.corrcoef(avg_end, np.roll(profile, i))[0, 1]
                if score > best_score:
                    best_score, best_key = score, f"{NOTES_LIST[i]} {mode}"
        return best_key, best_score
    except: return None, 0

def detect_sequential_cadence(df_tl, chroma_avg):
    if len(df_tl) < 3: return False, None
    notes = df_tl['Note'].tolist()
    for i in range(len(notes)-1):
        n_from, n_to = notes[i], notes[i+1]
        try:
            parts_f, parts_t = n_from.split(), n_to.split()
            root_f, mode_f = parts_f[0], parts_f[1].lower()
            root_t, mode_t = parts_t[0], parts_t[1].lower()
            idx_f, idx_t = NOTES_LIST.index(root_f), NOTES_LIST.index(root_t)
            if mode_f == 'major' and mode_t == 'minor':
                if idx_f == (idx_t + 7) % 12:
                    sensible_idx = (idx_t - 1) % 12
                    if chroma_avg[sensible_idx] > 0.18: return True, n_to
        except: continue
    return False, None

def detect_relative_key(n1, n2):
    try:
        c1, c2 = get_camelot_pro(n1), get_camelot_pro(n2)
        if c1 == "??" or c2 == "??": return False, n1
        v1, m1 = int(c1[:-1]), c1[-1]
        v2, m2 = int(c2[:-1]), c2[-1]
        if v1 == v2 and m1 != m2: return True, (n1 if m1 == 'A' else n2)
        return False, n1
    except: return False, n1

def upload_to_telegram(file_buffer, filename, caption, plot_bytes=None):
    try:
        file_buffer.seek(0)
        url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        response = requests.post(url_doc, files=files, data=data, timeout=30).json()
        if plot_bytes and response.get("ok"):
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(url_photo, files={'photo': ('graph.png', plot_bytes)}, data={'chat_id': CHAT_ID}, timeout=30)
        return response.get("ok", False)
    except: return False

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note, mode = parts[0], parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #888;">{note} {mode[:3].upper()} PIANO</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null;
    function playNote(freq, startTime) {{
        const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain();
        osc.type = 'triangle'; osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(0.4, startTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, startTime + 2.5);
        osc.connect(gain); gain.connect(audioCtx.destination);
        osc.start(startTime); osc.stop(startTime + 2.6);
    }}
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        this.innerText = '◼'; this.style.background = '#E74C3C';
        const isMinor = '{mode}' === 'minor';
        const intervals = isMinor ? [0, 3, 7, 12] : [0, 4, 7, 12];
        const now = audioCtx.currentTime;
        intervals.forEach((interval, index) => {{
            const freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
            playNote(freq, now + (index * 0.02));
        }});
        setTimeout(() => {{ this.innerText = '▶'; this.style.background = '#6366F1'; }}, 2500);
    }};
    </script>""", height=40)

# --- COEUR DE L'ANALYSE ---

@st.cache_data(show_spinner=False, max_entries=5)
def get_full_analysis(file_bytes, file_name):
    y_raw, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
    
    y_harm = librosa.effects.harmonic(y_raw, margin=3.0)
    y_filtered = apply_bandpass_filter(y_harm, sr)
    tuning = librosa.estimate_tuning(y=y_filtered, sr=sr)
    duration = librosa.get_duration(y=y_raw, sr=sr)

    intro_dur = min(15, duration * 0.15)
    y_intro = y_harm[:int(intro_dur * sr)]
    intro_chroma = librosa.feature.chroma_cens(y=y_intro, sr=sr)
    harmonic_intensity = np.mean(intro_chroma)
    intro_type = "🥁 Percussion" if harmonic_intensity < 0.15 else "🎹 Mélodique"
    energy_avg = np.mean(librosa.feature.rms(y=y_raw))

    def solve_key(chroma_feat):
        avg = np.mean(chroma_feat, axis=1)
        b_score, b_key = -1, ""
        for mode, profile in PROFILES.items():
            for i in range(12):
                score = np.corrcoef(avg, np.roll(profile, i))[0, 1]
                if score > b_score: b_score, b_key = score, f"{NOTES_LIST[i]} {mode}"
        return b_key, b_score, avg

    chroma_cens_gl = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    key_cens, _, avg_cens = solve_key(chroma_cens_gl)
    chroma_cqt_gl = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    key_cqt, _, avg_cqt = solve_key(chroma_cqt_gl)
    
    n1_global = Counter([key_cens, key_cqt]).most_common(1)[0][0]
    chroma_global_avg = (avg_cens + avg_cqt) / 2

    step, timeline_data = 6, []
    weighted_scores = Counter()
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        if len(y_seg) < 1024: continue
        res_key, b_score, _ = solve_key(librosa.feature.chroma_cens(y=y_seg, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y_seg))
        weight = int(rms * 100) + 5
        weighted_scores[res_key] += weight
        timeline_data.append({"Temps": start_t, "Note": res_key, "Confiance": round(float(b_score)*100, 1), "RMS": rms})

    df_tl = pd.DataFrame(timeline_data)
    df_fiable = df_tl[(df_tl['Confiance'] > 65) & (df_tl['RMS'] > 0.01)]
    
    # Extraire le TOP 3 des notes dominantes pour le verrou de sécurité
    top_3_notes = [item[0] for item in weighted_scores.most_common(3)]
    
    note_solide = df_fiable['Note'].mode()[0] if not df_fiable.empty else df_tl['Note'].mode()[0]
    occ_graph = (len(df_fiable[df_fiable['Note'] == note_solide]) / len(df_fiable)) * 100 if not df_fiable.empty else 0

    # --- ARBITRAGE EXPERT HIÉRARCHIQUE ---
    is_cad, cad_key = detect_sequential_cadence(df_tl, chroma_global_avg)
    final_chord_key, final_chord_conf = detect_final_chord(y_harm, sr)
    
    score_solide = validate_coherence(chroma_global_avg, note_solide)
    score_glob = validate_coherence(chroma_global_avg, n1_global)
    
    # 1. Décision de base (Stabilité vs Globalité)
    final_decision = note_solide if (occ_graph > 40 or score_solide > (score_glob - 0.02)) else n1_global
    
    # 2. VERROU DE SÉCURITÉ : BOOST POINT DE REPOS
    # On n'applique l'accord de fin que s'il est cohérent avec le reste du morceau (Top 3 notes)
    resolution_applied = False
    if final_chord_conf > 0.75:
        if final_chord_key in top_3_notes:
            final_decision = final_chord_key
            resolution_applied = True
    
    # 3. PRIORITÉ ABSOLUE CADENCE (Même si hors top 3, car c'est un événement structurel)
    if is_cad: final_decision = cad_key

    # 4. Correction finale Relatifs
    n2_alt = weighted_scores.most_common(2)[1][0] if len(weighted_scores) > 1 else n1_global
    is_rel, rel_pref = detect_relative_key(final_decision, n2_alt)
    if is_rel: final_decision = rel_pref

    final_conf = int(validate_coherence(chroma_global_avg, final_decision) * 100)
    bg = "linear-gradient(135deg, #1D976C, #93F9B9)" if final_conf > 80 else "linear-gradient(135deg, #2193B0, #6DD5ED)"
    tempo, _ = librosa.beat.beat_track(y=y_raw, sr=sr)
    
    fig_tg = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark")
    fig_tg.update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER})
    plot_img = fig_tg.to_image(format="png", width=800, height=400)

    res = {
        "file_name": file_name, "tempo": int(float(tempo)),
        "recommended": {"note": final_decision, "conf": final_conf, "bg": bg},
        "note_solide": note_solide, "solid_conf": int(df_tl[df_tl['Note'] == note_solide]['Confiance'].mean()) if not df_tl.empty else 0,
        "timeline": timeline_data, "is_cadence": is_cad, "is_relative": is_rel, "is_resolution": resolution_applied,
        "final_chord_val": final_chord_key,
        "duration": duration, "plot_img": plot_img, "tuning": round(tuning, 2),
        "intro_type": intro_type, "votes": {"CENS": key_cens, "CQT": key_cqt},
        "energy": round(energy_avg, 4)
    }
    del y_raw, y_harm, y_filtered; gc.collect()
    return res

# --- INTERFACE STREAMLIT ---
st.title("🎧 RCDJ228 key7 PRO")

with st.sidebar:
    st.header("⚙️ SYSTÈME")
    st.info("Moteur : HSS + Resolution Harm. (Secured) + Cadence V-i + Arbitrage")
    if st.button("🧹 RESET CACHE"):
        st.session_state.processed_files = {}
        st.session_state.order_list = []
        st.cache_data.clear()
        st.rerun()

if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []

files = st.file_uploader("📂 AUDIO FILES", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])
tabs = st.tabs(["🚀 ANALYSEUR", "📜 HISTORIQUE"])

with tabs[0]:
    if files:
        progress_text = st.empty()
        global_bar = st.progress(0)
        for index, f in enumerate(files):
            fid = f"{f.name}_{f.size}"
            if fid not in st.session_state.processed_files:
                progress_text.text(f"⏳ Analyse Harmonique M3 PRO : {f.name}...")
                f_bytes = f.read()
                res = get_full_analysis(f_bytes, f.name)
                if res:
                    conf_bar = "🟢" * (res['recommended']['conf'] // 10) + "⚪" * (10 - (res['recommended']['conf'] // 10))
                    tg_cap = (
                        f"🚀 *RAPPORT D'ANALYSE EXPERT M3 PRO*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📁 *FICHIER :* `{res['file_name']}`\n"
                        f"⏱ *DURÉE :* `{int(res['duration'] // 60)}m {int(res['duration'] % 60)}s`\n"
                        f"🥁 *TEMPO :* `{res['tempo']} BPM`\n"
                        f"🎹 *INTRO :* `{res['intro_type']}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"💎 *DÉCISION HARMONIQUE*\n"
                        f"主 *CLÉ FINALE :* `{res['recommended']['note'].upper()}`\n"
                        f"🧭 *CAMELOT :* `{get_camelot_pro(res['recommended']['note'])}`\n"
                        f"🎯 *FIABILITÉ :* `{res['recommended']['conf']}%`\n"
                        f"{conf_bar}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📡 *MOTEURS D'ARBITRAGE :*\n"
                        f"🔹 *Point Repos Final :* `{'✅ Confirmé ('+res['final_chord_val']+')' if res['is_resolution'] else '❌ Rejeté/Indéterminé'}`\n"
                        f"🔹 *Cadence V-i :* `{'✅ DÉTECTÉE' if res['is_cadence'] else '❌ Non'}`\n"
                        f"🔹 *Note Dominante :* `{res['note_solide']}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎧 *RCDJ228 Hkey M3 PRO Engine*"
                    )
                    upload_to_telegram(io.BytesIO(f_bytes), f.name, tg_cap, res["plot_img"])
                    del res["plot_img"]
                    st.session_state.processed_files[fid] = res
                    st.session_state.order_list.insert(0, fid)
                del f_bytes; gc.collect()
            global_bar.progress((index + 1) / len(files))
        progress_text.empty(); global_bar.empty()

        for fid in st.session_state.order_list:
            res = st.session_state.processed_files.get(fid)
            if res:
                with st.expander(f"📊 {res['file_name']}", expanded=True):
                    st.markdown(f'<div class="final-decision-box" style="background:{res["recommended"]["bg"]};"><h1>{res["recommended"]["note"]}</h1><h2>CAMELOT: {get_camelot_pro(res["recommended"]["note"])} • CERTITUDE: {res["recommended"]["conf"]}%</h2></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="solid-note-box">💎 ANALYSE FINALE : {res["final_chord_val"] if res["is_resolution"] else "Stabilité temporelle privilégiée (Accord de fin ignoré ou non-dominant)"}</div>', unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.markdown(f'<div class="metric-container">BPM<br><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
                    with c2: get_sine_witness(res["recommended"]["note"], fid)
                    with c3: st.markdown(f'<div class="metric-container">RÉSOLUTION<br><div class="value-custom">{"OUI" if res["is_resolution"] else "NON"}</div></div>', unsafe_allow_html=True)
                    with c4: st.markdown(f'<div class="metric-container">CADENCE V-i<br><div class="value-custom">{"BOOST" if res["is_cadence"] else "NON"}</div></div>', unsafe_allow_html=True)
                    
                    df_plot = pd.DataFrame(res['timeline'])
                    fig = px.line(df_plot, x="Temps", y="Note", template="plotly_dark", title="Stabilité Harmonique (Moteur M3)")
                    fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER})
                    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    if st.session_state.processed_files:
        st.dataframe(pd.DataFrame([{"Fichier": r["file_name"], "Note": r['recommended']['note'], "Camelot": get_camelot_pro(r['recommended']['note']), "BPM": r["tempo"]} for r in st.session_state.processed_files.values()]))

gc.collect()
