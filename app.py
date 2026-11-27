"""
OneiroVoice - multimodal morning voice-note analyzer
Run with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

import audio_engine as ae
import dream_graph as dg

st.set_page_config(page_title="OneiroVoice", page_icon="🌙", layout="wide", initial_sidebar_state="expanded")

NONE_LABEL = "🎙️ Use my own recording"

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(1200px 700px at 12% -10%, #1b1450 0%, rgba(11,16,48,0) 60%),
                radial-gradient(1000px 700px at 100% 0%, #3b0f4d 0%, rgba(11,16,48,0) 55%),
                linear-gradient(180deg, #04050d 0%, #090c22 100%);
    color: #e6e9ff;
}
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #070918; border-right: 1px solid rgba(129,140,248,0.18); }
h1, h2, h3, h4, h5, p, label, span, div { color: inherit; }
.hero-title {
    font-size: 2.5rem; font-weight: 800; letter-spacing: 0.5px; margin: 0;
    background: linear-gradient(90deg, #67e8f9, #a78bfa 45%, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #94a3b8; margin-top: 2px; font-size: 1.0rem; }
.card-title { font-weight: 700; letter-spacing: 0.4px; color: #c7d2fe; margin: 2px 0 6px 0; font-size: 1.02rem; }
.glass {
    background: linear-gradient(145deg, rgba(30,27,75,0.55), rgba(15,23,42,0.55));
    border: 1px solid rgba(129,140,248,0.25); border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 0 24px rgba(99,102,241,0.12); backdrop-filter: blur(6px);
}
.chip {
    display: inline-block; padding: 4px 11px; margin: 3px 4px 3px 0; border-radius: 999px;
    border: 1px solid rgba(129,140,248,0.45); background: rgba(67,56,202,0.22); color: #e0e7ff; font-size: 0.82rem;
}
.chip.hot { border-color: rgba(244,114,182,0.8); background: rgba(190,24,93,0.28); color: #ffe4f1;
            box-shadow: 0 0 10px rgba(244,114,182,0.45); }
.badge {
    display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 0.95rem;
    letter-spacing: 0.5px; color: #05060f;
}
.small { color: #94a3b8; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #f1f5f9; }
[data-testid="stMetricLabel"] p { color: #a5b4fc; }
div[data-testid="stMetric"] {
    background: rgba(30,27,75,0.45); border: 1px solid rgba(129,140,248,0.22);
    border-radius: 12px; padding: 8px 12px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid rgba(167,139,250,0.55);
    background: linear-gradient(90deg, rgba(79,70,229,0.55), rgba(192,38,211,0.45)); color: #fff;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #f0abfc; box-shadow: 0 0 14px rgba(240,171,252,0.45); }
.sim-row { padding: 6px 10px; margin-bottom: 6px; border-left: 3px solid #a78bfa; background: rgba(30,27,75,0.35);
           border-radius: 6px; font-size: 0.85rem; color: #dbe4ff; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
if "session_entries" not in st.session_state:
    st.session_state.session_entries = []
if "mic_nonce" not in st.session_state:
    st.session_state.mic_nonce = 0
if "preset" not in st.session_state:
    st.session_state.preset = NONE_LABEL


def reset_session() -> None:
    st.session_state.session_entries = []
    st.session_state.mic_nonce += 1
    st.session_state.preset = NONE_LABEL


# --------------------------------------------------------------------------------------
# Cached resources / data
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the acoustic ML model (first run trains it, ~30-60 s)...")
def get_audio_model():
    return ae.load_or_train_model()


@st.cache_data(show_spinner="Synthesizing preset voice note...")
def preset_wav(name: str) -> bytes:
    return ae.generate_preset_wav(name)


@st.cache_data(show_spinner="Extracting vocal biomarkers & classifying emotion...")
def run_audio_analysis(wav_bytes: bytes) -> dict:
    return ae.analyze_audio(wav_bytes, get_audio_model())


@st.cache_data(show_spinner="Transcribing with Whisper...")
def transcribe_cached(wav_bytes: bytes, model_name: str) -> dict:
    model, err = dg.load_whisper_model(model_name)
    if err:
        return {"text": "", "error": err}
    y, _ = ae.load_audio(wav_bytes)
    return dg.transcribe_audio(model, y)


def chips_html(rows) -> str:
    if not rows:
        return "<span class='small'>No recurring motifs yet.</span>"
    out = []
    for motif, count, hot in rows:
        icon = dg.MOTIF_ICONS.get(motif, "•")
        out.append(f"<span class='chip{' hot' if hot else ''}'>{icon} {motif} ×{count}</span>")
    return "".join(out)


# --------------------------------------------------------------------------------------
# Sidebar (static part)
# --------------------------------------------------------------------------------------
bundle = get_audio_model()

with st.sidebar:
    st.markdown("## 🌙 OneiroVoice")
    st.caption("Your voice at dawn carries two signals: how your body feels and what your mind dreamed.")
    st.markdown(
        "**What it does**\n\n"
        "1. **Vocal-to-Affect Avatar** - real acoustic biomarkers drive an ML emotion classifier and morph a 3D face.\n"
        "2. **Dream Constellation** - Whisper + MiniLM link today's dream to 10 days of history and score "
        "your Sleep-Stress Drift."
    )
    with st.expander("🧠 ML pipeline & architecture", expanded=False):
        st.code(
            "Audio (mic / preset)\n"
            " └─ librosa: MFCC · YIN F0 · RMS · ZCR · centroid\n"
            "     └─ StandardScaler + RandomForest\n"
            "        (+ rule calibration, 75/25 blend)\n"
            "         └─ P(Calm|Anxious|Fatigued|Startled)\n"
            "             └─ blendshapes → 3D Plotly face\n\n"
            "Whisper ASR → MiniLM embeddings\n"
            " └─ cosine graph (networkx, sim > thr)\n"
            "     └─ Drift Score 0-100\n"
            "         └─ 3D constellation",
            language="text",
        )
    st.markdown("**Model metrics**")
    cv = bundle.get("cv_accuracy", float("nan"))
    m1, m2 = st.columns(2)
    m1.metric("CV accuracy", "n/a" if cv != cv else f"{cv * 100:.1f}%")
    m2.metric("Train clips", f"{bundle.get('n_train', 0)}")
    st.caption(f"{len(bundle.get('feature_names', []))} acoustic features · RandomForest · trained {bundle.get('trained_at', '-')}. "
               "Trained on synthetic, augmented voice profiles - a research prototype, not a medical device.")

    st.markdown("---")
    st.markdown("**Settings**")
    use_whisper = st.toggle("Transcribe microphone audio with Whisper", value=True)
    whisper_name = st.selectbox("Whisper model", ["tiny", "base"], index=0)
    sim_thr = st.slider("Constellation similarity threshold", 0.20, 0.80, 0.40, 0.01,
                        help="Dreams are linked when cosine similarity exceeds this value.")
    st.markdown("---")
    st.markdown("**Session**")
    export_slot = st.container()
    st.button("🗑️ Reset session", on_click=reset_session, use_container_width=True)


def fill_export(latest: dict | None) -> None:
    payload = {
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sample_logs": dg.load_sample_logs(),
        "session_entries": st.session_state.session_entries,
        "latest_analysis": latest,
    }
    with export_slot:
        st.download_button("⬇️ Download session logs (JSON)", data=json.dumps(payload, indent=2),
                           file_name="oneirovoice_session_logs.json", mime="application/json",
                           use_container_width=True)
        st.caption(f"{len(st.session_state.session_entries)} saved session entr{'y' if len(st.session_state.session_entries) == 1 else 'ies'}")


# --------------------------------------------------------------------------------------
# Header + input bar
# --------------------------------------------------------------------------------------
st.markdown("<p class='hero-title'>🌙 OneiroVoice</p>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Multimodal morning voice-note intelligence - decode how you feel and what you dreamed.</p>",
            unsafe_allow_html=True)

in1, in2 = st.columns([1.25, 1.0], gap="large")
with in1:
    st.markdown("<div class='card-title'>🎙️ Record</div>", unsafe_allow_html=True)
    rec = None
    mic_key = f"mic_{st.session_state.mic_nonce}"
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Record 15s Morning Voice Note", key=mic_key)
    else:
        rec = st.file_uploader("Upload a 15 s voice note (WAV / FLAC / OGG)", type=["wav", "flac", "ogg"], key=mic_key)
with in2:
    st.markdown("<div class='card-title'>🧪 Or try a built-in scenario</div>", unsafe_allow_html=True)
    preset_choice = st.selectbox("Synthesized voice + dream test presets (no microphone needed)",
                                 [NONE_LABEL] + list(ae.PRESETS.keys()), key="preset")

wav_bytes = None
source = None
default_text = ""
if preset_choice != NONE_LABEL:
    wav_bytes = preset_wav(preset_choice)
    source = "preset"
    default_text = ae.PRESETS[preset_choice]["transcript"]
    with in2:
        st.audio(wav_bytes, format="audio/wav")
elif rec is not None:
    wav_bytes = rec.getvalue()
    source = "mic"

if wav_bytes is None:
    fill_export(None)
    st.markdown(
        "<div class='glass' style='margin-top:18px'>"
        "<div class='card-title'>Ready when you are</div>"
        "Record a 15-second morning voice note - describe your dream or how you feel - or pick one of the four "
        "built-in scenarios above. OneiroVoice will extract vocal biomarkers, predict your emotional state, "
        "morph a 3D avatar and place your dream inside a 10-day semantic constellation."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------------------
# Audio analysis
# --------------------------------------------------------------------------------------
try:
    res = run_audio_analysis(wav_bytes)
except ValueError as exc:
    st.error(str(exc))
    fill_export(None)
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Audio analysis failed: {type(exc).__name__}: {exc}")
    fill_export(None)
    st.stop()

h = hashlib.md5(wav_bytes).hexdigest()[:10]

# transcript (preset text / Whisper / manual)
if source == "mic":
    if use_whisper:
        tx = transcribe_cached(wav_bytes, whisper_name)
        default_text = tx.get("text") or ""
        if tx.get("error"):
            st.warning(f"{tx['error']} You can type or paste the dream text below instead.")
    else:
        st.info("Whisper is switched off - type or paste your dream text below.")

with st.expander("📝 Dream transcript (auto-filled - edit freely to update the constellation)", expanded=(source == "mic")):
    transcript = st.text_area("Transcript", value=default_text, key=f"tx_{source}_{h}", height=110,
                              label_visibility="collapsed")
transcript = transcript.strip()
tx_for_graph = transcript if transcript else "A morning voice note without a transcript."

today_entry = {
    "id": f"live-{h}",
    "date": dt.date.today().isoformat(),
    "transcript": tx_for_graph,
    "motifs": dg.extract_motifs(tx_for_graph),
    "vocal_stress": round(res["vocal_stress"], 1),
    "dominant_emotion": res["dominant"],
    "is_today": True,
}

# --------------------------------------------------------------------------------------
# Dream graph
# --------------------------------------------------------------------------------------
history = [e for e in dg.load_sample_logs() + st.session_state.session_entries if e.get("id") != today_entry["id"]]
history = sorted(history, key=lambda e: e.get("date", ""))
all_entries = history + [today_entry]

embedder, emb_err = dg.load_embedder()
emb, backend = dg.embed_entries(embedder, [e["transcript"] for e in all_entries])
sim = dg.similarity_matrix(emb)
thr = dg.effective_threshold(backend, sim_thr)
drift = dg.compute_drift_scores(all_entries, sim, thr)
G = dg.build_graph(all_entries, sim, thr, drift)
today_idx = len(all_entries) - 1
today_drift = drift[today_idx]
baseline = float(np.mean(drift[:-1])) if len(drift) > 1 else None
motif_rows = dg.motif_summary(all_entries, min_count=2)
similar = dg.most_similar(all_entries, sim, today_idx, k=3)

with st.sidebar:
    st.caption(f"Embedding backend: **{backend}** · link threshold {thr:.2f} · {G.number_of_edges()} edges")
    if emb_err:
        st.caption("⚠️ MiniLM could not be loaded (offline?). Using TF-IDF fallback.")

latest_summary = {
    "date": today_entry["date"],
    "source": source,
    "dominant_emotion": res["dominant"],
    "emotion_probabilities": {k: round(v, 4) for k, v in res["probs"].items()},
    "vocal_tension": round(res["tension"], 1),
    "vocal_stress": today_entry["vocal_stress"],
    "drift_score": round(today_drift, 1),
    "motifs": today_entry["motifs"],
    "transcript": today_entry["transcript"],
    "features": {k: round(float(v), 5) for k, v in res["features"].items()},
}
fill_export(latest_summary)

# --------------------------------------------------------------------------------------
# 3-column dashboard
# --------------------------------------------------------------------------------------
c1, c2, c3 = st.columns([1.0, 1.1, 1.25], gap="medium")
feats = res["features"]
dom = res["dominant"]
dom_color = ae.EMOTION_COLORS[dom]

with c1:
    st.markdown("<div class='card-title'>📈 Voice Signal & Pitch Contour</div>", unsafe_allow_html=True)
    st.plotly_chart(ae.make_wave_pitch_figure(res), use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div class='card-title'>🧬 Vocal Biomarkers</div>", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    k1.metric("Pitch Variance", f"{feats['pitch_std_hz']:.1f} Hz", f"CV {feats['pitch_cv'] * 100:.1f}%", delta_color="off")
    k2.metric("Vocal Tension", f"{res['tension']:.0f} / 100")
    k3, k4 = st.columns(2)
    k3.metric("Energy (RMS)", f"{feats['rms_db']:.1f} dBFS", f"mean {feats['rms_mean']:.3f}", delta_color="off")
    k4.metric("Mean Pitch", f"{feats['pitch_mean']:.0f} Hz", f"{feats['pitch_range_st']:.1f} st range", delta_color="off")
    st.progress(float(np.clip(res["tension"] / 100.0, 0, 1)), text=f"Vocal tension {res['tension']:.0f}%")
    with st.expander("All extracted features (librosa)"):
        rows = [(k, float(v)) for k, v in feats.items()]
        st.dataframe(pd.DataFrame(rows, columns=["feature", "value"]), use_container_width=True, hide_index=True, height=260)

with c2:
    st.markdown("<div class='card-title'>🫥 Live 3D Affect Avatar</div>", unsafe_allow_html=True)
    st.plotly_chart(ae.build_face_figure(res["shapes"], res["probs"]), use_container_width=True,
                    config={"displayModeBar": False})
    sh = res["shapes"]
    st.markdown(
        f"<span class='badge' style='background:{dom_color};box-shadow:0 0 16px {dom_color}'>"
        f"{dom.upper()} · {res['probs'][dom] * 100:.0f}%</span>"
        f"<div class='small' style='margin-top:6px'>brow tilt {sh['brow_tilt']:+.2f} · brow height {sh['brow_height']:+.2f} · "
        f"eye openness {sh['eye_open']:.2f} · mouth curve {sh['mouth_curve']:+.2f} · mouth open {sh['mouth_open']:.2f}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='card-title' style='margin-top:10px'>📊 Emotion Probabilities</div>", unsafe_allow_html=True)
    st.plotly_chart(ae.make_emotion_bar(res["probs"]), use_container_width=True, config={"displayModeBar": False})
    st.caption("Drag to orbit the avatar. Probabilities = 75% RandomForest + 25% rule-based calibration.")

with c3:
    st.markdown("<div class='card-title'>✨ Dream & Stress Constellation</div>", unsafe_allow_html=True)
    st.plotly_chart(dg.make_constellation(G), use_container_width=True, config={"displaylogo": False})
    st.caption("Nodes = mornings (colour & size = drift score). Glowing edges = semantically related dreams. ◆ = today.")
    st.plotly_chart(dg.make_drift_gauge(today_drift, baseline), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div class='small'>Today's drift: <b style='color:#f0abfc'>{dg.drift_label(today_drift)}</b>"
        + (f" · 10-day baseline {baseline:.0f}" if baseline is not None else "") + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='card-title' style='margin-top:10px'>🏷️ Recurring Motifs</div>", unsafe_allow_html=True)
    st.markdown(chips_html(motif_rows), unsafe_allow_html=True)
    if similar:
        st.markdown("<div class='card-title' style='margin-top:10px'>🔗 Closest past dreams</div>", unsafe_allow_html=True)
        for e, s in similar:
            snippet = e["transcript"][:95] + ("…" if len(e["transcript"]) > 95 else "")
            st.markdown(f"<div class='sim-row'><b>{e['date']}</b> · sim {s:.2f} · {e.get('dominant_emotion', '')}<br>{snippet}</div>",
                        unsafe_allow_html=True)

st.markdown("---")
sv1, sv2 = st.columns([1, 3])
with sv1:
    already = any(e.get("id") == today_entry["id"] for e in st.session_state.session_entries)
    if st.button("💾 Save entry to session log", disabled=already, use_container_width=True):
        saved = {k: v for k, v in today_entry.items() if k != "is_today"}
        st.session_state.session_entries.append(saved)
        st.toast("Entry saved - it will join the constellation history.", icon="🌙")
        st.rerun()
with sv2:
    st.caption("OneiroVoice is a research/hackathon prototype for self-reflection. It is not a medical or diagnostic tool.")
