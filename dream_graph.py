"""
OneiroVoice - dream & memory constellation engine
=================================================
* Whisper transcription (array input -> no ffmpeg needed) with clean error handling
* SentenceTransformer embeddings (all-MiniLM-L6-v2) with a TF-IDF fallback
* Cosine-similarity graph (networkx) + Sleep-Stress Drift Score (0-100)
* Futuristic dark 3D Plotly constellation + drift gauge + motif analytics
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_LOGS_PATH = os.path.join(HERE, "sample_logs.json")

EMOTION_STRESS = {"Calm": 0.0, "Anxious": 1.0, "Startled": 0.85, "Fatigued": 0.6}

MOTIF_LEXICON: Dict[str, List[str]] = {
    "water": ["water", "lake", "river", "rain", "flood", "flooded", "ocean", "sea", "wave", "swim", "drown"],
    "falling": ["fall", "falling", "fell", "drop", "stairs", "staircase", "cliff"],
    "exam": ["exam", "test", "paper", "papers", "formula", "syllabus", "marks"],
    "presentation": ["presentation", "interview", "speech", "stage", "audience", "pitch", "panel"],
    "time pressure": ["late", "clock", "deadline", "missed", "missing", "running", "rush", "hurry", "racing"],
    "corridor": ["hallway", "corridor", "door", "doors", "locked", "endless"],
    "flying": ["flying", "flew", "float", "floating", "soar"],
    "home": ["home", "house", "kitchen", "hostel", "room", "window"],
    "family": ["grandmother", "grandfather", "mother", "father", "sister", "brother", "family"],
    "friends": ["friend", "friends", "laugh", "laughing", "laughed"],
    "light": ["light", "sunrise", "sun", "glow", "dawn", "dusk", "golden"],
    "heaviness": ["heavy", "tired", "sleepy", "exhausted", "drag", "sand"],
    "teeth": ["teeth", "tooth", "crumbling"],
    "alarm": ["alarm", "siren", "bell", "ringing", "jerked", "jolted"],
    "lost": ["lost", "wander", "maze", "forgot", "forget", "stuck", "dead"],
    "train": ["train", "station", "bus", "platform"],
}
MOTIF_ICONS = {
    "water": "🌊", "falling": "🌀", "exam": "📝", "presentation": "🎤", "time pressure": "⏳",
    "corridor": "🚪", "flying": "🕊️", "home": "🏠", "family": "👵", "friends": "🫶",
    "light": "✨", "heaviness": "🪨", "teeth": "🦷", "alarm": "🚨", "lost": "🧭", "train": "🚆",
}
NEG_WORDS = {
    "anxious", "afraid", "fear", "panic", "racing", "locked", "heavy", "lost", "late", "missed", "drown",
    "flood", "alarm", "falling", "crumbling", "dark", "scream", "trapped", "stuck", "tired", "exhausted",
    "jolted", "gasping", "forgot", "helpless", "sweating", "tense", "dead", "pounding", "jerked", "stared",
}
DRIFT_SCALE = [[0.0, "#2dd4bf"], [0.35, "#60a5fa"], [0.65, "#c084fc"], [1.0, "#fb7185"]]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_sample_logs(path: str = SAMPLE_LOGS_PATH) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return sorted(data, key=lambda e: e.get("date", ""))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load sample_logs.json ({exc}). Starting with an empty history.")
        return []


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def extract_motifs(text: str) -> List[str]:
    tokens = set(_tokens(text))
    found = []
    for motif, words in MOTIF_LEXICON.items():
        if any(w in tokens for w in words):
            found.append(motif)
    return found


# --------------------------------------------------------------------------------------
# Models (cached)  - loaders return (model, error) so failures degrade gracefully
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Whisper speech model...")
def load_whisper_model(name: str = "tiny"):
    try:
        import whisper  # type: ignore
        return whisper.load_model(name), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Whisper unavailable ({type(exc).__name__}: {exc})"


@st.cache_resource(show_spinner="Loading SentenceTransformer (all-MiniLM-L6-v2)...")
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer("all-MiniLM-L6-v2"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"SentenceTransformer unavailable ({type(exc).__name__}: {exc})"


def transcribe_audio(model, y: np.ndarray) -> Dict[str, Optional[str]]:
    """Transcribe a 16 kHz mono float32 array. Passing the array directly avoids any ffmpeg dependency."""
    if model is None:
        return {"text": "", "error": "Whisper model is not loaded."}
    try:
        out = model.transcribe(np.asarray(y, dtype=np.float32), fp16=False, task="transcribe",
                               condition_on_previous_text=False, verbose=None)
        text = (out.get("text") or "").strip()
        if not text:
            return {"text": "", "error": "No speech was recognised in this recording."}
        return {"text": text, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "error": f"Transcription failed ({type(exc).__name__}: {exc})"}


# --------------------------------------------------------------------------------------
# Embeddings + similarity
# --------------------------------------------------------------------------------------
def embed_entries(embedder, texts: List[str]) -> Tuple[np.ndarray, str]:
    if embedder is not None:
        try:
            emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float64), "MiniLM (all-MiniLM-L6-v2)"
        except Exception:  # noqa: BLE001
            pass
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform(texts).toarray().astype(np.float64)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m, "TF-IDF fallback"


def similarity_matrix(emb: np.ndarray) -> np.ndarray:
    sim = emb @ emb.T
    return np.clip(sim, -1.0, 1.0)


def effective_threshold(backend: str, thr: float) -> float:
    """TF-IDF cosine values are systematically lower than MiniLM's, so the cut-off is scaled."""
    return float(thr) if backend.startswith("MiniLM") else round(float(thr) * 0.3, 3)


# --------------------------------------------------------------------------------------
# Sleep-Stress Drift Score
# --------------------------------------------------------------------------------------
def compute_drift_scores(entries: List[Dict], sim: np.ndarray, thr: float) -> List[float]:
    """
    Drift = 0.45*vocal stress + 0.20*emotion load + 0.15*stress of semantically linked dreams
          + 0.10*negative-language load + 0.10*short-term trend vs. the previous 5 mornings.
    """
    n = len(entries)
    vs = np.array([float(e.get("vocal_stress", 40.0)) for e in entries])
    scores: List[float] = []
    for i, e in enumerate(entries):
        emo = EMOTION_STRESS.get(e.get("dominant_emotion", "Calm"), 0.3) * 100.0
        nbr = [j for j in range(n) if j != i and sim[i, j] > thr]
        if nbr:
            w = np.array([sim[i, j] for j in nbr])
            nb = float(np.average(vs[nbr], weights=w))
        else:
            nb = float(vs[i])
        hits = sum(1 for tk in _tokens(e.get("transcript", "")) if tk in NEG_WORDS)
        neg = 100.0 * min(1.0, hits / 4.0)
        prev = vs[max(0, i - 5):i]
        if len(prev) >= 2:
            z = (vs[i] - prev.mean()) / (prev.std() + 8.0)
            trend = 50.0 + 25.0 * float(np.clip(z, -2, 2))
        else:
            trend = 50.0
        s = 0.45 * vs[i] + 0.20 * emo + 0.15 * nb + 0.10 * neg + 0.10 * trend
        scores.append(float(np.clip(s, 0, 100)))
    return scores


def drift_label(score: float) -> str:
    if score < 30:
        return "Low - restorative"
    if score < 50:
        return "Mild"
    if score < 70:
        return "Elevated"
    return "High - high stress load"


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
def build_graph(entries: List[Dict], sim: np.ndarray, thr: float, drift: List[float]) -> nx.Graph:
    G = nx.Graph()
    for i, e in enumerate(entries):
        G.add_node(i, date=e.get("date", ""), transcript=e.get("transcript", ""),
                   motifs=e.get("motifs", []), emotion=e.get("dominant_emotion", "Calm"),
                   vocal_stress=float(e.get("vocal_stress", 0)), drift=drift[i],
                   is_today=bool(e.get("is_today", False)))
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > thr:
                G.add_edge(i, j, weight=float(sim[i, j]))
    return G


def layout_3d(G: nx.Graph) -> Dict[int, np.ndarray]:
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        return {0: np.zeros(3)}
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, weight=d["weight"] ** 2)
    pos = nx.spring_layout(H, dim=3, weight="weight", seed=11, k=0.9, iterations=250)
    arr = np.array([pos[i] for i in range(n)])
    arr = arr - arr.mean(axis=0)
    arr = arr / (np.abs(arr).max() + 1e-9)
    return {i: arr[i] for i in range(n)}


def _short_date(d: str) -> str:
    try:
        return f"{int(d[8:10])} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(d[5:7]) - 1]}"
    except Exception:  # noqa: BLE001
        return d


def _hover(d: Dict) -> str:
    snippet = textwrap.fill(d["transcript"][:170] + ("..." if len(d["transcript"]) > 170 else ""), 48).replace("\n", "<br>")
    motifs = ", ".join(d["motifs"]) if d["motifs"] else "-"
    tag = " (TODAY)" if d["is_today"] else ""
    return (f"<b>{d['date']}{tag}</b><br>{d['emotion']} - drift {d['drift']:.0f}/100<br>"
            f"vocal stress {d['vocal_stress']:.0f}<br><i>motifs:</i> {motifs}<br>{snippet}")


def make_constellation(G: nx.Graph) -> go.Figure:
    pos = layout_3d(G)
    fig = go.Figure()
    rng = np.random.default_rng(3)

    # star field
    stars = rng.uniform(-1.7, 1.7, size=(160, 3))
    fig.add_trace(go.Scatter3d(x=stars[:, 0], y=stars[:, 1], z=stars[:, 2], mode="markers",
                               marker=dict(size=1.6, color="#cbd5e1", opacity=0.35),
                               hoverinfo="skip", showlegend=False))

    # glowing edges (3 strength bins x glow + core)
    bins = [(0.0, 0.5, "#38bdf8"), (0.5, 0.65, "#a78bfa"), (0.65, 1.01, "#f0abfc")]
    for lo, hi, col in bins:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        zs: List[Optional[float]] = []
        for u, v, d in G.edges(data=True):
            if lo <= d["weight"] < hi:
                for p in (pos[u], pos[v]):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                xs.append(None); ys.append(None); zs.append(None)
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=col, width=9), opacity=0.10,
                                       hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=col, width=2.5), opacity=0.75,
                                       hoverinfo="skip", showlegend=False))

    nodes = list(G.nodes(data=True))
    past = [(i, d) for i, d in nodes if not d["is_today"]]
    today = [(i, d) for i, d in nodes if d["is_today"]]

    if past:
        P = np.array([pos[i] for i, _ in past])
        sc = np.array([d["drift"] for _, d in past])
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                   marker=dict(size=14 + sc * 0.22, color=sc, colorscale=DRIFT_SCALE,
                                               cmin=0, cmax=100, opacity=0.16),
                                   hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
            text=[_short_date(d["date"]) for _, d in past], textposition="top center",
            textfont=dict(color="#cbd5e1", size=9),
            marker=dict(size=7 + sc * 0.09, color=sc, colorscale=DRIFT_SCALE, cmin=0, cmax=100,
                        line=dict(color="rgba(255,255,255,0.55)", width=1),
                        colorbar=dict(title=dict(text="Drift", font=dict(color="#cbd5e1", size=10)),
                                      orientation="h", thickness=7, len=0.6, x=0.5, y=-0.02,
                                      tickfont=dict(color="#94a3b8", size=9))),
            hovertext=[_hover(d) for _, d in past], hoverinfo="text", showlegend=False))

    if today:
        T = np.array([pos[i] for i, _ in today])
        ts = np.array([d["drift"] for _, d in today])
        fig.add_trace(go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="markers",
                                   marker=dict(size=30, color=ts, colorscale=DRIFT_SCALE, cmin=0, cmax=100, opacity=0.22),
                                   hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(
            x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="markers+text", text=["TODAY"] * len(today),
            textposition="top center", textfont=dict(color="#ffffff", size=11),
            marker=dict(size=15, symbol="diamond", color=ts, colorscale=DRIFT_SCALE, cmin=0, cmax=100,
                        line=dict(color="#ffffff", width=3)),
            hovertext=[_hover(d) for _, d in today], hoverinfo="text", showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#04050d", aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.0, z=0.8)), dragmode="orbit",
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=430, showlegend=False,
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#e2e8f0", size=11)),
    )
    return fig


def make_drift_gauge(score: float, baseline: Optional[float]) -> go.Figure:
    delta = None
    if baseline is not None:
        delta = dict(reference=float(baseline), increasing=dict(color="#fb7185"), decreasing=dict(color="#2dd4bf"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=float(score), delta=delta,
        number=dict(font=dict(color="#f1f5f9", size=38), suffix=""),
        title=dict(text="Sleep-Stress Drift Score", font=dict(color="#a5b4fc", size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#64748b", tickfont=dict(color="#94a3b8")),
            bar=dict(color="#e879f9", thickness=0.28),
            bgcolor="rgba(15,23,42,0.6)", borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(45,212,191,0.35)"),
                   dict(range=[30, 50], color="rgba(96,165,250,0.30)"),
                   dict(range=[50, 70], color="rgba(192,132,252,0.30)"),
                   dict(range=[70, 100], color="rgba(251,113,133,0.35)")],
            threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.8, value=float(score)),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230, margin=dict(l=20, r=20, t=50, b=10),
                      font=dict(color="#cbd5e1"))
    return fig


# --------------------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------------------
def motif_summary(entries: List[Dict], min_count: int = 2) -> List[Tuple[str, int, bool]]:
    """(motif, count, appears_today) for motifs occurring at least `min_count` times."""
    counts: Dict[str, int] = {}
    today_set: set = set()
    for e in entries:
        for m in e.get("motifs", []):
            counts[m] = counts.get(m, 0) + 1
            if e.get("is_today"):
                today_set.add(m)
    rows = [(m, c, m in today_set) for m, c in counts.items() if c >= min_count]
    return sorted(rows, key=lambda r: (-int(r[2]), -r[1], r[0]))


def most_similar(entries: List[Dict], sim: np.ndarray, idx: int, k: int = 3) -> List[Tuple[Dict, float]]:
    order = [j for j in np.argsort(-sim[idx]) if j != idx][:k]
    return [(entries[j], float(sim[idx, j])) for j in order]
