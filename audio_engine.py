"""
OneiroVoice - audio engine
==========================
* Audio loading / resampling
* Real acoustic feature extraction with librosa (MFCC, YIN pitch, RMS, ZCR, centroid)
* Self-training scikit-learn emotion classifier (trained on first run on synthetic,
  augmented voice clips that go through the *same* feature pipeline, saved as .pkl)
* Rule-calibrated blendshape parameters
* Procedural formant voice synthesizer for 4 built-in presets
* Plotly figures: morphing 3D face avatar, waveform/pitch plot, emotion bars
"""
from __future__ import annotations

import io
import os
import time
from typing import Dict, List, Tuple

import joblib
import librosa
import numpy as np
import plotly.graph_objects as go
import scipy.signal as sps
from plotly.subplots import make_subplots
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SR = 16000
MAX_SECONDS = 15.0
EMOTIONS = ["Calm", "Anxious", "Fatigued", "Startled"]
EMOTION_COLORS = {
    "Calm": "#2dd4bf",
    "Anxious": "#f59e0b",
    "Fatigued": "#818cf8",
    "Startled": "#fb7185",
}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oneirovoice_model.pkl")
MODEL_VERSION = 4
FEATURE_NAMES = [
    "pitch_mean", "pitch_cv", "pitch_range_st", "rms_mean", "rms_cv",
    "zcr_mean", "centroid_mean", "silence_ratio",
    "mfcc2_mean", "mfcc3_mean", "mfcc4_mean", "mfcc5_mean",
]
HOP = 256
FRAME = 1024


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _norm(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(a[k] * (1 - t) + b[k] * t)) for k in range(3))


def _rgba(c: str, a: float) -> str:
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a:.3f})"


def to_wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    pcm = np.clip(y, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Audio loading
# --------------------------------------------------------------------------------------
def load_audio(wav_bytes: bytes, sr: int = SR, max_seconds: float = MAX_SECONDS) -> Tuple[np.ndarray, int]:
    """Decode bytes -> mono float32 at `sr`, truncated to `max_seconds`."""
    try:
        y, _ = librosa.load(io.BytesIO(wav_bytes), sr=sr, mono=True, duration=max_seconds)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode the audio ({exc}). Please use WAV/FLAC/OGG audio.") from exc
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if y.size < int(0.5 * sr):
        raise ValueError("The recording is too short (< 0.5 s). Please record again.")
    y = y - float(np.mean(y))
    return y, sr


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------
def extract_features(y: np.ndarray, sr: int = SR) -> Tuple[Dict[str, float], np.ndarray, Dict[str, np.ndarray]]:
    """Returns (feature dict, classifier vector, tracks for plotting)."""
    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME, hop_length=HOP, n_mels=40)
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=HOP)

    n = int(min(len(rms), len(zcr), len(cent), mfcc.shape[1], len(f0)))
    rms, zcr, cent, mfcc, f0 = rms[:n], zcr[:n], cent[:n], mfcc[:, :n], f0[:n]

    p90 = float(np.percentile(rms, 90))
    thr = max(0.30 * p90, 1e-4)
    speech = rms > thr
    if speech.sum() < 5:
        speech = np.ones(n, dtype=bool)
    silence_ratio = float(1.0 - speech.mean())

    f0s = sps.medfilt(f0, 5)
    voiced = speech & (f0s > 70) & (f0s < 395)
    if voiced.sum() >= 5:
        fv = f0s[voiced]
        pitch_mean = float(np.mean(fv))
        pitch_std = float(np.std(fv))
        pitch_cv = pitch_std / (pitch_mean + 1e-9)
        p5, p95 = np.percentile(fv, [5, 95])
        pitch_range = float(12.0 * np.log2((p95 + 1e-9) / (p5 + 1e-9)))
    else:
        pitch_mean, pitch_std, pitch_cv, pitch_range = float(np.median(f0s)), 0.0, 0.0, 0.0

    r = rms[speech]
    rms_mean = float(np.mean(r))
    rms_cv = float(np.std(r) / (rms_mean + 1e-9))
    mfcc_mean = mfcc[:, speech].mean(axis=1)

    feats = {
        "pitch_mean": pitch_mean,
        "pitch_std_hz": pitch_std,
        "pitch_cv": float(pitch_cv),
        "pitch_range_st": pitch_range,
        "rms_mean": rms_mean,
        "rms_db": float(20.0 * np.log10(rms_mean + 1e-6)),
        "rms_cv": rms_cv,
        "zcr_mean": float(np.mean(zcr[speech])),
        "centroid_mean": float(np.mean(cent[speech])),
        "silence_ratio": silence_ratio,
    }
    for k in range(13):
        feats[f"mfcc{k + 1}_mean"] = float(mfcc_mean[k])
    vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)

    tracks = {
        "times": librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP),
        "rms": rms,
        "f0": np.where(voiced, f0s, np.nan),
        "voiced": voiced,
        "mfcc": mfcc,
    }
    return feats, vec, tracks


def vocal_tension(f: Dict[str, float]) -> float:
    """0-100 composite index of vocal strain (pitch instability, brightness, noisiness, dynamics)."""
    t = (
        0.30 * _norm(f["pitch_cv"], 0.03, 0.25)
        + 0.20 * _norm(f["pitch_range_st"], 2.0, 14.0)
        + 0.20 * _norm(f["centroid_mean"], 900.0, 3200.0)
        + 0.15 * _norm(f["zcr_mean"], 0.03, 0.16)
        + 0.15 * _norm(f["rms_cv"], 0.30, 1.10)
    )
    return float(100.0 * t)


def vocal_stress_score(probs: Dict[str, float], tension: float) -> float:
    w = {"Calm": 0.05, "Anxious": 1.0, "Fatigued": 0.6, "Startled": 0.85}
    emo = sum(probs[e] * w[e] for e in EMOTIONS)
    return float(np.clip(100.0 * (0.55 * emo + 0.45 * tension / 100.0), 0, 100))


# --------------------------------------------------------------------------------------
# Procedural formant voice synthesizer (used for presets AND classifier training)
# --------------------------------------------------------------------------------------
VOWELS = [
    (730, 1090, 2440),  # a
    (530, 1840, 2480),  # e
    (270, 2290, 3010),  # i
    (570, 840, 2410),   # o
    (300, 870, 2240),   # u
]
FORMANT_BW = (80.0, 110.0, 160.0)

PROFILES = {
    "Calm": dict(f0s=1.00, depth=(0.04, 0.08), jitter=(0.002, 0.006), tremor=(0.0, 0.006),
                 rate=(2.8, 3.6), energy=(0.10, 0.18), pause=(0.08, 0.16), bright=(0.95, 1.05),
                 breath=(0.02, 0.05), tilt=(1.6, 2.0), burst=(0.0, 0.10)),
    "Anxious": dict(f0s=1.10, depth=(0.10, 0.18), jitter=(0.010, 0.020), tremor=(0.02, 0.05),
                    rate=(4.6, 6.0), energy=(0.18, 0.30), pause=(0.04, 0.12), bright=(1.10, 1.30),
                    breath=(0.05, 0.10), tilt=(1.0, 1.4), burst=(0.15, 0.40)),
    "Fatigued": dict(f0s=0.90, depth=(0.01, 0.03), jitter=(0.003, 0.008), tremor=(0.0, 0.01),
                     rate=(1.8, 2.5), energy=(0.04, 0.09), pause=(0.30, 0.45), bright=(0.80, 0.92),
                     breath=(0.08, 0.14), tilt=(2.2, 2.8), burst=(0.0, 0.05)),
    "Startled": dict(f0s=1.25, depth=(0.20, 0.35), jitter=(0.010, 0.025), tremor=(0.01, 0.03),
                     rate=(3.5, 5.0), energy=(0.24, 0.38), pause=(0.20, 0.35), bright=(1.20, 1.40),
                     breath=(0.03, 0.07), tilt=(0.9, 1.3), burst=(0.70, 1.00)),
}


def sample_voice_params(emotion: str, rng: np.random.Generator) -> Dict[str, float]:
    p = PROFILES[emotion]
    out = {"f0": float(rng.uniform(105, 205) * p["f0s"])}
    for key in ("depth", "jitter", "tremor", "rate", "energy", "pause", "bright", "breath", "tilt", "burst"):
        lo, hi = p[key]
        out[key] = float(rng.uniform(lo, hi))
    return out


def _resonator(freq: float, bw: float, sr: int):
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * freq / sr
    return [1.0 - r], [1.0, -2.0 * r * np.cos(theta), r * r]


def synth_voice(duration: float, f0: float, depth: float, jitter: float, tremor: float, rate: float,
                energy: float, pause: float, bright: float, breath: float, tilt: float, burst: float,
                rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Source-filter speech-like synthesis with controllable prosody."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    syl_dur = 1.0 / max(rate, 0.5)

    # --- pitch contour: slow intonation + syllable accents + jitter + tremor
    ctrl_n = int(duration * 50) + 2
    ctrl_t = np.linspace(0, duration, ctrl_n)
    slow = gaussian_filter1d(rng.standard_normal(ctrl_n), 6)
    slow /= np.std(slow) + 1e-9
    fast = gaussian_filter1d(rng.standard_normal(ctrl_n), 1.2)
    fast /= np.std(fast) + 1e-9
    contour = np.interp(t, ctrl_t, depth * (0.8 * slow + 0.5 * fast))

    n_syl = int(duration / syl_dur) + 3
    syl_centers = (np.arange(n_syl) + 0.5) * syl_dur
    accents = rng.standard_normal(n_syl) * depth * 0.6
    accent = np.interp(t, syl_centers, accents)

    jit_t = np.linspace(0, duration, int(duration * 200) + 2)
    jit = np.interp(t, jit_t, rng.standard_normal(jit_t.size) * jitter)
    trem = tremor * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi))

    f0_t = np.clip(f0 * (1.0 + contour + accent + jit + trem), 60.0, 420.0)
    phase = 2.0 * np.pi * np.cumsum(f0_t) / sr

    # --- glottal-like source: harmonic series with spectral tilt
    src = np.zeros(n)
    for k in range(1, 20):
        src += np.sin(k * phase) / (k ** tilt)

    # --- syllable-wise formant filtering + amplitude envelope
    out = np.zeros(n)
    amp_env = np.zeros(n)
    cur = 0.0
    attack = 0.04 + (1.0 - burst) * 0.30
    while cur < duration:
        dur = syl_dur * rng.uniform(0.75, 1.25)
        s, e = int(cur * sr), min(n, int((cur + dur) * sr))
        if e - s < 64:
            break
        if rng.random() > pause:
            vowel = VOWELS[int(rng.integers(len(VOWELS)))]
            seg = src[s:e]
            for fm, bw in zip(vowel, FORMANT_BW):
                b, a = _resonator(fm * bright * (1.0 + rng.normal(0, 0.03)), bw, sr)
                seg = sps.lfilter(b, a, seg)
            seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
            x = np.linspace(0, 1, e - s)
            env = np.minimum(x / attack, 1.0) * np.minimum((1.0 - x) / 0.35, 1.0)
            env = np.clip(env, 0, 1) ** 1.2
            amp = energy * float(rng.lognormal(0.0, 0.12 + 0.45 * burst))
            out[s:e] += seg * env * amp
            amp_env[s:e] += env * amp
            cur += dur
        else:
            cur += dur * rng.uniform(1.0, 2.2)  # a pause

    # --- breath / aspiration noise (high-passed), modulated by the voiced envelope
    nb, na = sps.butter(2, 2000.0 / (sr / 2), btype="high")
    noise = sps.lfilter(nb, na, rng.standard_normal(n))
    out += breath * noise * amp_env * 1.5
    return np.clip(out, -0.98, 0.98).astype(np.float32)


def _augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = y * float(rng.uniform(0.7, 1.3))
    y = y + rng.standard_normal(y.size).astype(np.float32) * float(rng.uniform(0.0003, 0.003))
    return np.clip(y, -0.98, 0.98).astype(np.float32)


# --------------------------------------------------------------------------------------
# Built-in presets
# --------------------------------------------------------------------------------------
PRESETS: Dict[str, Dict] = {
    "🌅 Calm - Lake House Sunrise": {
        "emotion": "Calm", "seed": 101,
        "transcript": ("I was back at the lake house just before sunrise. The water was still and warm, "
                       "and my grandmother was making tea in the kitchen while light spilled across the floor. "
                       "I woke up feeling light and unhurried."),
    },
    "😰 Anxious - Missed Exam Dream": {
        "emotion": "Anxious", "seed": 202,
        "transcript": ("I was in an exam hall and the clock kept speeding up. My pen ran dry, I could not "
                       "remember a single formula, and everyone else was already handing in their papers. "
                       "I woke up with my heart racing."),
    },
    "🥱 Fatigued - Endless Hallway": {
        "emotion": "Fatigued", "seed": 303,
        "transcript": ("I was walking down a long hallway that never ended. Every door was locked and my legs "
                       "felt heavy like wet sand. I just wanted to lie down and sleep, and I still feel tired."),
    },
    "😱 Startled - Falling Staircase": {
        "emotion": "Startled", "seed": 404,
        "transcript": ("I was falling down a dark staircase and the steps kept disappearing under my feet. "
                       "Then an alarm rang and I jolted awake, gasping, with water everywhere on the floor."),
    },
}


def generate_preset_wav(name: str, duration: float = 15.0) -> bytes:
    cfg = PRESETS[name]
    rng = np.random.default_rng(cfg["seed"])
    params = sample_voice_params(cfg["emotion"], rng)
    y = synth_voice(duration, rng=rng, **params)
    y = _augment(y, rng)
    return to_wav_bytes(y)


# --------------------------------------------------------------------------------------
# ML classifier (trained on first run)
# --------------------------------------------------------------------------------------
def _train_model(n_per_class: int = 40, clip_seconds: float = 4.0, seed: int = 7) -> Dict:
    rng = np.random.default_rng(seed)
    X: List[np.ndarray] = []
    y_lab: List[int] = []
    for ci, emo in enumerate(EMOTIONS):
        for _ in range(n_per_class):
            params = sample_voice_params(emo, rng)
            clip = _augment(synth_voice(clip_seconds, rng=rng, **params), rng)
            _, vec, _ = extract_features(clip)
            X.append(vec)
            y_lab.append(ci)
    Xa, ya = np.vstack(X), np.array(y_lab)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
    ])
    try:
        cv_acc = float(np.mean(cross_val_score(pipe, Xa, ya, cv=5)))
    except Exception:  # noqa: BLE001
        cv_acc = float("nan")
    pipe.fit(Xa, ya)
    return {
        "version": MODEL_VERSION,
        "pipeline": pipe,
        "feature_names": FEATURE_NAMES,
        "emotions": EMOTIONS,
        "cv_accuracy": cv_acc,
        "n_train": int(len(ya)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }


def load_or_train_model(path: str = MODEL_PATH) -> Dict:
    if os.path.exists(path):
        try:
            bundle = joblib.load(path)
            if bundle.get("version") == MODEL_VERSION and bundle.get("feature_names") == FEATURE_NAMES:
                return bundle
        except Exception:  # noqa: BLE001
            pass
    bundle = _train_model()
    try:
        joblib.dump(bundle, path)
    except Exception:  # noqa: BLE001  (read-only file systems still work, just retrain next run)
        pass
    return bundle


def _rule_probs(f: Dict[str, float], tension: float) -> np.ndarray:
    tn = tension / 100.0
    e = _norm(f["rms_mean"], 0.02, 0.25)
    v = _norm(f["pitch_cv"], 0.03, 0.25)
    s = f["silence_ratio"]
    br = _norm(f["centroid_mean"], 900.0, 3000.0)
    z = np.array([
        1.2 * (1 - abs(tn - 0.3) / 0.7) + 0.5 * (1 - abs(e - 0.45)) - 0.5 * s,   # Calm
        1.6 * tn + 0.6 * v + 0.4 * br - 0.5 * s,                                  # Anxious
        1.6 * s + 1.0 * (1 - e) + 0.6 * (1 - v) - 0.8 * tn,                       # Fatigued
        1.2 * e + 1.0 * v + 0.9 * tn - 0.4 * s,                                   # Startled
    ])
    z = z * 2.0
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def predict_emotions(bundle: Dict, vec: np.ndarray, feats: Dict[str, float], tension: float):
    pipe = bundle["pipeline"]
    raw = pipe.predict_proba(vec.reshape(1, -1))[0]
    ml = np.zeros(len(EMOTIONS))
    for cls, p in zip(pipe.classes_, raw):
        ml[int(cls)] = p
    rule = _rule_probs(feats, tension)
    blend = 0.75 * ml + 0.25 * rule
    blend = blend / blend.sum()
    return (
        {e: float(blend[i]) for i, e in enumerate(EMOTIONS)},
        {e: float(ml[i]) for i, e in enumerate(EMOTIONS)},
        {e: float(rule[i]) for i, e in enumerate(EMOTIONS)},
    )


# --------------------------------------------------------------------------------------
# Blendshapes
# --------------------------------------------------------------------------------------
EMO_SHAPES = {
    "Calm":     dict(brow_height=0.05, brow_tilt=0.00, brow_arch=0.35, eye_open=0.62, mouth_curve=0.45, mouth_open=0.03, cheek_lift=0.50),
    "Anxious":  dict(brow_height=0.35, brow_tilt=0.85, brow_arch=0.20, eye_open=0.88, mouth_curve=-0.25, mouth_open=0.10, cheek_lift=0.00),
    "Fatigued": dict(brow_height=-0.45, brow_tilt=0.25, brow_arch=0.00, eye_open=0.28, mouth_curve=-0.45, mouth_open=0.05, cheek_lift=0.00),
    "Startled": dict(brow_height=1.00, brow_tilt=0.10, brow_arch=0.90, eye_open=1.00, mouth_curve=0.00, mouth_open=0.85, cheek_lift=0.00),
}


def compute_blendshapes(probs: Dict[str, float], tension: float, feats: Dict[str, float]) -> Dict[str, float]:
    keys = list(EMO_SHAPES["Calm"].keys())
    shapes = {k: sum(probs[e] * EMO_SHAPES[e][k] for e in EMOTIONS) for k in keys}
    tn = tension / 100.0
    shapes["brow_tilt"] += 0.35 * (tn - 0.4)
    shapes["brow_height"] += 0.15 * (tn - 0.4)
    shapes["eye_open"] += 0.15 * (tn - 0.5)
    shapes["mouth_open"] += 0.10 * max(0.0, tn - 0.5) + 0.05 * _norm(feats["rms_mean"], 0.05, 0.3)
    shapes["mouth_curve"] -= 0.15 * (tn - 0.4)
    ranges = {"brow_height": (-1, 1), "brow_tilt": (-1, 1), "brow_arch": (0, 1), "eye_open": (0.05, 1),
              "mouth_curve": (-1, 1), "mouth_open": (0, 1), "cheek_lift": (0, 1)}
    for k, (lo, hi) in ranges.items():
        shapes[k] = float(np.clip(shapes[k], lo, hi))
    shapes["tension"] = float(tn)
    return shapes


# --------------------------------------------------------------------------------------
# Full analysis
# --------------------------------------------------------------------------------------
def analyze_audio(wav_bytes: bytes, bundle: Dict) -> Dict:
    y, sr = load_audio(wav_bytes)
    feats, vec, tracks = extract_features(y, sr)
    tension = vocal_tension(feats)
    probs, ml, rule = predict_emotions(bundle, vec, feats, tension)
    dominant = max(probs, key=probs.get)
    return {
        "y": y, "sr": sr, "features": feats, "vector": vec, "tracks": tracks,
        "tension": tension, "probs": probs, "ml_probs": ml, "rule_probs": rule,
        "dominant": dominant, "vocal_stress": vocal_stress_score(probs, tension),
        "shapes": compute_blendshapes(probs, tension, feats),
        "duration": float(len(y) / sr),
    }


# --------------------------------------------------------------------------------------
# 3D expressive face
# --------------------------------------------------------------------------------------
XR, YR, ZR = 0.78, 1.0, 0.86
NU, NV = 52, 80


def _surf_z(x, y):
    t = 1.0 - (np.asarray(x) / XR) ** 2 - (np.asarray(y) / YR) ** 2
    return ZR * np.sqrt(np.clip(t, 0.0, None))


def _head_mesh(sh: Dict[str, float], color_hi: str):
    theta = np.linspace(0, np.pi, NU)
    phi = np.linspace(0, 2 * np.pi, NV, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    x = XR * np.sin(T) * np.sin(P)
    y = YR * np.cos(T)
    z = ZR * np.sin(T) * np.cos(P)
    front = np.clip(z / ZR, 0, 1)

    # shading from the undeformed ellipsoid normals
    nx_, ny_, nz_ = x / XR ** 2, y / YR ** 2, z / ZR ** 2
    nn = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2) + 1e-9
    light = np.array([0.35, 0.55, 1.0])
    light = light / np.linalg.norm(light)
    lam = np.clip((nx_ * light[0] + ny_ * light[1] + nz_ * light[2]) / nn, 0, 1)
    intensity = 0.12 + 0.88 * lam

    # deformations
    jaw_w = np.clip((-y - 0.10) / 0.60, 0, 1)
    y = y - sh["mouth_open"] * 0.16 * jaw_w
    cheek = np.exp(-(((np.abs(x) - 0.42) / 0.18) ** 2 + ((y + 0.12) / 0.20) ** 2)) * front
    z = z + 0.05 * sh["cheek_lift"] * cheek
    y = y + 0.03 * sh["cheek_lift"] * cheek
    z = z + 0.035 * np.exp(-((y - 0.40) / 0.07) ** 2) * front            # brow ridge
    for sx in (-1, 1):                                                     # eye sockets
        z = z - 0.03 * np.exp(-(((x - sx * 0.31) / 0.17) ** 2 + ((y - 0.17) / 0.10) ** 2)) * front
    z = z + 0.07 * np.exp(-((x / 0.10) ** 2 + ((y - 0.0) / 0.28) ** 2)) * front   # nose
    z = z - 0.015 * np.exp(-((x / 0.25) ** 2 + ((y + 0.42) / 0.08) ** 2)) * front  # mouth recess

    ii, jj = np.meshgrid(np.arange(NU - 1), np.arange(NV), indexing="ij")
    a = ii * NV + jj
    b = ii * NV + (jj + 1) % NV
    c = (ii + 1) * NV + jj
    d = (ii + 1) * NV + (jj + 1) % NV
    I = np.concatenate([a.ravel(), b.ravel()])
    J = np.concatenate([c.ravel(), c.ravel()])
    K = np.concatenate([b.ravel(), d.ravel()])

    colorscale = [[0.0, "#050816"], [0.5, _mix("#050816", color_hi, 0.38)], [1.0, _mix(color_hi, "#ffffff", 0.25)]]
    return go.Mesh3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), i=I, j=J, k=K,
        intensity=intensity.ravel(), colorscale=colorscale, showscale=False, opacity=0.95,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.35, roughness=0.5, fresnel=0.4),
        lightposition=dict(x=200, y=300, z=600), hoverinfo="skip", name="head",
    )


def _fan(xs, ys, zs, color, opacity=1.0):
    xs, ys, zs = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    X = np.append(xs, xs.mean())
    Y = np.append(ys, ys.mean())
    Z = np.append(zs, zs.mean())
    m = len(xs)
    return go.Mesh3d(
        x=X, y=Y, z=Z, i=np.full(m, m), j=np.arange(m), k=(np.arange(m) + 1) % m,
        color=color, opacity=opacity, flatshading=True, hoverinfo="skip", showscale=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    )


def _line(xs, ys, off, color, width, name="", dash=None):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return go.Scatter3d(
        x=xs, y=ys, z=_surf_z(xs, ys) + off, mode="lines",
        line=dict(color=color, width=width, dash=dash or "solid"),
        hoverinfo="skip", showlegend=False, name=name,
    )


def build_face_figure(sh: Dict[str, float], probs: Dict[str, float]) -> go.Figure:
    dominant = max(probs, key=probs.get)
    emo = EMOTION_COLORS[dominant]
    tn = sh["tension"]
    fig = go.Figure()
    fig.add_trace(_head_mesh(sh, emo))

    kp_x: List[float] = []
    kp_y: List[float] = []

    # --- eyes
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    h = 0.012 + 0.095 * sh["eye_open"]
    for sx in (-1, 1):
        cx, cy, w = sx * 0.31, 0.17, 0.14
        ex, ey = cx + w * np.cos(t), cy + h * np.sin(t)
        fig.add_trace(_fan(ex, ey, _surf_z(ex, ey) + 0.05, "#eef2ff", 1.0))
        tt = np.linspace(0, np.pi, 20)
        fig.add_trace(_line(cx + w * np.cos(tt), cy + h * np.sin(tt), 0.065, _mix(emo, "#ffffff", 0.2), 6, "lid"))
        fig.add_trace(_line(ex, ey, 0.065, "#c7d2fe", 3, "eye"))
        pupil = min(1.0, sh["eye_open"] * 2.2)
        px, py = cx, cy - 0.01
        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[float(_surf_z(px, py)) + 0.075], mode="markers",
            marker=dict(size=4 + 9 * pupil * (0.8 + 0.4 * sh["eye_open"]), color="#0b1020",
                        line=dict(color=emo, width=3)),
            hoverinfo="skip", showlegend=False))
        kp_x += [cx - w, cx + w]
        kp_y += [cy, cy]

    # --- eyebrows
    s = np.linspace(0, 1, 24)
    for sx in (-1, 1):
        bx = sx * (0.10 + 0.40 * s)
        by = (0.42 + 0.09 * sh["brow_height"] + 0.05 * sh["brow_arch"] * np.sin(np.pi * s)
              + 0.07 * sh["brow_tilt"] * (1.0 - 2.0 * s))
        fig.add_trace(_line(bx, by, 0.07, _mix("#e8eeff", emo, 0.25 + 0.5 * tn), 9, "brow"))
        kp_x += [bx[0], bx[-1]]
        kp_y += [by[0], by[-1]]

    # --- nose
    ny = np.linspace(0.20, -0.10, 12)
    fig.add_trace(_line(np.zeros_like(ny), ny, 0.14, "#9fb4ff", 3, "nose"))
    nxs = np.array([-0.09, -0.05, 0.0, 0.05, 0.09])
    nys = np.array([-0.11, -0.14, -0.15, -0.14, -0.11])
    fig.add_trace(_line(nxs, nys, 0.12, "#9fb4ff", 3, "nostrils"))
    kp_x += [0.0]
    kp_y += [-0.15]

    # --- mouth
    mw = 0.20 + 0.03 * max(sh["mouth_curve"], 0)
    u = np.linspace(-1, 1, 30)
    mx = u * mw
    my0 = -0.42
    base = my0 + sh["mouth_curve"] * 0.08 * u ** 2
    upper = base + 0.02 * (1 - u ** 2) * (0.4 + 0.6 * (1 - sh["mouth_open"]))
    lower = base - sh["mouth_open"] * 0.16 * (1 - u ** 2) - 0.018 * (1 - u ** 2)
    if sh["mouth_open"] > 0.06:
        px = np.concatenate([mx, mx[::-1]])
        py = np.concatenate([upper, lower[::-1]])
        fig.add_trace(_fan(px, py, _surf_z(px, py) + 0.06, "#1a0020", 1.0))
    fig.add_trace(_line(mx, upper, 0.075, "#ff8fb1", 7, "upper lip"))
    fig.add_trace(_line(mx, lower, 0.075, "#ff6f9c", 7, "lower lip"))
    kp_x += [mx[0], mx[-1], 0.0]
    kp_y += [upper[0], upper[-1], lower[15]]

    # --- vocal-tension forehead furrows
    if tn > 0.25:
        fx = np.linspace(-0.36, 0.36, 30)
        for k in range(3):
            fy = 0.64 + 0.075 * k + 0.012 * np.sin(fx * 14 + k)
            fig.add_trace(_line(fx, fy, 0.045, _rgba("#fb7185", min(1.0, 0.25 + tn)), 3, "furrow"))

    # --- key landmark points
    kx, ky = np.array(kp_x), np.array(kp_y)
    fig.add_trace(go.Scatter3d(
        x=kx, y=ky, z=_surf_z(kx, ky) + 0.1, mode="markers",
        marker=dict(size=3, color=emo, opacity=0.95), hoverinfo="skip", showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.0, 1.0]),
            yaxis=dict(visible=False, range=[-1.15, 1.15]),
            zaxis=dict(visible=False, range=[-1.0, 1.0]),
            aspectmode="data", bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=0.35, y=0.15, z=1.9), up=dict(x=0, y=1, z=0)),
            dragmode="orbit",
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=430, showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------------------
# 2D plots
# --------------------------------------------------------------------------------------
def make_wave_pitch_figure(res: Dict) -> go.Figure:
    y, sr, tr = res["y"], res["sr"], res["tracks"]
    step = max(1, len(y) // 3500)
    tw = np.arange(0, len(y), step) / sr
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Waveform + RMS energy envelope", "Pitch contour (YIN, voiced frames)"))
    fig.add_trace(go.Scatter(x=tw, y=y[::step], mode="lines", line=dict(color="#60a5fa", width=1),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.15)", name="waveform"), 1, 1)
    fig.add_trace(go.Scatter(x=tr["times"], y=tr["rms"], mode="lines", line=dict(color="#f472b6", width=2),
                             name="RMS"), 1, 1)
    fig.add_trace(go.Scatter(x=tr["times"], y=-tr["rms"], mode="lines", line=dict(color="#f472b6", width=2),
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=tr["times"], y=tr["f0"], mode="lines+markers", connectgaps=False,
                             line=dict(color="#2dd4bf", width=2), marker=dict(size=3), name="F0 (Hz)"), 2, 1)
    pm = res["features"]["pitch_mean"]
    if pm > 0:
        fig.add_hline(y=pm, line=dict(color="#fbbf24", dash="dash", width=1), row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,10,28,0.6)",
                      height=420, margin=dict(l=40, r=10, t=40, b=30), showlegend=False,
                      font=dict(color="#cbd5e1", size=11))
    fig.update_annotations(font=dict(size=12, color="#a5b4fc"))
    return fig


def make_emotion_bar(probs: Dict[str, float]) -> go.Figure:
    emos = EMOTIONS[::-1]
    vals = [probs[e] * 100 for e in emos]
    fig = go.Figure(go.Bar(
        x=vals, y=emos, orientation="h",
        marker=dict(color=[EMOTION_COLORS[e] for e in emos], line=dict(color="rgba(255,255,255,0.25)", width=1)),
        text=[f"{v:.0f}%" for v in vals], textposition="outside", textfont=dict(color="#e2e8f0"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=190, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 115], showgrid=False, visible=False),
                      yaxis=dict(showgrid=False), font=dict(color="#cbd5e1"))
    return fig
