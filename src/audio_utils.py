# src/audio_utils.py
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf

SR = 22050
N_MELS = 128
FIXED_TIME_FRAMES = 431
LABELS = ["makina", "newstyle"]  # amplía aquí cuando añadas más estilos

# --- Loader robusto ---
def safe_load_path(path, sr=SR, mono=True):
    """
    Intenta cargar con soundfile (WAV/FLAC/AIFF/OGG) y cae a librosa (audioread) si falla.
    Devuelve (y, sr).
    """
    try:
        data, srf = sf.read(path, always_2d=False)
        if data.ndim == 2 and mono:
            data = data.mean(axis=1)
        if srf != sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=srf, target_sr=sr)
        return data.astype(np.float32), sr
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=mono)
        return y.astype(np.float32), sr

def load_audio_wave(path, sr=SR, mono=True):
    return safe_load_path(path, sr=sr, mono=mono)

# --- Mel utils ---
def wav_to_logmel_from_wave(y, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS

def wav_to_logmel(path, sr=SR, n_mels=N_MELS):
    y, _ = load_audio_wave(path, sr=sr, mono=True)
    return wav_to_logmel_from_wave(y, sr=sr, n_mels=n_mels)

def pad_or_crop_mel(mel, fixed_frames=FIXED_TIME_FRAMES):
    T = mel.shape[1]
    if T < fixed_frames:
        mel = np.pad(mel, ((0,0),(0, fixed_frames - T)), mode="edge")
    else:
        mel = mel[:, :fixed_frames]
    return mel

def file_to_model_input(path):
    mel = wav_to_logmel(path, sr=SR, n_mels=N_MELS)
    mel = pad_or_crop_mel(mel, FIXED_TIME_FRAMES)
    x = mel[np.newaxis, ..., np.newaxis]
    return x

def segment_long_audio_for_model(path, segment_seconds=10, hop_seconds=10):
    y, sr = load_audio_wave(path, sr=SR, mono=True)
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < seg_len:
        return [], np.array([])
    starts = np.arange(0, len(y) - seg_len + 1, hop_len, dtype=int)
    windows = []
    for s in starts:
        clip = y[s:s+seg_len]
        mel = wav_to_logmel_from_wave(clip, sr=sr, n_mels=N_MELS)
        mel = pad_or_crop_mel(mel, FIXED_TIME_FRAMES)
        windows.append(mel[np.newaxis, ..., np.newaxis])
    return windows, (starts / sr)

# --- Visualización ---
def mel_figure(y, sr=SR, n_mels=N_MELS, title="Mel-espectrograma"):
    mel = wav_to_logmel_from_wave(y, sr=sr, n_mels=n_mels)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def overlay_gradcam_on_mel(mel_2d, heatmap_2d, labels, pred_idx, alpha=0.45):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel_2d, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap="magma")
    ax.imshow(heatmap_2d, aspect='auto', interpolation='nearest',
              extent=[0, mel_2d.shape[1], 0, mel_2d.shape[0]],
              origin='lower', alpha=alpha, cmap='jet')
    ax.set_title(f"Grad-CAM — contribuciones a: {labels[pred_idx].upper()}")
    fig.tight_layout()
    return fig

# --- Métricas básicas ---
def compute_basic_features(y, sr=SR):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {
        "bpm": float(np.asarray(tempo).squeeze()),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
    }

# --- Grad-CAM ---
def last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        except Exception:
            continue
    return None

def gradcam_heatmap(model, x, conv_layer_name, class_index, upsample_to=(N_MELS, FIXED_TIME_FRAMES)):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap[..., np.newaxis]
    heatmap = tf.image.resize(heatmap, upsample_to, method="bilinear").numpy().squeeze()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap

# --- Generador demo 5s ---
def generate_demo_wave(sr=SR, dur=5.0, bpm=150, f_kick=55.0):
    t = np.arange(int(sr*dur)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    beat_int = 60.0 / bpm
    for b in np.arange(0, dur, beat_int):
        idx = t >= b
        tb = t[idx] - b
        env = np.exp(-tb * 30.0).astype(np.float32)
        kick = (np.sin(2*np.pi*f_kick*tb).astype(np.float32)) * env
        y[idx] += 0.8 * kick
    rng = np.random.default_rng(7)
    for b in np.arange(0, dur, beat_int):
        start = b + beat_int/2.0
        if start >= dur: break
        mask = (t >= start) & (t < start + 0.04)
        hat = rng.normal(0, 1, mask.sum()).astype(np.float32)
        hat = np.diff(np.concatenate([[0.0], hat])).astype(np.float32)
        y[mask] += 0.25 * hat
    y = (0.9 * y / (np.max(np.abs(y)) + 1e-12)).astype(np.float32)
    return y
