# src/audio_utils.py
# ------------------------------------------------
# Utilidades de audio, features y visualización
# ------------------------------------------------
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

SR = 22050
N_MELS = 128
FIXED_TIME_FRAMES = 431  # ~10s con hop ≈512 (depende de parámetros de librosa)
LABELS = ["makina", "newstyle"]

# ---------- Carga y mels ----------
def load_audio_wave(path, sr=SR, mono=True):
    y, _ = librosa.load(path, sr=sr, mono=mono)
    return y, sr

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
    x = mel[np.newaxis, ..., np.newaxis]  # (1,128,431,1)
    return x

def segment_long_audio_for_model(path, segment_seconds=10, hop_seconds=10):
    """
    Devuelve: lista de ventanas ya con forma (1,128,431,1) y
    array de tiempos de inicio (s) para timeline.
    """
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

# ---------- Visualización ----------
def mel_figure(y, sr=SR, n_mels=N_MELS, title="Mel-espectrograma"):
    mel = wav_to_logmel_from_wave(y, sr=sr, n_mels=n_mels)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def overlay_gradcam_on_mel(mel_2d, heatmap_2d, labels, pred_idx, alpha=0.45):
    """
    mel_2d: (128, 431), heatmap_2d: (128, 431) en [0,1].
    Devuelve figura con overlay del heatmap sobre el mel.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel_2d, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap="magma")
    ax.imshow(heatmap_2d, aspect='auto', interpolation='nearest',
              extent=[0, mel_2d.shape[1], 0, mel_2d.shape[0]],
              origin='lower', alpha=alpha, cmap='jet')
    ax.set_title(f"Grad-CAM — contribuciones a: {labels[pred_idx].upper()}")
    fig.tight_layout()
    return fig

# ---------- Métricas de audio ----------
def compute_basic_features(y, sr=SR):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # BPM estimado
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {
        "bpm": float(tempo),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
    }

# ---------- Grad-CAM ----------
def last_conv_layer_name(model):
    """Encuentra el nombre de la última capa Conv2D del modelo."""
    for layer in reversed(model.layers):
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        except Exception:
            continue
    return None

def gradcam_heatmap(model, x, conv_layer_name, class_index, upsample_to=(N_MELS, FIXED_TIME_FRAMES)):
    """
    x: (1, H, W, 1). Devuelve heatmap 2D normalizada (H,W) en [0,1].
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]  # (h, w, channels)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    # Normalizar
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap[..., np.newaxis]

    # Redimensionar al tamaño del mel de entrada (N_MELS, FIXED_TIME_FRAMES)
    heatmap = tf.image.resize(heatmap, upsample_to, method="bilinear").numpy().squeeze()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap
