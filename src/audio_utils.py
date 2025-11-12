# src/audio_utils.py
# ------------------------------------------------
# Utilidades de audio para el clasificador Fabrik
# ------------------------------------------------
import numpy as np
import librosa

SR = 22050
N_MELS = 128
FIXED_TIME_FRAMES = 431  # ~10s con hop 512 aprox.
LABELS = ["makina", "newstyle"]  # índice 0 y 1

def wav_to_logmel(path, sr=SR, n_mels=N_MELS):
    """Carga un archivo de audio (WAV/MP3/OGG/M4A), calcula melspectrogram y lo pasa a dB."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS

def pad_or_crop_mel(mel, fixed_frames=FIXED_TIME_FRAMES):
    """Ajusta el mel (128, T) a (128, fixed_frames) por padding o recorte."""
    T = mel.shape[1]
    if T < fixed_frames:
        mel = np.pad(mel, ((0,0), (0, fixed_frames - T)), mode="edge")
    else:
        mel = mel[:, :fixed_frames]
    return mel

def file_to_model_input(path):
    """Convierte un archivo de audio a tensor listo para el modelo: (1,128,431,1)."""
    mel = wav_to_logmel(path)
    mel = pad_or_crop_mel(mel)
    x = mel[np.newaxis, ..., np.newaxis]
    return x

def segment_long_audio(path, segment_seconds=10, hop_seconds=10):
    """
    Divide un audio largo en ventanas (por defecto sin solape) y devuelve
    una lista de tensores (1,128,431,1) para predicción por ventanas.
    """
    y, _ = librosa.load(path, sr=SR, mono=True)
    seg_len = int(segment_seconds * SR)
    hop_len = int(hop_seconds * SR)
    if len(y) < seg_len:
        return []  # se tratará como clip único fuera

    starts = np.arange(0, len(y) - seg_len + 1, hop_len, dtype=int)
    windows = []
    for s in starts:
        clip = y[s:s+seg_len]
        S = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS)
        logS = librosa.power_to_db(S, ref=np.max)
        logS = pad_or_crop_mel(logS)
        windows.append(logS[np.newaxis, ..., np.newaxis])
    return windows

def probs_to_label(probs):
    """Convierte un vector de probabilidades a (nombre_de_etiqueta, idx)."""
    idx = int(np.argmax(probs))
    return LABELS[idx], idx
