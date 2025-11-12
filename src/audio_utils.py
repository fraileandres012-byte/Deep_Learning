# src/audio_utils.py
import numpy as np
import librosa

SR = 22050
N_MELS = 128
FIXED_TIME_FRAMES = 431  # ~10s con hop=512 aprox.
LABELS = ["makina", "newstyle"]  # índice 0 y 1

def wav_to_logmel(path, sr=SR, n_mels=N_MELS):
    # Intenta cargar WAV/MP3. Recomiendo WAV para evitar dependencias de ffmpeg
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS

def pad_or_crop_mel(mel, fixed_frames=FIXED_TIME_FRAMES):
    # mel: (128, T) -> (128, fixed_frames)
    T = mel.shape[1]
    if T < fixed_frames:
        mel = np.pad(mel, ((0,0), (0, fixed_frames - T)), mode="edge")
    else:
        mel = mel[:, :fixed_frames]
    return mel

def file_to_model_input(path):
    mel = wav_to_logmel(path)
    mel = pad_or_crop_mel(mel)
    x = mel[np.newaxis, ..., np.newaxis]  # (1, 128, 431, 1)
    return x

def segment_long_audio(path, segment_seconds=10, hop_seconds=10):
    """
    Divide un audio largo en ventanas (por defecto, no solapadas) y devuelve
    una lista de arrays model-ready (cada uno (1,128,431,1)).
    """
    y, _ = librosa.load(path, sr=SR, mono=True)
    seg_len = segment_seconds * SR
    hop_len = hop_seconds * SR
    starts = np.arange(0, len(y) - seg_len + 1, hop_len, dtype=int)
    windows = []
    for s in starts:
        clip = y[s:s+seg_len]
        S = librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS)
        logS = librosa.power_to_db(S, ref=np.max)
        logS = pad_or_crop_mel(logS)
        windows.append(logS[np.newaxis, ..., np.newaxis])
    return windows  # lista de (1,128,431,1)

def probs_to_label(probs):
    idx = int(np.argmax(probs))
    return LABELS[idx], idx
