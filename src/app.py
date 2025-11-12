# src/app.py
import sys
from pathlib import Path

# añade la carpeta raíz del repo al sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_utils import (
    file_to_model_input,
    segment_long_audio,
    probs_to_label,
    SR, N_MELS, FIXED_TIME_FRAMES, LABELS
)

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Fabrik DJ Style Classifier", page_icon="🎧", layout="centered")

st.title("🎧 Fabrik DJ Style Classifier")
st.write("Clasificador de **Makina** vs **Newstyle**. Sube un WAV/MP3 (~10s) o un audio largo (se segmenta).")

# Cargar modelo
@st.cache_resource(show_spinner=True)
def load_model(path=None, uploaded_bytes=None):
    if path and os.path.exists(path):
        return tf.keras.models.load_model(path)
    if uploaded_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_bytes)
            tmp.flush()
            return tf.keras.models.load_model(tmp.name)
    return None

default_model_path = "models/fabrik_makina_newstyle.h5"
model_file = None

col1, col2 = st.columns(2)
with col1:
    use_repo_model = st.toggle("Usar modelo del repo", value=os.path.exists(default_model_path))
with col2:
    uploaded_model = st.file_uploader("O sube un .h5", type=["h5"], accept_multiple_files=False)

if use_repo_model and os.path.exists(default_model_path):
    model = load_model(path=default_model_path)
elif uploaded_model is not None:
    model = load_model(uploaded_bytes=uploaded_model.read())
else:
    model = None

if model is None:
    st.warning("🔧 Sube un modelo `.h5` o añade `models/fabrik_makina_newstyle.h5` al repo.")
    st.stop()

# Subir audio
audio_file = st.file_uploader("Sube un audio (WAV recomendado; MP3 suele funcionar)", type=["wav","mp3","ogg","m4a"])
segment_seconds = st.slider("Segmento (s)", 5, 15, 10)
hop_seconds = st.slider("Salto (s)", 1, segment_seconds, segment_seconds)

if audio_file is not None:
    st.audio(audio_file)
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        # Si es corto (~segment_seconds), predicción directa
        windows = segment_long_audio(tmp_path, segment_seconds=segment_seconds, hop_seconds=hop_seconds)
        if len(windows) == 0:
            # fallback: tratarlo como clip 1x
            x = file_to_model_input(tmp_path)
            probs = model.predict(x, verbose=0)[0]
            label, _ = probs_to_label(probs)
            st.subheader(f"Predicción: **{label}**")
            st.write({LABELS[0]: float(probs[0]), LABELS[1]: float(probs[1])})
        else:
            # predecir cada ventana
            probs_list = []
            for w in windows:
                p = model.predict(w, verbose=0)[0]
                probs_list.append(p)
            probs_arr = np.stack(probs_list)
            mean_probs = probs_arr.mean(axis=0)
            label, _ = probs_to_label(mean_probs)

            st.subheader(f"Predicción global: **{label}**")
            st.write("Probabilidades (promedio):",
                     {LABELS[0]: float(mean_probs[0]), LABELS[1]: float(mean_probs[1])})

            # gráfico simple
            fig, ax = plt.subplots()
            ax.bar(LABELS, mean_probs)
            ax.set_ylim([0,1])
            ax.set_ylabel("Probabilidad")
            ax.set_title("Probabilidades promedio por clase")
            st.pyplot(fig)

            with st.expander("Ver detalle por segmento"):
                st.write("Segmentos:", len(probs_list))
                table = [{"segmento": i, LABELS[0]: float(p[0]), LABELS[1]: float(p[1])} for i,p in enumerate(probs_list)]
                st.dataframe(table)
    except Exception as e:
        st.error(f"No se pudo procesar el audio. Prueba con WAV. Error: {e}")
