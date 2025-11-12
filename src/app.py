# app.py
# -----------------------------------------------------------
# 🎧 Fabrik DJ Style Classifier — Pro (Makina vs Newstyle)
# -----------------------------------------------------------

import os, sys, io
from pathlib import Path

# Hacer importable "src/" tanto en local como en Streamlit Cloud
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# Estilo visual: tema y CSS sutil
st.set_page_config(page_title="Fabrik DJ Style Classifier", page_icon="🎛️", layout="wide")
st.markdown("""
<style>
/* Tipografía y tarjetas suaves */
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji','Segoe UI Emoji', 'Segoe UI Symbol'; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div.stButton > button, .stDownloadButton > button { border-radius: 12px; padding: 0.6rem 1rem; }
[data-testid="stMetricValue"] { font-weight: 800; }
.card { background: #0e1117; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 0.9rem 1rem; }
.small { font-size: 0.9rem; opacity: .9; }
.hr { border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent); margin: .6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Import utilidades de audio/vis
try:
    from src.audio_utils import (
        SR, N_MELS, FIXED_TIME_FRAMES, LABELS,
        file_to_model_input, segment_long_audio_for_model,
        mel_figure, overlay_gradcam_on_mel, compute_basic_features,
        last_conv_layer_name, gradcam_heatmap, load_audio_wave
    )
except ModuleNotFoundError:
    from audio_utils import (
        SR, N_MELS, FIXED_TIME_FRAMES, LABELS,
        file_to_model_input, segment_long_audio_for_model,
        mel_figure, overlay_gradcam_on_mel, compute_basic_features,
        last_conv_layer_name, gradcam_heatmap, load_audio_wave
    )

# Fallback para Streamlit antiguos sin st.toggle
def ui_toggle(label, default=False, key=None):
    if hasattr(st, "toggle"):
        return st.toggle(label, value=default, key=key)
    else:
        return st.checkbox(label, value=default, key=key)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎛️ Configuración")
    default_model_path = "models/fabrik_makina_newstyle.h5"
    use_repo_model = ui_toggle("Usar modelo del repo", default=os.path.exists(default_model_path), key="use_repo")
    uploaded_model = st.file_uploader("O sube un modelo (.h5)", type=["h5"], accept_multiple_files=False)
    st.markdown("---")
    seg_len = st.slider("Duración segmento (s)", 5, 15, 10)
    hop_len = st.slider("Salto entre segmentos (s)", 1, seg_len, seg_len)
    st.caption("Para 'casi tiempo real', usa segmentación de 5–10 s y hop pequeño (1–2 s).")

# Carga del modelo (cacheada)
@st.cache_resource(show_spinner=True)
def load_model(path=None, uploaded_bytes=None):
    import tempfile
    if path and os.path.exists(path):
        return tf.keras.models.load_model(path)
    if uploaded_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_bytes)
            tmp.flush()
            return tf.keras.models.load_model(tmp.name)
    return None

if use_repo_model and os.path.exists(default_model_path):
    model = load_model(path=default_model_path)
elif uploaded_model is not None:
    model = load_model(uploaded_bytes=uploaded_model.read())
else:
    model = None

st.title("🎧 Fabrik DJ Style Classifier — **Pro**")
st.markdown("Identifica **Makina** vs **Newstyle** con explicaciones visuales: **mel-espectrograma**, **Grad-CAM**, **timeline por segmentos** y **métricas de audio**.")

if model is None:
    st.warning("Sube un modelo `.h5` o añade `models/fabrik_makina_newstyle.h5` al repo.")
    st.stop()

# ---------- Carga de audio ----------
colA, colB = st.columns([2, 1])
with colA:
    audio_file = st.file_uploader("Sube un audio (WAV recomendado; MP3/OGG/M4A suelen funcionar)", type=["wav", "mp3", "ogg", "m4a"])
with colB:
    st.info("💡 Consejo: si el archivo es largo, la app lo segmenta automáticamente y promedia probabilidades. Puedes regular **segmento** y **salto** en la barra lateral.")

if not audio_file:
    st.stop()

st.audio(audio_file)

# Guardar temporalmente
import tempfile
with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
    tmp.write(audio_file.read())
    tmp_path = tmp.name

# ---------- Inferencia ----------
# 1) Segmentación (model-ready windows) y tiempos
windows, starts_s = segment_long_audio_for_model(tmp_path, segment_seconds=seg_len, hop_seconds=hop_len)

# 2) Predicción por ventanas o clip único
if len(windows) == 0:
    x = file_to_model_input(tmp_path)
    probs = model.predict(x, verbose=0)[0]
    probs_per_window = np.expand_dims(probs, 0)
    starts_s = np.array([0.0])
else:
    probs_list = [model.predict(w, verbose=0)[0] for w in windows]
    probs_per_window = np.stack(probs_list, axis=0)  # (num_windows, 2)

mean_probs = probs_per_window.mean(axis=0)
pred_idx = int(np.argmax(mean_probs))
pred_label = LABELS[pred_idx]
pred_conf = float(mean_probs[pred_idx])

# ---------- Cabecera de resultado ----------
met1, met2, met3 = st.columns(3)
with met1:
    st.metric("Predicción", f"{pred_label.upper()}")
with met2:
    st.metric("Confianza", f"{pred_conf:.2%}")
with met3:
    st.metric("Segmentos analizados", f"{len(probs_per_window)}")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- Pestañas visuales ----------
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Explicación & Evidencias", "📈 Timeline de segmentos", "🎨 Espectrograma & Grad-CAM", "🧪 Métricas de audio"])

# Tab 1: Explicación natural + evidencias
with tab1:
    st.subheader("🧠 ¿Por qué el modelo decide esto?")
    col1, col2 = st.columns([2, 1])

    # Texto explicativo con evidencias
    with col1:
        # Elegimos el segmento más seguro (máx. confianza)
        best_i = int(np.argmax(probs_per_window[:, pred_idx])) if len(probs_per_window) > 1 else 0
        best_start = starts_s[best_i] if len(starts_s) else 0.0
        best_prob = float(probs_per_window[best_i, pred_idx])

        st.write(
            f"- El modelo predice **{pred_label}** con **{pred_conf:.1%}** de confianza promedio.\n"
            f"- La mayor evidencia aparece en el segmento **#{best_i}** (t ≈ **{best_start:.1f}s**), "
            f"con confianza **{best_prob:.1%}** para **{pred_label}**.\n"
            f"- A continuación puedes ver el **mel-espectrograma** y una **mapa de atención (Grad-CAM)** que muestra las zonas de tiempo-frecuencia que más han contribuido."
        )

    # Mini tabla con top-3 segmentos
    with col2:
        if len(probs_per_window) > 1:
            top_idx = np.argsort(-probs_per_window[:, pred_idx])[:3]
            df_top = pd.DataFrame({
                "segmento": top_idx,
                "t_inicio (s)": starts_s[top_idx] if len(starts_s) else np.zeros_like(top_idx, dtype=float),
                f"p({pred_label})": probs_per_window[top_idx, pred_idx]
            })
            st.markdown("**Top-3 segmentos más convincentes**")
            st.dataframe(df_top, use_container_width=True, hide_index=True)

# Tab 2: Timeline por segmentos
with tab2:
    st.subheader("📈 Probabilidades por segmento")
    timeline_df = pd.DataFrame({
        "t_inicio_s": starts_s,
        LABELS[0]: probs_per_window[:, 0],
        LABELS[1]: probs_per_window[:, 1],
    })
    st.line_chart(timeline_df.set_index("t_inicio_s"))
    st.caption("Cada punto es un segmento de longitud fija. El promedio de toda la serie produce la predicción global.")

    csv = timeline_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV de probabilidades", data=csv, file_name="segment_probs.csv")

# Tab 3: Espectrograma & Grad-CAM
with tab3:
    st.subheader("🎨 Evidencia visual: mel-espectrograma + Grad-CAM")
    # Cargar audio y figura de mel del clip completo (hasta ~seg_len si es muy largo)
    y, sr = load_audio_wave(tmp_path, sr=SR)
    # Si el audio es largo, mostramos solo los primeros seg_len segundos para un mel global rápido
    max_samples = int(seg_len * sr)
    y_view = y[:max_samples] if len(y) > max_samples else y

    # Mel global (vista rápida)
    fig_mel = mel_figure(y_view, sr=sr, n_mels=N_MELS, title="Mel-espectrograma (vista parcial)")
    st.pyplot(fig_mel, use_container_width=True)

    # Grad-CAM sobre el segmento más convincente
    st.markdown("**Mapa de atención (Grad-CAM) sobre el segmento más representativo**")
    # Preparamos entrada de ese segmento concreto para Grad-CAM
    if len(windows) == 0:
        x_for_cam = file_to_model_input(tmp_path)
    else:
        x_for_cam = windows[best_i]  # (1,128,431,1)

    # Nombre de la última capa conv
    conv_name = last_conv_layer_name(model)
    if conv_name is None:
        st.warning("No se encontró una capa Conv2D en el modelo para Grad-CAM.")
    else:
        heat = gradcam_heatmap(model, x_for_cam, conv_name, class_index=pred_idx, upsample_to=(N_MELS, FIXED_TIME_FRAMES))
        fig_cam = overlay_gradcam_on_mel(x_for_cam[0, :, :, 0], heat, labels=LABELS, pred_idx=pred_idx)
        st.pyplot(fig_cam, use_container_width=True)
        st.caption("Las zonas en rojo/amarillo indican regiones tiempo-frecuencia que más contribuyeron a la clase predicha.")

# Tab 4: Métricas de audio (BPM, centroide, rolloff…)
with tab4:
    st.subheader("🧪 Métricas de audio")
    feats = compute_basic_features(y, sr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BPM estimado", f"{feats['bpm']:.1f}")
    c2.metric("Centroide espectral (Hz)", f"{feats['spectral_centroid_hz']:.0f}")
    c3.metric("Rolloff 85% (Hz)", f"{feats['spectral_rolloff_hz']:.0f}")
    c4.metric("Tasa de cruces por cero", f"{feats['zcr']:.3f}")
    st.markdown(
        "<div class='small'>Estas métricas no determinan por sí solas la clase, "
        "pero ayudan a entender el carácter del audio (tempo, brillo, contenido armónico, etc.).</div>",
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("© Fabrik DJ Style Classifier — Demo educativa. Para mejores resultados, entrena con más horas por estilo y valida por sesión/tema.")
