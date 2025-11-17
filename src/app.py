# app.py
# -----------------------------------------------------------
# 🎧 Fabrik DJ Style Classifier — Pro (Makina vs Newstyle)
# -----------------------------------------------------------

import os, sys, io, zipfile, tempfile, requests, traceback
from pathlib import Path

# Import path para "src/"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# Estilos
st.set_page_config(page_title="Fabrik DJ Style Classifier", page_icon="🎛️", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans'; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.1rem; }
div.stButton > button, .stDownloadButton > button { border-radius: 12px; padding: 0.55rem 1rem; }
[data-testid="stMetricValue"] { font-weight: 800; }
.hr { border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent); margin: .6rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Utilidades (desde src/audio_utils.py). Fallback al mismo nivel si no existe src/.
try:
    from src.audio_utils import (
        SR, N_MELS, FIXED_TIME_FRAMES, LABELS,
        file_to_model_input, segment_long_audio_for_model,
        mel_figure, overlay_gradcam_on_mel, compute_basic_features,
        last_conv_layer_name, gradcam_heatmap, load_audio_wave,
        generate_demo_wave, safe_load_path
    )
except ModuleNotFoundError:
    from audio_utils import (
        SR, N_MELS, FIXED_TIME_FRAMES, LABELS,
        file_to_model_input, segment_long_audio_for_model,
        mel_figure, overlay_gradcam_on_mel, compute_basic_features,
        last_conv_layer_name, gradcam_heatmap, load_audio_wave,
        generate_demo_wave, safe_load_path
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
    uploaded_model = st.file_uploader(
        "O sube un **modelo Keras** (.h5 o .keras, NO audio .h5)",
        type=["h5","keras"], accept_multiple_files=False
    )
    st.markdown("---")
    seg_len = st.slider("Duración segmento (s)", 5, 15, 10)
    hop_len = st.slider("Salto entre segmentos (s)", 1, seg_len, seg_len)
    debug_mode = ui_toggle("Modo depuración (mostrar detalles de archivo/errores)", default=False, key="debug")
    st.caption("Para menos latencia: 5–10 s de segmento y hop de 1–2 s.")

# ---------- Carga del modelo (con validación de .h5) ----------
@st.cache_resource(show_spinner=True)
def _load_model_from_path(path: str):
    import h5py
    # Si parece un HDF5 de audio, avisar
    try:
        with h5py.File(path, "r") as f:
            if "waveform" in f.keys() and "sr" in f.keys():
                raise ValueError("El archivo .h5 parece un AUDIO (datasets 'waveform'/'sr'), no un modelo Keras.")
    except OSError:
        # No es HDF5 (puede ser .keras o SavedModel empaquetado) → deja a Keras decidir
        pass
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=True)
def _load_model_from_bytes(uploaded_bytes: bytes, orig_name: str):
    import h5py
    suffix = Path(orig_name).suffix or ".h5"  # conserva extensión original (.keras/.h5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_bytes)
        tmp.flush()
        tmp_path = tmp.name
    # Validación de audio .h5
    try:
        with h5py.File(tmp_path, "r") as f:
            if "waveform" in f.keys() and "sr" in f.keys():
                raise ValueError("Has subido un .h5 de audio. Sube un modelo Keras guardado con model.save(...).")
    except OSError:
        pass
    return tf.keras.models.load_model(tmp_path)

# Decide fuente del modelo con manejo de errores amigable
model = None
try:
    if use_repo_model and os.path.exists(default_model_path):
        model = _load_model_from_path(default_model_path)
    elif uploaded_model is not None:
        model = _load_model_from_bytes(uploaded_model.getvalue(), uploaded_model.name)
except Exception as e:
    st.error(
        "❌ No se pudo cargar el modelo.\n\n"
        "Asegúrate de subir un **modelo Keras** válido (guardado con `model.save('…')`).\n\n"
        f"Detalle: {e}"
    )
    st.stop()

st.title("🎧 Fabrik DJ Style Classifier — **Pro**")
st.markdown("Identifica **Makina** vs **Newstyle** con mel, **Grad-CAM**, timeline por segmentos y métricas.")

if model is None:
    st.warning("Sube un modelo `.h5`/`.keras` o añade `models/fabrik_makina_newstyle.h5` al repo y activa 'Usar modelo del repo'.")
    st.stop()

# ---------- Entrada de audio: Archivo / URL / Demo interna (AJUSTE ROBUSTO) ----------
# Preselecciona demo (index=2). Cambia a index=0 si prefieres “Subir archivo”.
input_mode = st.radio(
    "Cómo quieres cargar el audio:",
    ["Subir archivo", "Pegar URL", "Usar demo (5 s)"],
    horizontal=True, index=2
)
tmp_path = None

# Formatos soportados SIN ffmpeg (Streamlit Cloud)
LOSSY_BLOCK = {".mp3", ".m4a", ".aac"}  # suelen fallar en cloud sin ffmpeg
LOSSLESS_OK = {".wav", ".wave", ".flac", ".ogg", ".oga", ".aiff", ".aif"}

def save_temp_bytes(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return tmp.name

try:
    if input_mode == "Subir archivo":
        file = st.file_uploader(
            "Sube un audio (WAV/FLAC/OGG/AIFF recomendados) o un ZIP con audio. MP3/M4A/AAC no soportados en este entorno.",
            type=None, accept_multiple_files=False
        )
        if not file:
            st.stop()

        raw = file.getvalue()  # lee UNA vez
        name = file.name or "audio.bin"
        ext = Path(name).suffix.lower()

        if ext == ".zip":
            try:
                z = zipfile.ZipFile(io.BytesIO(raw))
                member = None
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    ext_in = Path(info.filename).suffix.lower()
                    if ext_in in LOSSLESS_OK:
                        member = info
                        break
                if member is None:
                    raise ValueError("ZIP sin audios compatibles (usa WAV/FLAC/OGG/AIFF).")
                data = z.read(member)
                ext2 = Path(member.filename).suffix.lower()
                tmp_path = save_temp_bytes(data, ext2)
            except Exception as e:
                if debug_mode: st.exception(e)
                st.error(f"No se pudo leer el ZIP: {e}")
                st.stop()
        else:
            if ext in LOSSY_BLOCK:
                st.error("Este entorno no puede decodificar MP3/M4A/AAC. Convierte a WAV/FLAC/OGG/AIFF e inténtalo de nuevo.")
                st.stop()
            if ext not in LOSSLESS_OK:
                st.error(f"Formato no soportado ({ext or 'desconocido'}). Usa WAV/FLAC/OGG/AIFF o ZIP con audio.")
                st.stop()
            tmp_path = save_temp_bytes(raw, ext)

        try:
            st.audio(tmp_path)
        except Exception as e:
            if debug_mode: st.warning(f"Player no disponible: {e}")

        if debug_mode:
            st.write({"name": name, "ext": ext, "bytes": len(raw)})
            import soundfile as sf, librosa
            try:
                y_sf, sr_sf = sf.read(tmp_path, always_2d=False)
                st.success(f"[soundfile] OK — sr={sr_sf}, shape={np.shape(y_sf)}")
            except Exception as e:
                st.error(f"[soundfile] ERROR: {e}")
            try:
                y_lb, sr_lb = librosa.load(tmp_path, sr=SR, mono=True)
                st.success(f"[librosa] OK — sr={sr_lb}, len={len(y_lb)}")
            except Exception as e:
                st.error(f"[librosa] ERROR: {e}")

    elif input_mode == "Pegar URL":
        url = st.text_input("Pega una URL directa (audio o ZIP con audio):")
        if not url:
            st.stop()
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ext = Path(url.split("?")[0]).suffix.lower()

        if ext == ".zip":
            z = zipfile.ZipFile(io.BytesIO(r.content))
            member = None
            for info in z.infolist():
                if info.is_dir():
                    continue
                ext_in = Path(info.filename).suffix.lower()
                if ext_in in LOSSLESS_OK:
                    member = info
                    break
            if member is None:
                st.error("ZIP sin audios compatibles (usa WAV/FLAC/OGG/AIFF).")
                st.stop()
            data = z.read(member)
            ext2 = Path(member.filename).suffix.lower()
            tmp_path = save_temp_bytes(data, ext2)
        else:
            if ext in LOSSY_BLOCK:
                st.error("Este entorno no puede decodificar MP3/M4A/AAC desde URL. Usa WAV/FLAC/OGG/AIFF o ZIP.")
                st.stop()
            if ext not in LOSSLESS_OK:
                st.error(f"Extensión no soportada en URL ({ext or 'desconocida'}). Usa WAV/FLAC/OGG/AIFF o ZIP.")
                st.stop()
            tmp_path = save_temp_bytes(r.content, ext)

        try:
            st.audio(tmp_path)
        except Exception as e:
            if debug_mode: st.warning(f"Player no disponible: {e}")

        if debug_mode:
            st.write({"url": url, "ext": ext, "bytes": len(r.content)})

    else:  # Demo interna 5 s
        y = generate_demo_wave(SR, 5.0)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, y, SR, format='WAV', subtype='PCM_16')
            tmp_path = tmp.name
        st.audio(tmp_path)

except Exception as up_e:
    if debug_mode: st.exception(up_e)
    st.error("No se pudo recibir el audio. Prueba la **Demo (5 s)** o una **URL/ZIP** con WAV/FLAC/OGG/AIFF.")
    st.stop()

# ---------- Inferencia ----------
try:
    windows, starts_s = segment_long_audio_for_model(tmp_path, segment_seconds=seg_len, hop_seconds=hop_len)

    if len(windows) == 0:
        x = file_to_model_input(tmp_path)
        probs = model.predict(x, verbose=0)[0]
        probs_per_window = np.expand_dims(probs, 0)
        starts_s = np.array([0.0])
    else:
        probs_list = [model.predict(w, verbose=0)[0] for w in windows]
        probs_per_window = np.stack(probs_list, axis=0)

    mean_probs = probs_per_window.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_label = LABELS[pred_idx]
    pred_conf = float(mean_probs[pred_idx])

except Exception as inf_e:
    if debug_mode:
        st.error("❌ Error en inferencia / lectura del audio:")
        st.exception(inf_e)
        st.code(traceback.format_exc())
    else:
        st.error("❌ Error procesando el audio. Activa 'Modo depuración' en la barra lateral para ver detalles.")
    st.stop()

# Header
m1, m2, m3 = st.columns(3)
m1.metric("Predicción", pred_label.upper())
m2.metric("Confianza", f"{pred_conf:.2%}")
m3.metric("Segmentos", f"{len(probs_per_window)}")
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Explicación & evidencias", "📈 Timeline", "🎨 Mel & Grad-CAM", "🧪 Métricas"])

with tab1:
    st.subheader("🧠 ¿Por qué esta decisión?")
    best_i = int(np.argmax(probs_per_window[:, pred_idx])) if len(probs_per_window) > 1 else 0
    best_start = float(starts_s[best_i]) if len(starts_s) else 0.0
    best_prob = float(probs_per_window[best_i, pred_idx])

    c1, c2 = st.columns([2,1])
    with c1:
        st.write(
            f"- Predicción global: **{pred_label}** (**{pred_conf:.1%}**).\n"
            f"- Segmento más convincente: **#{best_i}** (t ≈ **{best_start:.1f}s**), "
            f"confianza **{best_prob:.1%}**.\n"
            f"- Debajo verás mel-espectrograma y **Grad-CAM** resaltando qué zonas tiempo-frecuencia pesaron más."
        )
    with c2:
        if len(probs_per_window) > 1:
            top_idx = np.argsort(-probs_per_window[:, pred_idx])[:3]
            df_top = pd.DataFrame({
                "segmento": top_idx,
                "t_inicio (s)": starts_s[top_idx] if len(starts_s) else np.zeros_like(top_idx, dtype=float),
                f"p({pred_label})": probs_per_window[top_idx, pred_idx]
            })
            st.markdown("**Top-3 segmentos**")
            st.dataframe(df_top, hide_index=True, use_container_width=True)

with tab2:
    st.subheader("📈 Probabilidades por segmento")
    # Dinámico para N clases
    timeline_df = pd.DataFrame({"t_inicio_s": starts_s})
    for i, lab in enumerate(LABELS):
        timeline_df[lab] = probs_per_window[:, i]
    st.line_chart(timeline_df.set_index("t_inicio_s"))
    st.caption("Se promedian las probabilidades para la predicción global.")
    st.download_button("⬇️ CSV de probabilidades", timeline_df.to_csv(index=False).encode("utf-8"),
                       file_name="segment_probs.csv")

with tab3:
    st.subheader("🎨 Mel-espectrograma & Grad-CAM")
    y, sr = load_audio_wave(tmp_path, sr=SR)
    max_samples = int(seg_len * sr)
    y_view = y[:max_samples] if len(y) > max_samples else y
    fig_mel = mel_figure(y_view, sr=sr, n_mels=N_MELS, title="Mel-espectrograma (vista parcial)")
    st.pyplot(fig_mel, use_container_width=True)

    st.markdown("**Mapa de atención (Grad-CAM)** del segmento más representativo")
    x_for_cam = file_to_model_input(tmp_path) if len(windows) == 0 else windows[best_i]
    conv_name = last_conv_layer_name(model)
    if conv_name is None:
        st.warning("No se encontró una capa Conv2D para aplicar Grad-CAM.")
    else:
        heat = gradcam_heatmap(model, x_for_cam, conv_name, class_index=pred_idx, upsample_to=(N_MELS, FIXED_TIME_FRAMES))
        fig_cam = overlay_gradcam_on_mel(x_for_cam[0, :, :, 0], heat, labels=LABELS, pred_idx=pred_idx)
        st.pyplot(fig_cam, use_container_width=True)
        st.caption("Rojo/amarillo = mayor contribución a la clase predicha.")

with tab4:
    st.subheader("🧪 Métricas básicas")
    feats = compute_basic_features(y, sr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BPM estimado", f"{feats['bpm']:.1f}")
    c2.metric("Centroide espectral (Hz)", f"{feats['spectral_centroid_hz']:.0f}")
    c3.metric("Rolloff 85% (Hz)", f"{feats['spectral_rolloff_hz']:.0f}")
    c4.metric("ZCR", f"{feats['zcr']:.3f}")
    st.caption("No determinan por sí solas la clase, pero ayudan a entender el carácter del audio.")
