import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from utils import (
    load_model,
    predict_image,
    format_nominal,
    speak,
    add_history
)

st.set_page_config(
    page_title="Deteksi Nominal Uang Rupiah",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS

with open("ui.css", encoding="utf-8") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

@st.cache_resource
def load_cnn():
    return load_model()

model, class_names = load_cnn()


# SESSION STATE

default_state = {
    "image": None,
    "camera_frame": None,
    "prediction": None,
    "confidence": 0,
    "probability": None,
    "inference": 0,
    "history": []
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# SIDEBAR

with st.sidebar:

    st.markdown("# 💵 Deteksi Uang")
    st.caption("CNN Classification")

    st.divider()

    mode = st.radio(
        "Mode",
        [
            "📁 Upload Gambar",
            "📷 Webcam"
        ]
    )

    st.divider()

    st.subheader("💰 Kelas")

    for cls in class_names:
        st.write("•", format_nominal(cls))


# HEADER

st.markdown("""
<div class="hero">

<h1>Deteksi Nominal Uang Rupiah</h1>

<p>
Aplikasi klasifikasi nominal uang Rupiah menggunakan
Convolutional Neural Network (CNN)
</p>

</div>
""", unsafe_allow_html=True)

top_left, top_right = st.columns(
    [1.5, 1],
    gap="large"
)


# PANEL KIRI

with top_left:

    st.markdown("## 📷 Input Gambar")


    # UPLOAD GAMBAR

    if mode == "📁 Upload Gambar":

        uploaded_file = st.file_uploader(
            "Upload gambar uang Rupiah",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file is not None:

            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            st.session_state.image = image


    # WEBCAM    

    else:

        class Camera(VideoProcessorBase):
            def __init__(self):
                self.frame = None

            def recv(self, frame):
                self.frame = frame.to_ndarray(format="bgr24")

                return frame

        ctx = webrtc_streamer(
            key="camera",
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            video_processor_factory=Camera,
            async_processing=True
        )

        st.info("Tekan tombol **Gunakan Foto** untuk mengambil frame dari kamera.")

        if st.button(
            "📸 Gunakan Foto",
            use_container_width=True
        ):

            if (
                ctx.video_processor is not None
                and ctx.video_processor.frame is not None):

                st.session_state.image = ctx.video_processor.frame.copy()
                st.success("Foto berhasil diambil.")

            else:

                st.warning("Kamera belum siap.")


    # PREVIEW 

    if st.session_state.image is not None:

        st.markdown("---")

        st.markdown("## 🖼️ Preview")

        preview = cv2.cvtColor(
            st.session_state.image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            preview,
            use_container_width=True
        )

        st.markdown("")

        if st.button(
            "🔍 Prediksi Nominal",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Sedang melakukan prediksi..."):

                (
                    prediction,
                    confidence,
                    probability,
                    inference
                ) = predict_image(
                    model,
                    st.session_state.image,
                    class_names
                )

                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.probability = probability
                st.session_state.inference = inference

                st.session_state.history = add_history(
                    st.session_state.history,
                    prediction,
                    confidence
                )

                speak(prediction)

                st.rerun()


# PANEL KANAN

with top_right:

    st.markdown("## 🤖 Hasil Prediksi")

    if st.session_state.prediction is None:

        st.info("Belum ada hasil prediksi.")

    else:

        nominal = format_nominal(st.session_state.prediction)
        confidence = st.session_state.confidence
        inference = st.session_state.inference
        probability = st.session_state.probability


        # HASIL PREDIKSI

        st.metric(
            label="💵 Nominal Terdeteksi",
            value=nominal
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                label="Inference",
                value=f"{inference:.2f} ms"
            )

        st.markdown("### 🎯 Tingkat Keyakinan")

        st.progress(confidence / 100)

        st.caption(f"{confidence:.2f}%")