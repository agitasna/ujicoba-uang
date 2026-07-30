import time
import cv2
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components

MODEL_PATH = "model/model_uang.keras"
LABEL_PATH = "model/labels.txt"

def load_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names


# PREPROCESS

def preprocess_image(image, img_size=(224, 224)):

    # BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32)

    # RESCALING
    image = np.expand_dims(image, axis=0)

    return image


# PREDKSI

def predict_image(model, image, class_names):
   
    image_input = preprocess_image(image)

    start = time.perf_counter()

    prediction = model.predict(
        image_input,
        verbose=0
    )[0]

    end = time.perf_counter()
    inference = (end - start) * 1000
    class_index = np.argmax(prediction)
    label = class_names[class_index]
    confidence = float(prediction[class_index]) * 100
    probability = prediction * 100

    return (
        label,
        confidence,
        probability,
        inference
    )

def format_nominal(label):
    mapping = {
        "1rb": "Rp1.000",
        "2rb": "Rp2.000",
        "5rb": "Rp5.000",
        "10rb": "Rp10.000",
        "20rb": "Rp20.000",
        "50rb": "Rp50.000",
        "100rb": "Rp100.000"
    }

    return mapping.get(label, label)


# TEXT TO SPEECH

def speak(label):
    speech = {
        "1rb": "Seribu rupiah",
        "2rb": "Dua ribu rupiah",
        "5rb": "Lima ribu rupiah",
        "10rb": "Sepuluh ribu rupiah",
        "20rb": "Dua puluh ribu rupiah",
        "50rb": "Lima puluh ribu rupiah",
        "100rb": "Seratus ribu rupiah"
    }

    text = speech.get(label, label)

    components.html(
        f"""
        <script>
        window.speechSynthesis.cancel();
        let msg = new SpeechSynthesisUtterance("{text}");
        msg.lang = "id-ID";
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


def add_history(history, nominal, confidence):
    
    from datetime import datetime

    history.insert(
        0,
        {
            "Nominal": format_nominal(nominal),
            "Confidence": f"{confidence:.2f}%",
            "Waktu": datetime.now().strftime("%H:%M:%S")
        }
    )

    return history[:10]