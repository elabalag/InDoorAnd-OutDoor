import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "indoor_outdoor_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def preprocess_image_pil(pil_img):
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_image_pil(pil_img):
    model, class_names = load_model_and_classes()
    x = preprocess_image_pil(pil_img)
    preds = model.predict(x)[0][0]

    if preds < 0.5:
        predicted_index = 0
        confidence = 1.0 - float(preds)
    else:
        predicted_index = 1
        confidence = float(preds)

    predicted_label = class_names[predicted_index]
    return predicted_label, confidence


def main():
    st.title("Drone Indoor/Outdoor Classifier")
    st.write(
        "Upload an image and the model will predict whether it is "
        "**indoor** or **outdoor** with a confidence score."
    )

    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            label, conf = predict_image_pil(img)
            st.markdown(f"**Predicted label:** {label}")
            st.markdown(f"**Confidence:** {conf:.2f}")


if __name__ == "__main__":
    main()
