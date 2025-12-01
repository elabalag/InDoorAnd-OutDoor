import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "indoor_outdoor_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

IMG_SIZE = (224, 224)


def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_image(img_path):
    model, class_names = load_model_and_classes()
    x = preprocess_image(img_path)

    preds = model.predict(x)[0][0]  # scalar 0–1

    if preds < 0.5:
        predicted_index = 0
        confidence = 1.0 - float(preds)
    else:
        predicted_index = 1
        confidence = float(preds)

    predicted_label = class_names[predicted_index]
    return predicted_label, confidence


if __name__ == "__main__":
    test_image_path = input("Enter image path: ").strip()
    label, conf = predict_image(test_image_path)
    print(f"Predicted label: {label}")
    print(f"Confidence: {conf:.2f}")
