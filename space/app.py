"""Gradio app for the Singapore food classifier (runs locally and on Hugging Face Spaces)."""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
IMG_SIZE = 224

model = tf.keras.models.load_model(HERE / "model.keras")
labels = json.loads((HERE / "labels.json").read_text())


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def predict(image):
    if image is None:
        return {}
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(np.asarray(image, dtype=np.float32), 0)  # raw 0-255; model does its own preprocessing
    probs = model.predict(x, verbose=0)[0]
    return {pretty(labels[i]): float(p) for i, p in enumerate(probs)}


examples_dir = HERE / "examples"
examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.exists() else None

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo of Singapore food"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Singapore Food Classifier",
    description=(
        "Identifies 10 Singapore hawker dishes: " + ", ".join(pretty(l) for l in labels) + ". "
        "Built with TensorFlow transfer learning (EfficientNetV2-B0)."
    ),
    examples=examples,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
