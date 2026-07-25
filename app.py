"""Flask REST API for ChilliGuard red chilli powder adulteration detection."""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"
MODEL_STATS_PATH = BASE_DIR / "model_stats.json"

IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
STATIC_OUTPUTS = {
    "confusion_matrix.png",
    "training_curves.png",
    "sample_predictions.png",
    "pca_visualization.png",
}

app = Flask(__name__, static_folder=None)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


model = None
model_load_error: str | None = None
class_names: list[str] = []
label_map: dict[str, str] = {}


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_artifacts() -> None:
    global model, model_load_error, class_names, label_map
    class_names = load_json_file(CLASS_NAMES_PATH, [])
    label_map = load_json_file(LABEL_MAP_PATH, {})
    try:
        if not MODEL_PATH.exists():
            model_load_error = f"Model file not found: {MODEL_PATH.name}"
            model = None
            return
        model = load_model(MODEL_PATH)
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = str(exc)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def error_response(message: str, status_code: int, **details):
    payload = {"error": message}
    payload.update(details)
    return jsonify(payload), status_code


def preprocess_image(file_storage) -> np.ndarray:
    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def parse_adulteration_level(class_name: str) -> int:
    normalized = class_name.lower()
    if "brick" in normalized and "reference" in normalized:
        return -1
    if "pure chilli" in normalized:
        return 0
    match = re.search(r"(\d+)\s*%", class_name)
    if match:
        return int(match.group(1))
    return 0


def risk_details(class_name: str) -> dict[str, Any]:
    mapping = {
        "Pure Chilli": {
            "risk_level": "Safe",
            "risk_color": "#22c55e",
            "health_advisory": "Sample appears genuine. Safe for consumption.",
            "is_safe": True,
            "recommendation": "No action needed. Product meets safety standards.",
        },
        "Low Adulteration": {
            "risk_level": "Low",
            "risk_color": "#eab308",
            "health_advisory": "Low level brick powder adulteration detected. Caution advised.",
            "is_safe": False,
            "recommendation": "Avoid regular consumption. Source from a certified supplier.",
        },
        "Moderate Adulteration": {
            "risk_level": "Moderate",
            "risk_color": "#f97316",
            "health_advisory": "Moderate adulteration detected. Not recommended for consumption.",
            "is_safe": False,
            "recommendation": "Do not consume. Report to local food safety authority.",
        },
        "High Adulteration": {
            "risk_level": "High",
            "risk_color": "#ef4444",
            "health_advisory": "High level adulteration detected. Seriously unsafe.",
            "is_safe": False,
            "recommendation": "Discard immediately. File a complaint with FSSAI.",
        },
        "Pure Brick (Reference)": {
            "risk_level": "Extreme",
            "risk_color": "#7f1d1d",
            "health_advisory": "This sample appears to be pure adulterant with no chilli powder.",
            "is_safe": False,
            "recommendation": "Do not consume under any circumstances.",
        },
    }
    return mapping.get(class_name, {
        "risk_level": "Unknown",
        "risk_color": "#6b7280",
        "health_advisory": "Unable to determine risk category.",
        "is_safe": False,
        "recommendation": "Verify sample through laboratory testing.",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return error_response("Model not loaded", 503, model_loaded=False, details=model_load_error)
    if not class_names:
        return error_response("Class names not loaded", 503, model_loaded=True)
    if "image" not in request.files:
        return error_response("Missing file field 'image'", 400)

    image_file = request.files["image"]
    if image_file.filename == "":
        return error_response("No image file selected", 400)
    if not allowed_file(image_file.filename):
        return error_response("Unsupported file type", 400, allowed_extensions=sorted(ALLOWED_EXTENSIONS))

    try:
        processed_image = preprocess_image(image_file)
    except UnidentifiedImageError:
        return error_response("Uploaded file is not a valid image", 400)
    except Exception as exc:
        return error_response("Image preprocessing failed", 400, details=str(exc))

    try:
        probabilities = model.predict(processed_image, verbose=0)[0]
    except Exception as exc:
        return error_response("Model prediction failed", 500, details=str(exc))

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    all_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(class_names)
    }

    CONFIDENCE_THRESHOLD = 0.70
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "predicted_class": "Invalid Sample",
            "confidence": confidence,
            "adulteration_level": -1,
            "is_safe": None,
            "risk_level": "Unknown",
            "risk_color": "#6b7280",
            "all_probabilities": all_probabilities,
            "health_advisory": "The uploaded image does not appear to be a red chilli powder sample. Please upload a close-up photograph of powder on a flat surface.",
            "recommendation": "Ensure the image shows only red chilli powder under good lighting.",
            "adulterant": "N/A",
            "valid_sample": False
        })

    adulteration_level = parse_adulteration_level(predicted_class)
    risk = risk_details(predicted_class)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "adulteration_level": adulteration_level,
        "is_safe": risk["is_safe"],
        "risk_level": risk["risk_level"],
        "risk_color": risk["risk_color"],
        "all_probabilities": all_probabilities,
        "health_advisory": risk["health_advisory"],
        "recommendation": risk["recommendation"],
        "adulterant": "Brick Powder",
        "valid_sample": True
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    if not MODEL_STATS_PATH.exists():
        return error_response("model_stats.json not found", 404)
    with MODEL_STATS_PATH.open("r", encoding="utf-8") as file:
        return jsonify(json.load(file))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/static/<path:filename>", methods=["GET"])
def static_outputs(filename: str):
    if filename not in STATIC_OUTPUTS:
        return error_response("Static file is not allowed", 404)
    file_path = BASE_DIR / filename
    if not file_path.exists():
        return error_response("Static file not found", 404, filename=filename)
    return send_from_directory(str(BASE_DIR), filename)


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    load_artifacts()