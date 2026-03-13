import base64
import io
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google import genai
import pandas as pd
from ultralytics import YOLO

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "global")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
RSNA_LABEL_CSV = r"E:/Documents/GitHub/MedVision/data/stage_2_train_labels.csv"
RSNA_TRAIN_IMAGE_DIR = r"E:/Documents/GitHub/MedVision/data/stage_2_train_images"
rsna_df = pd.read_csv(RSNA_LABEL_CSV)

DETECTION_MODEL_PATH = os.getenv(
    "DETECTION_MODEL_PATH",
    r"E:/Documents/GitHub/MedVision/runs/detect/train2/weights/best.pt"
)
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.05"))
det_model = YOLO(DETECTION_MODEL_PATH)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATED_DIR = BASE_DIR / "annotated"

UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}

app = Flask(__name__)
CORS(app)


class AppConfigError(Exception):
    pass


class FileValidationError(Exception):
    pass


class DicomProcessingError(Exception):
    pass


class AIAnalysisError(Exception):
    pass


def validate_config():
    if not PROJECT_ID:
        raise AppConfigError("Missing GCP_PROJECT_ID in .env")
    if not REGION:
        raise AppConfigError("Missing GCP_REGION in .env")
    if not MODEL_NAME:
        raise AppConfigError("Missing MODEL_NAME in .env")


def get_client():
    validate_config()
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=REGION,
    )


def get_extension(filename: str) -> str:
    if not filename or "." not in filename:
        raise FileValidationError("Invalid filename.")
    return os.path.splitext(filename)[1].lower()


def validate_file(filename: str, file_bytes: bytes) -> str:
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if not file_bytes:
        raise FileValidationError("Uploaded file is empty.")
    return ext


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if max_val == min_val:
        raise DicomProcessingError("Image has constant pixel values.")

    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def dicom_to_pil(dicom_bytes: bytes) -> Image.Image:
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    except Exception as e:
        raise DicomProcessingError(f"Failed to read DICOM: {e}") from e

    if not hasattr(ds, "PixelData"):
        raise DicomProcessingError("DICOM file has no PixelData.")

    try:
        image = ds.pixel_array
    except Exception as e:
        raise DicomProcessingError(f"Failed to read pixel array: {e}") from e

    if image.ndim == 3:
        image = image[0]

    image_uint8 = normalize_to_uint8(image)
    pil_image = Image.fromarray(image_uint8).convert("RGB")
    return pil_image


def bytes_to_pil(filename: str, file_bytes: bytes) -> Image.Image:
    ext = validate_file(filename, file_bytes)

    if ext == ".dcm":
        return dicom_to_pil(file_bytes)

    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise FileValidationError(f"Failed to open image: {e}") from e


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_yolo_inference(image: Image.Image):
    image_np = np.array(image)

    results = det_model.predict(
        source=image_np,
        conf=DETECTION_CONF,
        verbose=False
    )

    findings = []

    if not results:
        return findings

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return findings

    names = result.names

    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        label = names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = [int(v) for v in xyxy]

        findings.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        })

    return findings


def get_rsna_findings_by_patient_id(patient_id: str):
    rows = rsna_df[(rsna_df["patientId"] == patient_id) & (rsna_df["Target"] == 1)]

    findings = []
    for _, row in rows.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])

        findings.append({
            "label": "pneumonia",
            "confidence": 1.0,
            "bbox": [x, y, x + w, y + h]
        })

    return findings


def draw_findings_on_image(image: Image.Image, findings):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for finding in findings:
        x1, y1, x2, y2 = finding["bbox"]
        label = finding["label"]
        conf = finding["confidence"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, max(0, y1 - 20)), f"{label} ({conf:.2f})", fill="red")

    return annotated


def build_gemini_prompt(findings):
    findings_json = json.dumps(findings, ensure_ascii=False)

    return f"""
You are a medical AI assistant helping radiologists with preliminary screening.

A trained object detection model has analyzed a chest X-ray and produced the following findings:

{findings_json}

Your task is to summarize these model findings cautiously for clinician review.

Return ONLY valid JSON, with no markdown fences and no extra text.

Required schema:
{{
  "abnormality": "string",
  "risk_level": "low | medium | high",
  "confidence": 0.0,
  "explanation": "string",
  "requires_clinician_review": true,
  "disclaimer": "This AI output is for preliminary screening only and must be reviewed by a qualified clinician."
}}

Rules:
- Base your answer on the detection findings and the image only.
- Do not claim a definitive diagnosis.
- Do not invent extra findings that are not supported by the detections.
- Be cautious and clinically conservative.
- Keep the explanation short and clear.
""".strip()


def generate_explanation_with_gemini(image: Image.Image, findings):
    client = get_client()
    prompt = build_gemini_prompt(findings)
    image_bytes = pil_to_png_bytes(image)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded_image,
                            }
                        },
                        {
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        raise AIAnalysisError(f"Gemini request failed: {e}") from e

    text = getattr(response, "text", None)
    if not text:
        raise AIAnalysisError("Gemini returned empty output.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise AIAnalysisError(f"Gemini returned non-JSON output: {text}") from e


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MedVision MVP API running",
        "project_id": PROJECT_ID,
        "region": REGION,
        "model_name": MODEL_NAME,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/annotated/<filename>", methods=["GET"])
def get_annotated_file(filename):
    return send_from_directory(ANNOTATED_DIR, filename)


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        if "image" not in request.files:
            raise FileValidationError("No file field named 'image' found.")

        uploaded_file = request.files["image"]

        if not uploaded_file.filename:
            raise FileValidationError("No file selected.")

        file_bytes = uploaded_file.read()
        original_image = bytes_to_pil(uploaded_file.filename, file_bytes)

        findings = run_yolo_inference(original_image)
        annotated_image = draw_findings_on_image(original_image, findings)

        file_id = uuid.uuid4().hex
        annotated_filename = f"{file_id}.png"
        annotated_path = ANNOTATED_DIR / annotated_filename
        annotated_image.save(annotated_path, format="PNG")

        if findings:
            gemini_result = generate_explanation_with_gemini(original_image, findings)
        else:
            gemini_result = {
                "abnormality": "No pneumonia detected by the current screening model",
                "risk_level": "low",
                "confidence": 0.0,
                "explanation": "The current detection model did not identify any pneumonia regions above the configured confidence threshold. This does not rule out disease and still requires clinician review.",
                "requires_clinician_review": True,
                "disclaimer": "This AI output is for preliminary screening only and must be reviewed by a qualified clinician."
            }

        return jsonify({
            "success": True,
            "project_id": PROJECT_ID,
            "region": REGION,
            "model_name": MODEL_NAME,
            "filename": uploaded_file.filename,
            "annotated_image_url": f"http://127.0.0.1:8080/annotated/{annotated_filename}",
            "findings": findings,
            "result": gemini_result,
        }), 200

    except FileValidationError as e:
        return jsonify({
            "success": False,
            "error_type": "FileValidationError",
            "error": str(e),
        }), 400

    except DicomProcessingError as e:
        return jsonify({
            "success": False,
            "error_type": "DicomProcessingError",
            "error": str(e),
        }), 400

    except AIAnalysisError as e:
        return jsonify({
            "success": False,
            "error_type": "AIAnalysisError",
            "error": str(e),
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "UnhandledException",
            "error": str(e),
        }), 500


@app.route("/demo-rsna/<patient_id>", methods=["GET"])
def demo_rsna(patient_id):
    print("=== demo_rsna HIT ===")
    print("patient_id:", repr(patient_id))

    try:
        dicom_path = os.path.join(RSNA_TRAIN_IMAGE_DIR, f"{patient_id}.dcm")
        print("dicom_path:", dicom_path)
        print("exists:", os.path.exists(dicom_path))

        with open(dicom_path, "rb") as f:
            dicom_bytes = f.read()

        image = bytes_to_pil(f"{patient_id}.dcm", dicom_bytes)
        findings = get_rsna_findings_by_patient_id(patient_id)
        print("findings count:", len(findings))
        print("findings:", findings)

        if not findings:
            return jsonify({
                "success": False,
                "error": f"No positive findings found for patient_id={patient_id}"
            }), 404

        annotated_image = draw_findings_on_image(image, findings)

        file_id = uuid.uuid4().hex
        annotated_filename = f"{file_id}.png"
        annotated_path = ANNOTATED_DIR / annotated_filename
        annotated_image.save(annotated_path, format="PNG")

        gemini_result = generate_explanation_with_gemini(image, findings)

        return jsonify({
            "success": True,
            "patient_id": patient_id,
            "annotated_image_url": f"http://127.0.0.1:8080/annotated/{annotated_filename}",
            "findings": findings,
            "result": gemini_result
        }), 200

    except Exception as e:
        print("EXCEPTION:", repr(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print(app.url_map)
    print(f"PROJECT_ID = {PROJECT_ID}")
    print(f"REGION = {REGION}")
    print(f"MODEL_NAME = {MODEL_NAME}")
    app.run(host="127.0.0.1", port=8080, debug=True)