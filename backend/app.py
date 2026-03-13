import io
import json
import os
from typing import Tuple

import numpy as np
import pydicom
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai


load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "global")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


class AppConfigError(Exception):
    pass


class FileValidationError(Exception):
    pass


class DicomProcessingError(Exception):
    pass


class AIAnalysisError(Exception):
    pass


def validate_config() -> None:
    if not PROJECT_ID:
        raise AppConfigError("Missing GCP_PROJECT_ID in .env")
    if not REGION:
        raise AppConfigError("Missing GCP_REGION in .env")
    if not MODEL_NAME:
        raise AppConfigError("Missing MODEL_NAME in .env")


def get_file_extension(filename: str) -> str:
    if not filename or "." not in filename:
        raise FileValidationError("Uploaded file must have a valid filename and extension.")
    return os.path.splitext(filename)[1].lower()


def validate_file(filename: str, file_bytes: bytes) -> str:
    ext = get_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type: {ext}. Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if not file_bytes:
        raise FileValidationError("Uploaded file is empty.")
    return ext


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise DicomProcessingError("DICOM image contains no pixel data.")

    image = image.astype(np.float32)

    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if max_val == min_val:
        raise DicomProcessingError("DICOM image has constant pixel values and cannot be normalized.")

    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def dicom_to_png_bytes(dicom_bytes: bytes) -> bytes:
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    except Exception as e:
        raise DicomProcessingError(f"Failed to read DICOM file: {e}") from e

    if not hasattr(ds, "PixelData"):
        raise DicomProcessingError("DICOM file does not contain PixelData.")

    try:
        image = ds.pixel_array
    except Exception as e:
        raise DicomProcessingError(f"Failed to extract pixel array from DICOM: {e}") from e

    try:
        # 如果是多通道或多帧，这里做一个最简单处理
        if image.ndim == 3:
            # 常见情况：取第一帧
            image = image[0]

        image_uint8 = normalize_to_uint8(image)
        pil_image = Image.fromarray(image_uint8)

        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        raise DicomProcessingError(f"Failed to convert DICOM to PNG: {e}") from e


def image_bytes_for_model(filename: str, file_bytes: bytes) -> Tuple[bytes, str]:
    ext = validate_file(filename, file_bytes)

    if ext == ".dcm":
        png_bytes = dicom_to_png_bytes(file_bytes)
        return png_bytes, "image/png"

    if ext == ".png":
        return file_bytes, "image/png"

    if ext in {".jpg", ".jpeg"}:
        return file_bytes, "image/jpeg"

    raise FileValidationError("Unsupported image type after validation.")


def get_genai_client() -> genai.Client:
    try:
        validate_config()
        return genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=REGION,
        )
    except Exception as e:
        raise AIAnalysisError(f"Failed to initialize Gemini client: {e}") from e


def build_prompt() -> str:
    return """
You are a medical AI assistant helping radiologists with preliminary screening.

Analyze the uploaded medical image and return ONLY valid JSON.
Do not include markdown fences. Do not include extra explanation outside JSON.

Required JSON schema:
{
  "abnormality": "string",
  "risk_level": "low | medium | high",
  "confidence": 0.0,
  "explanation": "string",
  "requires_clinician_review": true,
  "disclaimer": "This AI output is for preliminary screening only and must be reviewed by a qualified clinician."
}

Rules:
- If the image quality is insufficient, state that clearly in "explanation".
- Never claim a definitive diagnosis.
- Keep the explanation concise and clinically cautious.
""".strip()


def analyze_image_with_gemini(filename: str, file_bytes: bytes) -> dict:
    prepared_bytes, mime_type = image_bytes_for_model(filename, file_bytes)
    prompt = build_prompt()
    client = get_genai_client()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": prepared_bytes
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
        )
    except Exception as e:
        raise AIAnalysisError(f"Gemini request failed: {e}") from e

    text = getattr(response, "text", None)
    if not text:
        raise AIAnalysisError("Gemini returned an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise AIAnalysisError(f"Gemini returned non-JSON output: {text}") from e

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MedVision API running",
        "project_id": PROJECT_ID,
        "region": REGION,
        "model_name": MODEL_NAME,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route("/config-check", methods=["GET"])
def config_check():
    try:
        validate_config()
        return jsonify({
            "success": True,
            "project_id": PROJECT_ID,
            "region": REGION,
            "model_name": MODEL_NAME
        })
    except AppConfigError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image_api():
    try:
        if "image" not in request.files:
            raise FileValidationError("No file field named 'image' was found in the request.")

        uploaded_file = request.files["image"]

        if not uploaded_file or not uploaded_file.filename:
            raise FileValidationError("No file selected.")

        file_bytes = uploaded_file.read()

        result = analyze_image_with_gemini(
            filename=uploaded_file.filename,
            file_bytes=file_bytes
        )

        return jsonify({
            "success": True,
            "project_id": PROJECT_ID,
            "region": REGION,
            "model_name": MODEL_NAME,
            "filename": uploaded_file.filename,
            "result": result
        }), 200

    except FileValidationError as e:
        return jsonify({
            "success": False,
            "error_type": "FileValidationError",
            "error": str(e)
        }), 400

    except DicomProcessingError as e:
        return jsonify({
            "success": False,
            "error_type": "DicomProcessingError",
            "error": str(e)
        }), 400

    except AIAnalysisError as e:
        return jsonify({
            "success": False,
            "error_type": "AIAnalysisError",
            "error": str(e)
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "UnhandledException",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print(f"PROJECT_ID = {PROJECT_ID}")
    print(f"REGION = {REGION}")
    print(f"MODEL_NAME = {MODEL_NAME}")
    app.run(host="127.0.0.1", port=8080, debug=True)