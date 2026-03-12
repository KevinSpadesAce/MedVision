import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google import genai

# load environment variables
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")

# initialize Gemini client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "MedVision AI backend running"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    description = data.get("description", "")

    prompt = f"""
You are a medical AI assistant helping radiologists detect possible abnormalities.

A doctor describes a possible observation from a chest X-ray:

"{description}"

Give:
1. Possible abnormality
2. Risk level (low / medium / high)
3. Short explanation
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return jsonify({
            "analysis": response.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
    print(PROJECT_ID)