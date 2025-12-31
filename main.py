from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz  # PyMuPDF
import re
import os
import tempfile
import json
import requests
from dotenv import load_dotenv

# -------------------- HUGGING FACE EMBEDDING API --------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"

HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------- APP SETUP --------------------

app = Flask(__name__)
CORS(app)
load_dotenv()

# -------------------- BASIC ROUTES --------------------

@app.route("/")
def index():
    return "<h1>Your Flask server is running!</h1>"

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "api": "huggingface"
    })

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# -------------------- PDF TEXT EXTRACTION --------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# -------------------- SUBJECT DETECTION --------------------

def detect_subject(text):
    subject_keywords = {
        'Physics': ['mechanics','optics','thermodynamics','kinematics','dynamics',
                    'relativity','electromagnetic','waves','energy','force'],
        'Chemistry': ['chemistry','mole','periodic','bonding','equilibrium','organic','inorganic'],
        'Mathematics': ['algebra','calculus','trigonometry','geometry','statistics'],
        'Biology': ['cell','genetics','ecology','physiology','photosynthesis'],
        'Psychology': ['psychology','behavior','mental','cognitive']
    }

    text_lower = text.lower()
    scores = {k: sum(text_lower.count(w) for w in v) for k, v in subject_keywords.items()}
    subject = max(scores, key=scores.get)
    return subject if scores[subject] > 2 else "General"

# -------------------- CLASS DETECTION --------------------

def detect_class_level(text):
    patterns = [
        (r'class\s+xi\b', '11'), (r'class\s+xii\b', '12'),
        (r'class\s+11\b', '11'), (r'class\s+12\b', '12'),
    ]
    for pattern, value in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return "Unknown"

# -------------------- TOPIC PARSER --------------------

def parse_topics_universal(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    topics = []
    for line in lines:
        if len(line) > 5 and len(line) < 120:
            topics.append(line)
    return list(dict.fromkeys(topics))

# -------------------- EMBEDDING GENERATOR (FIXED) --------------------

def generate_embedding_hf_api(text):
    try:
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": [text],   # ✅ MUST BE A LIST
                "options": {"wait_for_model": True}
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]  # return embedding vector
        else:
            print("❌ HF API error:", response.text)

        return None

    except Exception as e:
        print("❌ Embedding error:", e)
        return None

# -------------------- EMBEDDING PIPELINE --------------------

def generate_embeddings_hf(class_level, subject, topics, fallback_texts=None, full_text=""):
    items = []

    if not topics:
        topics = fallback_texts or []

    topics = topics[:50]  # safety limit

    for idx, topic in enumerate(topics):
        print(f"🔹 Embedding {idx+1}/{len(topics)}")

        text = f"{class_level} | {subject} | {topic}"
        embedding = generate_embedding_hf_api(text)

        if embedding:
            items.append({
                "class": class_level,
                "subject": subject,
                "topic": topic,
                "embedding": embedding
            })

    print(f"✅ Generated {len(items)} embeddings")
    return items

# -------------------- MAIN PDF PARSER --------------------

@app.route("/parse", methods=["POST"])
def parse_file():
    import time
    start = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        pdf_path = tmp.name

    try:
        text = extract_text_from_pdf(pdf_path)
        subject = detect_subject(text)
        class_level = detect_class_level(text)
        topics = parse_topics_universal(text)

        items = generate_embeddings_hf(
            class_level,
            subject,
            topics,
            fallback_texts=text.split("\n"),
            full_text=text
        )

        return jsonify({
            "class": class_level,
            "subject": subject,
            "topics": topics,
            "items": items,
            "processing_time": round(time.time() - start, 2)
        })

    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

# -------------------- START SERVER --------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
