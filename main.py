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

# Use Hugging Face Inference API instead of local model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM-L6-v2 output size
HF_API_URL = f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# Get HF API token from environment
HF_TOKEN = os.getenv("HF_TOKEN")# Set this in your .env or Hugging Face Space secrets

# -------------------- APP SETUP --------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

load_dotenv()

# -------------------- ROOT ROUTE --------------------

@app.route("/")
def index():
    return "<h1>Your Flask server is running!</h1>"

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": EMBEDDING_MODEL, "dimension": EMBEDDING_DIM, "api": "huggingface"})

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
        'Physics': [
            'mechanics', 'optics', 'thermodynamics', 'kinematics',
            'dynamics', 'relativity', 'electromagnetic', 'oscillations',
            'waves', 'motion', 'energy', 'power', 'force'
        ],
        'Chemistry': [
            'chemistry', 'mole concept', 'periodic table', 'chemical bonding',
            'thermochemistry', 'equilibrium', 'redox', 'hydrocarbons',
            'organic', 'inorganic', 'structure of atom'
        ],
        'Mathematics': [
            'algebra', 'calculus', 'trigonometry', 'geometry', 'statistics',
            'probability', 'matrices', 'determinants', 'vectors'
        ],
        'Biology': [
            'cell biology', 'genetics', 'ecology', 'physiology',
            'biochemistry', 'taxonomy', 'photosynthesis', 'respiration'
        ],
        'Psychology': [
            'psychology', 'cognition', 'behavior', 'counseling',
            'personality', 'mental health', 'therapy'
        ]
    }

    text_lower = text.lower()
    scores = {subject: 0 for subject in subject_keywords}

    for subject, keywords in subject_keywords.items():
        for kw in keywords:
            scores[subject] += text_lower.count(kw)

    best_subject = max(scores, key=scores.get)
    if scores[best_subject] > 2:
        return best_subject
    return "General"


# -------------------- CLASS LEVEL DETECTION --------------------

def detect_class_level(text):
    patterns = [
        (r'class\s+xi\b', '11'),
        (r'class\s+11\b', '11'),
        (r'class\s+xii\b', '12'),
        (r'class\s+12\b', '12'),
        (r'grade\s+xi\b', '11'),
        (r'grade\s+11\b', '11'),
        (r'grade\s+xii\b', '12'),
        (r'grade\s+12\b', '12')
    ]

    for pattern, class_level in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return class_level

    return "Unknown"


# -------------------- TOPIC PARSING --------------------

def parse_topics_universal(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    topics = []

    unit_pattern = re.compile(r'Unit[\s–-]*([IVXLC\d]+)\s*[:\-]?\s*(.+)', re.IGNORECASE)
    chapter_pattern = re.compile(r'Chapter[\s–-]*(\d+)\s*[:\-]?\s*(.+)', re.IGNORECASE)

    subtopic_patterns = [
        re.compile(r'^\d+\.\s+(.+)'),
        re.compile(r'^[•\-*]\s+(.+)'),
        re.compile(r'^[a-z]\)\s+(.+)'),
        re.compile(r'^[IVX]+\.\s+(.+)'),
    ]

    for line in lines:
        if any(term in line.lower() for term in
               ['page', 'period', 'mark', 'hours', 'reference',
                'book', 'textbook', 'examination', 'evaluation', 'scheme']):
            continue

        unit_match = unit_pattern.match(line)
        if unit_match:
            topics.append(f"Unit {unit_match.group(1)}: {unit_match.group(2).strip()}")
            continue

        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            topics.append(f"Chapter {chapter_match.group(1)}: {chapter_match.group(2).strip()}")
            continue

        for pattern in subtopic_patterns:
            sub_match = pattern.match(line)
            if sub_match:
                topic = re.sub(r'\s+', ' ', sub_match.group(1)).strip()
                if 3 <= len(topic) <= 150:
                    topics.append(topic)
                break

    return list(dict.fromkeys(topics))  # unique


# -------------------- TEXT CHUNKING --------------------

def chunk_text(text, max_chars=1500, overlap=200):
    if not text or len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind(".")
            if last_period != -1:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


# -------------------- HUGGING FACE API EMBEDDING GENERATION --------------------

def generate_embedding_hf_api(text):
    """Generate embedding using Hugging Face Inference API (384 dimensions)"""
    try:
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()
            # Handle different response formats
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    return embedding[0]  # Return first embedding if batch
                return embedding
            return embedding
        else:
            print(f"❌ HF API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("❌ Embedding API error:", e)
        return None


def generate_embeddings_hf(class_level, subject, topics, fallback_texts=None, full_text=""):
    items = []

    if full_text and topics:
        print(f"📊 Processing {len(topics)} topics with HF API...")
        
        # Limit topics if too many to avoid rate limits
        max_topics = 50  # Process max 50 topics
        if len(topics) > max_topics:
            print(f"⚠️ Too many topics ({len(topics)}). Processing first {max_topics}.")
            topics = topics[:max_topics]

        for topic_idx, topic in enumerate(topics):
            # Progress indicator
            if topic_idx % 5 == 0:
                print(f"  📍 Progress: {topic_idx}/{len(topics)} topics processed...")
            
            topic_text = f"{class_level} | {subject} | {topic}"
            chunks = chunk_text(topic_text)

            for chunk_idx, chunk in enumerate(chunks):
                embedding = generate_embedding_hf_api(chunk)

                if embedding:
                    items.append({
                        "class": class_level,
                        "subject": subject,
                        "topic": topic,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "embedding": embedding
                    })

        print(f"✅ Generated {len(items)} HF API embeddings from {len(topics)} topics")
    else:
        texts = [f"{class_level} | {subject} | {t}" for t in topics] or fallback_texts
        
        # Limit fallback texts too
        if len(texts) > 50:
            print(f"⚠️ Too many texts ({len(texts)}). Processing first 50.")
            texts = texts[:50]

        for idx, text in enumerate(texts):
            if idx % 10 == 0:
                print(f"  📍 Progress: {idx}/{len(texts)} embeddings...")
                
            embedding = generate_embedding_hf_api(text)
            if embedding:
                items.append({
                    "class": class_level,
                    "subject": subject,
                    "topic": text,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "text": text,
                    "embedding": embedding
                })

        print(f"✅ Generated {len(items)} simple embeddings")

    return items


# -------------------- PDF PARSE ROUTE --------------------

@app.route("/parse", methods=["POST"])
def parse_file():
    import time
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    print(f"\n{'='*60}")
    print(f"📄 Processing: {file.filename}")
    print(f"{'='*60}")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        filepath = tmp_file.name
        file.save(filepath)

    try:
        print("📖 Step 1: Extracting text from PDF...")
        text = extract_text_from_pdf(filepath)
        print(f"✅ Extracted {len(text)} characters")
        
        print("🔍 Step 2: Detecting subject and class...")
        subject = detect_subject(text)
        class_level = detect_class_level(text)
        print(f"✅ Detected: Class {class_level}, Subject: {subject}")
        
        print("📋 Step 3: Parsing topics...")
        topics = parse_topics_universal(text)
        print(f"✅ Found {len(topics)} topics")

        if subject == "Unknown":
            subject = "General"
        if class_level == "Unknown":
            class_level = "11"

        fallback_lines = [line.strip() for line in text.split('\n') if line.strip()][:10]

        print("🧠 Step 4: Generating embeddings with HF API (MiniLM-L6-v2, 384 dims)")
        embed_start = time.time()
        items = generate_embeddings_hf(
            class_level,
            subject,
            topics,
            fallback_texts=fallback_lines,
            full_text=text
        )
        embed_time = time.time() - embed_start
        print(f"✅ Embedding generation completed in {embed_time:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Total processing time: {total_time:.2f} seconds")
        print(f"{'='*60}\n")

        return jsonify({
            "class": class_level,
            "subject": subject,
            "topics": topics,
            "items": items,
            "message": f"Generated {len(items)} local embeddings for {subject} Class {class_level}",
            "processing_time": round(total_time, 2)
        })

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


# -------------------- QUIZ GENERATION (using Groq LLM) --------------------

from groq import Groq

# Initialize Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        topic = data.get("topic", "")
        subject = data.get("subject", "General")
        num_questions = data.get("num_questions", 5)

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        print(f"🎯 Generating quiz for: {topic} ({subject})")

        # Create prompt for Groq
        prompt = f"""Generate a multiple-choice quiz for the following topic:

Topic: {topic}
Subject: {subject}
Number of Questions: {num_questions}

Requirements:
- Create {num_questions} multiple-choice questions
- Each question should have 4 options (A, B, C, D)
- Include the correct answer for each question
- Questions should test understanding, not just memorization
- Difficulty should be appropriate for high school students (Class 11-12)

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct": 0,
    "explanation": "Brief explanation of why this is correct"
  }}
]

CRITICAL: Return ONLY the JSON array, no explanations or text outside JSON."""

        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert quiz generator. You MUST respond with valid JSON only. No markdown, no explanations, just the JSON array."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        response_text = chat_completion.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        response_text = response_text.replace("```json\n", "").replace("```\n", "").replace("```", "")
        
        # Parse JSON response
        try:
            quiz_data = json.loads(response_text)
            
            # Handle both array and object with "quiz" key
            if isinstance(quiz_data, dict) and "quiz" in quiz_data:
                quiz = quiz_data["quiz"]
            elif isinstance(quiz_data, list):
                quiz = quiz_data
            else:
                quiz = [quiz_data]  # Wrap single question in array
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response: {response_text[:500]}")
            return jsonify({"error": "Failed to parse quiz response"}), 500

        print(f"✅ Generated {len(quiz)} questions")
        
        return jsonify({
            "quiz": quiz,
            "topic": topic,
            "subject": subject
        })

    except Exception as e:
        print(f"❌ Quiz generation error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------- MAIN --------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)
