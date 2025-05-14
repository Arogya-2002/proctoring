from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import tempfile
import docx
import pdfplumber
import numpy as np
from faster_whisper import WhisperModel
from google.generativeai import GenerativeModel, configure
import soundfile as sf
import logging

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

configure(api_key="AIzaSyDS0SXbtLKaawt2IdjTezO8HzsaSoM6RJM")
gemini_model = GenerativeModel("gemini-1.5-flash")
logging.basicConfig(filename='ai_interview.log', level=logging.INFO)
model = WhisperModel("base", compute_type="int8")

def extract_resume_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Only PDF or DOCX allowed")

def transcribe_audio(audio_path):
    audio_data, sample_rate = sf.read(audio_path)
    segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
    return " ".join([seg.text for seg in segments])

def generate_question(resume_text, previous_answer=None):
    prompt = (
        f"Generate one interview question at a time  based on this resume:\n{resume_text}" if not previous_answer
        else f"Ask a follow-up technical question based on this resume and previous answer.\nResume:\n{resume_text}\nAnswer:\n{previous_answer}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "Sorry, I couldn't generate the next question."

def generate_feedback(resume_text, responses):
    compiled = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in responses])
    prompt = f"Provide constructive feedback for this interview. Resume:\n{resume_text}\nInterview log:\n{compiled}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Feedback generation error: {e}")
        return "Feedback generation failed."

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "resume" not in request.files:
        return redirect(url_for("index"))
    file = request.files["resume"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    resume_text = extract_resume_text(filepath)
    return jsonify({"resume_text": resume_text})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    resume_text = data["resume"]
    previous_answer = data.get("previous_answer")
    question = generate_question(resume_text, previous_answer)
    return jsonify({"question": question})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    text = transcribe_audio(filepath)
    return jsonify({"transcription": text})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    resume_text = data["resume"]
    responses = data["responses"]
    fb = generate_feedback(resume_text, responses)
    return jsonify({"feedback": fb})

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080) 