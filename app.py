from flask import Flask, render_template, request, jsonify
from extractor import extract_action_items, validate_action_items
from pathlib import Path

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/extract")
def extract():
    transcript = request.form.get("transcript", "").strip()

    uploaded = request.files.get("file")
    if uploaded and uploaded.filename:
        if not uploaded.filename.lower().endswith(".txt"):
            return jsonify({"error": "Please upload a .txt meeting transcript."}), 400
        transcript = uploaded.read().decode("utf-8", errors="replace").strip()

    if not transcript:
        return jsonify({"error": "Please paste a transcript or upload a .txt file."}), 400

    items = extract_action_items(transcript)
    validated = validate_action_items(items)

    return jsonify({
        "items": validated,
        "count": len(validated)
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
