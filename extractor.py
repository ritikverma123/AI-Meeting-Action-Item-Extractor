import json
import re
from datetime import datetime
from difflib import SequenceMatcher

# Step 3: transformer model for information extraction.
# The project tries to load FLAN-T5-small. If the model is unavailable,
# the built-in extraction fallback keeps the demo runnable.
MODEL_NAME = "google/flan-t5-small"
_generator = None

def get_generator():
    global _generator
    if _generator is not None:
        return _generator
    try:
        from transformers import pipeline
        _generator = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME
        )
    except Exception:
        _generator = False
    return _generator

def clean_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def segment_transcript(transcript):
    """Step 2: clean and segment by speaker and sentence."""
    lines = [clean_text(x) for x in transcript.splitlines() if clean_text(x)]
    segments = []
    for line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z0-9 ._-]{0,50}):\s*(.+)$", line)
        if match:
            speaker, content = match.group(1).strip(), match.group(2).strip()
        else:
            speaker, content = "Unknown", line

        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            sentence = clean_text(sentence)
            if sentence:
                segments.append({"speaker": speaker, "text": sentence})
    return segments

def _date_from_text(text):
    patterns = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{4})\b",
        r"\b(?:by|on|before|due)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"\b(?:by|on|before|due)\s+(tomorrow|today|next week|Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1)
    return None

def _owner_from_text(speaker, text):
    # Explicit assignment: "Rahul will..." / "Priya, please..."
    patterns = [
        r"\b([A-Z][a-z]{2,20})\s+(?:will|should|needs to|is going to|can)\b",
        r"\b(?:assign(?:ed)?|owner(?: is)?|responsibility(?: is)?)\s+(?:to|for)\s+([A-Z][a-z]{2,20})\b",
        r"\b([A-Z][a-z]{2,20}),\s*please\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    if speaker != "Unknown":
        return speaker
    return None

def _task_from_text(text):
    # Remove common assignment/deadline language while keeping the actual task.
    task = text
    task = re.sub(
        r"^(?:please\s+)?(?:assign(?:ed)?\s+to\s+\w+|[A-Z][a-z]{2,20},\s*)",
        "",
        task,
        flags=re.I
    )
    task = re.sub(r"\b(?:by|before|on|due)\s+(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z0-9 ,]+)$", "", task, flags=re.I)
    return clean_text(task).rstrip(". ")

def _is_action_sentence(text):
    action_words = [
        "will", "needs to", "need to", "should", "must", "please",
        "action item", "assigned", "complete", "finish", "prepare",
        "send", "create", "review", "update", "schedule", "fix",
        "deploy", "submit", "design", "test", "share", "call"
    ]
    lower = text.lower()
    return any(w in lower for w in action_words)

def _heuristic_extract(segments):
    items = []
    for seg in segments:
        text = seg["text"]
        if not _is_action_sentence(text):
            continue
        task = _task_from_text(text)
        owner = _owner_from_text(seg["speaker"], text)
        deadline = _date_from_text(text)
        confidence = 0.55
        if owner:
            confidence += 0.15
        if deadline:
            confidence += 0.15
        if any(w in text.lower() for w in ["will", "must", "assigned", "needs to"]):
            confidence += 0.10
        confidence = min(round(confidence, 2), 0.99)

        items.append({
            "task": task,
            "owner": owner,
            "deadline": deadline,
            "status": "Open",
            "confidence": confidence
        })
    return items

def _llm_extract(segments):
    generator = get_generator()
    if not generator:
        return []

    grouped = {}
    for s in segments:
        grouped.setdefault(s["speaker"], []).append(s["text"])

    prompt = """Extract only clear action items from the meeting transcript below.
Return ONLY a JSON array. Each object must contain:
task, owner, deadline, status, confidence.
Use null when owner or deadline is missing.
Status must be Open, In Progress, or Done.
Confidence must be a number from 0 to 1.
Do not invent people or dates.

Transcript:
""" + "\n".join(f"{speaker}: {' '.join(parts)}" for speaker, parts in grouped.items())

    try:
        output = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        match = re.search(r"\[[\s\S]*\]", output)
        if not match:
            return []
        data = json.loads(match.group(0))
        if not isinstance(data, list):
            return []
        cleaned = []
        for item in data:
            if not isinstance(item, dict) or not item.get("task"):
                continue
            cleaned.append({
                "task": clean_text(str(item.get("task"))),
                "owner": item.get("owner"),
                "deadline": item.get("deadline"),
                "status": item.get("status") or "Open",
                "confidence": float(item.get("confidence", 0.5))
            })
        return cleaned
    except Exception:
        return []

def extract_action_items(transcript):
    segments = segment_transcript(transcript)
    # Step 3: prefer transformer extraction; fallback keeps the application usable.
    items = _llm_extract(segments)
    if not items:
        items = _heuristic_extract(segments)
    return items

def _valid_date(value):
    if not value:
        return True
    value = str(value).strip()
    known_words = {"today", "tomorrow", "next week", "monday", "tuesday",
                   "wednesday", "thursday", "friday", "saturday", "sunday"}
    if value.lower() in known_words:
        return True
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
               "%d %B %Y", "%d %b %Y"]
    return any(_try_date(value, fmt) for fmt in formats)

def _try_date(value, fmt):
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False

def _duplicate(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.88

def validate_action_items(items):
    """Step 5: validation for dates, missing owners, and duplicate tasks."""
    valid = []
    for item in items:
        task = clean_text(item.get("task"))
        if not task:
            continue

        owner = item.get("owner")
        deadline = item.get("deadline")
        status = item.get("status") if item.get("status") in {"Open", "In Progress", "Done"} else "Open"

        warnings = []
        if not owner:
            warnings.append("Missing owner")
        if deadline and not _valid_date(deadline):
            warnings.append("Check date format")

        is_dup = any(_duplicate(task, x["task"]) for x in valid)
        if is_dup:
            warnings.append("Duplicate task")
            continue

        try:
            confidence = round(max(0.0, min(1.0, float(item.get("confidence", 0.5)))), 2)
        except Exception:
            confidence = 0.5

        valid.append({
            "task": task,
            "owner": owner or "Unassigned",
            "deadline": deadline or "Not specified",
            "status": status,
            "confidence": confidence,
            "warnings": warnings
        })
    return valid

def evaluate_extraction(predicted, expected):
    """Step 6: simple accuracy evaluation against annotated action items."""
    expected_tasks = [clean_text(x["task"]).lower() for x in expected]
    predicted_tasks = [clean_text(x["task"]).lower() for x in predicted]

    matched = 0
    used = set()
    for p in predicted_tasks:
        best_i, best_score = None, 0
        for i, e in enumerate(expected_tasks):
            if i in used:
                continue
            score = SequenceMatcher(None, p, e).ratio()
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= 0.70:
            matched += 1
            used.add(best_i)

    precision = matched / len(predicted_tasks) if predicted_tasks else 0
    recall = matched / len(expected_tasks) if expected_tasks else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    return {
        "matched": matched,
        "predicted": len(predicted_tasks),
        "expected": len(expected_tasks),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    }
