# AI Meeting Action-Item Extractor

## Project description
Create an AI assistant that converts meeting transcripts into structured action items containing the task, owner, deadline, and confidence.

This project follows the six requested steps **without changing them**:

1. Gather or generate meeting transcripts with annotated action items.
2. Clean and segment transcripts by speaker and sentence.
3. Use an LLM or transformer model for information extraction.
4. Design a structured output schema: task, person, date, and status.
5. Add validation rules for dates, missing owners, and duplicate tasks.
6. Evaluate extraction accuracy and build a simple upload-to-results interface.

## Technologies
- Python
- Flask
- Hugging Face Transformers
- FLAN-T5-small transformer model
- HTML/CSS/JavaScript
- JSON annotated dataset

## Folder structure

```text
AI_Meeting_Action_Item_Extractor/
│
├── app.py
├── extractor.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   └── annotated_transcripts.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Step 1 — Gather or generate meeting transcripts with annotated action items

The file `data/annotated_transcripts.json` contains sample meeting transcripts and their expected action items.

You can add more records in the same format.

## Step 2 — Clean and segment transcripts by speaker and sentence

`segment_transcript()`:
- removes extra spaces
- detects `Speaker: sentence`
- separates multiple sentences
- stores speaker and sentence separately

## Step 3 — Use an LLM or transformer model for information extraction

The project uses the Hugging Face transformer model:

`google/flan-t5-small`

The extractor asks the model to return JSON containing task, owner, deadline, status, and confidence.

If the transformer model cannot be downloaded or loaded, a local rule-based fallback is used so the web application can still demonstrate the complete workflow.

## Step 4 — Structured output schema

Every extracted item follows:

```json
{
  "task": "Prepare the project report",
  "owner": "Rahul",
  "deadline": "10/10/2026",
  "status": "Open",
  "confidence": 0.85
}
```

## Step 5 — Validation rules

The application checks:
- missing owners
- invalid or suspicious dates
- duplicate tasks
- allowed status values
- confidence range from 0 to 1

## Step 6 — Evaluation and upload-to-results interface

### Start the web application

```bash
pip install -r requirements.txt
python app.py
```

Open:

`http://127.0.0.1:5000`

Paste a transcript or upload a `.txt` file and click **Extract Action Items**.

### Run evaluation

```bash
python evaluate.py
```

The evaluation reports:
- precision
- recall
- F1 score
- matched items
- predicted items
- expected items

## Example transcript

```text
Anita: We need to finalize the project report by 10/10/2026.
Rahul: I will prepare the report and send it to the team.
Priya: Please review the final report by Friday.
Anita: We also need to schedule the client demo next week.
```

## Important note

On first use, Hugging Face may download the FLAN-T5-small model. Internet access is required for the first model download. After the model is cached locally, it can be reused.
