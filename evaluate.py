import json
from extractor import extract_action_items, validate_action_items, evaluate_extraction

with open("data/annotated_transcripts.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

all_predicted = []
all_expected = []

for record in dataset:
    predicted = validate_action_items(extract_action_items(record["transcript"]))
    all_predicted.extend(predicted)
    all_expected.extend(record["action_items"])

result = evaluate_extraction(all_predicted, all_expected)
print("\nAI Meeting Action-Item Extractor Evaluation")
print("-------------------------------------------")
print(f"Expected action items : {result['expected']}")
print(f"Predicted action items: {result['predicted']}")
print(f"Matched action items  : {result['matched']}")
print(f"Precision             : {result['precision']}")
print(f"Recall                : {result['recall']}")
print(f"F1 score              : {result['f1']}")
