import json, os

run_dir = r"artifacts\albert-checkpoint-5000" 

LABELS_28 = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

run_card = {
    "run_id": "albert_model",
    "family": "transformer",
    "model_name": "albert-base-v2",
    "tokenizer_name": "albert-base-v2",
    "params": {"max_len": 128},
    "metrics": {},                       
    "label_classes": LABELS_28
}
with open(os.path.join(run_dir, "run_card.json"), "w") as f:
    json.dump(run_card, f, indent=2)

print("Wrote run_card.json to", run_dir)
