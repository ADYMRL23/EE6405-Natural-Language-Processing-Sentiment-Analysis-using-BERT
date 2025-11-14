# Streamlit FRONTEND to browse checkpoints and run single-text predictions

import os, json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------- App config
st.set_page_config(page_title="Sentiment/Emotion Viewer", layout="wide")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -------------------- Utilities
def discover_runs(art_dir: str = ARTIFACTS_DIR) -> Dict[str, Dict]:
    found = {}
    if not os.path.isdir(art_dir):
        return found
    for name in os.listdir(art_dir):
        rd = os.path.join(art_dir, name)
        rc_path = os.path.join(rd, "run_card.json")
        if os.path.isdir(rd) and os.path.isfile(rc_path):
            try:
                with open(rc_path) as f:
                    meta = json.load(f)
                meta["_path"] = rd
                found[meta["run_id"]] = meta
            except Exception:
                pass
    return found

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False

def load_predictor(run_card: Dict):
    """Return (predict_fn, predict_proba_fn, class_names, family) for the selected checkpoint."""
    path = run_card["_path"]
    class_names = run_card["label_classes"]
    family = run_card["family"]

    if family == "baseline":
        pipe = joblib.load(os.path.join(path, "model.joblib"))
        def predict(texts: List[str]) -> np.ndarray:
            return pipe.predict(texts)
        def predict_proba(texts: List[str]) -> np.ndarray:
            if hasattr(pipe, "predict_proba"):
                return pipe.predict_proba(texts)
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(texts)
                scores = np.atleast_2d(scores)
                e = np.exp(scores - scores.max(axis=-1, keepdims=True))
                return e / e.sum(axis=-1, keepdims=True)
            preds = pipe.predict(texts)
            probs = np.zeros((len(preds), len(class_names)))
            for i, p in enumerate(preds): probs[i, int(p)] = 1.0
            return probs
        return predict, predict_proba, class_names, family

    # ----- Prefer labels from config.json
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            if "id2label" in cfg and "num_labels" in cfg:
                class_names = [cfg["id2label"][str(i)] for i in range(cfg["num_labels"])]
        except Exception:
            pass

    # ----- Tokenizer source
    has_tok = any(os.path.exists(os.path.join(path, fn)) for fn in ("tokenizer.json","vocab.txt","spiece.model"))
    tok_src = path if has_tok else run_card.get("tokenizer_name", path)

    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()

    device = torch.device("cuda" if _cuda_ok() else "cpu")
    try:
        mdl.to(device)
    except Exception:
        device = torch.device("cpu"); mdl.to(device)

    max_len = int(run_card.get("params", {}).get("max_len", 128))

    def predict(texts: List[str]) -> np.ndarray:
        enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits
        return logits.argmax(-1).cpu().numpy()

    def predict_proba(texts: List[str]) -> np.ndarray:
        enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    return predict, predict_proba, class_names, family

# -------------------- Load all checkpoints
ALL_RUNS = discover_runs()

st.sidebar.title("Models")
if not ALL_RUNS:
    st.sidebar.warning("No checkpoints found in ./artifacts/. Drop your trained runs there.")
run_ids_sorted = sorted(ALL_RUNS.keys())
run_choice = st.sidebar.selectbox("Select checkpoint/run", run_ids_sorted, index=0 if run_ids_sorted else None)

st.title("Sentiment / Emotion Model Viewer")
st.caption("Select a checkpoint and run a prediction on your text. See all runs and metrics in the second tab.")

# Tabs: only Predict + All Runs
tab_predict, tab_all = st.tabs(["Predict", "All Runs (Metrics)"])

# -------------------- PREDICT (single text only)
with tab_predict:
    if not run_choice:
        st.info("Pick a checkpoint on the left.")
    else:
        rc = ALL_RUNS[run_choice]
        st.markdown(f"**Selected run:** `{rc['run_id']}` • **Model:** `{rc['model_name']}` • **Classes:** {', '.join(rc['label_classes'])}")
        predict, predict_proba, classes, family = load_predictor(rc)

        text = st.text_area("Type your text here", "I respect you", height=120)
        if st.button("Predict"):
            probs = predict_proba([text])[0]
            pred_id = int(np.argmax(probs))
            st.success(f"Prediction: **{classes[pred_id]}**")

            dfp = pd.DataFrame({"class": classes, "probability": probs})
            st.bar_chart(dfp.set_index("class"))

# -------------------- ALL RUNS
with tab_all:
    st.subheader("All saved checkpoints (from artifacts/)")
    if not ALL_RUNS:
        st.info("No checkpoints found.")
    else:
        rows = []
        for rid, rc in ALL_RUNS.items():
            m = rc.get("metrics", {})
            rows.append({
                "run_id": rid,
                "family": rc.get("family"),
                "model": rc.get("model_name"),
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "n_classes": len(rc.get("label_classes", [])),
                "path": rc.get("_path")
            })
        df_runs = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        st.dataframe(df_runs, use_container_width=True)
        st.download_button("Download runs table (CSV)", df_runs.to_csv(index=False), "all_runs.csv", "text/csv")