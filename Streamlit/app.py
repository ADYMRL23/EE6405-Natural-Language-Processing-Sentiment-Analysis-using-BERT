# Streamlit FRONTEND to browse checkpoints and run predictions

import os, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import joblib
import torch
from transformers import pipeline

# -------------------- App config
st.set_page_config(page_title="Sentiment/Emotion Prediction", layout="wide")
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

def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False

@st.cache_resource
def _cached_pipe(model_dir: str, tokenizer_src: str, use_cuda: bool):
    return pipeline(
        task="text-classification",
        model=model_dir,
        tokenizer=tokenizer_src,
        device=0 if use_cuda else -1,
        return_all_scores=True,   # get scores for all labels
        truncation=True
    )

def load_predictor(run_card: Dict):
    """
    Return (predict, predict_proba, class_names, family)
    using a HF pipeline.
    """
    path = run_card["_path"]
    family = "transformer"

    # Tokenizer source uses checkpoint first, else fall back to run_card.
    has_tok = any(
        os.path.exists(os.path.join(path, fn))
        for fn in ("tokenizer.json", "vocab.txt", "spiece.model")
    )
    tok_src = path if has_tok else run_card.get("tokenizer_name", path)

    pipe = _cached_pipe(path, tok_src, _cuda_ok())

    # Prefer labels from the model config (ground truth)
    cfg = pipe.model.config
    class_names = run_card.get("label_classes", [])
    try:
        num_labels = int(getattr(cfg, "num_labels", len(class_names) or 0))
        id2label = getattr(cfg, "id2label", None)
        if id2label and num_labels:
            class_names = [
                id2label[i] if isinstance(id2label, dict) and i in id2label
                else id2label[str(i)] if isinstance(id2label, dict) and str(i) in id2label
                else class_names[i] if i < len(class_names)
                else f"LABEL_{i}"
                for i in range(num_labels)
            ]
        elif not class_names and num_labels:
            class_names = [f"LABEL_{i}" for i in range(num_labels)]
    except Exception:
        if not class_names:
            class_names = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                        "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                        "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
                        "pride","realization","relief","remorse","sadness","surprise","neutral"] # just a fallback

    name_to_idx = {n: i for i, n in enumerate(class_names)}

    def _row_to_probs(row):
        """
        row is a list of dicts from the pipeline:
          [{"label": "admiration", "score": 0.12}, ...]
        Convert to a fixed-length vector aligned to class_names.
        """
        vec = np.zeros(len(class_names), dtype=np.float32)
        for d in row:
            lab = d["label"]
            j = name_to_idx.get(lab)
            if j is None and lab.startswith("LABEL_"):
                try:
                    j = int(lab.split("_")[-1])
                except Exception:
                    j = None
            if j is not None and 0 <= j < len(vec):
                vec[j] = float(d["score"])
        s = float(vec.sum())
        if s > 0:
            vec /= s  # make sure sums to 1
        return vec

    def predict_proba(texts: List[str]) -> np.ndarray:
        outs = pipe(texts)
        return np.vstack([_row_to_probs(row) for row in outs])

    def predict(texts: List[str]) -> np.ndarray:
        return predict_proba(texts).argmax(axis=-1)

    return predict, predict_proba, class_names, family

# -------------------- Load all checkpoints
ALL_RUNS = discover_runs()

st.sidebar.title("Models")
if not ALL_RUNS:
    st.sidebar.warning("No checkpoints found in ./artifacts/. Drop your trained runs there.")
run_ids_sorted = sorted(ALL_RUNS.keys())
run_choice = st.sidebar.selectbox("Select checkpoint/run", run_ids_sorted, index=0 if run_ids_sorted else None)

st.title("Sentiment / Emotion Prediction")
st.caption("Select a checkpoint and run a prediction on your text. See all runs and metrics in the second tab.")

# Tabs
tab_predict, tab_all = st.tabs(["Prediction", "Metrics (All models)"])

# -------------------- PREDICT
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

            #replace with words instead of label numbers
            Emotions_28 = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
                        "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
                        "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
                        "pride","realization","relief","remorse","sadness","surprise","neutral"]

            display_classes = Emotions_28 if classes and classes[0].upper().startswith("LABEL_") else classes

            st.success(f"Prediction: **{display_classes[pred_id]}**")
            dfp = pd.DataFrame({"class": display_classes, "probability": probs})
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
                # "family": rc.get("family"),
                "model": rc.get("model_name"),
                "tokenizer": rc.get("tokenizer_name"),
                # "accuracy": m.get("accuracy"),
                "n_classes": len(rc.get("label_classes", [])),
                "path": rc.get("_path")
            })
        df_runs = pd.DataFrame(rows)
        st.dataframe(df_runs, use_container_width=True)
        st.download_button("Download runs table (CSV)", df_runs.to_csv(index=False), "all_runs.csv", "text/csv")

        st.markdown("### Training loss plot")

        IMG_PATH = Path(__file__).with_name("training_loss.jpg")

        if IMG_PATH.exists():
            # st.image(str(IMG_PATH), caption="Training Loss vs Epochs")
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.image(str(IMG_PATH), caption="Training Loss vs Epochs", width=800)
        else:
            st.error(f"Image not found: {IMG_PATH}")