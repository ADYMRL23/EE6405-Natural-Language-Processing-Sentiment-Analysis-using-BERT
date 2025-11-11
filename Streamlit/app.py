# Streamlit FRONTEND to browse checkpoints, run predictions, and view metrics.

import os, json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

import nltk, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoModelForMultipleChoice

# -------------------- App config
st.set_page_config(page_title="Sentiment/Emotion Viewer", layout="wide")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Ensure NLTK bits exist (used only for optional light cleaning)
for pkg, res in [("stopwords", "corpora/stopwords"),
                 ("punkt", "tokenizers/punkt"),
                 ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(pkg)

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

def load_predictor(run_card: Dict):
    """Return (predict_fn, predict_proba_fn, class_names, family)"""
    path = run_card["_path"]
    class_names = run_card["label_classes"]
    family = run_card["family"]

    if family == "baseline":
        pipe = joblib.load(os.path.join(path, "model.joblib"))
        def predict(texts: List[str]) -> np.ndarray:
            return pipe.predict(texts)
        def predict_proba(texts: List[str]) -> np.ndarray:
            # not all sklearn classifiers expose predict_proba
            if hasattr(pipe, "predict_proba"):
                return pipe.predict_proba(texts)
            # fallback: decision_function -> softmax
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(texts)
                scores = np.atleast_2d(scores)
                return softmax(scores)
            # final fallback: one-hot of predict
            preds = pipe.predict(texts)
            probs = np.zeros((len(preds), len(class_names)))
            for i, p in enumerate(preds): probs[i, int(p)] = 1.0
            return probs
        return predict, predict_proba, class_names, family

    # Transformer
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)

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

def light_clean(texts: List[str],
                lower=True, rm_urls=True, rm_punct=True, rm_stop=False, lemm=False):
    sw = set(stopwords.words("english")) if rm_stop else set()
    lemmatizer = WordNetLemmatizer() if lemm else None

    def _rm_urls(t):  # very light URL removal
        return pd.Series([t]).str.replace(r"http\S+|www\.\S+", "", regex=True).iloc[0]
    def _rm_punct(t):
        return t.translate(str.maketrans("", "", string.punctuation))

    out = []
    for t in texts:
        t = "" if pd.isna(t) else str(t)
        if lower: t = t.lower()
        if rm_urls: t = _rm_urls(t)
        if rm_punct: t = _rm_punct(t)
        toks = t.split()
        if rm_stop: toks = [w for w in toks if w not in sw]
        if lemm: toks = [lemmatizer.lemmatize(w) for w in toks]
        out.append(" ".join(toks))
    return out

def map_labels_to_ids(labels: pd.Series, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert string labels to the checkpoint's class indices.
    Returns (y_ids, keep_mask) where keep_mask marks rows successfully mapped.
    """
    name_to_id = {name: i for i, name in enumerate(class_names)}
    y_ids = []
    keep = []
    for v in labels.astype(str).tolist():
        if v in name_to_id:
            y_ids.append(name_to_id[v]); keep.append(True)
        else:
            # try case-insensitive match
            k = next((name_to_id[n] for n in class_names if n.lower() == v.lower()), None)
            if k is None:
                keep.append(False); y_ids.append(-1)
            else:
                keep.append(True); y_ids.append(k)
    return np.array(y_ids), np.array(keep, dtype=bool)

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

# ---- Zero-shot (Multiple-Choice) helpers ----
@st.cache_resource
def load_mc_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForMultipleChoice.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()
    return tok, mdl, device

def predict_mc(tokenizer, model, device, sentences, choices, prompt_template="The closest sentiment is: {text}", max_len=256):
    """
    Multiple-Choice zero-shot:
    For each sentence S and label list [c1..cK], build K pairs:
    (stem(S), c_i) -> logits over K, argmax is prediction.
    """
    import torch.nn.functional as F
    preds, probs_all = [], []
    with torch.no_grad():
        for s in sentences:
            stem = prompt_template.format(text=s)
            stems = [stem] * len(choices)
            enc = tokenizer(stems, choices, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}  # (1, num_choices, seq_len)
            out = model(**enc)
            logits = out.logits.squeeze(0)            # (num_choices,)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds.append(int(probs.argmax()))
            probs_all.append(probs)
    return np.array(preds), np.vstack(probs_all)

# -------------------- Load all checkpoints
ALL_RUNS = discover_runs()

st.sidebar.title("Models")
if not ALL_RUNS:
    st.sidebar.warning("No checkpoints found in ./artifacts/. Drop your trained runs there.")
run_ids_sorted = sorted(ALL_RUNS.keys())
run_choice = st.sidebar.selectbox("Select checkpoint/run", run_ids_sorted, index=0 if run_ids_sorted else None)

# Optional light cleaning toggles
st.sidebar.title("Input Cleaning (applies to predictions)")
lower = st.sidebar.checkbox("Lowercase", True)
rm_urls = st.sidebar.checkbox("Remove URLs", True)
rm_punct = st.sidebar.checkbox("Remove punctuation", True)
rm_stop  = st.sidebar.checkbox("Remove stopwords", False)
lemm     = st.sidebar.checkbox("Lemmatize", False)

st.title("Sentiment / Emotion Model Viewer")
st.caption("Load checkpoints, run predictions on datasets or free text, and view metrics.")

# Tabs
tab_overview, tab_predict, tab_eval, tab_zero, tab_all = st.tabs(
    ["Checkpoint Overview", "Predict", "Evaluate on Dataset", "Zero-shot (Hub)", "All Runs (Metrics)"]
)

# -------------------- OVERVIEW
with tab_overview:
    if not run_choice:
        st.info("Pick a checkpoint on the left.")
    else:
        rc = ALL_RUNS[run_choice]
        st.subheader(f"Run: `{rc['run_id']}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Family", rc["family"])
        c2.metric("Model", rc["model_name"])
        c3.metric("Classes", len(rc["label_classes"]))
        st.write("**Label classes:**", ", ".join(rc["label_classes"]))
        st.write("**Params:**", rc.get("params", {}))
        st.write("**Stored training metrics (from run_card):**", rc.get("metrics", {}))
        st.code(rc.get("_path", ""), language="bash")

# -------------------- PREDICT
with tab_predict:
    st.subheader("Single text")
    if not run_choice:
        st.info("Pick a checkpoint on the left.")
    else:
        rc = ALL_RUNS[run_choice]
        predict, predict_proba, classes, family = load_predictor(rc)

        text = st.text_area("Type text here", "I love this product, it works great!")
        if st.button("Predict"):
            cleaned = light_clean([text], lower, rm_urls, rm_punct, rm_stop, lemm)
            probs = predict_proba(cleaned)[0]
            pred_id = int(np.argmax(probs))
            st.success(f"Prediction: **{classes[pred_id]}**")
            # show probability table
            dfp = pd.DataFrame({"class": classes, "probability": probs})
            st.bar_chart(dfp.set_index("class"))

    st.markdown("---")
    st.subheader("Batch (CSV)")
    st.write("Upload a CSV (choose **text** column; **label** is optional for metrics).")
    csv = st.file_uploader("Upload CSV", type=["csv"], key="pred_csv")
    if csv is not None and run_choice:
        df = pd.read_csv(csv)
        st.dataframe(df.head())
        text_col = st.selectbox("Select TEXT column", list(df.columns))
        label_col = st.selectbox("Select LABEL column (optional)", ["<none>"] + list(df.columns))
        if st.button("Run batch predictions"):
            rc = ALL_RUNS[run_choice]
            predict, predict_proba, classes, _ = load_predictor(rc)
            texts = light_clean(df[text_col].astype(str).tolist(), lower, rm_urls, rm_punct, rm_stop, lemm)
            probs = predict_proba(texts)
            preds = probs.argmax(axis=-1)
            out = df.copy()
            out["prediction"] = [classes[int(i)] for i in preds]
            for j, cname in enumerate(classes):
                out[f"p_{cname}"] = probs[:, j]
            st.dataframe(out.head(30))
            st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")

            if label_col != "<none>":
                y_ids, keep = map_labels_to_ids(df[label_col], classes)
                kept = out[keep]
                if kept.empty:
                    st.warning("None of the labels matched the checkpoint's classes; cannot compute metrics.")
                else:
                    y_true = y_ids[keep]
                    y_pred = preds[keep]
                    mets = compute_metrics(y_true, y_pred)
                    st.write("**Metrics on your CSV:**", {"accuracy": mets["accuracy"], "macro_f1": mets["macro_f1"]})
                    plot_confusion(y_true, y_pred, classes, title="Confusion (on uploaded CSV)")
                    st.json(mets["report"])

# -------------------- EVALUATE on one of the 4 datasets (upload local copies)
with tab_eval:
    st.subheader("Evaluate on a dataset (upload the CSV for each)")
    ds_choice = st.selectbox(
        "Dataset",
        [
            "Emotions Dataset (Kaggle)",
            "Genius Lyrics (Kaggle)",
            "Financial Sentiment (Kaggle)",
            "Mental Health Sentiment (Kaggle)",
            "Custom CSV"
        ]
    )
    up = st.file_uploader("Upload the dataset CSV", type=["csv"], key="eval_csv")
    if up is not None and run_choice:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        # you pick columns based on the file
        text_col = st.selectbox("Select TEXT column", list(df.columns), key="eval_text")
        label_col = st.selectbox("Select LABEL column", list(df.columns), key="eval_label")
        if st.button("Evaluate now"):
            rc = ALL_RUNS[run_choice]
            predict, predict_proba, classes, _ = load_predictor(rc)
            texts = light_clean(df[text_col].astype(str).tolist(), lower, rm_urls, rm_punct, rm_stop, lemm)
            preds = predict(texts)
            y_ids, keep = map_labels_to_ids(df[label_col], classes)
            if keep.sum() == 0:
                st.warning("Labels did not match the checkpoint's class names; cannot compute metrics.")
            else:
                y_true = y_ids[keep]; y_pred = preds[keep]
                mets = compute_metrics(y_true, y_pred)
                st.write("**Accuracy:**", round(mets["accuracy"], 4), " **Macro-F1:**", round(mets["macro_f1"], 4))
                plot_confusion(y_true, y_pred, classes, title="Confusion matrix (evaluated)")
                st.json(mets["report"])
            # Quick sample predictions
            st.markdown("#### Sample predictions")
            show_n = min(20, len(df))
            sample = pd.DataFrame({
                "text": df[text_col].astype(str).head(show_n),
                "true_label": df[label_col].astype(str).head(show_n),
                "pred_label": [classes[int(i)] for i in preds[:show_n]]
            })
            st.dataframe(sample)

# -------------------- ZERO-SHOT (loads from HF hub, no checkpoints needed)
with tab_zero:
    st.subheader("Zero-shot (Hugging Face Hub) — Multiple-Choice")

    colA, colB = st.columns([2,1])
    with colA:
        hub_model = st.text_input("HF model name", value="xlnet/xlnet-base-cased",
                                  help="Examples: xlnet/xlnet-base-cased, bert-base-uncased")
    with colB:
        max_len_zero = st.number_input("Max seq length", value=256, step=8)

    choices_str = st.text_area("Label choices (comma-separated, order = class id)",
                               "neutral,happiness,sadness,worry,love")
    choices_zero = [c.strip() for c in choices_str.split(",") if c.strip()]
    prompt_template = st.text_input("Prompt template", value="The closest sentiment is: {text}",
                                    help="Use {text} where the input sentence should go.")

    if st.button("Load zero-shot model"):
        with st.spinner("Downloading/loading model from HF..."):
            try:
                tok_z, mdl_z, dev_z = load_mc_model(hub_model)
                st.session_state._zero_tok = tok_z
                st.session_state._zero_mdl = mdl_z
                st.session_state._zero_dev = dev_z
                st.success(f"Model ready on {dev_z}.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.markdown("---")
    ztab_free, ztab_csv = st.tabs(["Single text", "CSV dataset"])

    # ---- Single text
    with ztab_free:
        text_z = st.text_area("Your sentence", "I feel great about this new job!")
        if st.button("Predict (zero-shot single)"):
            if "_zero_tok" not in st.session_state:
                st.warning("Click 'Load zero-shot model' first.")
            elif not choices_zero:
                st.warning("Please provide at least one label choice.")
            else:
                preds, probs = predict_mc(
                    st.session_state._zero_tok, st.session_state._zero_mdl, st.session_state._zero_dev,
                    [text_z], choices_zero, prompt_template, max_len=max_len_zero
                )
                pid = int(preds[0])
                st.success(f"Prediction: **{choices_zero[pid]}**")
                dfp = pd.DataFrame({"class": choices_zero, "probability": probs[0]})
                st.bar_chart(dfp.set_index("class"))

    # ---- CSV dataset
    with ztab_csv:
        st.write("Upload a CSV. Pick the text column. If you also pick a label column, we’ll compute metrics.")
        up_z = st.file_uploader("CSV file (zero-shot)", type=["csv"], key="zs_csv")
        if up_z is not None:
            dfz = pd.read_csv(up_z)
            st.dataframe(dfz.head())
            text_col_z = st.selectbox("Text column", list(dfz.columns), key="zs_text_col")
            label_col_z = st.selectbox("Label column (optional)", ["<none>"] + list(dfz.columns), key="zs_label_col")
            limit_z = st.slider("Evaluate first N rows", 10, min(2000, len(dfz)), min(200, len(dfz)))

            if st.button("Run zero-shot on CSV"):
                if "_zero_tok" not in st.session_state:
                    st.warning("Click 'Load zero-shot model' first.")
                elif not choices_zero:
                    st.warning("Please provide label choices.")
                else:
                    texts_z = dfz[text_col_z].astype(str).head(limit_z).tolist()
                    preds_z, probs_z = predict_mc(
                        st.session_state._zero_tok, st.session_state._zero_mdl, st.session_state._zero_dev,
                        texts_z, choices_zero, prompt_template, max_len=max_len_zero
                    )
                    outz = dfz.head(limit_z).copy()
                    outz["prediction"] = [choices_zero[i] for i in preds_z]
                    for j, c in enumerate(choices_zero):
                        outz[f"p_{c}"] = probs_z[:, j]
                    st.dataframe(outz.head(50))
                    st.download_button("Download predictions CSV", outz.to_csv(index=False),
                                       "predictions_zero_shot.csv", "text/csv")

                    # Optional metrics if labels provided
                    if label_col_z != "<none>":
                        # simple case-insensitive exact match of labels to our choices
                        name2id = {c.lower(): i for i, c in enumerate(choices_zero)}
                        y_true, keep_idx = [], []
                        for i, v in enumerate(dfz[label_col_z].astype(str).head(limit_z)):
                            k = name2id.get(v.lower())
                            if k is not None:
                                y_true.append(k); keep_idx.append(i)
                        if y_true:
                            y_true = np.array(y_true)
                            y_pred = preds_z[np.array(keep_idx)]
                            acc = accuracy_score(y_true, y_pred)
                            mf1 = f1_score(y_true, y_pred, average="macro")
                            st.success(f"Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")
                            plot_confusion(y_true, y_pred, choices_zero, title="Confusion (Zero-shot)")
                            st.json(classification_report(y_true, y_pred, target_names=choices_zero, output_dict=True))
                        else:
                            st.info("Couldn’t align labels to choices; metrics skipped.")

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
                "classes": ", ".join(rc.get("label_classes", [])),
                "path": rc.get("_path")
            })
        df_runs = pd.DataFrame(rows).sort_values(["macro_f1","accuracy"], ascending=[False, False])
        st.dataframe(df_runs, use_container_width=True)
        st.download_button("Download runs table (CSV)", df_runs.to_csv(index=False), "all_runs.csv", "text/csv")
