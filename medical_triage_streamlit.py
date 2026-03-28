import ast
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


def parse_messages_blob(blob):
    if isinstance(blob, list):
        parsed = blob
    elif isinstance(blob, str):
        try:
            parsed = ast.literal_eval(blob)
        except Exception:
            parsed = []
    else:
        parsed = []
    return parsed if isinstance(parsed, list) else []


def extract_user_text(messages_blob):
    msgs = parse_messages_blob(messages_blob)
    parts = []
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            txt = str(msg.get("content", "")).strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts)


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    ds = load_dataset("sweatSmile/medical-symptom-triage-conversational")
    train_df = ds["train"].to_pandas().copy()
    val_df = ds["validation"].to_pandas().copy()

    for frame in (train_df, val_df):
        frame["text"] = frame["messages"].map(extract_user_text)
        frame["clean_text"] = frame["text"].map(clean_text)
        frame.drop(frame[frame["clean_text"] == ""].index, inplace=True)
        frame.reset_index(drop=True, inplace=True)

    min_spec_count = 50
    counts = train_df["specialty"].value_counts()
    stable_spec = set(counts[counts >= min_spec_count].index.tolist())

    def map_spec(x):
        return x if x in stable_spec else "Other"

    train_df["specialty_merged"] = train_df["specialty"].map(map_spec)
    val_df["specialty_merged"] = val_df["specialty"].map(map_spec)
    return train_df, val_df


def _fit_specialty_model(train_df: pd.DataFrame):
    # Top-K ranking model: cosine similarity to class centroids.
    try:
        from sentence_transformers import SentenceTransformer

        emb_id = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = SentenceTransformer(emb_id)
        X_train = embedder.encode(train_df["clean_text"].tolist(), convert_to_numpy=True, show_progress_bar=False)

        le_spec = LabelEncoder()
        y_train = le_spec.fit_transform(train_df["specialty_merged"])
        class_ids = np.arange(len(le_spec.classes_))
        centroids = np.vstack([X_train[y_train == cid].mean(axis=0) for cid in class_ids])

        return {
            "kind": "minilm_centroid",
            "embedder": embedder,
            "vectorizer": None,
            "label_encoder": le_spec,
            "class_ids": class_ids,
            "centroids": centroids,
            "source_name": "MiniLM + cosine centroid",
        }
    except Exception:
        vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
        X_train_sparse = vec.fit_transform(train_df["clean_text"])

        le_spec = LabelEncoder()
        y_train = le_spec.fit_transform(train_df["specialty_merged"])
        class_ids = np.arange(len(le_spec.classes_))
        X_train = X_train_sparse.toarray()
        centroids = np.vstack([X_train[y_train == cid].mean(axis=0) for cid in class_ids])

        return {
            "kind": "tfidf_centroid",
            "embedder": None,
            "vectorizer": vec,
            "label_encoder": le_spec,
            "class_ids": class_ids,
            "centroids": centroids,
            "source_name": "TF-IDF + cosine centroid (fallback)",
        }


@st.cache_resource(show_spinner=False)
def train_models(train_df: pd.DataFrame):
    urg_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_urg = urg_vec.fit_transform(train_df["clean_text"])
    le_urg = LabelEncoder()
    y_train_urg = le_urg.fit_transform(train_df["urgency"])
    urg_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    urg_model.fit(X_train_urg, y_train_urg)

    specialty_pack = _fit_specialty_model(train_df)
    return urg_vec, urg_model, le_urg, specialty_pack


def _vectorize_query(text: str, specialty_pack: dict) -> np.ndarray:
    if specialty_pack["kind"] == "minilm_centroid":
        return specialty_pack["embedder"].encode([text], convert_to_numpy=True, show_progress_bar=False)
    return specialty_pack["vectorizer"].transform([text]).toarray()


def topk_specialty(text: str, specialty_pack: dict, top_k: int = 3) -> List[dict]:
    query_vec = _vectorize_query(text, specialty_pack)
    sims = cosine_similarity(query_vec, specialty_pack["centroids"])[0]
    top_pos = np.argsort(sims)[-top_k:][::-1]

    rows = []
    for pos in top_pos:
        cls_id = int(specialty_pack["class_ids"][pos])
        lbl = specialty_pack["label_encoder"].inverse_transform([cls_id])[0]
        rows.append({"specialty": lbl, "score": float(sims[pos])})
    return rows


def eval_specialty_ranking(val_df: pd.DataFrame, specialty_pack: dict, top_k: int = 3) -> dict:
    y_raw = val_df["specialty_merged"].astype(str).values
    le = specialty_pack["label_encoder"]

    known_mask = np.isin(y_raw, le.classes_)
    if not known_mask.any():
        return {"mcc": np.nan, "macro_f1": np.nan, "top3_accuracy": np.nan, "n_eval": 0}

    X_eval_text = val_df.loc[known_mask, "clean_text"].tolist()
    y_true = le.transform(y_raw[known_mask])

    if specialty_pack["kind"] == "minilm_centroid":
        X_eval = specialty_pack["embedder"].encode(X_eval_text, convert_to_numpy=True, show_progress_bar=False)
    else:
        X_eval = specialty_pack["vectorizer"].transform(X_eval_text).toarray()

    sims = cosine_similarity(X_eval, specialty_pack["centroids"])
    top1_pos = np.argmax(sims, axis=1)
    y_pred = specialty_pack["class_ids"][top1_pos]

    top3_pos = np.argsort(sims, axis=1)[:, -top_k:]
    top3_ids = specialty_pack["class_ids"][top3_pos]
    top3_hits = [int(y) in ids for y, ids in zip(y_true, top3_ids)]

    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_accuracy": float(np.mean(top3_hits)),
        "n_eval": int(len(y_true)),
    }


st.set_page_config(page_title="Medical Symptom Triage", layout="wide")
st.title("Medical Symptom Triage")
st.caption("Primary output: urgency class. Secondary output: Top-3 specialty suggestions.")

try:
    train_df, val_df = load_data()
    urg_vec, urg_model, le_urg, specialty_pack = train_models(train_df)
    specialty_eval = eval_specialty_ranking(val_df, specialty_pack, top_k=3)
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Train rows", len(train_df))
col2.metric("Validation rows", len(val_df))
col3.metric("Specialty source", specialty_pack["source_name"])

with st.expander("Specialty interpretation metrics", expanded=False):
    st.write("- MCC -> does strict top-1 single-label make sense?")
    st.write("- Top-3 accuracy -> does ranking formulation work?")
    st.write("- Macro F1 -> auxiliary class-balance signal.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "MCC (top-1)": specialty_eval["mcc"],
                    "Macro F1 (top-1)": specialty_eval["macro_f1"],
                    "Top-3 accuracy": specialty_eval["top3_accuracy"],
                    "Validation rows used": specialty_eval["n_eval"],
                }
            ]
        ),
        use_container_width=True,
    )

st.subheader("Patient text input")
user_input = st.text_area("Describe symptoms:", height=160)

if user_input.strip():
    cleaned = clean_text(user_input)
    X_urg = urg_vec.transform([cleaned])

    urg_pred_id = int(urg_model.predict(X_urg)[0])
    urg_pred_lbl = le_urg.inverse_transform([urg_pred_id])[0]
    urg_prob = float(np.max(urg_model.predict_proba(X_urg)[0])) if hasattr(urg_model, "predict_proba") else None

    spec_top3 = topk_specialty(cleaned, specialty_pack, top_k=3)
    best_score = spec_top3[0]["score"] if spec_top3 else -1.0

    st.subheader("Triage output")
    c1, c2 = st.columns(2)
    c1.metric("Predicted urgency (primary)", urg_pred_lbl)
    c2.metric("Urgency confidence", f"{urg_prob:.2f}" if urg_prob is not None else "n/a")

    st.subheader("Top-3 likely specialties (secondary)")
    st.dataframe(pd.DataFrame(spec_top3), use_container_width=True)

    if best_score < 0.5:
        st.warning("uncertain -> check multiple specialties")
