import os
import random

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit.components.v1 import html

load_dotenv()

# -------------------------------
# Basic setup
# -------------------------------
st.set_page_config(page_title="Earth Observation QA", page_icon="🌍", layout="centered")
st.title("🌍 Earth Observation QA")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
SAMPLE_CSV_PATH = os.getenv("SAMPLE_CSV_PATH", "generated_questions.csv")

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

st.sidebar.subheader("Model")
st.sidebar.selectbox(
    "Select model",
    options=["mistral-small-2503"],
    index=0,
    help="Other models are disabled in this demo.",
)

st.sidebar.subheader("Datasources")
kg = st.sidebar.checkbox(
    "Knowledge Graph",
    value=True,
    help="EO publications and datasets",
)
st.sidebar.checkbox(
    "Webdata (disabled)",
    value=False,
    disabled=True,
)
datasource = "KG" if kg else "None"

# -------------------------------
# Stateless info + clickable examples
# -------------------------------
st.markdown("> **Note:** This bot is **stateless**")

def load_samples(path: str):
    try:
        df = pd.read_csv(path)
        if "question" in df.columns:
            return df["question"].dropna().tolist()
    except Exception:
        pass
    # fallback examples
    return [
        "What satellites observe vegetation indices?",
        "How can I access Sentinel-2 imagery for land cover mapping?",
        "What are the main datasets used for climate monitoring?",
        "How do I find open EO datasets?",
        "What is the difference between L1C and L2A products?",
    ]


samples = load_samples(SAMPLE_CSV_PATH)
examples = random.sample(samples, min(3, len(samples)))

st.markdown("###### Example Questions")
cols = st.columns(len(examples))
# --- store click in session state ---


for col, q in zip(cols, examples):
    with col:
        html(
            f"""
            <div style="text-align:center;">
                <p
                    
                    style="
                        width: 100%;
                        background-color: #f0f2f6;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        padding: 8px;
                        cursor: pointer;
                        font-size: 0.9rem;">
                    {q}
                </p>
            </div>
            """,
            height=70,  # 👈 still valid here
        )
# -------------------------------
# QA section
# -------------------------------
st.subheader("🤖 QA bot")
# --- Text area uses session state value ---
question = st.text_area(
    "",
    key="question_input",
    placeholder="Type your EO question here...",)

ask = st.button("Ask")

def ask_backend(q: str, model: str = "mistral-small-2503", ds: str = "KG") -> str:
    try:
        payload = {"model": model, "question": q, "datasource": ds}
        r = requests.post(f"{BACKEND_URL}/ask_question", json=payload, timeout=400)
        if r.status_code == 200:
            return r.json().get("answer", "No answer field in response.")
        return f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Request failed: {e}"

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("☕ Thinking... Grab a coffee while I work on this!"):
            answer = ask_backend(question.strip(), "mistral-small-2503", datasource)
        st.markdown("### 🧠 Answer")
        st.write(answer)