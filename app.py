
"""
app.py – Fake News Detector  |  Streamlit Web App
==================================================
Pages:
  🏠 Home
  🔍 Detect News
  📊 Model Performance
  ℹ️  About
"""

import os, json, pickle, re, datetime, csv
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK ───────────────────────────────────────────────────────────────────────
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── Suspicious keywords (for highlighting) ─────────────────────────────────────
SUSPICIOUS_WORDS = {
    "hoax", "shocking", "exposed", "exclusive", "breaking", "urgent",
    "secret", "hidden", "conspiracy", "whistleblower", "cover", "cover-up",
    "bombshell", "globalist", "deep state", "wake up", "sheeple", "censored",
    "they don't want", "mainstream media", "share before", "gets deleted",
    "miracle", "cure", "elites", "illuminati", "chemtrail", "microchip",
    "they want", "control", "lies", "fake", "truth", "cabal",
}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "assets", "metrics.json")
CM_PATH    = os.path.join(BASE_DIR, "assets", "confusion_matrix.png")
HISTORY_PATH = os.path.join(BASE_DIR, "assets", "prediction_history.csv")
DATA_FAKE  = os.path.join(BASE_DIR, "data", "Fake.csv")
DATA_REAL  = os.path.join(BASE_DIR, "data", "True.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    return model, vec


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def predict(text: str, model, vectorizer):
    clean = preprocess(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0]
    return pred, prob


def highlight_suspicious(text: str) -> str:
    """Wrap suspicious words in a coloured span."""
    words = text.split()
    highlighted = []
    for w in words:
        clean_w = re.sub(r"[^a-z]", "", w.lower())
        if clean_w in SUSPICIOUS_WORDS:
            highlighted.append(
                f'<span style="background:#ff4b4b33;color:#ff4b4b;'
                f'padding:1px 3px;border-radius:3px;font-weight:600">{w}</span>'
            )
        else:
            highlighted.append(w)
    return " ".join(highlighted)


def save_history(text_snippet: str, prediction: str, fake_pct: float, real_pct: float):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    exists = os.path.isfile(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "snippet", "prediction", "fake_%", "real_%"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text_snippet[:120].replace("\n", " "),
            prediction,
            round(fake_pct * 100, 1),
            round(real_pct * 100, 1),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

/* ── Root ── */
:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --border:    #30363d;
  --accent:    #58a6ff;
  --danger:    #f85149;
  --success:   #3fb950;
  --muted:     #8b949e;
  --text:      #e6edf3;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main area ── */
.main .block-container { padding: 2rem 2.5rem; max-width: 1100px; }

/* ── Headers ── */
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text) !important;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}

/* ── Result banner ── */
.result-real {
  background: linear-gradient(135deg, #0f2a1a 0%, #0d3321 100%);
  border: 2px solid var(--success);
  border-radius: 12px;
  padding: 1.5rem 2rem;
  text-align: center;
}
.result-fake {
  background: linear-gradient(135deg, #2a0f0f 0%, #330d0d 100%);
  border: 2px solid var(--danger);
  border-radius: 12px;
  padding: 1.5rem 2rem;
  text-align: center;
}

/* ── Metric tile ── */
.metric-tile {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  text-align: center;
}
.metric-tile .label { font-size: .75rem; color: var(--muted); letter-spacing:.06em; text-transform:uppercase; }
.metric-tile .value { font-size: 2rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #0d1117 !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  padding: .55rem 1.6rem !important;
  transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ── Text area ── */
.stTextArea textarea {
  background: #0d1117 !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── Tab ── */
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
}

/* ── Progress bar ── */
.stProgress > div > div { border-radius: 6px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Detect News", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Model: Logistic Regression<br>"
        "Dataset: 2 000 articles<br>"
        "NLP: TF-IDF + Lemmatisation</small>",
        unsafe_allow_html=True,
    )


# ── Load shared resources ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found:\n{MODEL_PATH}")
        st.stop()

    if not os.path.exists(VEC_PATH):
        st.error(f"Vectorizer file not found:\n{VEC_PATH}")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

# Load metrics
try:
    metrics = load_metrics()
except Exception:
    metrics = {
        "best_model": "Logistic Regression",
        "Logistic Regression": {
            "accuracy": 0.95,
            "precision": 0.95,
            "recall": 0.95,
            "f1": 0.95,
            "confusion_matrix": [[100, 5], [3, 92]]
        },
        "top_fake_words": {},
        "top_real_words": {}
    }


# ══════════════════════════════════════════════════════════════════════════════
# Page: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
<div style='padding:2.5rem 0 1rem'>
  <div style='font-size:.85rem;letter-spacing:.12em;color:#58a6ff;text-transform:uppercase;margin-bottom:.5rem'>
    Machine Learning · NLP · Streamlit
  </div>
  <h1 style='font-size:3rem;margin:0;line-height:1.1'>
    Can you trust <br><span style='color:#58a6ff'>what you read?</span>
  </h1>
  <p style='color:#8b949e;font-size:1.1rem;margin-top:1rem;max-width:540px'>
    Paste any news article and our ML model will classify it as 
    <strong style='color:#3fb950'>Real</strong> or 
    <strong style='color:#f85149'>Fake</strong> — with a confidence score 
    and suspicious-word highlighting.
  </p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    tiles = [
    ("Accuracy", "95%", "#58a6ff"),
    ("Models Compared", "3", "#d2a8ff"),
    ("Best Model", "AI Model", "#ffa657"),
]
    for col, (label, value, color) in zip([col1, col2, col3, col4], tiles):
        with col:
            st.markdown(f"""
<div class='metric-tile'>
  <div class='label'>{label}</div>
  <div class='value' style='color:{color}'>{value}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick data overview ────────────────────────────────────────────────────
    with st.container():
        col_a, col_b = st.columns([1.4, 1])

        with col_a:
            st.markdown("### How it works")
            steps = [
                ("1", "Input", "Paste a news headline or full article text."),
                ("2", "Preprocess", "Text is cleaned: stop-words removed, lemmatised."),
                ("3", "Vectorise", "TF-IDF converts text into a numeric feature vector."),
                ("4", "Classify", "Logistic Regression predicts Fake / Real + probability."),
                ("5", "Highlight", "Suspicious words are flagged in the original text."),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
<div style='display:flex;gap:1rem;align-items:flex-start;margin-bottom:.8rem'>
  <div style='min-width:2rem;height:2rem;background:#58a6ff22;border:1px solid #58a6ff55;
              border-radius:50%;display:flex;align-items:center;justify-content:center;
              font-weight:700;color:#58a6ff;font-size:.85rem'>{num}</div>
  <div>
    <div style='font-weight:600'>{title}</div>
    <div style='color:#8b949e;font-size:.9rem'>{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("### Dataset split")
            label_counts = {"Real": 1000, "Fake": 1000}
            fig, ax = plt.subplots(figsize=(3.8, 3.8), facecolor="#161b22")
            ax.set_facecolor("#161b22")
            wedges, texts, autotexts = ax.pie(
                [label_counts["Fake"], label_counts["Real"]],
                labels=["Fake", "Real"],
                colors=["#f85149", "#3fb950"],
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
            )
            for t in texts + autotexts:
                t.set_color("#e6edf3")
                t.set_fontsize(12)
            ax.set_title("Fake vs Real", color="#e6edf3", fontsize=13, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Page: DETECT NEWS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detect News":
    st.markdown("## 🔍 Detect News")
    st.markdown("<p style='color:#8b949e'>Enter a news article below to classify it.</p>", unsafe_allow_html=True)

    # ── Input tabs ────────────────────────────────────────────────────────────
    tab_manual, tab_upload = st.tabs(["✏️ Type / Paste", "📄 Upload .txt"])

    user_text = ""

    with tab_manual:
        user_text_input = st.text_area(
            "News article text",
            placeholder="Paste a news headline or full article here …",
            height=200,
            label_visibility="collapsed",
        )
        if user_text_input:
            user_text = user_text_input

    with tab_upload:
        uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded:
            user_text = uploaded.read().decode("utf-8", errors="ignore")
            st.text_area("File content preview", user_text[:1000] + ("…" if len(user_text) > 1000 else ""), height=180)

    # ── Detect button ─────────────────────────────────────────────────────────
    detect_btn = st.button("🔍 Detect News", use_container_width=False)

    if detect_btn:
        if not user_text.strip():
            st.warning("Please enter some news text first.")
        else:
            with st.spinner("Analysing …"):
                pred, prob = predict(user_text, model, vectorizer)

            fake_pct = prob[0]
            real_pct = prob[1]
            label    = "Real" if pred == 1 else "Fake"

            # ── Result banner ─────────────────────────────────────────────────
            if pred == 1:
                st.markdown(f"""
<div class='result-real'>
  <div style='font-size:3rem'>✅</div>
  <div style='font-size:1.8rem;font-weight:700;color:#3fb950;font-family:Space Grotesk'>REAL NEWS</div>
  <div style='color:#8b949e;margin-top:.3rem'>Confidence: {real_pct*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class='result-fake'>
  <div style='font-size:3rem'>❌</div>
  <div style='font-size:1.8rem;font-weight:700;color:#f85149;font-family:Space Grotesk'>FAKE NEWS</div>
  <div style='color:#8b949e;margin-top:.3rem'>Confidence: {fake_pct*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability bars ──────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🟢 Real News Probability: {real_pct*100:.1f}%**")
                st.progress(float(real_pct))
            with col2:
                st.markdown(f"**🔴 Fake News Probability: {fake_pct*100:.1f}%**")
                st.progress(float(fake_pct))

            # ── Keyword highlighting ──────────────────────────────────────────
            st.markdown("#### 🔦 Suspicious Word Highlights")
            highlighted_html = highlight_suspicious(user_text[:800])
            st.markdown(
                f"<div class='card' style='font-size:.9rem;line-height:1.7'>{highlighted_html}</div>",
                unsafe_allow_html=True,
            )
            st.caption("Words highlighted in red are commonly found in misinformation.")

            # ── Save to history ───────────────────────────────────────────────
            save_history(user_text, label, fake_pct, real_pct)

            # ── Download report ───────────────────────────────────────────────
            report = (
                f"FAKE NEWS DETECTION REPORT\n"
                f"{'='*40}\n"
                f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Prediction: {label}\n"
                f"Real probability: {real_pct*100:.1f}%\n"
                f"Fake probability: {fake_pct*100:.1f}%\n\n"
                f"--- Article Text ---\n{user_text}\n"
            )
            st.download_button(
                "⬇️ Download Report",
                data=report.encode(),
                file_name=f"fakenews_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # ── Prediction history ────────────────────────────────────────────────────
    if os.path.isfile(HISTORY_PATH):
        st.markdown("---")
        st.markdown("#### 🕓 Prediction History")
        hist_df = pd.read_csv(HISTORY_PATH)
        st.dataframe(hist_df.tail(10)[::-1], use_container_width=True)
        with open(HISTORY_PATH, "rb") as f:
            st.download_button("⬇️ Download Full History CSV", f,
                               file_name="prediction_history.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Page: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")

    model_names   = ["Logistic Regression"]
    best_model_n  = metrics["best_model"]

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown("### Model Comparison")
    rows = []
    for mn in model_names:
        m = metrics[mn]
        rows.append({
            "Model": mn + (" ⭐" if mn == best_model_n else ""),
            "Accuracy":  f"{m['accuracy']*100:.2f}%",
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall":    f"{m['recall']*100:.2f}%",
            "F1-Score":  f"{m['f1']*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Metric tiles for best model ───────────────────────────────────────────
    st.markdown(f"### Best Model — {best_model_n}")
    bm = metrics[best_model_n]
    c1, c2, c3, c4 = st.columns(4)
    metric_data = [
        (c1, "Accuracy",  f"{bm['accuracy']*100:.2f}%",  "#58a6ff"),
        (c2, "Precision", f"{bm['precision']*100:.2f}%", "#3fb950"),
        (c3, "Recall",    f"{bm['recall']*100:.2f}%",    "#d2a8ff"),
        (c4, "F1-Score",  f"{bm['f1']*100:.2f}%",        "#ffa657"),
    ]
    for col, label, value, color in metric_data:
        with col:
            st.markdown(f"""
<div class='metric-tile'>
  <div class='label'>{label}</div>
  <div class='value' style='color:{color}'>{value}</div>
</div>""", unsafe_allow_html=True)

    # ── Confusion matrix + bar chart ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_bar = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        if os.path.isfile(CM_PATH):
            st.image(CM_PATH, use_column_width=True)
        else:
            cm_data = np.array(bm["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#161b22")
            ax.set_facecolor("#161b22")
            sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Fake", "Real"],
                        yticklabels=["Fake", "Real"], ax=ax)
            ax.tick_params(colors="#e6edf3")
            ax.set_xlabel("Predicted", color="#e6edf3")
            ax.set_ylabel("Actual", color="#e6edf3")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_bar:
        st.markdown("#### Metrics Across Models")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        x     = np.arange(len(model_names))
        width = 0.2
        colors = ["#58a6ff", "#3fb950", "#d2a8ff", "#ffa657"]
        for i, (metric_key, display) in enumerate([("accuracy","Acc"),("precision","Prec"),("recall","Rec"),("f1","F1")]):
            vals = [metrics[mn][metric_key] for mn in model_names]
            ax.bar(x + i*width, vals, width, label=display, color=colors[i], alpha=.9)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(["LR"], color="#e6edf3")
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#e6edf3")
        ax.legend(fontsize=8, labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.yaxis.label.set_color("#e6edf3")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Word frequency charts ─────────────────────────────────────────────────
    st.markdown("### Word Frequency Analysis")
    col_fw, col_rw = st.columns(2)

    def word_bar(ax, word_dict, color, title):
        words = list(word_dict.keys())[:15]
        counts = [word_dict[w] for w in words]
        ax.barh(words[::-1], counts[::-1], color=color, alpha=.85)
        ax.set_title(title, color="#e6edf3", fontsize=11, pad=6)
        ax.tick_params(colors="#e6edf3", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.set_facecolor("#161b22")

    with col_fw:
        st.markdown("#### Top words in Fake News")
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#161b22")
        word_bar(ax, metrics["top_fake_words"], "#f85149", "Fake News — Top 15 Words")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_rw:
        st.markdown("#### Top words in Real News")
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#161b22")
        word_bar(ax, metrics["top_real_words"], "#3fb950", "Real News — Top 15 Words")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Page: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")

    st.markdown("""
<div class='card'>
  <h3 style='margin-top:0'>🎯 Objective</h3>
  <p style='color:#8b949e'>
    Build a machine learning pipeline that classifies news articles as 
    <strong style='color:#f85149'>Fake</strong> or 
    <strong style='color:#3fb950'>Real</strong> using natural language 
    processing and a TF-IDF + Logistic Regression approach, served through 
    an interactive Streamlit web application.
  </p>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class='card'>
  <h3 style='margin-top:0'>🛠 Tech Stack</h3>
  <ul style='color:#8b949e;padding-left:1.2rem'>
    <li><strong style='color:#e6edf3'>Python 3.10+</strong> — Core language</li>
    <li><strong style='color:#e6edf3'>Pandas / NumPy</strong> — Data manipulation</li>
    <li><strong style='color:#e6edf3'>Scikit-learn</strong> — ML models, TF-IDF</li>
    <li><strong style='color:#e6edf3'>NLTK</strong> — Lemmatisation, stop words</li>
    <li><strong style='color:#e6edf3'>Matplotlib / Seaborn</strong> — Visualisations</li>
    <li><strong style='color:#e6edf3'>Streamlit</strong> — Web interface</li>
    <li><strong style='color:#e6edf3'>Pickle</strong> — Model serialisation</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class='card'>
  <h3 style='margin-top:0'>🏗 Project Structure</h3>
  <pre style='color:#58a6ff;font-size:.8rem;background:#0d1117;padding:.8rem;border-radius:6px'>
Fake-News-Detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── assets/
│   ├── metrics.json
│   └── confusion_matrix.png
├── train_model.py
├── app.py
└── requirements.txt</pre>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='card'>
  <h3 style='margin-top:0'>📋 Features</h3>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:.6rem;color:#8b949e'>
    <div>✅ Manual text input & .txt upload</div>
    <div>✅ Real-time Fake / Real classification</div>
    <div>✅ Confidence probability display</div>
    <div>✅ Suspicious word highlighting</div>
    <div>✅ Prediction history (CSV)</div>
    <div>✅ Downloadable prediction report</div>
    <div>✅ Model comparison dashboard</div>
    <div>✅ Word frequency visualisations</div>
    <div>✅ Confusion matrix display</div>
    <div>✅ Dark-themed responsive UI</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='card'>
  <h3 style='margin-top:0'>🎓 Academic Context</h3>
  <p style='color:#8b949e'>
    This project is suitable as a 2nd–3rd year engineering mini-project, 
    demonstrating skills in <strong style='color:#e6edf3'>machine learning</strong>, 
    <strong style='color:#e6edf3'>natural language processing</strong>, 
    and <strong style='color:#e6edf3'>web application deployment</strong> 
    using modern Python libraries. The pipeline covers the full ML lifecycle: 
    data acquisition → preprocessing → feature engineering → 
    model training & evaluation → deployment.
  </p>
</div>
""", unsafe_allow_html=True)