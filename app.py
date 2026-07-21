"""
Book Recommendation Chatbot — Streamlit App
Run with: streamlit run chatbot_app.py
"""

import streamlit as st
import os, re, pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
# ✅ FIX 1: Removed exit() that was killing the app at startup

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BookBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0e0c;
    color: #f0ebe1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1814;
    border-right: 1px solid #2e2b25;
}
[data-testid="stSidebar"] * { color: #c9c0af !important; }

/* ── Main area ── */
.main .block-container { padding: 2rem 3rem; max-width: 900px; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    color: #f0ebe1;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #8a7f6e;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.03em;
}
.accent { color: #c8a96e; }

/* ── Chat bubbles ── */
.bubble-user {
    background: #c8a96e;
    color: #0f0e0c;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px auto;
    max-width: 75%;
    font-size: 0.95rem;
    font-weight: 500;
    width: fit-content;
    margin-left: auto;
}
.bubble-bot {
    background: #1e1c18;
    border: 1px solid #2e2b25;
    color: #e8e0d0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 0.93rem;
    line-height: 1.65;
}
.chat-label {
    font-size: 0.72rem;
    color: #5a5248;
    margin: 3px 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.chat-label-right { text-align: right; }

/* ── Category badge ── */
.cat-badge {
    display: inline-block;
    background: #c8a96e;
    color: #0f0e0c;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.conf-text {
    font-size: 0.78rem;
    color: #6b6357;
    margin-left: 8px;
}

/* ── Recommendation table ── */
.rec-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 0.88rem;
}
.rec-table th {
    background: #2a2721;
    color: #c8a96e;
    padding: 8px 12px;
    text-align: left;
    font-weight: 500;
    letter-spacing: 0.04em;
    font-size: 0.78rem;
    text-transform: uppercase;
    border-bottom: 1px solid #3a3630;
}
.rec-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #252220;
    color: #d4ccbc;
    vertical-align: top;
}
.rec-table tr:last-child td { border-bottom: none; }
.rec-table tr:hover td { background: #1a1814; }

/* ── Chat container ── */
.chat-container {
    background: #13120f;
    border: 1px solid #2a2721;
    border-radius: 16px;
    padding: 20px;
    min-height: 350px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 1.5rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #2a2721;
    margin: 1.5rem 0;
}

/* ── Input styling override ── */
.stTextInput input {
    background: #1a1814 !important;
    border: 1px solid #3a3630 !important;
    border-radius: 10px !important;
    color: #f0ebe1 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 2px rgba(200,169,110,0.15) !important;
}

/* ── Button ── */
.stButton button {
    background: #c8a96e !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: #d4b97e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(200,169,110,0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1a1814;
    border: 1px dashed #3a3630;
    border-radius: 12px;
    padding: 10px;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #1a1814;
    border: 1px solid #2a2721;
    border-radius: 12px;
    padding: 16px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #13120f; }
::-webkit-scrollbar-thumb { background: #3a3630; border-radius: 3px; }

/* ── Status ── */
.status-thinking {
    color: #c8a96e;
    font-size: 0.85rem;
    font-style: italic;
    padding: 6px 0;
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #1e1c18 0%, #1a1814 100%);
    border: 1px solid #2e2b25;
    border-left: 3px solid #c8a96e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    color: #c9c0af;
    line-height: 1.6;
}

/* ── Image preview ── */
.img-preview {
    border-radius: 8px;
    border: 1px solid #2e2b25;
    max-width: 120px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "outputs"
IMAGES_DIR  = "images"
IMG_SIZE    = (224, 224)
MAX_LEN     = 64
CATEGORIES  = ["baby books", "cooking", "japanese", "kittens"]

STOP_WORDS  = set(stopwords.words("english"))
LEAKY_WORDS = {
    "cooking","cook","recipe","recipes","food","kitchen","chef",
    "japanese","japan","manga","anime","tokyo",
    "kitten","kittens","cat","cats","kitty",
    "baby","babies","infant","toddler","child","children"
}

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    cnn3 = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "cnn3_efficientnet.keras"),
        safe_mode=False,
        compile=False
    )

    rnn2 = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "rnn2_bigru.keras"),
        safe_mode=False,
        compile=False
    )

    le               = pickle.load(open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "rb"))
    tokenizer_keras  = pickle.load(open(os.path.join(OUTPUT_DIR, "keras_tokenizer.pkl"), "rb"))
    df               = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_books_dataset.csv"))
    class_names      = sorted(os.listdir(IMAGES_DIR))

    return cnn3, rnn2, le, tokenizer_keras, df, class_names

# ── Prediction helpers ────────────────────────────────────────────────────────
def clean_text(text):
    """Used during training only — strips leaky category words to prevent overfitting."""
    words = text.lower().split()
    words = [w for w in words if w not in LEAKY_WORDS]
    words = [w for w in words if w not in STOP_WORDS]
    words = [re.sub(r"[^a-z]", "", w) for w in words]
    words = [w for w in words if len(w) > 2]
    return " ".join(words)

def clean_text_inference(text):
    """Used at inference time — keeps category keywords so the model can use them."""
    words = text.lower().split()
    words = [w for w in words if w not in STOP_WORDS]   # stopwords only, no LEAKY_WORDS
    words = [re.sub(r"[^a-z]", "", w) for w in words]
    words = [w for w in words if len(w) > 2]
    return " ".join(words)

def predict_text(text, rnn2, tokenizer_keras, le):
    # ✅ FIX: use clean_text_inference so category keywords are NOT stripped
    clean = clean_text_inference(text)
    seq   = pad_sequences(
        tokenizer_keras.texts_to_sequences([clean]),
        maxlen=MAX_LEN, padding="post", truncating="post"
    )
    prob = rnn2.predict(seq, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return le.inverse_transform([idx])[0], float(prob[idx])

def predict_image(img_pil, cnn3, class_names):
    img  = img_pil.convert("RGB").resize(IMG_SIZE)
    arr  = np.expand_dims(np.array(img, dtype=np.float32), 0)
    prob = cnn3.predict(arr, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return class_names[idx], float(prob[idx])

# ✅ FIX 2: Completely rewrote recs_to_html — broken row-splitting logic removed
def get_recommendations(category, df, n=5):
    # ✅ FIX 5: Guard against missing/wrong column name and empty results
    if "subject" not in df.columns:
        raise ValueError(f"Expected 'subject' column in CSV, got: {list(df.columns)}")
    subset = df[df["subject"] == category].copy()
    if subset.empty:
        return pd.DataFrame(columns=["title", "authors", "rating"])
    top = subset.nlargest(n, "rating")[["title", "authors", "rating"]]
    return top.reset_index(drop=True)

def recs_to_html(recs):
    if recs.empty:
        return "<p style='color:#6b6357;font-style:italic'>No books found for this category.</p>"

    rows = ""
    for i, r in enumerate(recs.itertuples()):
        rating_val = float(r.rating) if not np.isnan(r.rating) else 0.0
        filled     = int(round(rating_val))
        stars      = "★" * filled + "☆" * (5 - filled)
        rows += (
            f"<tr>"
            f"<td style='color:#5a5248;font-size:0.8rem'>{i + 1}</td>"
            f"<td>{r.title}</td>"
            f"<td>{r.authors}</td>"
            f"<td style='color:#c8a96e;letter-spacing:1px'>{stars} {rating_val:.1f}</td>"
            f"</tr>"
        )

    return (
        '<table class="rec-table">'
        "<tr><th>#</th><th>Title</th><th>Author(s)</th><th>Rating</th></tr>"
        + rows
        + "</table>"
    )

# ── App layout ────────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 20px'>
        <div style='font-family:Playfair Display,serif;font-size:1.4rem;
                    color:#c8a96e;font-weight:700;margin-bottom:4px'>📚 BookBot</div>
        <div style='font-size:0.78rem;color:#5a5248;text-transform:uppercase;
                    letter-spacing:0.08em'>Intelligent Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem;color:#8a7f6e;margin-bottom:8px'>MODELS ACTIVE</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem;line-height:2'>
        🖼️ <b>CNN</b> — EfficientNetB0<br>
        💬 <b>RNN</b> — Bidirectional GRU<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem;color:#8a7f6e;margin-bottom:8px'>CATEGORIES</div>", unsafe_allow_html=True)
    for cat in CATEGORIES:
        icons = {"baby books": "👶", "cooking": "🍳", "japanese": "🗾", "kittens": "🐱"}
        st.markdown(f"<div style='font-size:0.88rem;padding:3px 0'>{icons.get(cat,'📖')} {cat.title()}</div>", unsafe_allow_html=True)

    st.markdown("---")
    n_recs = st.slider("Recommendations", min_value=3, max_value=10, value=5)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area
st.markdown("""
<div class='hero-title'>Book<span class='accent'>Bot</span></div>
<div class='hero-sub'>Describe what you want to read — or upload a cover to find similar books</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    try:
        cnn3, rnn2, le, tokenizer_keras, df, class_names = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Could not load models: {e}")
        models_loaded = False

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat display ──────────────────────────────────────────────────────────────
chat_html = ""

if not st.session_state.messages:
    chat_html += """
    <div class='welcome-card'>
        Welcome! I can recommend books across 4 categories:
        <b style='color:#c8a96e'>Baby Books</b>,
        <b style='color:#c8a96e'>Cooking</b>,
        <b style='color:#c8a96e'>Japanese</b>, and
        <b style='color:#c8a96e'>Kittens</b>.<br><br>
        Type something like <i>"suggest me cooking books"</i>
        or upload a book cover image below.
    </div>
    """

for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += "<div class='chat-label chat-label-right'>You</div>"
        if msg.get("is_image"):
            chat_html += f"<div class='bubble-user'>📷 {msg['filename']}</div>"
        else:
            chat_html += f"<div class='bubble-user'>{msg['content']}</div>"
    else:
        chat_html += "<div class='chat-label'>BookBot</div>"
        chat_html += f"<div class='bubble-bot'>{msg['content']}</div>"

st.markdown(f"<div class='chat-container'>{chat_html}</div>", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        label="message",
        placeholder="e.g. I want Japanese culture books...",
        label_visibility="collapsed",
        key="text_input"
    )

with col2:
    send_clicked = st.button("Send →", use_container_width=True)

st.markdown("<div style='margin:0.5rem 0;text-align:center;color:#3a3630;font-size:0.8rem'>— or upload a book cover —</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload book cover",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

# ── Handle text input ─────────────────────────────────────────────────────────
# ✅ FIX 3: Changed `send_clicked or user_input` to `send_clicked and user_input.strip()`
#            to prevent triggering on every keystroke or empty input
if (send_clicked and user_input.strip()) and models_loaded:
    text = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": text})

    try:
        cat, conf = predict_text(text, rnn2, tokenizer_keras, le)
        recs      = get_recommendations(cat, df, n=n_recs)
        reply = (
            f"<span class='cat-badge'>{cat.upper()}</span><br><br>"
            f"Here are my top picks for you:<br>"
            + recs_to_html(recs)
        )
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"

    st.session_state.messages.append({"role": "bot", "content": reply})
    st.rerun()

# ── Handle image input ────────────────────────────────────────────────────────
if uploaded_file is not None and models_loaded:
    # Avoid re-processing the same file
    if st.session_state.get("last_upload") != uploaded_file.name:
        st.session_state["last_upload"] = uploaded_file.name

        img_pil = Image.open(uploaded_file)

        # Show preview
        st.image(img_pil, width=150, caption="Uploaded cover")

        st.session_state.messages.append({
            "role": "user",
            "content": f"📷 {uploaded_file.name}",
            "is_image": True,
            "filename": uploaded_file.name
        })

        try:
            cat, conf = predict_image(img_pil, cnn3, class_names)
            recs      = get_recommendations(cat, df, n=n_recs)
            reply = (
                f"<span class='cat-badge'>{cat.upper()}</span><br><br>"
                f"Books from the same category:<br>"
                + recs_to_html(recs)
            )
        except Exception as e:
            reply = f"Sorry, something went wrong: {e}"

        st.session_state.messages.append({"role": "bot", "content": reply})
        st.rerun()

# ── Footer stats ──────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Categories", "4")
with c2: st.metric("Books", f"{len(df):,}" if models_loaded else "—")
with c3: st.metric("CNN Model", "EfficientNetB0")
with c4: st.metric("RNN Model", "BiGRU")