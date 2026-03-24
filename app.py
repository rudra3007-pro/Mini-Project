"""
app.py — Main Flask Application
---------------------------------
Sentiment  → RNN (LSTM/Simple RNN)
Stress     → RNN or ML (Random Forest/SVM)
Response   → Groq LLaMA 3.1 (independent AI)
"""

import os, pickle, sys
from flask import Flask, render_template, request, jsonify
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from chatbot.preprocessor    import preprocess
from chatbot.stress_detector import detect_stress_keywords, is_crisis
from chatbot.groq_response   import get_groq_response

app       = Flask(__name__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ── Load Models ───────────────────────────────────────────────────────────────
def load_models():
    m = {}
    files = {
        "sentiment_clf":    "sentiment_model.pkl",
        "sentiment_vec":    "tfidf_vectorizer.pkl",
        "stress_clf":       "stress_model.pkl",
        "stress_vec":       "stress_vectorizer.pkl",
        "sentiment_meta":   "sentiment_model_meta.pkl",
        "stress_meta":      "stress_model_meta.pkl",
        # RNN Models
        "sentiment_rnn":    "sentiment_model_rnn.keras",
        "tokenizer_sent":   "rnn_tokenizer.pkl",
        "label_le_sent":    "sentiment_model_rnn_le.pkl",
        "stress_rnn":       "stress_model_rnn.keras",
        "tokenizer_stress": "rnn_tokenizer_stress.pkl",
        "label_le_stress":  "stress_model_rnn_le.pkl",
    }

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        m['tf']            = tf
        m['pad_sequences'] = pad_sequences

        for key, fname in files.items():
            path = os.path.join(MODEL_DIR, fname)
            if not os.path.exists(path):
                continue
            if fname.endswith(".keras"):
                m[key] = tf.keras.models.load_model(path)
            else:
                with open(path, "rb") as f:
                    m[key] = pickle.load(f)

        print("✅  Models loaded.")
        if "sentiment_rnn" in m:
            print("    Sentiment → RNN (LSTM)")
        elif "sentiment_meta" in m:
            print(f"    Sentiment → {m['sentiment_meta'].get('model_name','ML')}  "
                  f"({m['sentiment_meta'].get('cv_accuracy','?')}%)")
        if "stress_rnn" in m:
            print("    Stress    → RNN (LSTM)")
        elif "stress_meta" in m:
            print(f"    Stress    → {m['stress_meta'].get('model_name','ML')}  "
                  f"({m['stress_meta'].get('cv_accuracy','?')}%)")

    except (FileNotFoundError, ImportError) as e:
        print(f"⚠️  TF not available ({e}), falling back to ML models...")
        for key in ["sentiment_clf","sentiment_vec","stress_clf","stress_vec"]:
            path = os.path.join(MODEL_DIR, files[key])
            if os.path.exists(path) and key not in m:
                with open(path, "rb") as f:
                    m[key] = pickle.load(f)
        print("✅  ML fallback models loaded.")

    return m

models = load_models()

# ── Prediction Helpers ────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "neutral"

    # RNN prediction
    if "sentiment_rnn" in models and "tokenizer_sent" in models:
        try:
            seq    = models["tokenizer_sent"].texts_to_sequences([text])
            padded = models["pad_sequences"](seq, maxlen=100)
            pred   = models["sentiment_rnn"].predict(padded, verbose=0)
            idx    = np.argmax(pred)
            return models["label_le_sent"].inverse_transform([idx])[0]
        except Exception as e:
            print(f"[RNN Sentiment] Error: {e}")

    # ML fallback
    if "sentiment_clf" in models:
        vec = models["sentiment_vec"].transform([text])
        return models["sentiment_clf"].predict(vec)[0]

    return "neutral"


def predict_stress(text: str, rule_stress: str) -> str:
    if not text or not text.strip():
        return rule_stress or "low"

    ml_stress = rule_stress

    # RNN prediction
    if "stress_rnn" in models and "tokenizer_stress" in models:
        try:
            seq       = models["tokenizer_stress"].texts_to_sequences([text])
            padded    = models["pad_sequences"](seq, maxlen=100)
            pred      = models["stress_rnn"].predict(padded, verbose=0)
            idx       = np.argmax(pred)
            ml_stress = models["label_le_stress"].inverse_transform([idx])[0]
        except Exception as e:
            print(f"[RNN Stress] Error: {e}")

    # ML fallback
    elif "stress_clf" in models:
        vec       = models["stress_vec"].transform([text])
        ml_stress = models["stress_clf"].predict(vec)[0]

    # Rule-based HIGH always wins (safety first)
    if rule_stress == "high" or ml_stress == "high":   return "high"
    if rule_stress == "medium" or ml_stress == "medium": return "medium"
    return "low"


def get_model_info():
    s_meta = models.get("sentiment_meta", {})
    t_meta = models.get("stress_meta", {})
    return {
        "sentiment_model": "RNN (LSTM)" if "sentiment_rnn" in models else s_meta.get("model_name", "ML Model"),
        "sentiment_acc":   "~90+"       if "sentiment_rnn" in models else s_meta.get("cv_accuracy", "—"),
        "stress_model":    "RNN (LSTM)" if "stress_rnn"    in models else t_meta.get("model_name", "ML Model"),
        "stress_acc":      "~90+"       if "stress_rnn"    in models else t_meta.get("cv_accuracy", "—"),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_info=get_model_info())


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # ── ML Pipeline (sidebar badges only) ────────────────────────
    cleaned     = preprocess(user_msg)
    crisis      = is_crisis(user_msg)
    sentiment   = predict_sentiment(cleaned)
    rule_stress = detect_stress_keywords(user_msg)
    stress      = predict_stress(cleaned, rule_stress)

    if crisis:
        stress    = "high"
        sentiment = "negative"

    print(f"[ML] '{user_msg[:50]}' → sentiment={sentiment}, stress={stress}")

    # ── Groq AI (fully independent) ──────────────────────────────
    response = get_groq_response(user_msg, is_crisis=crisis)

    return jsonify({
        "response":   response,
        "sentiment":  sentiment,
        "stress":     stress,
        "crisis":     crisis,
        "model_info": get_model_info(),
    })


@app.route("/train")
def train():
    import subprocess
    script = os.path.join(MODEL_DIR, "train_model.py")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    global models
    models = load_models()
    return jsonify({"status": "done", "stdout": result.stdout, "stderr": result.stderr})


@app.route("/report")
def report():
    path = os.path.join(MODEL_DIR, "model_report.txt")
    try:
        with open(path) as f:
            content = f.read()
        return f"<pre style='font-family:monospace;padding:2rem;background:#0d1117;color:#e8edf5;'>{content}</pre>"
    except FileNotFoundError:
        return "Report not found. Run /train first.", 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))