"""
app.py — Main Flask Application
---------------------------------
Sentiment  → RNN (LSTM via ONNX runtime)  ← no TensorFlow, ~80 MB
Stress     → RNN (LSTM via ONNX runtime)
Response   → Groq LLaMA 3.1
"""

import os, pickle, sys, threading
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

try:
    from chatbot.preprocessor    import preprocess
    from chatbot.stress_detector import detect_stress_keywords, is_crisis
    from chatbot.groq_response   import get_groq_response
except ImportError as e:
    raise RuntimeError(
        f"[MindEase] Failed to import chatbot module: {e}\n"
        "Make sure chatbot/ package exists and dependencies are installed."
    ) from e

app       = Flask(__name__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

_models      = None
_models_lock = threading.Lock()

def load_models() -> dict:
    m = {}

    # ── ONNX Runtime ───────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        onnx_files = {
            "sentiment_rnn": "sentiment_model_rnn.onnx",
            "stress_rnn":    "stress_model_rnn.onnx",
        }
        for key, fname in onnx_files.items():
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                m[key] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                print(f"✅  Loaded ONNX model: {fname}")
            else:
                print(f"⚠️  ONNX model missing: {fname}")
    except ImportError:
        print("⚠️  onnxruntime not installed. RNN models disabled.")

    # ── Tokenizers + label encoders ────────────────────────────────────────────
    pkl_files = {
        "tokenizer_sent":   "rnn_tokenizer.pkl",
        "label_le_sent":    "sentiment_model_rnn_le.pkl",
        "tokenizer_stress": "rnn_tokenizer_stress.pkl",
        "label_le_stress":  "stress_model_rnn_le.pkl",
        "sentiment_clf":    "sentiment_model.pkl",
        "sentiment_vec":    "tfidf_vectorizer.pkl",
        "stress_clf":       "stress_model.pkl",
        "stress_vec":       "stress_vectorizer.pkl",
        "sentiment_meta":   "sentiment_model_meta.pkl",
        "stress_meta":      "stress_model_meta.pkl",
    }
    for key, fname in pkl_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    m[key] = pickle.load(f)
                print(f"✅  Loaded: {fname}")
            except Exception as e:
                print(f"❌  Failed to load {fname}: {e}")
        else:
            print(f"⚠️  Missing: {fname}")

    if not m:
        print("⚠️  No models loaded — rule-based fallback only.")
    return m


def get_models() -> dict:
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                _models = load_models()
    return _models


def reload_models() -> dict:
    global _models
    with _models_lock:
        _models = load_models()
    return _models

# ── ONNX inference helpers ────────────────────────────────────────────────────

MAX_LEN = 100

def _tokenize_and_pad(text: str, tokenizer) -> np.ndarray:
    if isinstance(tokenizer, dict):
        word_index = tokenizer["word_index"]
        num_words  = tokenizer.get("num_words", 5000)
        words = text.lower().split()
        s = [word_index[w] for w in words if w in word_index and word_index[w] < num_words]
    else:
        seq = tokenizer.texts_to_sequences([text])
        s   = seq[0] if seq else []

    if len(s) >= MAX_LEN:
        padded = s[:MAX_LEN]
    else:
        padded = [0] * (MAX_LEN - len(s)) + s
    return np.array([padded], dtype=np.float32)


def _run_onnx(session, input_array: np.ndarray) -> int:
    input_name = session.get_inputs()[0].name
    output     = session.run(None, {input_name: input_array})
    return int(np.argmax(output[0], axis=1)[0])

# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "neutral"
    m = get_models()

    if "sentiment_rnn" in m and "tokenizer_sent" in m and "label_le_sent" in m:
        try:
            arr = _tokenize_and_pad(text, m["tokenizer_sent"])
            idx = _run_onnx(m["sentiment_rnn"], arr)
            return m["label_le_sent"].inverse_transform([idx])[0]
        except Exception as e:
            print(f"[ONNX Sentiment] Error: {e} — falling back to ML")

    if "sentiment_clf" in m and "sentiment_vec" in m:
        try:
            vec = m["sentiment_vec"].transform([text])
            return m["sentiment_clf"].predict(vec)[0]
        except Exception as e:
            print(f"[ML Sentiment] Error: {e}")

    return "neutral"


def predict_stress(text: str, rule_stress: str) -> str:
    if not text or not text.strip():
        return rule_stress or "low"
    m         = get_models()
    ml_stress = rule_stress

    if "stress_rnn" in m and "tokenizer_stress" in m and "label_le_stress" in m:
        try:
            arr       = _tokenize_and_pad(text, m["tokenizer_stress"])
            idx       = _run_onnx(m["stress_rnn"], arr)
            ml_stress = m["label_le_stress"].inverse_transform([idx])[0]
        except Exception as e:
            print(f"[ONNX Stress] Error: {e} — falling back to ML")

    elif "stress_clf" in m and "stress_vec" in m:
        try:
            vec       = m["stress_vec"].transform([text])
            ml_stress = m["stress_clf"].predict(vec)[0]
        except Exception as e:
            print(f"[ML Stress] Error: {e}")

    if rule_stress == "high"   or ml_stress == "high":   return "high"
    if rule_stress == "medium" or ml_stress == "medium": return "medium"
    return "low"


def get_model_info() -> dict:
    m      = get_models()
    s_meta = m.get("sentiment_meta", {})
    t_meta = m.get("stress_meta", {})
    return {
        "sentiment_model": "RNN/LSTM (ONNX)" if "sentiment_rnn" in m else s_meta.get("model_name", "ML Model"),
        "sentiment_acc":   "~90+"            if "sentiment_rnn" in m else s_meta.get("cv_accuracy", "—"),
        "stress_model":    "RNN/LSTM (ONNX)" if "stress_rnn"    in m else t_meta.get("model_name", "ML Model"),
        "stress_acc":      "~90+"            if "stress_rnn"    in m else t_meta.get("cv_accuracy", "—"),
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

    try:
        cleaned     = preprocess(user_msg)
        crisis      = is_crisis(user_msg)
        sentiment   = predict_sentiment(cleaned)
        rule_stress = detect_stress_keywords(user_msg)
        stress      = predict_stress(cleaned, rule_stress)

        if crisis:
            stress    = "high"
            sentiment = "negative"

        print(f"[ML] '{user_msg[:50]}' → sentiment={sentiment}, stress={stress}")
        response = get_groq_response(user_msg, is_crisis=crisis)

    except Exception as e:
        print(f"⚠️  /chat error: {e}")
        return jsonify({"error": "Internal server error. Please try again."}), 500

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
    if not os.path.exists(script):
        return jsonify({"error": f"train_model.py not found at {script}"}), 404
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=300
    )
    reload_models()
    return jsonify({
        "status": "done" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    })


@app.route("/report")
def report():
    path = os.path.join(MODEL_DIR, "model_report.txt")
    try:
        with open(path) as f:
            content = f.read()
        return (
            f"<pre style='font-family:monospace;padding:2rem;"
            f"background:#0d1117;color:#e8edf5;'>{content}</pre>"
        )
    except FileNotFoundError:
        return "Report not found. Run /train first.", 404


@app.route("/health")
def health():
    m            = get_models()
    sentiment_ok = "sentiment_rnn" in m or "sentiment_clf" in m
    stress_ok    = "stress_rnn"    in m or "stress_clf"    in m
    return jsonify({
        "status": "healthy" if (sentiment_ok and stress_ok) else "degraded",
        "engine": "onnx" if "sentiment_rnn" in m else "ml-fallback",
        "models": {
            "sentiment": "onnx" if "sentiment_rnn" in m else
                         "ml"   if "sentiment_clf" in m else "missing",
            "stress":    "onnx" if "stress_rnn"    in m else
                         "ml"   if "stress_clf"    in m else "missing",
        },
    }), 200


@app.route("/debug")
def debug():
    m       = get_models()
    results = {}
    results["models_loaded"]    = list(m.keys())
    if "sentiment_rnn" in m:
        sess = m["sentiment_rnn"]
        results["onnx_input_name"] = sess.get_inputs()[0].name
        results["onnx_input_type"] = sess.get_inputs()[0].type
    if "tokenizer_sent" in m:
        tok = m["tokenizer_sent"]
        results["test_sequence"] = tok.texts_to_sequences(["I am happy"])
    results["sentiment"] = predict_sentiment("I am so happy today")
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))