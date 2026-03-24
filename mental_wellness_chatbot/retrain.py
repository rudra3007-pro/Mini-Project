"""
retrain.py — Delete old models and retrain fresh.
python retrain.py
"""
import os, subprocess, sys

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

all_files = [
    # ML files
    "sentiment_model.pkl", "tfidf_vectorizer.pkl",
    "stress_model.pkl", "stress_vectorizer.pkl",
    "sentiment_model_meta.txt", "stress_model_meta.txt",
    "sentiment_model_meta.pkl", "stress_model_meta.pkl",
    "version.txt", "model_report.txt", "vocabulary.pkl",
    # RNN PyTorch files
    "rnn_sentiment_model.pt", "rnn_sentiment_best.pt",
    "rnn_tokenizer.pkl", "rnn_label_encoder.pkl",
    "rnn_sentiment_meta.txt", "rnn_config.pkl",
    # RNN Keras files
    "rnn_sentiment_model.keras", "rnn_sentiment_best.keras",
    "sentiment_model_rnn.keras", "stress_model_rnn.keras",
    "sentiment_model_rnn_le.pkl", "stress_model_rnn_le.pkl",
    "rnn_tokenizer_stress.pkl",
    # MLP files
    "rnn_sentiment_model.pkl",
]

print("=" * 50)
print("  MindEase — Model Retraining Script")
print("=" * 50)

print("\n🗑  Deleting old model files...")
for f in all_files:
    path = os.path.join(MODEL_DIR, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"   ✅ Deleted: {f}")
    else:
        print(f"   ⏭  Skipping: {f}")

print("\n🔄 Step 1/2 — Retraining ML models (Stress - Random Forest)...")
r1 = subprocess.run([sys.executable, os.path.join(MODEL_DIR, "train_model.py")], capture_output=False)
if r1.returncode != 0:
    print("\n❌ ML Training failed.")
    sys.exit(1)

print("\n🔄 Step 2/2 — Retraining RNN model (Sentiment)...")
r2 = subprocess.run([sys.executable, os.path.join(MODEL_DIR, "train_rnn.py")], capture_output=False)
if r2.returncode != 0:
    print("\n❌ RNN Training failed.")
    sys.exit(1)

with open(os.path.join(MODEL_DIR, "version.txt"), "w") as f:
    f.write("v2\n")

print("\n" + "=" * 50)
print("  ✅ All models retrained successfully!")
print("  Now run: python app.py")
print("=" * 50)