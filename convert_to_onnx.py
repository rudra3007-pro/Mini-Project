"""
convert_to_onnx.py — Keras 3 compatible conversion
Uses SavedModel as intermediate format since tf2onnx doesn't support Keras 3 directly.
"""
import os, sys, shutil, subprocess
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

CONVERSIONS = [
    ("sentiment_model_rnn.keras", "sentiment_model_rnn.onnx"),
    ("stress_model_rnn.keras",    "stress_model_rnn.onnx"),
]

for keras_fname, onnx_fname in CONVERSIONS:
    keras_path    = os.path.join(MODEL_DIR, keras_fname)
    onnx_path     = os.path.join(MODEL_DIR, onnx_fname)
    saved_model_dir = os.path.join(MODEL_DIR, keras_fname.replace(".keras", "_saved"))

    if not os.path.exists(keras_path):
        print(f"⚠️  Skipping {keras_fname} — file not found")
        continue

    print(f"\n🔄  Converting {keras_fname} → {onnx_fname} ...")

    try:
        import keras
        import tensorflow as tf

        # Step 1: Load Keras 3 model
        model = keras.saving.load_model(keras_path)
        print(f"    ✅ Loaded model with keras {keras.__version__}")

        # Step 2: Export as SavedModel (tf2onnx understands this format)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)

        tf.saved_model.save(model, saved_model_dir)
        print(f"    ✅ Exported to SavedModel: {saved_model_dir}")

        # Step 3: Convert SavedModel → ONNX using tf2onnx CLI
        result = subprocess.run([
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", saved_model_dir,
            "--output", onnx_path,
            "--opset", "13",
            "--inputs-as-float",   # force float32 input (not int32)
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            print(f"    ❌ tf2onnx error:\n{result.stderr}")
            continue

        # Step 4: Verify
        import onnx
        loaded = onnx.load(onnx_path)
        onnx.checker.check_model(loaded)
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"    ✅ Saved {onnx_fname}  ({size_mb:.1f} MB)")

        # Step 5: Cleanup SavedModel temp dir
        shutil.rmtree(saved_model_dir)
        print(f"    🗑  Cleaned up temp SavedModel")

    except Exception as e:
        print(f"    ❌ Failed: {e}")

print("\n✅ Done!")