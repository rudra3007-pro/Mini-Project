"""
train_rnn.py — Fixed for Keras 3.5 + TF 2.17
"""
import os, pickle, pandas as pd, numpy as np
import tensorflow as tf

# ── Keras 3 compatible imports ─────────────────────────────────────────────
try:
    # Keras 3.x standalone
    from keras.src.legacy.preprocessing.text import Tokenizer
    from keras.src.legacy.preprocessing.sequence import pad_sequences
except ImportError:
    # Fallback for older setups
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MAX_WORDS     = 5000
MAX_LEN       = 100
EMBEDDING_DIM = 64


def load_data():
    data_files = [f for f in os.listdir(BASE_DIR) if f.startswith("dataset") and f.endswith(".csv")]
    print(f"Loading data from: {data_files}")
    dfs = [pd.read_csv(os.path.join(BASE_DIR, f)) for f in data_files]
    df  = pd.concat(dfs, ignore_index=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    print(f"Total unique samples: {len(df)}")
    return df


def balance_classes(df, text_col, label_col):
    max_count = df[label_col].value_counts().max()
    parts = []
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        if len(subset) < max_count:
            subset = resample(subset, replace=True, n_samples=max_count, random_state=42)
        parts.append(subset)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def train_rnn(X, y, model_name, tokenizer_name, label_name):
    print(f"\nTraining RNN: {label_name}")

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_padded  = pad_sequences(sequences, maxlen=MAX_LEN)

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_encoded, test_size=0.2, random_state=42)

    # ── Keras 3: Embedding no longer accepts input_length ─────────────────
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM),   # <-- removed input_length
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.build(input_shape=(None, MAX_LEN))   # <-- explicit build instead
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    keras_path = os.path.join(BASE_DIR, model_name.replace(".pkl", ".keras"))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3,
        restore_best_weights=True, verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        keras_path, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    )

    model.fit(
        X_train, y_train,
        epochs=20, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    model.save(keras_path)
    print(f"Saved model → {keras_path}")

    with open(os.path.join(BASE_DIR, tokenizer_name), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(BASE_DIR, model_name.replace(".pkl", "_le.pkl")), "wb") as f:
        pickle.dump(le, f)

    print(f"Saved tokenizer → {tokenizer_name}")


def main():
    df = load_data()

    df_sent = balance_classes(df, "text", "sentiment")
    print(f"Balanced sentiment samples: {len(df_sent)}")
    train_rnn(df_sent["text"], df_sent["sentiment"],
              "sentiment_model_rnn.pkl", "rnn_tokenizer.pkl", "Sentiment")

    df_stress = balance_classes(df, "text", "stress_level")
    print(f"Balanced stress samples: {len(df_stress)}")
    train_rnn(df_stress["text"], df_stress["stress_level"],
              "stress_model_rnn.pkl", "rnn_tokenizer_stress.pkl", "Stress")


if __name__ == "__main__":
    main()