import pandas as pd
import os
import kagglehub

# ── Auto-download dataset ─────────────────────────────────────────────────────
print("Downloading Kaggle dataset...")
KAGGLE_DIR = kagglehub.dataset_download("prajwalkanade/sentiment-analysis-word-lists-dataset")
print(f"Dataset downloaded to: {KAGGLE_DIR}")

POS_FILE = os.path.join(KAGGLE_DIR, 'positive-words.txt')
NEG_FILE = os.path.join(KAGGLE_DIR, 'negative-words.txt')

def load_words(filepath):
    words = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return words
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith(';') and len(word) > 1:
                words.append(word)
    return words

pos_words = load_words(POS_FILE)
neg_words = load_words(NEG_FILE)
print(f"Loaded {len(pos_words)} positive and {len(neg_words)} negative words.")

data = []
# Positive
for word in pos_words:
    data.append((word, "positive", "low"))
    data.append((f"i feel {word}", "positive", "low"))

# Negative
for word in neg_words:
    data.append((word, "negative", "high"))
    data.append((f"i am {word}", "negative", "high"))

df = pd.DataFrame(data, columns=["text", "sentiment", "stress_level"])

# ── Universal save path (same folder as this script) ─────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(BASE_DIR, "dataset_kaggle.csv")

df.to_csv(save_path, index=False)
print(f"Kaggle dataset saved with {len(df)} samples → {save_path}")