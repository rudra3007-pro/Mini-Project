import pandas as pd
import glob
import os
import pickle

# Path to the directory containing dataset*.csv files
DATASET_DIR = 'model'
VOCAB_PATH = 'model/vocabulary.pkl'

# Find all CSV files that match 'dataset*.csv'
dataset_files = glob.glob(os.path.join(DATASET_DIR, 'dataset*.csv'))

vocab = set()

for file in dataset_files:
    print(f"Reading {file} for vocabulary...")
    df = pd.read_csv(file)
    if 'text' in df.columns:
        for text in df['text'].astype(str):
            # Simple tokenization by space and basic cleaning
            tokens = text.lower().split()
            for t in tokens:
                # Remove non-alphabetic characters
                clean_t = "".join([c for c in t if c.isalpha()])
                if clean_t:
                    vocab.add(clean_t)

with open(VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab, f)

print(f"Vocabulary built with {len(vocab)} unique words and saved to {VOCAB_PATH}")
