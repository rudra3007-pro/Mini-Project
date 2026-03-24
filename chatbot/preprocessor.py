"""
preprocessor.py
---------------
Cleans and normalizes raw user text before feeding it to the ML model.
IMPORTANT: Negation words are PRESERVED to avoid flipping sentiment.
"""

import re, os, pickle, itertools

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","only","own","same","so","than","too",
    "very","s","t","can","will","just","should","now","d","ll","m","o",
    "re","ve","y","ain","ma",
    # ⚠️ NOTE: negation words deliberately EXCLUDED from stopwords:
    # not, no, nor, never, dont, wont, cant, isnt, arent, wasnt, weren't etc.
}

# Negation contractions to expand BEFORE cleaning
NEGATION_MAP = {
    "cant":     "cannot",
    "wont":     "will not",
    "dont":     "do not",
    "doesnt":   "does not",
    "didnt":    "did not",
    "isnt":     "is not",
    "arent":    "are not",
    "wasnt":    "was not",
    "werent":   "were not",
    "hasnt":    "has not",
    "havent":   "have not",
    "hadnt":    "had not",
    "wouldnt":  "would not",
    "shouldnt": "should not",
    "couldnt":  "could not",
    "can't":    "cannot",
    "won't":    "will not",
    "don't":    "do not",
    "doesn't":  "does not",
    "didn't":   "did not",
    "isn't":    "is not",
    "aren't":   "are not",
    "wasn't":   "was not",
    "weren't":  "were not",
    "hasn't":   "has not",
    "haven't":  "have not",
    "hadn't":   "had not",
    "wouldn't": "would not",
    "shouldn't":"should not",
    "couldn't": "could not",
    "i'm":      "i am",
    "i've":     "i have",
    "i'll":     "i will",
    "i'd":      "i would",
    "it's":     "it is",
}

# Load vocabulary
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(os.path.dirname(BASE_DIR), "model", "vocabulary.pkl")
VOCABULARY = set()
if os.path.exists(VOCAB_PATH):
    try:
        with open(VOCAB_PATH, "rb") as f:
            VOCABULARY = pickle.load(f)
    except Exception:
        pass


def normalize_emphatic(word: str) -> str:
    if word in VOCABULARY:
        return word
    groups = re.findall(r'((.)\2+)', word)
    if groups:
        group_chars = [g[1] for g in groups]
        template    = word
        for i, (full, char) in enumerate(groups):
            template = template.replace(full, f"{{{i}}}", 1)
        num_groups = len(group_chars)
        if num_groups <= 4:
            for combo in itertools.product([1, 2], repeat=num_groups):
                values    = [char * n for char, n in zip(group_chars, combo)]
                candidate = template.format(*values)
                if candidate in VOCABULARY:
                    return candidate
    for suffix in ['ing', 'ed', 'ly', 's']:
        if word.endswith(suffix):
            base = word[:-len(suffix)]
            if base in VOCABULARY:
                return base
    return re.sub(r'(.)\1{2,}', r'\1\1', word)


def preprocess(text: str) -> str:
    """Return cleaned text suitable for TF-IDF/RNN vectorization."""
    text = text.lower()

    # Expand negation contractions FIRST
    for contraction, expansion in NEGATION_MAP.items():
        text = text.replace(contraction, expansion)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens         = text.split()
    cleaned_tokens = []

    for t in tokens:
        t_normalized = normalize_emphatic(t)
        if t_normalized in STOPWORDS or len(t_normalized) <= 1:
            continue
        cleaned_tokens.append(t_normalized)

    return " ".join(cleaned_tokens)