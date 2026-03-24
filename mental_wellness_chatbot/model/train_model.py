"""
train_model.py
--------------
Trains multiple ML models and auto-selects the best one:
  - Logistic Regression
  - Naive Bayes (MultinomialNB)
  - Support Vector Machine (SVM)
  - Random Forest

Both sentiment and stress level classifiers are trained this way.
Best accuracy model is saved as the active model.
"""

import os, pickle, pandas as pd, numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import MultinomialNB
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import classification_report, accuracy_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.utils            import resample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, solver="lbfgs", random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.5),
    "SVM":                 LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

def load_data():
    """Finds all dataset*.csv files in the model directory and merges them."""
    data_files = [f for f in os.listdir(BASE_DIR) if f.startswith("dataset") and f.endswith(".csv")]
    print(f"Loading data from: {data_files}")
    
    dfs = []
    for fname in data_files:
        path = os.path.join(BASE_DIR, fname)
        df = pd.read_csv(path)
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(inplace=True)
    combined_df.drop_duplicates(subset=['text'], inplace=True)
    combined_df["text"] = combined_df["text"].astype(str).str.strip()
    
    print(f"Total unique samples: {len(combined_df)}")
    return combined_df

def balance_classes(df, text_col, label_col):
    """Oversample minority classes to balance the dataset."""
    max_count = df[label_col].value_counts().max()
    parts = []
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        if len(subset) < max_count:
            subset = resample(subset, replace=True, n_samples=max_count, random_state=42)
        parts.append(subset)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

def train_best_model(X, y, model_name, vectorizer_name, label):
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"{'='*60}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
    )

    X_vec = vectorizer.fit_transform(X)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_name  = ""
    best_clf   = None
    results    = {}

    for name, clf in MODELS.items():
        try:
            scores = cross_val_score(clf, X_vec, y, cv=cv, scoring="accuracy")
            mean   = scores.mean()
            std    = scores.std()
            results[name] = mean
            print(f"  {name:<25} CV Accuracy: {mean*100:.2f}% ± {std*100:.2f}%")
            if mean > best_score:
                best_score = mean
                best_name  = name
                best_clf   = clf
        except Exception as e:
            print(f"  {name} failed: {e}")

    print(f"\n  ✅ Best Model: {best_name}  ({best_score*100:.2f}%)")

    # Final train on full data
    best_clf.fit(X_vec, y)

    # Test set evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_v = vectorizer.transform(X_train)
    X_test_v  = vectorizer.transform(X_test)
    best_clf_test = MODELS[best_name].__class__(**MODELS[best_name].get_params())
    best_clf_test.fit(X_train_v, y_train)
    y_pred = best_clf_test.predict(X_test_v)
    print(f"\n  Test Set Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save
    with open(os.path.join(BASE_DIR, model_name), "wb") as f:
        pickle.dump(best_clf, f)
    with open(os.path.join(BASE_DIR, vectorizer_name), "wb") as f:
        pickle.dump(vectorizer, f)

    # Save model name for reference
    meta_path = os.path.join(BASE_DIR, model_name.replace(".pkl", "_meta.txt"))
    with open(meta_path, "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"CV Accuracy: {best_score*100:.2f}%\n")
        for k, v in results.items():
            f.write(f"{k}: {v*100:.2f}%\n")

    print(f"  Saved → {model_name}, {vectorizer_name}")
    return best_name, best_score

def main():
    print("Loading & balancing dataset...")
    df = load_data()
    print(f"Original samples: {len(df)}")

    # Balance sentiment
    df_sent = balance_classes(df, "text", "sentiment")
    print(f"Balanced sentiment samples: {len(df_sent)}")
    print(df_sent["sentiment"].value_counts())

    # Balance stress
    df_stress = balance_classes(df, "text", "stress_level")
    print(f"\nBalanced stress samples: {len(df_stress)}")
    print(df_stress["stress_level"].value_counts())

    best_sent,   acc_sent   = train_best_model(
        df_sent["text"], df_sent["sentiment"],
        "sentiment_model.pkl", "tfidf_vectorizer.pkl", "Sentiment Classification")

    best_stress, acc_stress = train_best_model(
        df_stress["text"], df_stress["stress_level"],
        "stress_model.pkl", "stress_vectorizer.pkl", "Stress Level Detection")

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Sentiment  → {best_sent:<25} {acc_sent*100:.2f}%")
    print(f"  Stress     → {best_stress:<25} {acc_stress*100:.2f}%")
    print(f"\n✅  Training complete. Best models saved.")

if __name__ == "__main__":
    main()
