# MindEase — AI-Based Mental Wellness Chatbot

**GLA University | B.Tech CSE Mini Project 2025-26**  
Supervised by Dr. Rahul Pradhan

---

## 📁 Project Structure

```
mental_wellness_chatbot/
├── app.py                         # Flask app (main entry point)
├── model/
│   ├── dataset.csv                # Training data
│   ├── train_model.py             # Train ML models
│   ├── sentiment_model.pkl        # (generated after training)
│   ├── tfidf_vectorizer.pkl       # (generated after training)
│   ├── stress_model.pkl           # (generated after training)
│   └── stress_vectorizer.pkl      # (generated after training)
├── chatbot/
│   ├── preprocessor.py            # Text cleaning
│   ├── stress_detector.py         # Rule-based stress detection
│   └── response_generator.py      # Empathetic response repo
├── templates/
│   └── index.html                 # Chat UI
├── static/
│   ├── css/style.css
│   └── js/chat.js
└── requirements.txt
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python model/train_model.py
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open browser
```
http://localhost:5000
```

---

## 🧠 How It Works

1. **User Input** → Raw text entered in chat
2. **Preprocessing** → Lowercase, remove punctuation, tokenize, remove stopwords
3. **TF-IDF Vectorization** → Convert text to numerical features
4. **Sentiment Classification** → Logistic Regression predicts Positive / Neutral / Negative
5. **Stress Detection** → ML model + keyword rules → Low / Medium / High
6. **Crisis Check** → Detects self-harm keywords, triggers emergency response
7. **Response Generation** → Empathetic reply from predefined repository

---

## ⚠️ Disclaimer
This chatbot is NOT a medical device or clinical tool.
It provides emotional support only. For clinical help, contact a licensed professional.
