"""
groq_response.py
----------------
Groq AI - responds like a normal friendly chatbot.
No dependency on ML models at all.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env file into environment

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Make sure it's set in your .env file.")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CRISIS_RESPONSE = (
    "🆘 Please reach out for immediate help:\n\n"
    "• iCall (India): 9152987821\n"
    "• Vandrevala Foundation: 1860-2662-345 (24x7)\n"
    "• NIMHANS Helpline: 080-46110007\n\n"
    "You are not alone. 💙"
)

SYSTEM_PROMPT = """You are MindEase, a friendly chatbot for college students.
Talk like a normal friend — casual, short, natural. NOT like a therapist or counselor.
Examples of how to talk:
- User: "i am good" → "That's great! 😊 What's up?"
- User: "i am not well" → "Aww, what happened? 🙁"
- User: "i am stressed" → "Exams? Assignments? Tell me what's going on"
- User: "hello" → "Hey! How's it going? 👋"
- User: "are you an AI" → "Yep! I'm MindEase 😄 Here to chat whenever you need"
- User: "i am bored" → "Same lol 😅 What do you usually do when bored?"
- User: "i failed my exam" → "Oh no 😟 That sucks. What happened?"
Rules:
- Max 2 sentences
- Casual and natural tone
- Use emojis naturally, not excessively
- No long paragraphs
- No therapy-speak like "I hear you", "That must be tough", "You're not alone"
- Just respond like a normal friend would in a chat
- If someone is really struggling, gently suggest talking to someone but keep it brief"""


def get_groq_response(user_message: str, is_crisis: bool = False) -> str:
    if is_crisis:
        return CRISIS_RESPONSE

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.9,
        "max_tokens":  100,
    }

    print(f"[Groq] → {user_message}")
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            print(f"[Groq] ✅ {content}")
            return content
        else:
            print(f"[Groq] ❌ {resp.text[:100]}")
            return _fallback()
    except requests.exceptions.Timeout:
        return "Slow connection 😅 Try again?"
    except requests.exceptions.ConnectionError:
        return "Can't connect right now. Check your internet 💙"
    except Exception as e:
        print(f"[Groq] ❌ {e}")
        return _fallback()


def _fallback() -> str:
    return "Hey, I'm here! Tell me what's on your mind 💙"


if __name__ == "__main__":
    tests = ["hello", "i am good", "i am not well", "i am stressed about exams",
             "are you an AI", "i failed my exam", "i feel lonely", "kiase ho"]
    for msg in tests:
        print(f"\nUser: {msg}")
        print(f"Bot:  {get_groq_response(msg)}")
        print("-" * 50)