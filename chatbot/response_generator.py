"""
response_generator.py
---------------------
Selects an empathetic, supportive response from a predefined repository
based on sentiment + stress level. Never provides clinical advice or diagnosis.
"""

import random

RESPONSES = {
    ("positive", "low"): [
        "That's wonderful to hear! 🌟 It sounds like things are going really well for you. Keep nurturing that positivity — you deserve every bit of it!",
        "I'm so glad you're feeling good! 😊 Your energy and optimism are truly inspiring. Keep up the great work!",
        "How lovely! It's great to see you thriving. Remember to celebrate these moments — they matter a lot.",
        "You're doing amazing! 🎉 Stay connected to what makes you feel this way, it's clearly working for you."
    ],
    ("positive", "medium"): [
        "You seem to have a positive outlook even with some challenges around. That resilience is admirable! 💪",
        "It's great that you're staying positive despite some pressure. Just remember to take breaks and breathe. You've got this!",
        "Positivity in the middle of challenges is a superpower. Keep going — you're managing really well! 🌈"
    ],
    ("positive", "high"): [
        "I can sense some struggle beneath the surface even if you're putting on a brave face. It's okay to not be okay sometimes. 💙",
        "You're trying hard to stay positive, and that's admirable. But don't forget — it's okay to ask for help when things feel heavy."
    ],
    ("neutral", "low"): [
        "Sounds like a calm, regular day. Sometimes ordinary days are the most restful. How are you taking care of yourself today? 🍃",
        "A steady day is perfectly fine! Sometimes just going through the routine is exactly what we need.",
        "It's good that you're feeling balanced. Is there anything you'd like to talk about or something you're looking forward to? 😊"
    ],
    ("neutral", "medium"): [
        "It sounds like you're managing things, but there might be some underlying pressure. Would you like to talk about what's on your mind? 🤝",
        "Sometimes feeling neutral with some stress in the background can be draining. Make sure to carve out a little time for yourself today. 🌿",
        "You seem to be holding it together. If anything feels heavy, I'm here to listen without judgment."
    ],
    ("neutral", "high"): [
        "Even if you're feeling neutral on the surface, high stress can take a toll over time. Please consider speaking to someone you trust or a counselor at your college. 💙",
        "It's okay to feel a mix of things. But please don't ignore stress signals — you deserve support and care."
    ],
    ("negative", "low"): [
        "I'm sorry you're not feeling your best. 💙 Sometimes we have off days — that's completely okay. What would make today even a little better for you?",
        "It's okay to feel a bit down sometimes. Be gentle with yourself today. Maybe try something small that brings you comfort. 🍵",
        "Tough moments pass, even when they don't feel like it. I'm here if you want to talk. You're not alone. 🌷"
    ],
    ("negative", "medium"): [
        "I hear you — things seem tough right now. 💙 You're not alone in feeling this way. Many students go through similar struggles. Have you tried talking to a friend or counselor?",
        "It's clear you're going through something difficult. Remember: it's okay to reach out for support. Your feelings are valid and you deserve care.",
        "College life can feel overwhelming sometimes. Please be kind to yourself. Small steps count — even just drinking water or taking a walk can help. 🌿",
        "Thank you for sharing how you feel. Stress and sadness are real, and they deserve attention. Is there something specific that's been bothering you? I'm here to listen."
    ],
    ("negative", "high"): [
        "I'm really concerned about you right now. 💙 What you're feeling is serious and you deserve real support. Please reach out to iCall: 9152987821 or your college counselor as soon as possible. You are not alone.",
        "What you're going through sounds incredibly hard. Please don't carry this alone. Talk to someone you trust, or contact iCall (India): 9152987821. You matter, and help is available.",
        "I'm here with you. These feelings are very intense and it's important you speak to a mental health professional. Please contact Vandrevala Foundation: 1860-2662-345 (24x7). You deserve support and care. 💙",
        "Thank you for trusting me with this. Please know: you are not alone, and what you feel can get better with the right support. Reach out to NIMHANS helpline: 080-46110007. Your life has value. 🌟"
    ],
}

CRISIS_RESPONSE = (
    "🆘 I'm very concerned about what you've shared. Please reach out for immediate help:\n\n"
    "• iCall (India): 9152987821\n"
    "• Vandrevala Foundation: 1860-2662-345 (24x7)\n"
    "• NIMHANS Helpline: 080-46110007\n"
    "• Snehi: 044-24640050\n\n"
    "You are not alone. Your life matters deeply. Please talk to someone right now. 💙"
)

# Specific intent responses
INTENT_RESPONSES = {
    "greeting": [
        "Hello! 👋 How's your day going?",
        "Hi there! I'm here if you want to talk or share anything. 😊",
        "Hey! It's good to see you. How are you feeling today?",
        "Hello! I'm MindEase, your companion. What's on your mind? 💙"
    ],
    "how_are_you": [
        "I'm doing well, thank you for asking! 🤖 Just here and ready to support you. How about you?",
        "I'm functioning perfectly and feeling ready to listen! How's your day been so far?",
        "I'm here and happy to be chatting with you! 😊 How are you feeling right now?"
    ],
    "identity": [
        "I'm MindEase, your personal mental wellness companion. 🧠 I'm designed to listen, support, and help you navigate your feelings.",
        "I'm an AI companion created to provide emotional support and wellness tips. I'm not a doctor, but I'm a great listener! 💙"
    ],
    "capabilities": [
        "I can help you track your mood, manage stress, and provide a safe space to vent or share your thoughts. 🌈",
        "I'm here to provide empathetic support, wellness tips, and a listening ear whenever you need it. 🤝"
    ]
}

def match_intent(msg: str) -> str:
    msg = msg.lower().strip()
    # Greetings
    if any(word in msg for word in ["hi", "hello", "hey", "hlo", "namaste"]):
        if len(msg.split()) < 3: # Short greetings only
            return "greeting"
    # How are you (and follow-ups)
    if any(phrase in msg for phrase in ["how are you", "how are u", "how's it going", "what about u", "how about you"]):
        return "how_are_you"
    # Identity
    if "who are you" in msg or "what is your name" in msg or "who is mindease" in msg:
        return "identity"
    # Capabilities
    if "what can you do" in msg or "how can you help" in msg or "what are you for" in msg:
        return "capabilities"
    
    return None

def get_response(user_msg: str, sentiment: str, stress_level: str, is_crisis: bool = False) -> str:
    if is_crisis:
        return CRISIS_RESPONSE

    # Check for specific intents first
    intent = match_intent(user_msg)
    if intent and intent in INTENT_RESPONSES:
        return random.choice(INTENT_RESPONSES[intent])

    key = (sentiment, stress_level)
    options = RESPONSES.get(key)

    if not options:
        # fallback
        options = RESPONSES.get((sentiment, "low"), [
            "Thank you for sharing. I'm here to listen and support you. 💙"
        ])

    return random.choice(options)
