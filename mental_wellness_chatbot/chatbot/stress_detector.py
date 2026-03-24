"""
stress_detector.py
------------------
Rule-based stress level detector using keyword matching.
Acts as a secondary signal alongside the ML stress model.
"""

HIGH_KEYWORDS = [
    "suicidal","suicide","kill myself","end my life","harm myself","self harm",
    "no point","want to die","hopeless","worthless","panic attack","can't breathe",
    "unbearable","trapped","give up","disappear","never come back","depressed",
    "breakdown","crying every night","burnout","burned out","failing everything"
]

MEDIUM_KEYWORDS = [
    "stressed","stress","anxious","anxiety","worried","overwhelmed","pressure",
    "deadline","assignment","exam","test","nervous","scared","afraid","failing",
    "behind","struggle","struggling","exhausted","tired","sleepless","can't sleep",
    "procrastinating","unmotivated","frustrated","lonely","disconnected","sad"
]

LOW_KEYWORDS = [
    "okay","fine","alright","normal","decent","average","calm","regular","manage"
]

def detect_stress_keywords(text: str) -> str:
    """
    Returns 'high', 'medium', or 'low' based on keyword presence.
    High-risk keywords take priority.
    """
    text_lower = text.lower()

    for kw in HIGH_KEYWORDS:
        if kw in text_lower:
            return "high"

    medium_count = sum(1 for kw in MEDIUM_KEYWORDS if kw in text_lower)
    if medium_count >= 1:
        return "medium"

    return "low"

def is_crisis(text: str) -> bool:
    """Returns True if the text contains crisis/self-harm indicators."""
    crisis_keywords = [
        "suicidal","suicide","kill myself","end my life","harm myself",
        "self harm","want to die","no reason to live","disappear forever"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in crisis_keywords)
