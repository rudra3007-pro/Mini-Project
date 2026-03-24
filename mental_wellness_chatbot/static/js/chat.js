"use strict";

const messagesEl   = document.getElementById("messages");
const inputEl      = document.getElementById("user-input");
const sendBtn      = document.getElementById("send-btn");
const typingEl     = document.getElementById("typing");
const badgeSent    = document.getElementById("badge-sentiment");
const badgeStress  = document.getElementById("badge-stress");
const clearBtn     = document.getElementById("clear-btn");

// ── Auto-resize textarea ──────────────────────────────────────────────────
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
});

// ── Send on Enter (Shift+Enter = newline) ─────────────────────────────────
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);
clearBtn.addEventListener("click", clearChat);

// ── Helpers ───────────────────────────────────────────────────────────────

function getTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}

function appendMessage(text, role, extra = {}) {
  const wrap = document.createElement("div");
  wrap.classList.add("msg", role === "user" ? "user-msg" : "bot-msg");
  if (extra.crisis) wrap.classList.add("crisis-msg");

  const avatar = document.createElement("div");
  avatar.classList.add("msg-avatar");
  avatar.textContent = role === "user" ? "👤" : "🧠";

  const bubble = document.createElement("div");
  bubble.classList.add("msg-bubble");
  bubble.innerHTML = text.replace(/\n/g, "<br/>");

  const time = document.createElement("span");
  time.classList.add("msg-time");
  time.textContent = getTime();
  bubble.appendChild(time);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function updateBadges(sentiment, stress) {
  badgeSent.textContent  = sentiment ? sentiment.charAt(0).toUpperCase() + sentiment.slice(1) : "—";
  badgeStress.textContent = stress   ? stress.charAt(0).toUpperCase() + stress.slice(1) : "—";

  badgeSent.className  = "badge " + (sentiment || "");
  badgeStress.className = "badge " + (stress    || "");
}

function setLoading(state) {
  sendBtn.disabled = state;
  typingEl.classList.toggle("hidden", !state);
  if (state) scrollToBottom();
}

// ── Main send function ────────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  appendMessage(text, "user");
  inputEl.value = "";
  inputEl.style.height = "auto";
  setLoading(true);

  try {
    const res  = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });

    if (!res.ok) throw new Error("Server error " + res.status);

    const data = await res.json();

    setLoading(false);
    appendMessage(data.response, "bot", { crisis: data.crisis });
    updateBadges(data.sentiment, data.stress);

  } catch (err) {
    setLoading(false);
    appendMessage(
      "⚠️ Sorry, I couldn't connect to the server. Please make sure Flask is running.",
      "bot"
    );
    console.error(err);
  }
}

// ── Clear chat ────────────────────────────────────────────────────────────

function clearChat() {
  messagesEl.innerHTML = "";
  updateBadges(null, null);
  appendMessage(
    "Chat cleared. 🌿 Feel free to start a new conversation. I'm here for you! 💙",
    "bot"
  );
}
