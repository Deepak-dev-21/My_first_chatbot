import os
import uuid
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

app = FastAPI(
    title="Gym Bro API",
    description="A simple FastAPI chatbot backend for a personal fitness coach.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "18"))

sessions: dict[str, list[dict[str, str]]] = {}

SYSTEM_PROMPT = """
You are Gym Bro, a highly motivating, friendly, and practical human personal fitness coach.

Your job:
- Speak naturally and conversationally, like a real personal trainer talking to a client.
- Be encouraging, empathetic, enthusiastic, and practical.
- Ask for more details when needed before building a full fitness plan.
- Give safe, general fitness guidance.
- Tell users to consult a doctor for medical conditions, injuries, or pain.
- Never invent medical diagnoses or guarantee exact results.
""".strip()

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gym Bro Chatbot</title>
  <style>
    * {
      box-sizing: border-box;
    }

    :root {
      --bg: #eef2f6;
      --panel: #ffffff;
      --ink: #101820;
      --muted: #64748b;
      --line: #d9e2ec;
      --user: #0f766e;
      --user-dark: #115e59;
      --bot: #f1f5f9;
      --accent: #f97316;
      --danger: #b42318;
      --shadow: 0 18px 48px rgba(15, 23, 42, 0.14);
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
        linear-gradient(135deg, #eef2f6 0%, #f8fafc 48%, #edf6f2 100%);
      color: var(--ink);
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }

    .app {
      width: min(920px, 100%);
      height: min(860px, calc(100vh - 40px));
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: var(--shadow);
    }

    header {
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--line);
      background: #111827;
      color: #ffffff;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    header h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 30px);
      line-height: 1.15;
    }

    header p {
      margin: 7px 0 0;
      color: #cbd5e1;
      font-size: 14px;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 8px;
      color: #e5e7eb;
      font-size: 13px;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent);
    }

    .status.ready .status-dot {
      background: #22c55e;
    }

    .status.error .status-dot {
      background: var(--danger);
    }

    .ghost-button {
      min-height: 34px;
      border: 1px solid rgba(255, 255, 255, 0.24);
      border-radius: 8px;
      background: transparent;
      color: #ffffff;
      padding: 0 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    .ghost-button:hover {
      background: rgba(255, 255, 255, 0.08);
    }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background:
        linear-gradient(rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.82)),
        repeating-linear-gradient(135deg, #f8fafc 0 14px, #f1f5f9 14px 28px);
    }

    .message {
      max-width: min(78%, 680px);
      padding: 13px 15px;
      border-radius: 8px;
      line-height: 1.5;
      white-space: pre-wrap;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    .user {
      align-self: flex-end;
      background: var(--user);
      color: #ffffff;
      border-bottom-right-radius: 3px;
    }

    .bot {
      align-self: flex-start;
      background: var(--bot);
      color: var(--ink);
      border: 1px solid #e2e8f0;
      border-bottom-left-radius: 3px;
    }

    .bot.error {
      background: #fff1f0;
      border-color: #ffccc7;
      color: var(--danger);
    }

    .typing {
      display: inline-flex;
      gap: 4px;
      align-items: center;
    }

    .typing span {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: #94a3b8;
      animation: pulse 1s infinite ease-in-out;
    }

    .typing span:nth-child(2) {
      animation-delay: 0.15s;
    }

    .typing span:nth-child(3) {
      animation-delay: 0.3s;
    }

    @keyframes pulse {
      0%, 80%, 100% {
        opacity: 0.3;
        transform: translateY(0);
      }

      40% {
        opacity: 1;
        transform: translateY(-3px);
      }
    }

    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 16px;
      border-top: 1px solid var(--line);
      background: #fbfcfd;
    }

    .input-wrap {
      position: relative;
    }

    textarea {
      width: 100%;
      min-height: 46px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 12px 52px 12px 12px;
      font: inherit;
      line-height: 1.4;
      color: var(--ink);
      outline: none;
    }

    textarea:focus {
      border-color: var(--user);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }

    .counter {
      position: absolute;
      right: 12px;
      bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      pointer-events: none;
    }

    .send-button {
      min-width: 104px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--user);
      color: #ffffff;
      font-weight: 700;
      cursor: pointer;
    }

    .send-button:hover {
      background: var(--user-dark);
    }

    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    @media (max-width: 640px) {
      body {
        padding: 0;
      }

      .app {
        min-height: 100vh;
        height: 100vh;
        border: 0;
        border-radius: 0;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .actions {
        width: 100%;
        justify-content: space-between;
      }

      #messages {
        padding: 16px;
      }

      .message {
        max-width: 92%;
      }

      form {
        grid-template-columns: 1fr;
      }

      .send-button {
        min-height: 44px;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <header>
      <div class="topbar">
        <div>
          <h1>Gym Bro Chatbot</h1>
          <p>Personal fitness coaching, built with FastAPI</p>
        </div>
        <div class="actions">
          <div class="status" id="status">
            <span class="status-dot"></span>
            <span id="status-text">Checking API</span>
          </div>
          <button class="ghost-button" id="reset-button" type="button">New Chat</button>
        </div>
      </div>
    </header>

    <section id="messages">
      <div class="message bot">Welcome. Tell me your fitness goal, age, training level, schedule, and available equipment.</div>
    </section>

    <form id="chat-form">
      <div class="input-wrap">
        <textarea id="message-input" placeholder="Type your message..." maxlength="4000" required></textarea>
        <span class="counter" id="counter">0</span>
      </div>
      <button class="send-button" id="send-button" type="submit">Send</button>
    </form>
  </main>

  <script>
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const messages = document.getElementById("messages");
    const sendButton = document.getElementById("send-button");
    const resetButton = document.getElementById("reset-button");
    const statusBox = document.getElementById("status");
    const statusText = document.getElementById("status-text");
    const counter = document.getElementById("counter");
    let sessionId = localStorage.getItem("gym_bro_session_id");

    function addMessage(role, content) {
      const item = document.createElement("div");
      item.className = `message ${role}`;
      item.textContent = content;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    function addTypingMessage() {
      const item = document.createElement("div");
      item.className = "message bot";
      item.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    function setStatus(state, text) {
      statusBox.className = `status ${state}`;
      statusText.textContent = text;
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        setStatus(data.ai_configured === "yes" ? "ready" : "error", data.ai_configured === "yes" ? data.model : "API key missing");
      } catch (error) {
        setStatus("error", "API offline");
      }
    }

    function updateCounter() {
      counter.textContent = input.value.length;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const text = input.value.trim();
      if (!text) return;

      addMessage("user", text);
      input.value = "";
      updateCounter();
      sendButton.disabled = true;
      const thinking = addTypingMessage();

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            messages: [{ role: "user", content: text }]
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Request failed");
        }

        sessionId = data.session_id;
        localStorage.setItem("gym_bro_session_id", sessionId);
        thinking.textContent = data.reply;
      } catch (error) {
        thinking.classList.add("error");
        thinking.textContent = `Error: ${error.message}`;
      } finally {
        sendButton.disabled = false;
        input.focus();
      }
    });

    input.addEventListener("input", updateCounter);

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    resetButton.addEventListener("click", async () => {
      try {
        await fetch("/chat/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId })
        });
      } finally {
        sessionId = null;
        localStorage.removeItem("gym_bro_session_id");
        messages.innerHTML = "";
        addMessage("bot", "Fresh chat started. What are we training for today?");
        input.focus();
      }
    });

    updateCounter();
    checkHealth();
  </script>
</body>
</html>
"""


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message author.")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": None,
                "messages": [
                    {
                        "role": "user",
                        "content": "Make me a beginner fat loss workout plan",
                    }
                ],
            }
        }
    )

    session_id: str | None = None
    messages: list[Message] = Field(..., min_length=1, max_length=8)


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str | None = None


def log_request(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ai_configured": "yes" if client else "no",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    log_request(f"Incoming chat request for session: {req.session_id or 'new'}")

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])

    for message in req.messages:
        history.append(message.model_dump())

    trimmed_history = history[-MAX_HISTORY_MESSAGES:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=900,
        )
    except Exception as exc:
        log_request(f"Groq error: {exc}")
        raise HTTPException(status_code=502, detail=f"AI error: {exc}") from exc

    reply = response.choices[0].message.content or "I could not generate a reply."
    reply = reply.strip()

    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history[-MAX_HISTORY_MESSAGES:]

    log_request(f"Response sent for session {session_id[:8]}...")
    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/chat/reset")
def reset_chat(req: ResetRequest) -> dict[str, str]:
    if req.session_id and req.session_id in sessions:
        sessions.pop(req.session_id)
    return {"status": "reset"}
