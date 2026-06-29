"""
main.py — FastAPI application entry point for the StayFlow AI backend.

StayFlow AI is the AegeanStay Hotels Guest Support Copilot: a multilingual
(EL/EN) assistant focused only on hotel guest support topics (bookings, check-in
and check-out, cancellations and refunds, amenities, payments, room issues,
transport, and escalation to the hotel staff).

The chat API is wired to the agent flow (agent.py), which uses the RAG layer
(rag.py) and Gemini to produce grounded answers.
- GET  /         -> serves the simple chat UI (frontend/index.html).
- GET  /health   -> health check for hosting (Render).
- POST /chat     -> passes the guest message to run_agent() and returns its
                    structured decision (answer | clarify | fallback | escalate).
"""

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent

# Path to the frontend chat UI. It lives in frontend/, one level above backend/.
_FRONTEND_INDEX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
    "index.html",
)

app = FastAPI(
    title="StayFlow AI — AegeanStay Hotels Guest Support Copilot",
    description="Multilingual (EL/EN) guest support copilot for AegeanStay Hotels.",
)


# --- Request / response models (Pydantic validation) ------------------------


class ChatRequest(BaseModel):
    # The guest's message.
    message: str
    # Preferred language: "auto" (detect), "en", or "el". Defaults to "auto".
    language: str = "auto"


class ChatResponse(BaseModel):
    # The assistant's reply.
    answer: str
    # What the agent decided to do (answer, clarify, fallback, escalate).
    decision: str
    # The language used for the reply ("en" or "el").
    language: str
    # The detected hotel guest support intent.
    intent: str
    # Knowledge base sources used for the answer.
    sources: List[str]
    # How well the answer is grounded in sources (high | medium | low | not_checked).
    evidence_level: str
    # Summary handed to the hotel staff when escalating (None otherwise).
    escalation_summary: Optional[str] = None


# --- Routes -----------------------------------------------------------------


@app.get("/")
def root():
    # Serve the chat UI so the page and the /chat API share the same origin.
    if os.path.isfile(_FRONTEND_INDEX):
        return FileResponse(_FRONTEND_INDEX)
    # Fallback if the frontend file is missing for some reason.
    return JSONResponse(
        {"message": "StayFlow AI backend is running. Frontend file not found."}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    # Hand the guest message to the agent and return its structured result.
    return run_agent(request.message, request.language)
