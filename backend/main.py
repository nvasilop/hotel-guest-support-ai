"""
main.py — FastAPI application entry point for the StayFlow AI backend.

StayFlow AI is the AegeanStay Hotels Guest Support Copilot: a multilingual
(EL/EN) assistant focused only on hotel guest support topics (bookings, check-in
and check-out, cancellations and refunds, amenities, payments, room issues,
transport, and escalation to the hotel staff).

The chat API is wired to the agent flow (agent.py), which uses the RAG layer
(rag.py) and Gemini to produce grounded answers.
- GET  /         -> serves the simple chat UI (frontend/index.html).
- GET  /dashboard -> serves the KPI dashboard page (frontend/dashboard.html).
- GET  /health   -> health check for hosting (Render).
- POST /chat     -> passes the guest message to run_agent() and returns its
                    structured decision (answer | clarify | fallback | escalate).
- GET  /metrics  -> KPIs from the simulated chat logs (data/sample_chat_logs.csv).
"""

import os
import sys
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent

# Project root: one level above backend/. Built from this file's location, so it
# works no matter which folder the server is started from (e.g. on Render).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the frontend chat UI. It lives in frontend/, one level above backend/.
_FRONTEND_INDEX = os.path.join(_PROJECT_ROOT, "frontend", "index.html")

# Path to the KPI dashboard page (also in frontend/).
_FRONTEND_DASHBOARD = os.path.join(_PROJECT_ROOT, "frontend", "dashboard.html")

# The KPI calculation lives in data/compute_kpis.py, so the script and the
# /metrics endpoint share the same code. Add data/ to the import path to use it.
sys.path.append(os.path.join(_PROJECT_ROOT, "data"))
from compute_kpis import compute_kpis  # noqa: E402

# Path to the simulated chat logs used by /metrics.
_CHAT_LOGS_CSV = os.path.join(_PROJECT_ROOT, "data", "sample_chat_logs.csv")

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


@app.get("/dashboard")
def dashboard():
    # Serve the KPI dashboard page. It loads its numbers from /metrics.
    if os.path.isfile(_FRONTEND_DASHBOARD):
        return FileResponse(_FRONTEND_DASHBOARD)
    return JSONResponse({"error": "Dashboard file not found."}, status_code=404)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    # Hand the guest message to the agent and return its structured result.
    return run_agent(request.message, request.language)


@app.get("/metrics")
def metrics():
    # Return the KPIs from the simulated chat logs as JSON.
    if not os.path.isfile(_CHAT_LOGS_CSV):
        return JSONResponse(
            {"error": "Chat logs file not found (data/sample_chat_logs.csv)."},
            status_code=404,
        )
    return compute_kpis(_CHAT_LOGS_CSV)
