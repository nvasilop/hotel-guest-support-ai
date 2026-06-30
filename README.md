# StayFlow AI — Hotel Guest Support Copilot

A small, multilingual (Greek & English) hotel guest support AI assistant for a
fictional hotel group called **AegeanStay Hotels**.

## Live Demo

Try the hosted version of StayFlow AI here:

https://hotel-guest-support-ai.onrender.com 

## Project Overview

StayFlow AI is a focused prototype of a guest support copilot for AegeanStay
Hotels, a fictional group of seaside hotels in Greece. Guests can ask questions
in Greek or English about their stay, and the assistant answers using a small
hotel knowledge base, makes sensible agent-flow decisions, and hands off to the
hotel staff when needed.

It is intentionally small and easy to read — the goal is to demonstrate the
thinking behind a guest support assistant, not to ship a production system.

## Why I Built This

This project was built to demonstrate hands-on understanding of Conversational
AI & Agent Flow work, specifically:

- **Assistant behavior design** — tone, scope, and rules defined in a playbook.
- **RAG grounding** — answers are grounded in a hotel knowledge base, not invented.
- **Clarification logic** — asking a focused follow-up when a request is vague.
- **Fallback logic** — responding honestly when information is missing or out of scope.
- **Escalation / human handoff** — routing urgent, emotional, or sensitive issues
  to hotel staff.
- **Multilingual Greek–English support** — detecting and replying in the guest's
  language.
- **Evaluation scenarios** — a decision-based test set that checks the agent flow.

## What the Assistant Can Do

- Answer hotel guest questions from a small knowledge base (RAG + Gemini).
- Ask clarification questions when the request is ambiguous.
- Fall back gracefully when information is missing or out of scope.
- Escalate urgent, emotional, or sensitive issues to the hotel staff.
- Return useful metadata with every reply: `decision`, `intent`, `language`,
  `evidence_level`, `sources`, and an `escalation_summary` when relevant.

## Architecture

```
User (Greek/English)
   │
   ▼
Frontend (frontend/index.html)
   │  HTTP POST /chat
   ▼
FastAPI (backend/main.py)
   │
   ▼
Agent Flow (backend/agent.py)
   ├── RAG retrieval (backend/rag.py)  → relevant chunks from kb/
   ├── Playbook (backend/playbook.py)  → tone, rules, templates
   └── Gemini                          → grounded answer
   │
   ▼
Response with metadata (answer, decision, intent, language, evidence_level, sources, escalation_summary)
```

## Agent Flow

1. **Detect language** (Greek or English) and reply in the same language.
2. **Classify intent** (booking, check_in_checkout, cancellation_refund,
   amenities, payments, room_issue, transport, ambiguous, sensitive_private,
   out_of_scope).
3. **Check ambiguity** — vague messages get a clarification question.
4. **Check sensitive/private or urgent issues** — these escalate to hotel staff.
5. **Retrieve relevant hotel documents** with RAG (Gemini embeddings + cosine
   similarity over numpy).
6. **Generate a grounded answer** with Gemini using only the retrieved context.
7. **Return** the answer plus `decision`, `intent`, `language`, `evidence_level`,
   `sources`, and `escalation_summary`.

For clarify, fallback, and escalate decisions, the assistant uses ready-made
playbook templates and does not call RAG or Gemini.

## Knowledge Base

The hotel support documents live in `kb/` (each bilingual EL/EN):

- `hotel_faq.md` — general hotel FAQ (contact, pets, families, accessibility).
- `booking_policy.md` — how to book, deposits, minimum stay, group bookings.
- `checkin_checkout.md` — check-in/check-out times, late checkout, luggage.
- `cancellations_refunds.md` — cancellation windows, refunds, no-shows.
- `amenities_services.md` — breakfast, Wi-Fi, pool, spa, parking, transfers.

## Example Questions

- What time is check-in?
- Can I cancel my booking?
- Is breakfast included?
- I need help
- Το δωμάτιό μου έχει πρόβλημα
- Το δωμάτιό μου δεν έχει κλιματισμό και είμαι πολύ θυμωμένος

## Evaluation

The evaluation set in `eval/test_cases.json` contains **22 decision-based test
cases** covering factual questions (EN/EL), ambiguous requests, cancellations,
amenities, room issues, urgent/emotional issues, out-of-scope questions,
sensitive requests, and hallucination traps.

Latest result:

- **Total:** 22
- **Passed:** 22
- **Failed:** 0
- **Pass rate:** 100%
- **Intent matches:** 22/22

This is a focused prototype, **not a production-ready system**. The evaluation
checks whether the assistant makes the correct **agent-flow decision**
(answer / clarify / fallback / escalate) — it does not measure full production
quality such as answer fluency, factual completeness, or robustness at scale.

See `eval/evaluation_report.md` for a fuller write-up.

## How to Run Locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
uvicorn main:app --reload
```

Then open:

```
http://127.0.0.1:8000/
```

## How to Run Evaluation

From the project root:

```bash
python eval/evaluate.py
```

This runs all test cases through the agent, prints a summary, and saves detailed
results to `eval/results.json`.

## Security Note

Do not commit your `.env` file or API keys. The project uses environment
variables for Gemini API access. Create your local `backend/.env` from
`backend/.env.example` and add your `GEMINI_API_KEY` there — `.env` files are
already excluded via `.gitignore`.

## Limitations

- Uses **fictional** hotel data for AegeanStay Hotels.
- The **knowledge base is small** and intentionally simple.
- Intent classification is **simple and rule-based** (keyword-driven).
- There is **no real booking system / CRM integration**.
- There is **no authentication or database**.
- The evaluation is **decision-based**, not a full LLM-as-judge assessment.

## Future Improvements

- Connect to a real booking/CRM system for live reservation data.
- Add LangGraph (or similar) for more complex, multi-step flows.
- Build an admin evaluation dashboard to track results over time.
- Add real human handoff integration (e.g. ticketing or live chat).
- Expand the knowledge base with more hotels and policies.
- Improve multilingual intent detection beyond keyword rules.
