# StayFlow AI — Hotel Guest Support Copilot

A small, multilingual (Greek & English) hotel guest support AI assistant for a
fictional hotel group called **AegeanStay Hotels**.

## Live Demo

- Hotel website (demo): https://hotel-guest-support-ai.onrender.com/hotel
- All properties: https://hotel-guest-support-ai.onrender.com/properties
- Chat with StayFlow AI: https://hotel-guest-support-ai.onrender.com/
- KPI Dashboard: https://hotel-guest-support-ai.onrender.com/dashboard

Start from the hotel website and click “Chat with our assistant” to talk to the bot. The first load can take up to a minute because the free server sleeps when nobody uses it.

## Project Overview

StayFlow AI is a focused prototype of a guest support assistant for AegeanStay
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

Besides the chat page, `frontend/` now also has `hotel.html` and `properties.html`, a small demo website for AegeanStay Hotels. I made sure their information (check-in/check-out times, amenities, cancellation policy, contact) matches the files in `kb/`.

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

## KPI Dashboard

You can see the live dashboard here: https://hotel-guest-support-ai.onrender.com/dashboard

### What I did

I created 1,000 fake guest chats and saved them in `data/sample_chat_logs.csv`.
The script `data/generate_sample_logs.py` makes this file, and the script
`data/compute_kpis.py` reads it and prints the KPIs. A KPI is just a number that
tells you if something is working well. I did this to show how a hotel could
check if the bot really helps the front desk. The server works out the KPIs at
`/metrics`, and the page at `/dashboard` shows them.

### Real numbers (from my test set)

- **Intent accuracy: 22/22.** The bot found the right topic of the question every time.
- **Correct decision: 22/22.** The bot picked the right next step every time
  (answer, ask a follow-up question, say it can't help, or pass the chat to staff).

This is a small test set that I wrote myself. It shows the logic works on these
cases. It does not prove the bot is perfect.

### Simulated numbers

These come from running `compute_kpis.py` on the 1,000 fake chats.

| KPI | What it means | Result |
| --- | --- | --- |
| Automation rate | How often the bot answered the question by itself. | 67.5% |
| Escalation rate | How often the bot passed the chat to a real person at the hotel. | 7.5% |
| Fallback rate | How often the bot said it couldn't help, because it had no info or the question was not about the hotel. | 12.6% |
| Clarification rate | How often the bot asked the guest a follow-up question first. | 12.4% |
| CSAT | The average guest rating, from 1 (bad) to 5 (great). CSAT means "customer satisfaction". | 3.75 / 5 |

### Most common guest questions

1. Booking: 18.3%
2. Check-in / check-out: 15.5%
3. Amenities (breakfast, pool, parking and so on): 15.2%
4. Room problems: 9.9%
5. Cancellations and refunds: 9.7%

### What I learned from the numbers

- **Guests are happier when the bot answers.** The average rating was 4.25 when
  the bot answered, but only 2.60 after a fallback and 2.56 after a clarification.
  → I would add more info to the knowledge base, so the bot can answer more often.
- **Many fallbacks are real hotel questions.** 69 of the 126 fallbacks were about
  normal hotel topics, not off-topic questions. That means the knowledge base has gaps.
  → I would fill the gaps for check-in/check-out (14), amenities (13) and booking (13) first.
- **Most room problems need a question first.** 85 of the 99 room problems got a
  clarification. Only the 14 urgent ones went straight to staff.
  → I would add quick buttons (AC, Wi-Fi, cleaning, noise), so the guest can pick the problem in one tap.

### How to run it yourself

```bash
python data/generate_sample_logs.py   # creates the CSV (same result every time)
python data/compute_kpis.py           # prints the KPIs
```

### Note

The simulated numbers come from fake data I made for this demo, not from real guests.

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
