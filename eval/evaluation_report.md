# StayFlow AI — Evaluation Report

**Project:** StayFlow AI — Hotel Guest Support Copilot
**Fictional hotel group:** AegeanStay Hotels

## Evaluation Goal

The goal of this evaluation is to check whether the assistant makes the correct
**agent-flow decision** for a given guest message. Each decision is one of:

- `answer` — answer the question from the hotel knowledge base.
- `clarify` — ask a focused follow-up when the request is ambiguous.
- `fallback` — respond honestly when the information is missing or out of scope.
- `escalate` — hand off urgent, emotional, or sensitive issues to hotel staff.

This is **not** an LLM-as-judge evaluation. We simply compare the expected
decision (and intent) against the assistant's actual output. This keeps the
evaluation deterministic, transparent, and easy to explain.

## Categories Tested

The 22 test cases cover a realistic mix of guest support situations:

- Clear factual questions in **English** (e.g. "What time is check-in?")
- Clear factual questions in **Greek** (e.g. "Τι ώρα είναι η αναχώρηση;")
- **Cancellation / refund** questions
- **Amenities** questions (breakfast, pool, parking)
- **Transport** questions (airport transfer)
- **Payments** questions (deposit)
- **Ambiguous** requests in English and Greek (e.g. "I need help")
- **Room issue** requests (e.g. "Το δωμάτιό μου έχει πρόβλημα")
- **Urgent / emotional** room issues (anger, water leak, no electricity)
- **Sensitive / private** requests (e.g. another guest's card details)
- **Out-of-scope** questions (e.g. yesterday's football result)
- **Hallucination traps** (e.g. "Do you have a casino on the moon?")

## Results

| Metric | Value |
| --- | --- |
| Total cases | 22 |
| Passed | 22 |
| Failed | 0 |
| Pass rate | 100% |
| Intent matches | 22/22 |

Detailed, per-case output is saved to `eval/results.json` when the evaluation is
run via `python eval/evaluate.py`.

## What the Results Show

- The agent flow reliably distinguishes between **answering**, **clarifying**,
  **falling back**, and **escalating** across the tested scenarios.
- **Multilingual handling** works for both Greek and English inputs in the test
  set, including detecting the language and routing intent correctly.
- **Safety-oriented behavior** is working as intended: sensitive/private requests
  and urgent or emotional room issues are escalated rather than answered.
- **Grounding and scope control** hold up against out-of-scope questions and
  hallucination traps, which fall back instead of inventing hotel policies.

## Limitations of This Evaluation

- It measures **decision correctness only** — not answer fluency, factual
  completeness, tone, or robustness under heavy load.
- The test set is **small (22 cases)** and uses **fictional** hotel data.
- `answer` cases depend on live Gemini embeddings and generation, so they require
  network access and a valid API key; transient API issues could affect them.
- Intent classification is **rule-based**, so the evaluation partly reflects the
  keyword rules rather than a general language-understanding model.
- There is **no LLM-as-judge** scoring of answer quality.

## Next Improvements

- Expand the test set with more edge cases and adversarial inputs.
- Add answer-quality checks (e.g. keyword grounding or an optional LLM-as-judge).
- Track results over time in a simple dashboard.
- Test more languages and mixed-language messages.
- Add regression tests that run automatically on each change.
