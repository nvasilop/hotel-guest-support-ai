"""
generate_sample_logs.py — Create fake hotel guest chat logs for analysis.

Writes sample_chat_logs.csv (next to this script) with 1,000 rows and the
columns: conversation_id, language, intent, decision, rating.

Only the Python standard library is used. random.seed(42) makes the output
identical every time the script runs.

Run:
    python data/generate_sample_logs.py
"""

import csv
import random
from pathlib import Path

# Fixed seed -> same numbers on every run.
random.seed(42)

NUM_CONVERSATIONS = 1000
OUTPUT_FILE = Path(__file__).parent / "sample_chat_logs.csv"

# Language mix: about 55% Greek, 45% English.
LANGUAGES = ["el", "en"]
LANGUAGE_WEIGHTS = [55, 45]

# Intents from backend/agent.py with their share (in %).
# The weights add up to 100. The most common ones are booking,
# check_in_checkout and amenities.
INTENT_WEIGHTS = {
    "booking": 18,
    "check_in_checkout": 17,
    "amenities": 16,
    "cancellation_refund": 9,
    "transport": 8,
    "payments": 8,
    "room_issue": 9,          # mostly clarify, urgent ones escalate
    "sensitive_private": 6,   # always escalate
    "ambiguous": 5,           # always clarify
    "out_of_scope": 4,        # always fallback
}

# Intents whose decision is always the same.
FIXED_DECISIONS = {
    "sensitive_private": "escalate",
    "ambiguous": "clarify",
    "out_of_scope": "fallback",
}

# For the other (support topic) intents: mostly "answer", sometimes "fallback"
# (when the knowledge base has no good match). Topic intents are 76% of all
# traffic, so a ~8% fallback rate gives about 70% answer / 6% fallback overall.
TOPIC_FALLBACK_RATE = 0.08

# Room issues: like agent.py, most get a clarification question ("what is
# wrong?"). Only urgent/safety ones (water leak, no power, fire) escalate.
ROOM_ISSUE_URGENT_RATE = 0.15

# Rating (1-5) weights per decision.
RATING_WEIGHTS = {
    "answer":   [2, 3, 10, 40, 45],   # mostly 4-5
    "escalate": [5, 15, 35, 35, 10],  # mostly 3-4
    "fallback": [15, 40, 35, 7, 3],   # mostly 2-3
    "clarify":  [10, 40, 38, 9, 3],   # mostly 2-3
}
RATINGS = [1, 2, 3, 4, 5]


def pick_decision(intent: str) -> str:
    """Return the decision for an intent, following the agent's logic."""
    if intent in FIXED_DECISIONS:
        return FIXED_DECISIONS[intent]
    if intent == "room_issue":
        if random.random() < ROOM_ISSUE_URGENT_RATE:
            return "escalate"
        return "clarify"
    if random.random() < TOPIC_FALLBACK_RATE:
        return "fallback"
    return "answer"


def pick_rating(decision: str) -> int:
    """Return a 1-5 rating based on the decision."""
    return random.choices(RATINGS, weights=RATING_WEIGHTS[decision])[0]


def main() -> None:
    intents = list(INTENT_WEIGHTS.keys())
    intent_weights = list(INTENT_WEIGHTS.values())

    rows = []
    for i in range(1, NUM_CONVERSATIONS + 1):
        language = random.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS)[0]
        intent = random.choices(intents, weights=intent_weights)[0]
        decision = pick_decision(intent)
        rating = pick_rating(decision)
        rows.append({
            "conversation_id": f"CONV-{i:04d}",
            "language": language,
            "intent": intent,
            "decision": decision,
            "rating": rating,
        })

    # Write the rows to the CSV file.
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conversation_id", "language", "intent", "decision", "rating"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} conversations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
