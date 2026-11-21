"""
compute_kpis.py — Compute support KPIs from the simulated chat logs.

Reads sample_chat_logs.csv (next to this script) and computes:
  - total conversations
  - automation / escalation / fallback / clarification rates
  - average rating (CSAT, 1-5), overall and per decision
  - share per intent and per language
  - fallbacks on in-scope intents (knowledge base gaps)
  - room issues: how many got clarify vs escalate

The calculation lives in compute_kpis(), which returns a plain dict. It is used
in two places:
  - this script, which prints the KPIs (python data/compute_kpis.py)
  - the backend GET /metrics endpoint, which returns them as JSON

Only the Python standard library is used.
"""

import csv
from collections import Counter
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "sample_chat_logs.csv"

DECISIONS = ["answer", "escalate", "fallback", "clarify"]


def percent(part: int, total: int) -> float:
    """Return part/total as a percentage rounded to 1 decimal, e.g. 67.5."""
    if total == 0:
        return 0.0
    return round(part / total * 100, 1)


def average(values: list) -> float:
    """Return the average of a list of numbers rounded to 2 decimals."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def load_rows(csv_path: Path = INPUT_FILE) -> list:
    """Read the CSV file and return its rows as a list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Ratings are stored as text in the CSV, so convert them to numbers.
    for row in rows:
        row["rating"] = int(row["rating"])
    return rows


def compute_kpis(csv_path: Path = INPUT_FILE) -> dict:
    """Read the chat logs and return all KPIs as a dict."""
    rows = load_rows(csv_path)
    total = len(rows)

    decisions = Counter(row["decision"] for row in rows)
    intents = Counter(row["intent"] for row in rows)
    languages = Counter(row["language"] for row in rows)

    # Average rating per decision, plus how many chats had that decision.
    rating_per_decision = {}
    for decision in DECISIONS:
        ratings = [row["rating"] for row in rows if row["decision"] == decision]
        rating_per_decision[decision] = {
            "average": average(ratings),
            "conversations": len(ratings),
        }

    # A fallback on an in-scope intent means the guest asked a real hotel
    # question but the knowledge base had no good answer.
    fallbacks = [row for row in rows if row["decision"] == "fallback"]
    in_scope = [row for row in fallbacks if row["intent"] != "out_of_scope"]
    in_scope_per_intent = Counter(row["intent"] for row in in_scope)

    # Most room issues get a follow-up question; only urgent ones go to staff.
    room = Counter(row["decision"] for row in rows if row["intent"] == "room_issue")

    return {
        "total_conversations": total,
        "rates": {
            "automation": percent(decisions["answer"], total),
            "escalation": percent(decisions["escalate"], total),
            "fallback": percent(decisions["fallback"], total),
            "clarification": percent(decisions["clarify"], total),
        },
        "csat": average([row["rating"] for row in rows]),
        "rating_per_decision": rating_per_decision,
        # Most common intent first.
        "intents": {
            intent: {"count": count, "percent": percent(count, total)}
            for intent, count in intents.most_common()
        },
        "languages": {
            language: {"count": languages[language], "percent": percent(languages[language], total)}
            for language in ["el", "en"]
        },
        "kb_gaps": {
            "total_fallbacks": len(fallbacks),
            "in_scope_fallbacks": len(in_scope),
            "in_scope_percent_of_fallbacks": percent(len(in_scope), len(fallbacks)),
            "per_intent": dict(in_scope_per_intent.most_common()),
        },
        "room_issues": {
            "total": sum(room.values()),
            "clarify": room["clarify"],
            "escalate": room["escalate"],
        },
    }


def main() -> None:
    """Print the KPIs in a readable format."""
    kpis = compute_kpis()
    rates = kpis["rates"]

    # --- Overview ---
    print("=== Overview ===")
    print(f"Total conversations: {kpis['total_conversations']}")
    print(f"Automation rate (answer):    {rates['automation']:.1f}%")
    print(f"Escalation rate:             {rates['escalation']:.1f}%")
    print(f"Fallback rate:               {rates['fallback']:.1f}%")
    print(f"Clarification rate:          {rates['clarification']:.1f}%")

    # --- Ratings ---
    print()
    print("=== Ratings (CSAT, 1-5) ===")
    print(f"Average rating overall: {kpis['csat']:.2f}")
    print("Average rating per decision:")
    for decision, info in kpis["rating_per_decision"].items():
        print(f"  {decision:<10} {info['average']:.2f}  ({info['conversations']} conversations)")

    # --- Intents (most common first) ---
    print()
    print("=== Share per intent ===")
    for intent, info in kpis["intents"].items():
        print(f"  {intent:<20} {info['count']:>4}  {info['percent']:.1f}%")

    # --- Languages ---
    print()
    print("=== Share per language ===")
    for language, info in kpis["languages"].items():
        print(f"  {language:<4} {info['count']:>4}  {info['percent']:.1f}%")

    # --- Knowledge base gaps ---
    gaps = kpis["kb_gaps"]
    print()
    print("=== Knowledge base gaps ===")
    print(f"Total fallbacks:              {gaps['total_fallbacks']}")
    print(f"In-scope fallbacks (KB gaps): {gaps['in_scope_fallbacks']}  "
          f"({gaps['in_scope_percent_of_fallbacks']:.1f}% of fallbacks)")
    print("In-scope fallbacks per intent:")
    for intent, count in gaps["per_intent"].items():
        print(f"  {intent:<20} {count:>4}")

    # --- Room issues ---
    room = kpis["room_issues"]
    print()
    print("=== Room issues ===")
    print(f"Room issues: {room['total']}  "
          f"(clarify: {room['clarify']}, escalate: {room['escalate']})")


if __name__ == "__main__":
    main()
