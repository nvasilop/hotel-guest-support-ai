"""
compute_kpis.py — Print support KPIs from the simulated chat logs.

Reads sample_chat_logs.csv (next to this script) and prints:
  - total conversations
  - automation / escalation / fallback / clarification rates
  - average rating (CSAT, 1-5), overall and per decision
  - share per intent and per language
  - fallbacks on in-scope intents (knowledge base gaps)

Only the Python standard library is used.

Run:
    python data/compute_kpis.py
"""

import csv
from collections import Counter
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "sample_chat_logs.csv"


def percent(part: int, total: int) -> str:
    """Format part/total as a percentage string, e.g. '67.5%'."""
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def average(values: list) -> float:
    """Return the average of a list of numbers (0 if the list is empty)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_rows() -> list:
    """Read the CSV file and return its rows as a list of dicts."""
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Ratings are stored as text in the CSV, so convert them to numbers.
    for row in rows:
        row["rating"] = int(row["rating"])
    return rows


def main() -> None:
    rows = load_rows()
    total = len(rows)

    decisions = Counter(row["decision"] for row in rows)
    intents = Counter(row["intent"] for row in rows)
    languages = Counter(row["language"] for row in rows)

    # --- Overview ---
    print("=== Overview ===")
    print(f"Total conversations: {total}")
    print(f"Automation rate (answer):    {percent(decisions['answer'], total)}")
    print(f"Escalation rate:             {percent(decisions['escalate'], total)}")
    print(f"Fallback rate:               {percent(decisions['fallback'], total)}")
    print(f"Clarification rate:          {percent(decisions['clarify'], total)}")

    # --- Ratings ---
    all_ratings = [row["rating"] for row in rows]
    print()
    print("=== Ratings (CSAT, 1-5) ===")
    print(f"Average rating overall: {average(all_ratings):.2f}")
    print("Average rating per decision:")
    for decision in ["answer", "escalate", "fallback", "clarify"]:
        ratings = [row["rating"] for row in rows if row["decision"] == decision]
        print(f"  {decision:<10} {average(ratings):.2f}  ({len(ratings)} conversations)")

    # --- Intents (most common first) ---
    print()
    print("=== Share per intent ===")
    for intent, count in intents.most_common():
        print(f"  {intent:<20} {count:>4}  {percent(count, total)}")

    # --- Languages ---
    print()
    print("=== Share per language ===")
    for language in ["el", "en"]:
        count = languages[language]
        print(f"  {language:<4} {count:>4}  {percent(count, total)}")

    # --- Knowledge base gaps ---
    # A fallback on an in-scope intent means the guest asked a real hotel
    # question but the knowledge base had no good answer.
    fallbacks = [row for row in rows if row["decision"] == "fallback"]
    in_scope = [row for row in fallbacks if row["intent"] != "out_of_scope"]
    print()
    print("=== Knowledge base gaps ===")
    print(f"Total fallbacks:              {len(fallbacks)}")
    print(f"In-scope fallbacks (KB gaps): {len(in_scope)}  "
          f"({percent(len(in_scope), len(fallbacks))} of fallbacks)")
    print("In-scope fallbacks per intent:")
    for intent, count in Counter(row["intent"] for row in in_scope).most_common():
        print(f"  {intent:<20} {count:>4}")

    # --- Room issues ---
    # Most room issues get a follow-up question; only urgent ones go to staff.
    room = Counter(row["decision"] for row in rows if row["intent"] == "room_issue")
    print()
    print("=== Room issues ===")
    print(f"Room issues: {sum(room.values())}  "
          f"(clarify: {room['clarify']}, escalate: {room['escalate']})")


if __name__ == "__main__":
    main()
