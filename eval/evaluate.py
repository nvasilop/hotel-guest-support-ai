"""
evaluate.py — Simple evaluation harness for StayFlow AI, the AegeanStay Hotels
Guest Support Copilot.

What it does (kept deliberately simple, no LLM-as-judge):
1. Loads the test cases from eval/test_cases.json.
2. Runs each user input through the real agent flow (backend/agent.py).
3. Compares the expected decision with the actual decision (pass/fail).
   It also reports whether the intent matched, as extra information.
4. Prints a short summary (total / passed / failed / pass rate).
5. Saves detailed results to eval/results.json.

Run it from the project root:
    python eval/evaluate.py
"""

import json
import os
import sys

# --- Make the backend importable and load the API key ----------------------
# This file is in eval/. The agent lives in backend/. We add backend/ to the
# import path so "from agent import run_agent" works no matter where we run from.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

# Load GEMINI_API_KEY from backend/.env so "answer" cases can run from the root.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from agent import run_agent  # noqa: E402

TEST_CASES_PATH = os.path.join(HERE, "test_cases.json")
RESULTS_PATH = os.path.join(HERE, "results.json")


def load_test_cases(path: str = TEST_CASES_PATH) -> list:
    """Load the list of test cases from the JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def run_evaluation() -> dict:
    """Run every test case and collect the results."""
    cases = load_test_cases()
    results = []
    passed = 0

    for case in cases:
        # Ask the agent what it would do for this input.
        response = run_agent(case["user_input"], case.get("language", "auto"))

        actual_decision = response["decision"]
        actual_intent = response["intent"]

        # A case passes if the decision matches what we expected.
        decision_ok = actual_decision == case["expected_decision"]
        intent_ok = actual_intent == case.get("expected_intent")

        if decision_ok:
            passed += 1

        results.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "user_input": case["user_input"],
                "expected_decision": case["expected_decision"],
                "actual_decision": actual_decision,
                "decision_pass": decision_ok,
                "expected_intent": case.get("expected_intent"),
                "actual_intent": actual_intent,
                "intent_pass": intent_ok,
                "answer": response["answer"],
            }
        )

    total = len(cases)
    failed = total - passed
    pass_rate = round((passed / total) * 100, 1) if total else 0.0
    intent_matches = sum(1 for r in results if r["intent_pass"])

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_percent": pass_rate,
            "intent_matches": intent_matches,
        },
        "results": results,
    }


def main() -> None:
    report = run_evaluation()
    summary = report["summary"]

    # Print a per-case line so failures are easy to spot.
    print("StayFlow AI — evaluation results\n")
    for r in report["results"]:
        mark = "PASS" if r["decision_pass"] else "FAIL"
        print(
            f"[{mark}] #{r['id']:>2} {r['category']:<16} "
            f"expected={r['expected_decision']:<8} actual={r['actual_decision']:<8} "
            f"(intent: {r['actual_intent']})"
        )

    # Print the overall summary.
    print("\n--- Summary ---")
    print(f"Total:     {summary['total']}")
    print(f"Passed:    {summary['passed']}")
    print(f"Failed:    {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate_percent']}%")
    print(f"Intent matches: {summary['intent_matches']}/{summary['total']}")

    # Save the detailed results for later inspection.
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
